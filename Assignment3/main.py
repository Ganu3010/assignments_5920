from matplotlib.pyplot import plot

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'Models/CommaAiModel_lr3e-4.pth'

# --- 1. Dataset ---
class RoverDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.transform = transform
        
        valid_img_paths = []
        valid_indices = []

        print("Checking for missing images...")
        # Iterate through the dataframe to find which files actually exist
        for idx, row in dataframe.iterrows():
            path = row['Path']
            name = os.path.basename(path)
            
            # Check local directory first, then fallback to the full path
            actual_path = name if os.path.exists(name) else path
            
            if os.path.exists(actual_path):
                valid_img_paths.append(actual_path)
                valid_indices.append(idx)
        
        # Filter the dataframe to keep only rows where images were found
        filtered_df = dataframe.loc[valid_indices]
        
        self.img_paths = valid_img_paths
        self.labels  = filtered_df[['SteerAngle', 'Throttle']].values.astype('float32')
        self.scalars = filtered_df[['Speed', 'Yaw']].values.astype('float32')

        skipped = len(dataframe) - len(valid_indices)
        print(f"Done! Found {len(valid_indices)} images. Skipped {skipped} missing files.")

    def __len__(self):
        # This now returns the number of found images, not the original DF length
        return len(self.img_paths)

    def __getitem__(self, idx):
        # This will only be called for indices that exist
        image = Image.open(self.img_paths[idx]).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        labels  = torch.tensor(self.labels[idx],  dtype=torch.float32)
        scalars = torch.tensor(self.scalars[idx], dtype=torch.float32)
        return image, scalars, labels

# --- 2. Models ---

class PilotNet(nn.Module):
    def __init__(self, dropout_p=0.3):
        super(PilotNet, self).__init__()
        self.norm = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(3, 24, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(24, 36, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(36, 48, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, stride=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(27456, 100)
        self.drop1 = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(100, 50)
        self.drop2 = nn.Dropout(p=dropout_p)
        self.fc3 = nn.Linear(50, 10)
        self.drop3 = nn.Dropout(p=dropout_p)
        self.control_out = nn.Linear(10, 2)

    def forward(self, x, scalars=None):
        x = self.norm(x)
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = F.elu(self.conv5(x))
        x = self.flatten(x)
        x = self.drop1(F.elu(self.fc1(x)))
        x = self.drop2(F.elu(self.fc2(x)))
        x = self.drop3(F.elu(self.fc3(x)))
        return self.control_out(x)


class CommaAiModel(nn.Module):
    def __init__(self, num_scalars=2):
        super(CommaAiModel, self).__init__()

        self.norm = nn.BatchNorm2d(3)

        # Conv backbone — (160,320) -> (20,40) after 3 stride-2 layers
        self.conv1 = nn.Conv2d(3,  16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()

        # Scalar branch (Speed + Yaw)
        self.scalar_fc = nn.Linear(num_scalars, 16)

        # Fusion: 51200 (image) + 16 (scalars)
        self.fc1     = nn.Linear(51200 + 16, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2     = nn.Linear(512, 128)
        self.output  = nn.Linear(128, 2)

    def forward(self, x, scalars):
        x = self.norm(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)

        s = F.relu(self.scalar_fc(scalars))

        x = torch.cat([x, s], dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.output(x)


# --- 3. Shared DataLoader factory ---
def make_loader(df, transform, batch_size, shuffle):
    dataset = RoverDataset(df, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )


# --- 4. Evaluation helper ---
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, scalars, labels in loader:
            images  = images.to(device, non_blocking=True)
            scalars = scalars.to(device, non_blocking=True)
            labels  = labels.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                loss = criterion(model(images, scalars), labels)
            total_loss += loss.item()
    model.train()
    return total_loss / len(loader)


# --- 5. Training Loop ---
def train_model(patience=20):
    df = pd.read_csv("Training/processed_robot_log.csv").sample(frac=0.25)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    transform = transforms.Compose([
        # transforms.Resize((160, 320)),
        transforms.ToTensor(),
    ])

    train_loader = make_loader(train_df, transform, batch_size=512, shuffle=True)
    val_loader   = make_loader(val_df,   transform, batch_size=512, shuffle=False)

    model     = CommaAiModel(num_scalars=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=3e-6)
    criterion = nn.MSELoss()
    scaler    = torch.amp.GradScaler(enabled=device.type == 'cuda')

    best_val_loss     = float('inf')
    epochs_no_improve = 0
    epochs            = 100

    print(f"Starting training on {device} (patience={patience})...")

    for epoch in tqdm(range(epochs), ncols=0):
        model.train()
        running_loss = 0.0

        for images, scalars, labels in train_loader:
            images  = images.to(device, non_blocking=True)
            scalars = scalars.to(device, non_blocking=True)
            labels  = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                loss = criterion(model(images, scalars), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        scheduler.step()
        train_loss = running_loss / len(train_loader)
        val_loss   = evaluate(model, val_loader, criterion)
        current_lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {current_lr:.2e} | No improve: {epochs_no_improve}/{patience}")

        if val_loss < best_val_loss:
            best_val_loss     = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  -> New best saved (val_loss={best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n  Early stopping at epoch {epoch+1}.")
                break

    torch.save(model.state_dict(), MODEL_PATH)


# --- 6. Test ---
def test_model(weights_path=MODEL_PATH, test_csv="Testing/01/processed_robot_log.csv"):
    df = pd.read_csv(test_csv)

    transform = transforms.Compose([
        transforms.Resize((160, 320)),
        transforms.ToTensor(),
    ])

    test_loader = make_loader(df, transform, batch_size=512, shuffle=False)

    model = CommaAiModel(num_scalars=2).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    criterion = nn.MSELoss()
    test_loss = evaluate(model, test_loader, criterion)

    print(f"\nTest Loss (MSE): {test_loss:.4f}")
    print(f"Test RMSE:       {test_loss**0.5:.4f}")
    return test_loss

def plot_outputs(model, loader):
    import cv2 as cv
    import numpy as np

    model.to(device)
    model.eval()

    with torch.no_grad():
        for images, scalars, labels in loader:
            images  = images.to(device)
            scalars = scalars.to(device)

            outputs = model(images, scalars).cpu().numpy()
            labels  = labels.cpu().numpy()
            images  = images.cpu()

            for i in range(min(1000, len(outputs))):
                # (3, H, W) float [0,1]  ->  (H, W, 3) uint8 [0,255]  ->  BGR for OpenCV
                img = images[i].permute(1, 2, 0).numpy()
                img = (img * 255).clip(0, 255).astype('uint8')
                img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
                img = img.copy()  # Make contiguous for cv.putText

                pred_steer, pred_throttle = outputs[i]
                true_steer, true_throttle = labels[i]

                cv.putText(img, f"Pred: steer={pred_steer:.2f}  throttle={pred_throttle:.2f}",
                           (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                cv.putText(img, f"True: steer={true_steer:.2f}  throttle={true_throttle:.2f}",
                           (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

                cv.imwrite(f"outputs/Sample_{i+1}.jpg", img)

    cv.destroyAllWindows()

if __name__ == "__main__":
    model = CommaAiModel(num_scalars=2).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    df = pd.read_csv("Testing/01/processed_robot_log.csv")
    transform = transforms.Compose([
        transforms.Resize((160, 320)),
        transforms.ToTensor(),
    ])
    loader = make_loader(df, transform, batch_size=512, shuffle=False)
    plot_outputs(model, loader)


# if __name__ == "__main__":
#     train_model()
#     for i in ['01', '08', '14', '21', '26', '43']:
#         print(f"\n--- Evaluating on Testing/{i}/processed_robot_log.csv ---")
#         test_model(test_csv=f"Testing/{i}/processed_robot_log.csv")