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
MODEL_PATH = 'CommaAiModel.pth'
# --- 1. Dataset ---
class RoverDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.transform = transform
        self.labels = dataframe[['SteerAngle', 'Throttle']].values.astype('float32')

        # Resolve paths once upfront, not on every __getitem__ call
        self.img_paths = []
        for path in dataframe['Path']:
            name = os.path.basename(path)
            self.img_paths.append(name if os.path.exists(name) else path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = Image.open(self.img_paths[idx]).convert('RGB')
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)
        if self.transform:
            image = self.transform(image)
        return image, labels


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

    def forward(self, x):
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
    def __init__(self):
        super(CommaAiModel, self).__init__()

        self.norm = nn.BatchNorm2d(3)

        # 3x3 conv layers with stride 2 — after 3 layers on (160,320): -> (20,40)
        self.conv1 = nn.Conv2d(3,  16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)

        self.flatten = nn.Flatten()

        # 64 * 20 * 40 = 51200
        self.fc1     = nn.Linear(51200, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2     = nn.Linear(512, 128)
        self.output  = nn.Linear(128, 2)

    def forward(self, x):
        x = self.norm(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
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
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                loss = criterion(model(images), labels)
            total_loss += loss.item()
    model.train()
    return total_loss / len(loader)


# --- 5. Training Loop ---
def train_model(patience=10):
    df = pd.read_csv("Training/processed_robot_log.csv")
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    transform = transforms.Compose([
        transforms.Resize((160, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_loader = make_loader(train_df, transform, batch_size=512, shuffle=True)
    val_loader   = make_loader(val_df,   transform, batch_size=512, shuffle=False)

    model     = CommaAiModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.MSELoss()
    scaler    = torch.amp.GradScaler(enabled=device.type == 'cuda')

    best_val_loss    = float('inf')
    epochs_no_improve = 0
    epochs           = 100

    print(f"Starting training on {device} (patience={patience})...")

    for epoch in tqdm(range(epochs)):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        val_loss   = evaluate(model, val_loader, criterion)

        print(f"Epoch {epoch+1}/{epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | No improve: {epochs_no_improve}/{patience}")

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
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_loader = make_loader(df, transform, batch_size=512, shuffle=False)

    model = CommaAiModel().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    criterion = nn.MSELoss()
    test_loss = evaluate(model, test_loader, criterion)

    print(f"\nTest Loss (MSE): {test_loss:.4f}")
    print(f"Test RMSE:       {test_loss**0.5:.4f}")
    return test_loss


if __name__ == "__main__":
    train_model()
    for i in ['01', '08', '14', '21', '26', '43']:
        print(f"\n--- Testing on file {i} ---")
        test_model(test_csv=f"Testing/{i}/processed_robot_log.csv")