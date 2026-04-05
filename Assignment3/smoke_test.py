import os
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from main import RoverDataset, PilotNet,  CommaAiModel
# ── paste your Dataset/Model classes here, or import them ──
# from train import RoverDataset, PilotNet

TRAIN_CSV = "Training/processed_robot_log.csv"
TEST_CSV  = "Testing/01/processed_robot_log.csv"

transform = transforms.Compose([
    transforms.Resize((160, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def check(label, condition, detail=""):
    status = "✅ PASS" if condition else "❌ FAIL"
    print(f"  {status}  {label}" + (f" — {detail}" if detail else ""))
    return condition

def section(title):
    print(f"\n{'═'*50}\n  {title}\n{'═'*50}")

all_passed = True

# ── 1. CSV FILES ──────────────────────────────────────────
section("1. CSV Files")
for csv_path in [TRAIN_CSV, TEST_CSV]:
    ok = check(f"Exists: {csv_path}", os.path.isfile(csv_path))
    all_passed &= ok
    if ok:
        df = pd.read_csv(csv_path)
        check(f"  Non-empty rows",        len(df) > 0,              f"{len(df)} rows")
        check(f"  Column 'Path'",         'Path'       in df.columns)
        check(f"  Column 'SteerAngle'",   'SteerAngle' in df.columns)
        check(f"  Column 'Throttle'",     'Throttle'   in df.columns)
        check(f"  No NaNs",               df[['Path','SteerAngle','Throttle']].isnull().sum().sum() == 0,
              f"{df[['Path','SteerAngle','Throttle']].isnull().sum().sum()} NaNs found")

# ── 2. IMAGE FILES ────────────────────────────────────────
section("2. Image Files (first 5 per CSV)")
for csv_path in [TRAIN_CSV, TEST_CSV]:
    if not os.path.isfile(csv_path):
        continue
    df = pd.read_csv(csv_path)
    print(f"\n  Checking {csv_path}:")
    for _, row in df.head(5).iterrows():
        img_path = row['Path']
        img_name = os.path.basename(img_path)
        found = os.path.isfile(img_name) or os.path.isfile(img_path)
        resolved = img_name if os.path.isfile(img_name) else img_path
        ok = check(f"  Image: {img_name}", found, f"resolved → {resolved}")
        all_passed &= ok
        if found:
            try:
                img = Image.open(resolved).convert('RGB')
                check(f"    Readable & RGB", True, f"{img.size[0]}×{img.size[1]}")
            except Exception as e:
                check(f"    Readable & RGB", False, str(e))
                all_passed = False

# ── 3. DATASET & DATALOADER ───────────────────────────────
section("3. Dataset & DataLoader (1 batch)")
if os.path.isfile(TRAIN_CSV):
    df = pd.read_csv(TRAIN_CSV)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    try:
        ds = RoverDataset(train_df, transform=transform)
        check("RoverDataset created",     True, f"{len(ds)} samples")
        loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
        images, labels = next(iter(loader))
        check("DataLoader yields batch",  True,  f"images={tuple(images.shape)}, labels={tuple(labels.shape)}")
        check("Image tensor shape",       images.shape == (4, 3, 160, 320), str(tuple(images.shape)))
        check("Labels shape",             labels.shape == (4, 2),           str(tuple(labels.shape)))
    except Exception as e:
        check("Dataset/DataLoader", False, str(e))
        all_passed = False

# ── 4. MODEL FORWARD PASS ─────────────────────────────────
section("4. Model Forward Pass")
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    check("Device", True, str(device))
    model1  = PilotNet().to(device)
    dummy  = torch.randn(4, 3, 160, 320).to(device)
    output = model1(dummy)
    model2  = CommaAiModel().to(device)
    output2 = model2(dummy)
    check("Forward pass",    True,                      f"input={tuple(dummy.shape)}")
    check("Output shape (PilotNet)",    output.shape == (4, 2),    str(tuple(output.shape)))
    check("Output shape (CommaAiModel)",    output2.shape == (4, 2),    str(tuple(output2.shape)))
except Exception as e:
    check("Model forward pass", False, str(e))
    all_passed = False

# ── 5. OUTPUT DIRECTORY ───────────────────────────────────
section("5. Write Permissions (model save location)")
try:
    test_path = "rover_model_test_write.pth"
    torch.save({}, test_path)
    check("Can write .pth to current dir", os.path.isfile(test_path))
    os.remove(test_path)
except Exception as e:
    check("Can write .pth to current dir", False, str(e))
    all_passed = False

# ── SUMMARY ───────────────────────────────────────────────
print(f"\n{'═'*50}")
print(f"  {'✅ ALL CHECKS PASSED' if all_passed else '❌ SOME CHECKS FAILED — fix above before training'}")
print(f"{'═'*50}\n")