import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# --- Paths & configs ---
EXCEL_PATH = "../dataset/data_mapped.xlsx"
OPENCV_CSV = "../dataset/opencv_measurements.csv"
IMAGE_SIZE = 224
BATCH_SIZE = 16
RANDOM_STATE = 42

# --- Load mapped Excel ---
df = pd.read_excel(EXCEL_PATH)
numeric_cols = ['Oblique body length (cm)', 'Withers height(cm)',
                'Heart girth(cm)', 'Hip length (cm)', 'Body weight (kg)']

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=numeric_cols).reset_index(drop=True)
print(f"Rows after cleaning numeric columns: {len(df)}")

# --- Load OpenCV features ---
cv_df = pd.read_csv(OPENCV_CSV)

# Make sure the image column only contains base filenames (without paths/extensions)
cv_df['image_base'] = cv_df['image'].apply(lambda x: os.path.splitext(os.path.basename(str(x)))[0])
side_cv = cv_df[cv_df['view']=='side'].set_index('image_base')
back_cv = cv_df[cv_df['view']=='back'].set_index('image_base')

# --- Map OpenCV features to main DataFrame ---
missing_side, missing_back = 0, 0
for col in ['body_length','withers_height','chest_width','hip_width']:
    side_values = []
    back_values = []
    for path in df['side_image']:
        key = os.path.splitext(os.path.basename(path))[0]
        if key in side_cv.index:
            side_values.append(side_cv.at[key, col])
        else:
            side_values.append(np.nan)
            missing_side += 1
    for path in df['back_image']:
        key = os.path.splitext(os.path.basename(path))[0]
        if key in back_cv.index:
            back_values.append(back_cv.at[key, col])
        else:
            back_values.append(np.nan)
            missing_back += 1
    df[f'side_{col}'] = side_values
    df[f'back_{col}'] = back_values

cv_features = [f'side_{c}' for c in ['body_length','withers_height','chest_width','hip_width']] + \
              [f'back_{c}' for c in ['body_length','withers_height','chest_width','hip_width']]

print(f"Missing side feature values: {missing_side}, Missing back feature values: {missing_back}")

# Drop rows with missing OpenCV features
df = df.dropna(subset=cv_features).reset_index(drop=True)
print(f"Rows after merging OpenCV features: {len(df)}")

# --- All features used in numeric/OpenCV branch ---
all_features = numeric_cols + cv_features

# --- Scale numeric + OpenCV features ---
scaler = StandardScaler()
df[all_features] = scaler.fit_transform(df[all_features])

# --- Train/Val/Test split ---
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=RANDOM_STATE)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=RANDOM_STATE)
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# --- Image transformations ---
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# --- Dataset class ---
class ATCDataset(Dataset):
    def __init__(self, df, feature_cols, transform=None):
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        side_img = Image.open(row['side_image']).convert('RGB')
        back_img = Image.open(row['back_image']).convert('RGB')
        if self.transform:
            side_img = self.transform(side_img)
            back_img = self.transform(back_img)
        features = torch.tensor(row[self.feature_cols].values.astype(np.float32))
        target = torch.tensor(row['Body weight (kg)'], dtype=torch.float32)
        return {'side_img': side_img, 'back_img': back_img, 'features': features, 'target': target}

# --- DataLoaders ---
train_dataset = ATCDataset(train_df, all_features, transform=train_transform)
val_dataset = ATCDataset(val_df, all_features, transform=eval_transform)
test_dataset = ATCDataset(test_df, all_features, transform=eval_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("DataLoaders ready with OpenCV features included.")
