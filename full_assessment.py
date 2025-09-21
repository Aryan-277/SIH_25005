import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from model import HybridATCNet
import joblib

# --- Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Paths ---
HYBRID_MODEL_PATH = "../results/hybrid_model_best.pth"
RF_MODEL_PATH = "../results/rf_regressor.pkl"

# --- Load models ---
cnn_model = HybridATCNet(num_numeric_features=13, output_dim=1).to(DEVICE)
cnn_model.load_state_dict(torch.load(HYBRID_MODEL_PATH, map_location=DEVICE))
cnn_model.eval()

rf_model = joblib.load(RF_MODEL_PATH)

# --- Image transform ---
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# --- Feature columns used during training ---
all_numeric_cols = [
    'Oblique body length (cm)','Withers height(cm)',
    'Heart girth(cm)','Hip length (cm)','Body weight (kg)',  # placeholder if unknown
    'side_body_length','side_withers_height','side_chest_width','side_hip_width',
    'back_body_length','back_withers_height','back_chest_width','back_hip_width'
]

def load_image(path):
    img = Image.open(path).convert("RGB")
    return image_transforms(img).unsqueeze(0)  # Add batch dim

def prepare_numeric_features(numeric_dict):
    """
    numeric_dict: dictionary with keys matching some/all numeric columns
    Adds missing columns as 0 to match trained model input
    """
    features = []
    for col in all_numeric_cols:
        features.append(float(numeric_dict.get(col, 0)))
    return torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # [1, num_features]

def assess_animal(side_img_path, back_img_path, numeric_dict):
    """
    side_img_path, back_img_path: image paths
    numeric_dict: dictionary with available numeric + OpenCV features
    """
    side_img = load_image(side_img_path).to(DEVICE)
    back_img = load_image(back_img_path).to(DEVICE)
    features = prepare_numeric_features(numeric_dict).to(DEVICE)

    # --- CNN Prediction ---
    with torch.no_grad():
        pred_weight_cnn = cnn_model(side_img, back_img, features).item()

    # --- RF Prediction ---
    rf_input = prepare_numeric_features(numeric_dict).numpy()
    pred_weight_rf = rf_model.predict(rf_input)[0]

    # --- Derived body metrics ---
    oblique = numeric_dict.get('Oblique body length (cm)', 0)
    hip = numeric_dict.get('Hip length (cm)', 1)  # avoid divide by zero
    chest = numeric_dict.get('Heart girth(cm)', 0)

    muscle_ratio = chest / hip
    chest_hip_ratio = chest / hip

    fitness_status = "Fit for breeding" if 1.0 <= muscle_ratio <= 2.0 else "Check further"

    result = {
        "pred_weight_cnn": round(pred_weight_cnn,2),
        "pred_weight_rf": round(pred_weight_rf,2),
        "muscle_ratio": round(muscle_ratio,2),
        "chest_hip_ratio": round(chest_hip_ratio,2),
        "fitness_status": fitness_status
    }

    return result

# --- Example usage ---
if __name__ == "__main__":
    numeric_example = {
        'Oblique body length (cm)': 161,
        'Withers height(cm)': 124,
        'Heart girth(cm)': 190,
        'Hip length (cm)': 47,
        # OpenCV features (example)
        'side_body_length': 160,
        'side_withers_height': 123,
        'side_chest_width': 65,
        'side_hip_width': 40,
        'back_body_length': 161,
        'back_withers_height': 124,
        'back_chest_width': 60,
        'back_hip_width': 42
    }

    side_img = "../dataset/images/side/1.png"
    back_img = "../dataset/images/back/1.png"

    res = assess_animal(side_img, back_img, numeric_example)
    print(res)
