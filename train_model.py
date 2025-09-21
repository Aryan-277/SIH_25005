import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

from preprocessing_dataset import train_loader, val_loader, test_loader, all_features
from model import HybridATCNet

# --- Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Config ---
NUM_NUMERIC_FEATURES = len(all_features)
OUTPUT_DIM = 1          # Regression (weight)
EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 1e-5
PATIENCE = 7

# --- Results directory (robust) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "../results")
os.makedirs(SAVE_DIR, exist_ok=True)

# --- Initialize model ---
model = HybridATCNet(num_numeric_features=NUM_NUMERIC_FEATURES, output_dim=OUTPUT_DIM).to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

best_val_loss = np.inf
patience_counter = 0

print("Starting training...")

# --- Training loop ---
for epoch in range(EPOCHS):
    model.train()
    train_losses = []

    for batch in train_loader:
        optimizer.zero_grad()
        side = batch['side_img'].to(DEVICE)
        back = batch['back_img'].to(DEVICE)
        features = batch['features'].to(DEVICE)
        target = batch['target'].to(DEVICE)

        output = model(side, back, features)
        loss = criterion(output.squeeze(), target)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    # --- Validation ---
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            side = batch['side_img'].to(DEVICE)
            back = batch['back_img'].to(DEVICE)
            features = batch['features'].to(DEVICE)
            target = batch['target'].to(DEVICE)

            output = model(side, back, features)
            val_losses.append(criterion(output.squeeze(), target).item())

    val_loss = np.mean(val_losses)
    print(f"Epoch {epoch+1:02d}/{EPOCHS}: Train Loss={np.mean(train_losses):.4f}, Val Loss={val_loss:.4f}")

    # --- Early stopping ---
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "hybrid_model_best.pth"))
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

# --- Load best model ---
model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "hybrid_model_best.pth")))
model.eval()

# --- Test evaluation ---
all_preds, all_targets = [], []
with torch.no_grad():
    for batch in test_loader:
        side = batch['side_img'].to(DEVICE)
        back = batch['back_img'].to(DEVICE)
        features = batch['features'].to(DEVICE)
        target = batch['target'].to(DEVICE)

        output = model(side, back, features)
        all_preds.extend(output.squeeze().cpu().numpy())
        all_targets.extend(target.cpu().numpy())

rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
print(f"\nHybrid CNN + Numeric Test RMSE: {rmse:.3f}")

# --- Optional Decision Forest ---
print("\nTraining Decision Forest on numeric + OpenCV features...")

train_features, train_targets = [], []
for batch in train_loader:
    train_features.append(batch['features'].cpu().numpy())
    train_targets.append(batch['target'].cpu().numpy())
X_train = np.vstack(train_features)
y_train = np.hstack(train_targets)

test_features, test_targets = [], []
for batch in test_loader:
    test_features.append(batch['features'].cpu().numpy())
    test_targets.append(batch['target'].cpu().numpy())
X_test = np.vstack(test_features)
y_test = np.hstack(test_targets)

rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, rf_preds))
print(f"Decision Forest Test RMSE: {rmse_rf:.3f}")

joblib.dump(rf, os.path.join(SAVE_DIR, "rf_regressor.pkl"))
print(f"Random Forest model saved to {SAVE_DIR}/rf_regressor.pkl")
