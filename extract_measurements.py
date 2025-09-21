import os
import cv2
import numpy as np
import pandas as pd

# --- CONFIGURATION ---
SIDE_FOLDER = "../dataset/images/side"
BACK_FOLDER = "../dataset/images/back"
OUTPUT_CSV = "../dataset/opencv_measurements.csv"

# Helper function to extract measurements
def extract_measurements(image_path):
    """
    Returns a dict of measurements: body_length, withers_height, chest_width, hip_width
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Thresholding to separate animal from background
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None  # Could not detect
    
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Bounding rectangle
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Measurements in pixels (can scale later if reference provided)
    measurements = {
        "body_length": w,
        "withers_height": h
    }
    
    # Optional: chest width (widest horizontal line at ~30% from top)
    top = y + int(0.3*h)
    slice_row = thresh[top, x:x+w]
    chest_width = np.sum(slice_row == 255)
    measurements["chest_width"] = chest_width
    
    # Optional: hip width (widest horizontal line at ~80% from top)
    hip_row_idx = y + int(0.8*h)
    if hip_row_idx >= thresh.shape[0]:
        hip_row_idx = thresh.shape[0]-1
    slice_row = thresh[hip_row_idx, x:x+w]
    hip_width = np.sum(slice_row == 255)
    measurements["hip_width"] = hip_width
    
    return measurements

# --- Process side images ---
data = []
for img_file in sorted(os.listdir(SIDE_FOLDER), key=lambda x: int(os.path.splitext(x)[0])):
    if not img_file.lower().endswith((".png", ".jpg", ".jpeg")):
        continue
    path = os.path.join(SIDE_FOLDER, img_file)
    meas = extract_measurements(path)
    if meas:
        meas['image'] = path
        meas['view'] = 'side'
        data.append(meas)

# --- Process back images ---
for img_file in sorted(os.listdir(BACK_FOLDER), key=lambda x: int(os.path.splitext(x)[0])):
    if not img_file.lower().endswith((".png", ".jpg", ".jpeg")):
        continue
    path = os.path.join(BACK_FOLDER, img_file)
    meas = extract_measurements(path)
    if meas:
        meas['image'] = path
        meas['view'] = 'back'
        data.append(meas)

# --- Save to CSV ---
df = pd.DataFrame(data)
df.to_csv(OUTPUT_CSV, index=False)
print(f"OpenCV measurements saved to {OUTPUT_CSV}")
