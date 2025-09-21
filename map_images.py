import os
import pandas as pd

EXCEL_PATH = "../dataset/data.xlsx"
SIDE_FOLDER = "../dataset/images/side"
BACK_FOLDER = "../dataset/images/back"
OUTPUT_PATH = "../dataset/data_mapped.xlsx"

def get_sorted_images(folder):
    """Return list of image paths sorted by numeric filename."""
    images = [f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    images.sort(key=lambda x: int(os.path.splitext(x)[0]))
    return [os.path.join(folder, f) for f in images]

if __name__ == "__main__":
    df = pd.read_excel(EXCEL_PATH)
    side_images = get_sorted_images(SIDE_FOLDER)
    back_images = get_sorted_images(BACK_FOLDER)

    min_len = min(len(df), len(side_images), len(back_images))
    if len(df) != min_len:
        print(f"Warning: Mismatch detected! Mapping only first {min_len} rows.")

    df = df.iloc[:min_len].copy()
    df['side_image'] = side_images[:min_len]
    df['back_image'] = back_images[:min_len]

    df.to_excel(OUTPUT_PATH, index=False)
    print(f"Mapping completed. Saved to {OUTPUT_PATH}")
