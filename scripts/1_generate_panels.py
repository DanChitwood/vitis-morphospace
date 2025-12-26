import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# --- CONFIG ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "inference_pol_cart_aligned"
OUTPUT_DIR = PROJECT_ROOT / "panels" # We will create this folder
OUTPUT_DIR.mkdir(exist_ok=True)

# Target Width for the final composite (small enough for GitHub, clear enough for eyes)
TARGET_WIDTH = 800 

def apply_fake_inferno(gray):
    # Quick way to apply a heatmap without Matplotlib overhead
    return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)

def create_panel(uid):
    try:
        def get_img(mode, sub):
            path = DATA_DIR / mode / sub / f"{uid}.png"
            img = cv2.imread(str(path))
            if sub == "ect": # Apply color to ECT
                img = apply_fake_inferno(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            return cv2.resize(img, (400, 400))

        # Row 1: Cartesian
        c_viz = get_img("cartesian", "viz")
        c_rgb = get_img("cartesian", "rgb")
        c_ect = get_img("cartesian", "ect")
        
        # Row 2: Polar
        p_viz = get_img("polar", "viz")
        p_rgb = get_img("polar", "rgb")
        p_ect = get_img("polar", "ect")

        # Stack them: [Viz, RGB, ECT]
        top_row = np.hstack((c_viz, c_rgb, c_ect))
        bottom_row = np.hstack((p_viz, p_rgb, p_ect))
        combined = np.vstack((top_row, bottom_row))

        # Resize for web efficiency and save as compressed JPG
        final = cv2.resize(combined, (TARGET_WIDTH, int(TARGET_WIDTH * 0.66)))
        cv2.imwrite(str(OUTPUT_DIR / f"{uid}.jpg"), final, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        return True
    except:
        return False

def main():
    # Get UIDs from your existing metadata
    meta_df = pd.read_csv(PROJECT_ROOT / "data" / "master_mask_datasheet.csv")
    uids = meta_df.apply(lambda x: f"{x['image_id']}_{int(x['component_id'])}", axis=1).unique()

    print(f"🖼️ Generating {len(uids)} composite panels...")
    for uid in tqdm(uids):
        create_panel(uid)

if __name__ == "__main__":
    main()