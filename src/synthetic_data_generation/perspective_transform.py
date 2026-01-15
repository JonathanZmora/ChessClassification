import cv2
import numpy as np
import json
import os
import glob

# CONFIGURATION
MARGIN_PX = 3
OUTPUT_SIZE = 480

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def process_images(render_dir):
    json_files = glob.glob(os.path.join(render_dir, "*.json"))
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        img_path = json_file.replace('.json', '.png')
        if not os.path.exists(img_path): continue
            
        image = cv2.imread(img_path)
        if image is None: continue

        print(f"Processing {os.path.basename(img_path)}...")

        # 1. Source points (Inner Board Corners from Blender)
        src_pts = np.array(data['corners'], dtype="float32")
        src_pts = order_points(src_pts)

        # 2. Destination points (With Margin!)
        # Instead of mapping to 0 and W, we map to MARGIN and W-MARGIN
        m = MARGIN_PX
        w = OUTPUT_SIZE
        
        dst_pts = np.array([
            [m, m],             # Top-left (padded)
            [w - m, m],         # Top-right
            [w - m, w - m],     # Bottom-right
            [m, w - m]          # Bottom-left
        ], dtype="float32")

        # 3. Warp
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(image, M, (w, w))

        out_path = img_path.replace('.png', '_warped.png')
        cv2.imwrite(out_path, warped)

if __name__ == "__main__":
    process_images("./renders")
