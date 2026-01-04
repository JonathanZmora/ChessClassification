import cv2
import numpy as np
import json
import os
import glob

def order_points(pts):
    """
    Sorts coordinates into order: [top-left, top-right, bottom-right, bottom-left]
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # Top-left has smallest sum
    rect[2] = pts[np.argmax(s)] # Bottom-right has largest sum

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # Top-right has smallest difference
    rect[3] = pts[np.argmax(diff)] # Bottom-left has largest difference
    return rect

def process_images(render_dir):
    print("inside process")
    # Find all generated JSON files
    json_files = glob.glob(os.path.join(render_dir, "*.json"))
    
    for json_file in json_files:
        # Load the metadata
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Load the corresponding image
        img_path = json_file.replace('.json', '.png')
        if not os.path.exists(img_path):
            continue
            
        image = cv2.imread(img_path)
        if image is None: 
            continue

        print(f"Processing {os.path.basename(img_path)}...")

        # 1. Source points (from Blender)
        src_pts = np.array(data['corners'], dtype="float32")
        src_pts = order_points(src_pts)

        # 2. Destination points (A perfect square)
        # You can choose the output size, e.g., 1024x1024
        W, H = 1024, 1024
        dst_pts = np.array([
            [0, 0],
            [W - 1, 0],
            [W - 1, H - 1],
            [0, H - 1]
        ], dtype="float32")

        # 3. Calculate Perspective Transform Matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # 4. Warp the image
        warped = cv2.warpPerspective(image, M, (W, H))

        # Save the result
        out_path = img_path.replace('.png', '_warped.png')
        cv2.imwrite(out_path, warped)
        print(f" -> Saved {os.path.basename(out_path)}")

if __name__ == "__main__":
    # Point this to your renders folder
    process_images("./renders")
