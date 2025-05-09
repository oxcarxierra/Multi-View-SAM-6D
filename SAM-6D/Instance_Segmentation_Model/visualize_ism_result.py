import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# === CONFIG ===
scene_id = "000001"
base_dir = os.getcwd()
result_dir = os.path.join(base_dir, "log", "sam", "predictions", "tless", "result_tless")
image_dir = f"/home/ohseun/workspace/SAM-6D/SAM-6D/Data/BOP/tless/test_primesense/{scene_id}/rgb"
output_dir = os.path.join(base_dir, "visualizations")

os.makedirs(output_dir, exist_ok=True)

# === Loop through all result files ===
for fname in os.listdir(result_dir):
    if fname.endswith(".npz") and "_runtime" not in fname:
        result_path = os.path.join(result_dir, fname)
        data = np.load(result_path, allow_pickle=True)

        # Get image ID and category data
        image_id = str(data["image_id"].item()).zfill(6)
        rgb_path = os.path.join(image_dir, f"{image_id}.png")
        if not os.path.exists(rgb_path):
            print(f"[!] Skipping: {rgb_path} not found")
            continue

        image = cv2.imread(rgb_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bboxes = data["bbox"]
        scores = data["score"]
        category_ids = data["category_id"]

        # Draw all boxes
        for i, box in enumerate(bboxes):
            x, y, w, h = map(int, box)
            label = f"Obj {category_ids[i]}, Score: {scores[i]:.2f}"
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # Save output
        out_path = os.path.join(output_dir, fname.replace(".npz", "_vis.png"))
        plt.imsave(out_path, image)
        print(f"[✓] Saved: {out_path}")