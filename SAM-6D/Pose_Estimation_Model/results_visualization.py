import os
import cv2
import json
import numpy as np
import pandas as pd
import trimesh

# --- Paths ---
rgb_base = "/home/hemeyer/SAM-6D/SAM-6D/Data/BOP/tless/test/000001/rgb"
model_dir = "/home/hemeyer/SAM-6D/SAM-6D/Data/BOP/tless/models_cad"
csv_pred_path = "/home/hemeyer/SAM-6D/SAM-6D/Pose_Estimation_Model/log/multiview_pose_estimation_model_base_multiview_id0/tless_eval_iter000000/result_tless.csv"
json_gt_path = "/home/hemeyer/SAM-6D/SAM-6D/Data/BOP/tless/test/000001/scene_gt.json"
json_cam_path = "/home/hemeyer/SAM-6D/SAM-6D/Data/BOP/tless/test/000001/scene_camera.json"
output_dir = "/home/hemeyer/SAM-6D/SAM-6D/Pose_Estimation_Model/Visualizations"
os.makedirs(output_dir, exist_ok=True)

# --- Fixed distortion (assumed zero for BOP)
dist_coeffs = np.zeros(5)

# --- Overlay colors ---
gt_color = (100, 200, 255)     # light blue for GT
pred_color = (255, 60, 60)     # red for predictions
mesh_cache = {}

# --- Load mesh and cache ---
def load_mesh(obj_id):
    if obj_id in mesh_cache:
        return mesh_cache[obj_id]
    path = os.path.join(model_dir, f"obj_{obj_id:06d}.ply")
    mesh = trimesh.load(path, process=False)
    mesh_cache[obj_id] = mesh
    return mesh

# --- Overlay object using projected mesh triangles ---
def render_object_overlay(img, K, R, t, obj_id, color, alpha=0.4):
    mesh = load_mesh(obj_id)
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)

    transformed = (R @ verts.T + t).T
    pts_2d, _ = cv2.projectPoints(transformed, np.zeros(3), np.zeros(3), K, dist_coeffs)
    pts_2d = pts_2d.squeeze()

    if np.isnan(pts_2d).any() or np.isinf(pts_2d).any():
        return

    pts_2d = pts_2d.astype(np.int32)
    overlay = img.copy()
    for tri in faces:
        pts = pts_2d[tri]
        if np.any(pts < 0) or np.any(pts[:, 0] >= img.shape[1]) or np.any(pts[:, 1] >= img.shape[0]):
            continue
        cv2.fillConvexPoly(overlay, pts, color)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

# --- Load data ---
with open(json_gt_path, 'r') as f:
    gt_data = json.load(f)

with open(json_cam_path, 'r') as f:
    camera_data = json.load(f)

df_pred = pd.read_csv(csv_pred_path, header=None)
image_ids = sorted(df_pred[1].unique())

# --- Loop through each image ---
for image_id in image_ids:
    rgb_path = os.path.join(rgb_base, f"{image_id:06d}.png")
    img = cv2.imread(rgb_path)
    if img is None:
        print(f"⚠️  Skipping missing image: {rgb_path}")
        continue

    # Load intrinsics
    cam = camera_data[str(image_id)]
    K = np.array(cam["cam_K"]).reshape(3, 3)

    # --- PREDICTED POSES ---
    # 👇 COMMENT OUT THIS BLOCK to hide predictions
    df_img_pred = df_pred[df_pred[1] == image_id]
    for _, row in df_img_pred.iterrows():
        try:
            obj_id = int(row[2])
            R = np.array([float(x) for x in row[4].strip().split()]).reshape(3, 3)
            t = np.array([float(x) for x in row[5].strip().split()]).reshape(3, 1)

            # Uncomment below if your prediction is in world coordinates
            # R_w2c = np.array(cam["cam_R_w2c"]).reshape(3, 3)
            # t_w2c = np.array(cam["cam_t_w2c"]).reshape(3, 1)
            # R = R_w2c @ R
            # t = R_w2c @ t + t_w2c

            render_object_overlay(img, K, R, t, obj_id, pred_color, alpha=0.4)
        except Exception as e:
            print(f"❌ Prediction error: obj {obj_id} image {image_id}: {e}")

    # --- GROUND TRUTH POSES ---
    # 👇 COMMENT OUT THIS BLOCK to hide GT
    if str(image_id) in gt_data:
        for entry in gt_data[str(image_id)]:
            try:
                obj_id = entry["obj_id"]
                R = np.array(entry["cam_R_m2c"]).reshape(3, 3)
                t = np.array(entry["cam_t_m2c"]).reshape(3, 1)
                render_object_overlay(img, K, R, t, obj_id, gt_color, alpha=0.3)
            except Exception as e:
                print(f"❌ GT error: obj {obj_id} image {image_id}: {e}")

    # Save image
    out_path = os.path.join(output_dir, f"pose_overlay_{image_id:06d}.png")
    cv2.imwrite(out_path, img)
    print(f"✅ Saved: {out_path}")
