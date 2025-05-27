import os
import json
import numpy as np
import pandas as pd
import trimesh
from scipy.spatial import cKDTree

# Configuration paths
model_dir = "/home/hemeyer/SAM-6D/SAM-6D/Data/BOP/tless/models_cad"
model_info_path = "/home/hemeyer/SAM-6D/SAM-6D/Data/BOP/tless/models_cad/models_info.json"
csv_pred_path = "/home/hemeyer/SAM-6D/SAM-6D/Pose_Estimation_Model/log/multiview_pose_estimation_model_base_multiview_id0/tless_eval_iter000000/result_tless.csv"
json_gt_path = "/home/hemeyer/SAM-6D/SAM-6D/Data/BOP/tless/test/000001/scene_gt.json"

# Load predicted poses
df_pred = pd.read_csv(csv_pred_path, header=None)

# Load GT poses
with open(json_gt_path, 'r') as f:
    gt_data = json.load(f)

# Load object diameters
with open(model_info_path, 'r') as f:
    model_info = json.load(f)
obj_diameters = {int(k): v["diameter"] for k, v in model_info.items()}

# Load object models (cached)
mesh_cache = {}
def load_model_points(obj_id):
    if obj_id not in mesh_cache:
        path = os.path.join(model_dir, f"obj_{obj_id:06d}.ply")
        mesh = trimesh.load(path, process=False)
        mesh_cache[obj_id] = np.asarray(mesh.vertices)
    return mesh_cache[obj_id]

# Metrics
thresholds = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
recall_hits = np.zeros(len(thresholds), dtype=int)
total = 0
axis_errors = []
rot_errors = []
trans_errors = []

# Organize GT by image_id
gt_by_image = {int(k): v for k, v in gt_data.items()}

for _, row in df_pred.iterrows():
    image_id = int(row[1])
    obj_id = int(row[2])
    R_pred = np.array([float(x) for x in row[4].strip().split()]).reshape(3, 3)
    t_pred = np.array([float(x) for x in row[5].strip().split()]).reshape(3, 1)

    if image_id not in gt_by_image:
        continue

    # Match GT pose
    gt_pose = next((gt for gt in gt_by_image[image_id] if gt["obj_id"] == obj_id), None)
    if gt_pose is None:
        continue

    R_gt = np.array(gt_pose["cam_R_m2c"]).reshape(3, 3)
    t_gt = np.array(gt_pose["cam_t_m2c"]).reshape(3, 1)
    model_points = load_model_points(obj_id)

    # Transform model
    pred_pts = (R_pred @ model_points.T + t_pred).T
    gt_pts = (R_gt @ model_points.T + t_gt).T

    # ADD-S via KDTree
    tree = cKDTree(gt_pts)
    dists, _ = tree.query(pred_pts, k=1)
    add_s = np.mean(dists)

    diameter = obj_diameters[obj_id]
    for i, frac in enumerate(thresholds):
        if add_s < diameter * frac:
            recall_hits[i] += 1

    # Axis errors
    offset_mm = np.abs((t_pred - t_gt).squeeze())
    axis_errors.append(offset_mm)

    # Rotation error (degrees)
    R_diff = R_pred @ R_gt.T
    trace = np.clip((np.trace(R_diff) - 1) / 2.0, -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(trace))
    rot_errors.append(angle_deg)

    # Translation error (mm)
    trans_error = np.linalg.norm(t_pred - t_gt)
    trans_errors.append(trans_error)

    total += 1

# Final stats
recalls = (recall_hits / total * 100) if total > 0 else np.zeros(len(thresholds))
average_recall = recalls.mean() if total > 0 else 0
axis_errors = np.array(axis_errors)
mean_axis_error = axis_errors.mean(axis=0) if len(axis_errors) else [0, 0, 0]
mean_rot_error = np.mean(rot_errors) if rot_errors else 0
mean_trans_error = np.mean(trans_errors) if trans_errors else 0

# Display metrics
metrics_df = pd.DataFrame({
    "Metric": [
        "Recall @ 0.02d", "Recall @ 0.05d", "Recall @ 0.1d",
        "Recall @ 0.2d", "Recall @ 0.3d", "Recall @ 0.5d",
        "Average Recall",
        "Mean Rotation error (°)", "Mean Translation error (mm)",
        "Mean X error (mm)", "Mean Y error (mm)", "Mean Z error (mm)"
    ],
    "Value": [
        f"{recalls[0]:.2f}%", f"{recalls[1]:.2f}%", f"{recalls[2]:.2f}%",
        f"{recalls[3]:.2f}%", f"{recalls[4]:.2f}%", f"{recalls[5]:.2f}%",
        f"{average_recall:.2f}%",
        f"{mean_rot_error:.2f}", f"{mean_trans_error:.2f}",
        f"{mean_axis_error[0]:.2f}", f"{mean_axis_error[1]:.2f}", f"{mean_axis_error[2]:.2f}"
    ]
})

print("\n🔍 Evaluation Results:")
print(metrics_df.to_string(index=False))
