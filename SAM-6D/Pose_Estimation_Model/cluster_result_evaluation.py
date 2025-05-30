import pandas as pd

# GT 목록: scene_id -> 정답 obj_id 리스트
gt_dict = {
    1: [2, 25, 29, 30],
    2: [5, 6, 7],
    3: [5, 8, 11, 12, 18],
    4: [5, 8, 26, 28],
    5: [1, 4, 9, 10, 27],
    6: [6, 7, 11, 11, 12],
    7: [1, 3, 13, 14, 15, 16, 17, 18],
    8: [19, 20, 21, 22, 23, 24],
    9: [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4],
    10: [19, 20, 21, 22, 23, 24],
}

# 예측 결과 불러오기
df = pd.read_csv("log/multiview_pose_estimation_model_base_multiview_id9/tless_eval_iter000000/cluster_result.csv")

# 비교 및 평가
total = 0
correct = 0
cluster_count_correct = 0

from ast import literal_eval  # 문자열 → 리스트로 안전하게 변환

for _, row in df.iterrows():
    scene_id = int(row["scene_id"])
    gt_obj_id = sorted(gt_dict[scene_id])
    pred_obj_ids = sorted(literal_eval(row["obj_ids"]))  # 문자열 → 리스트 → 정렬
    if len(gt_obj_id) == len(pred_obj_ids):
        cluster_count_correct += 1
    for obj_id in gt_obj_id:
        total += 1
        if obj_id in pred_obj_ids:
            correct += 1
            pred_obj_ids.remove(obj_id)

accuracy = correct / total * 100
print(f"\n✅ score: {correct}/{total} : {accuracy:.2f}%")
print(f"✅ cluster that has correct number of objects: {cluster_count_correct}/{df.shape[0]} : {cluster_count_correct / df.shape[0] * 100:.2f}%")