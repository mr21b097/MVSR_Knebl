import os
import cv2
import yaml
import numpy as np
import open3d as o3d
import trimesh
from pathlib import Path

# ---------- Parameters ----------
img_ids = [str(i) for i in range(10)]
rgb_dirs = [Path("images/train"), Path("images/val")]
label_dirs = [Path("labels/train"), Path("labels/val")]
depth_dir = Path("data/depth")
models_dir = Path("models")
render_ref_dir = Path("data/rgb")  # aktuell nicht genutzt
data_yaml = Path("data.yaml")
output_dir = Path("out_vis")
output_dir.mkdir(exist_ok=True)

depth_scale = 1000.0
fx = fy = 1066.778
cx = 312.9869
cy = 241.3109

np.random.seed(0)

# ---------- Helper functions ----------
def load_class_names(yaml_path):
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)["names"]

def yolo_to_bbox_px(yolo_line, img_w, img_h):
    cls, cx_, cy_, w, h = map(float, yolo_line.split())
    x1 = int((cx_ - w/2) * img_w)
    y1 = int((cy_ - h/2) * img_h)
    x2 = int((cx_ + w/2) * img_w)
    y2 = int((cy_ + h/2) * img_h)
    return max(0,x1), max(0,y1), min(img_w-1,x2), min(img_h-1,y2), int(cls)

def orthonormalize_rotation(rot):
    u, _, vt = np.linalg.svd(rot)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r

# ---------- Pointcloud / model loading ----------
def depth_crop_to_pointcloud(depth_img, bbox):
    x1, y1, x2, y2 = bbox
    h, w = depth_img.shape[:2]
    x1 = max(0, min(w-1, x1)); x2 = max(0, min(w-1, x2))
    y1 = max(0, min(h-1, y1)); y2 = max(0, min(h-1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    mask = np.zeros(depth_img.shape, dtype=bool)
    mask[y1:y2, x1:x2] = True
    ys, xs = np.where(mask & (depth_img > 0))
    if len(xs) == 0:
        return None
    zs = depth_img[ys, xs].astype(np.float32) / depth_scale
    xs_3d = (xs - cx) * zs / fx
    ys_3d = (ys - cy) * zs / fy
    pts = np.stack((xs_3d, ys_3d, zs), axis=-1)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    if len(pts) >= 3:
        pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
    return pc

def load_model_pointcloud_and_mesh(model_path, sample_points=5000):
    mesh = trimesh.load(model_path)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
    mesh.apply_scale(1.0 / 1000.0)
    points, _ = trimesh.sample.sample_surface(mesh, sample_points)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
    return pc, mesh

def estimate_initial_pose_from_bbox(x1, y1, x2, y2, depth_img):
    cx_box = (x1 + x2) / 2.0
    cy_box = (y1 + y2) / 2.0
    h, w = depth_img.shape[:2]
    ix = int(np.clip(cx_box, 0, w-1))
    iy = int(np.clip(cy_box, 0, h-1))
    depth_center = float(depth_img[iy, ix]) / depth_scale
    if depth_center <= 0:
        region = depth_img[y1:y2, x1:x2]
        nz = region[region>0]
        if nz.size == 0:
            depth_center = 0.5
        else:
            depth_center = float(np.median(nz)) / depth_scale
    X = (cx_box - cx) * depth_center / fx
    Y = (cy_box - cy) * depth_center / fy
    init = np.eye(4)
    init[:3, 3] = [X, Y, depth_center]
    return init

# ---------- RANSAC + ICP ----------
def prepare_point_cloud_for_fpfh(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=100)
    )
    return pcd_down, fpfh

def run_ransac_global_registration(source_pc, target_pc, voxel_size=0.005):
    src_down, src_fpfh = prepare_point_cloud_for_fpfh(source_pc, voxel_size)
    tgt_down, tgt_fpfh = prepare_point_cloud_for_fpfh(target_pc, voxel_size)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, tgt_down, src_fpfh, tgt_fpfh,
        mutual_filter=True,
        max_correspondence_distance=voxel_size * 2,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.95),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel_size * 2)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(500000, 500)
    )
    return result

def run_icp_with_init(source_pc, target_pc, init_pose):
    reg_point = o3d.pipelines.registration.registration_icp(
        source_pc, target_pc, 0.05, init_pose,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
    )
    reg_fine = o3d.pipelines.registration.registration_icp(
        source_pc, target_pc, 0.02, reg_point.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )
    return reg_fine.transformation

# ---------- Flip scoring ----------
def rot180_matrix(axis):
    if axis == 'x':
        return np.array([[1,0,0],[0,-1,0],[0,0,-1]])
    if axis == 'y':
        return np.array([[-1,0,0],[0,1,0],[0,0,-1]])
    if axis == 'z':
        return np.array([[-1,0,0],[0,-1,0],[0,0,1]])
    return np.eye(3)

def evaluate_transform_score(T_candidate, source_pc, target_pc, max_samples=1000, normal_weight=0.03):
    src_pts = np.asarray(source_pc.points)
    if src_pts.shape[0] == 0:
        return 1e9
    src_norms = np.asarray(source_pc.normals) if np.asarray(source_pc.normals).size else None
    n = src_pts.shape[0]
    if n > max_samples:
        idx = np.random.choice(n, max_samples, replace=False)
        src_s = src_pts[idx]
        src_n_s = src_norms[idx] if src_norms is not None else None
    else:
        src_s = src_pts
        src_n_s = src_norms if src_norms is not None else None
    ones = np.ones((src_s.shape[0], 1))
    homo = np.hstack([src_s, ones])
    transformed = (T_candidate @ homo.T).T[:, :3]
    Rm = T_candidate[:3, :3]
    if src_n_s is not None:
        transformed_normals = (Rm @ src_n_s.T).T
    else:
        transformed_normals = None
    tgt_norms = np.asarray(target_pc.normals) if np.asarray(target_pc.normals).size else None
    kdt = o3d.geometry.KDTreeFlann(target_pc)
    dists, normal_penalties = [], []
    for i, p in enumerate(transformed):
        if p[2] <= 0:
            dists.append(1e3); normal_penalties.append(1.0); continue
        k, idx, dist2 = kdt.search_knn_vector_3d(p, 1)
        if k > 0:
            d = np.sqrt(dist2[0])
            dists.append(d)
            if transformed_normals is not None and tgt_norms is not None:
                n_src = transformed_normals[i]; n_tgt = tgt_norms[idx[0]]
                if np.linalg.norm(n_src) == 0 or np.linalg.norm(n_tgt) == 0:
                    normal_penalties.append(1.0)
                else:
                    dot = abs(np.dot(n_src / np.linalg.norm(n_src), n_tgt / np.linalg.norm(n_tgt)))
                    normal_penalties.append(1.0 - dot)
            else:
                normal_penalties.append(1.0)
        else:
            dists.append(1e3); normal_penalties.append(1.0)
    return float(np.median(dists)) + normal_weight * float(np.mean(normal_penalties))

def resolve_180_flips(T_orig, source_pc, target_pc):
    T_orig = np.array(T_orig, copy=True)
    candidates = [T_orig.copy()]
    for ax in ['x','y','z']:
        flip = np.eye(4); flip[:3,:3] = rot180_matrix(ax)
        candidates.append(T_orig @ flip)
    best_score, best_T = 1e12, T_orig
    for T_c in candidates:
        score = evaluate_transform_score(T_c, source_pc, target_pc)
        if score < best_score:
            best_score, best_T = score, T_c
    return best_T

# ---------- Visualization ----------
def draw_poses(img, T, target_pc):
    pts = np.asarray(target_pc.points)
    if pts.shape[0] == 0:
        return img
    center = pts.mean(axis=0)
    axis_length = max(np.linalg.norm(pts.max(axis=0)-pts.min(axis=0))*0.18, 0.025)
    Rm = T[:3,:3]
    axes_pts = [center, center+Rm[:,0]*axis_length, center+Rm[:,1]*axis_length, center+Rm[:,2]*axis_length]
    proj = lambda p: (int((p[0]*fx/p[2])+cx), int((p[1]*fy/p[2])+cy))
    if center[2] <= 0:
        return img
    origin_pix = proj(center)
    for axis_end, color in zip(axes_pts[1:], [(0,0,255),(0,255,0),(255,0,0)]):
        if axis_end[2] > 0:
            cv2.line(img, origin_pix, proj(axis_end), color, 2)
    cv2.circle(img, origin_pix, 3, (255,255,255), -1)
    return img

# ---------- Main ----------
class_names = load_class_names(data_yaml)

for img_id in img_ids:
    rgb_path = None; label_path = None
    for rdir, ldir in zip(rgb_dirs, label_dirs):
        if (rdir / f"{img_id}.png").exists():
            rgb_path = rdir / f"{img_id}.png"
            label_path = ldir / f"{img_id}.txt"
            break
    depth_path = depth_dir / f"{img_id}.png"
    if not rgb_path or not label_path.exists() or not depth_path.exists():
        print(f"[WARN] Missing files for {img_id}")
        continue

    rgb_img = cv2.imread(str(rgb_path))
    depth_img = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    img_h, img_w = rgb_img.shape[:2]
    with open(label_path, "r") as f:
        labels = f.readlines()

    for det_idx, line in enumerate(labels):
        x1, y1, x2, y2, cls_id = yolo_to_bbox_px(line, img_w, img_h)
        obj_name = class_names[cls_id]
        model_path = models_dir / f"{obj_name}.obj"
        if not model_path.exists():
            print(f"[WARN] Missing model {model_path}"); continue

        target_pc = depth_crop_to_pointcloud(depth_img, (x1, y1, x2, y2))
        if target_pc is None or len(target_pc.points) < 10:
            print(f"[WARN] not enough depth points for {img_id} det{det_idx}"); continue

        source_pc, mesh = load_model_pointcloud_and_mesh(model_path)
        init_pose = estimate_initial_pose_from_bbox(x1, y1, x2, y2, depth_img)

        T_init = init_pose
        try:
            ransac_result = run_ransac_global_registration(source_pc, target_pc)
            if ransac_result.fitness > 0.3:
                T_init = ransac_result.transformation
        except Exception as e:
            print("[WARN] RANSAC failed:", e)

        T_icp = run_icp_with_init(source_pc, target_pc, T_init)
        T_best = resolve_180_flips(T_icp, source_pc, target_pc)
        T_best[:3,:3] = orthonormalize_rotation(T_best[:3,:3])

        np.savetxt(output_dir / f"{img_id}_det{det_idx}_pose.txt", T_best, fmt="%.6f")
        cv2.rectangle(rgb_img, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(rgb_img, obj_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        rgb_img = draw_poses(rgb_img, T_best, target_pc)

    cv2.imwrite(str(output_dir / f"{img_id}_vis.png"), rgb_img)
    print("[INFO] wrote", output_dir / f"{img_id}_vis.png")

print("[DONE] All images processed.")
