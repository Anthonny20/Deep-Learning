
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import os
import trimesh

# ---------- Função best_fit_transform ----------
def best_fit_transform(A, B):
    assert A.shape == B.shape
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = centroid_B.T - R @ centroid_A.T
    return R, t.reshape(3, 1)

# ---------- Função ICP do zero ----------
def ICP(M, S, init_pose=None, iterations=50, tolerance=1e-6, distance_threshold=1.0):
    if init_pose is None:
        R = np.eye(3)
        t = np.zeros((3, 1))
    else:
        R, t = init_pose

    prev_error = float('inf')

    for iteration in range(iterations):
        M_transformed = (R @ M.T).T + t.T
        tree = KDTree(S)
        distances, indices = tree.query(M_transformed)
        S_matched = S[indices]

        mask = distances < distance_threshold
        if np.sum(mask) < 3:
            break

        R_new, t_new = best_fit_transform(M_transformed[mask], S_matched[mask])
        R = R_new @ R
        t = R_new @ t + t_new

        mean_error = np.mean(distances)
        if abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    return R, t

# ---------- Aplicação na sequência ----------
kitti_dataset = '/seu/caminho/KITTI-Sequence'
gt_path = '/seu/caminho/ground_truth.npy'

point_clouds = []
for root, _, files in os.walk(kitti_dataset):
    for file in sorted(files):
        if file.endswith('.obj'):
            mesh = trimesh.load(os.path.join(root, file))
            point_clouds.append(np.asarray(mesh.vertices))

ground_truth = np.load(gt_path)

trajectory = [np.eye(4)]
for i in range(len(point_clouds) - 1):
    A = point_clouds[i]
    B = point_clouds[i + 1]
    R, t = ICP(A, B, iterations=50, distance_threshold=1.0)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()
    T_absolute = trajectory[-1] @ T
    trajectory.append(T_absolute)

# ---------- Alinhamento com Ground Truth ----------
icp_estimated = np.array([T[:3, 3] for T in trajectory])
gt_positions = np.array([gt[:3, 3] for gt in ground_truth[:len(icp_estimated)]])
R_align, t_align = best_fit_transform(icp_estimated, gt_positions)
icp_estimated_aligned = np.array([(R_align @ p.reshape(3, 1) + t_align).flatten() for p in icp_estimated])
offset = gt_positions[0] - icp_estimated_aligned[0]
icp_estimated_aligned += offset

# ---------- Plot ----------
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(icp_estimated_aligned[:, 0], icp_estimated_aligned[:, 1], icp_estimated_aligned[:, 2], label='ICP Manual Alinhado', color='blue')
ax.plot(gt_positions[:, 0], gt_positions[:, 1], gt_positions[:, 2], label='Ground Truth', color='green')
ax.set_title("Trajetória Estimada com ICP (Manual) vs Ground Truth")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.legend()
plt.show()
