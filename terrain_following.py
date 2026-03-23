import trimesh
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

print("加载 Mesh...")
mesh = trimesh.load("D:/GitHub/mouse-3d-reconstruction/SU.ply")

bounds = mesh.bounds
print(f"范围 X: {bounds[0][0]:.2f} ~ {bounds[1][0]:.2f}")
print(f"范围 Y: {bounds[0][1]:.2f} ~ {bounds[1][1]:.2f}")
print(f"范围 Z: {bounds[0][2]:.2f} ~ {bounds[1][2]:.2f}")

# 创建高度图
print("\n创建高度图...")
x_min, x_max = bounds[0][0], bounds[1][0]
z_min, z_max = bounds[0][2], bounds[1][2]

step = 0.3
x_coords = np.arange(x_min + step, x_max - step, step)
z_coords = np.arange(z_min + step, z_max - step, step)

height_map = {}
ray_direction = [0, -1, 0]
total_samples = len(x_coords) * len(z_coords)
sampled = 0

for x in x_coords:
    for z in z_coords:
        ray_origin = [x, bounds[1][1] + 0.5, z]
        locations, _, _ = mesh.ray.intersects_location(
            ray_origins=[ray_origin],
            ray_directions=[ray_direction]
        )
        if len(locations) > 0:
            height_map[(x, z)] = max(loc[1] for loc in locations)

        sampled += 1
        if sampled % 100 == 0:
            print(f"进度: {sampled}/{total_samples}")

print(f"\n成功采样 {len(height_map)} 个点")

if len(height_map) == 0:
    print("错误：没有采样到任何点")
    exit()

# 从中心区域开始
x_vals = sorted(set(x for x, z in height_map.keys()))
z_vals = sorted(set(z for x, z in height_map.keys()))
mid_x = x_vals[len(x_vals) // 2]
mid_z = z_vals[len(z_vals) // 2]
start_pos = min(height_map.keys(), key=lambda p: (p[0] - mid_x) ** 2 + (p[1] - mid_z) ** 2)
start_x, start_z = start_pos
start_y = height_map[start_pos] + 0.05

print(f"\n起始位置（中心）: ({start_x:.2f}, {start_y:.2f}, {start_z:.2f})")

# 行走
print("\n开始行走...")
current_x, current_z = start_x, start_z
current_y = start_y
step_size = 0.2
angle = 0

# 记录轨迹
trajectory = []

for i in range(50):
    if i % 10 == 0:
        angle = np.random.uniform(0, 2 * np.pi)

    dx = step_size * np.cos(angle)
    dz = step_size * np.sin(angle)
    next_x = current_x + dx
    next_z = current_z + dz

    best_key = None
    best_dist = step_size * 2

    for (hx, hz), height in height_map.items():
        dist = ((next_x - hx) ** 2 + (next_z - hz) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_key = (hx, hz)

    if best_key and best_dist < step_size:
        current_x, current_z = best_key
        current_y = height_map[best_key] + 0.05
        trajectory.append([current_x, current_y, current_z])
        if i % 10 == 0:
            print(f"步数 {i}: 位置 ({current_x:.2f}, {current_y:.2f}, {current_z:.2f})")
    else:
        angle += np.pi
        print(f"步数 {i}: 边缘，掉头")

print("\n完成！")

# 画轨迹图
print("\n正在生成轨迹图...")
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 画点云（采样一部分，避免太密集）
points = mesh.vertices
step_plot = max(1, len(points) // 5000)
ax.scatter(points[::step_plot, 0], points[::step_plot, 1], points[::step_plot, 2],
           c='gray', s=0.5, alpha=0.3)

# 画轨迹
if len(trajectory) > 0:
    traj = np.array(trajectory)
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'r-', linewidth=2, label='行走轨迹')
    ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], c='green', s=50, marker='o', label='起点')
    ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], c='red', s=50, marker='x', label='终点')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.title('小球在鼠标模型上的行走轨迹')
plt.show()

print("轨迹图已显示")