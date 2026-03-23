import open3d as o3d

## 点云→Mesh

ply_file = 'D:/GitHub/mouse-3d-reconstruction/su.ply'

# 读 PLY 点云
pcd = o3d.io.read_point_cloud(ply_file)
print(f"点数: {len(pcd.points)}")

# 如果点数为0，说明没读到文件
if len(pcd.points) == 0:
    print("错误：没读到点云文件，请检查路径和文件名")
    exit()

# 计算法向量
print("计算法向量...")
pcd.estimate_normals()

# 泊松重建
print("泊松重建中（可能需要几分钟）...")
mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)

# 保存到当前文件夹
o3d.io.write_triangle_mesh('D:/GitHub/mouse-3d-reconstruction/SU.ply', mesh)
print("Mesh 已保存为 SU_mesh.ply")

# 可视化
print("打开可视化窗口...")
o3d.visualization.draw_geometries([mesh], window_name="鼠标 Mesh", width=800, height=600)