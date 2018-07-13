#conda install pillow matplotlib
import py3d as py3d
import copy
import numpy as np
import matplotlib.pyplot as plt

def get_pcd_from_rgbd_image_index(i):
    color_name = "E:\\Project\\LightFieldGiga\\data\\result\\%04d_left.jpg" % i
    depth_name = "E:\\Project\\LightFieldGiga\\data\\result\\%04d_depth.png" % i
    print(color_name)
    print(depth_name)

    color_raw = py3d.read_image(color_name)
    depth_raw = py3d.read_image(depth_name)
    rgbd_image = py3d.create_rgbd_image_from_color_and_depth(
        color_raw, depth_raw, 100, 300, False);
    # print(rgbd_image)

    pinhole_camera_intrinsic = py3d.read_pinhole_camera_intrinsic(
        "E:\\Project\\LightFieldGiga\\SparseToDenseStereo\\camera.json")
    pcd = py3d.create_point_cloud_from_rgbd_image(rgbd_image,
            pinhole_camera_intrinsic)
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    return pcd, rgbd_image

def get_smooth_pcd_from_rgbd_image_index(i):
    color_name = "E:\\Project\\LightFieldGiga\\data\\result\\%04d_left.jpg" % i
    depth_name = "E:\\Project\\LightFieldGiga\\data\\result\\%04d_depth_refine.png" % i
    print(color_name)
    print(depth_name)

    color_raw = py3d.read_image(color_name)
    depth_raw = py3d.read_image(depth_name)
    rgbd_image = py3d.create_rgbd_image_from_color_and_depth(
        color_raw, depth_raw, 100, 300, False);
    # print(rgbd_image)

    pinhole_camera_intrinsic = py3d.read_pinhole_camera_intrinsic(
        "E:\\Project\\LightFieldGiga\\SparseToDenseStereo\\camera.json")
    pcd = py3d.create_point_cloud_from_rgbd_image(rgbd_image,
            pinhole_camera_intrinsic)
    # Flip it, otherwise the pointcloud will be upside down
    # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    return pcd, rgbd_image

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    py3d.draw_geometries([source_temp, target_temp])

def draw_registration_result_original_color(source, target, transformation):
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    py3d.draw_geometries([source_temp, target])

def registrate_two_color_pcd(pcd1, pcd2):
    voxel_radius = [ 4, 2, 1 ];
    max_iter = [ 50, 30, 14 ];
    current_transformation = np.identity(4)
    # print("Colored point cloud registration")

    for scale in range(3):
        iter = max_iter[scale]
        radius = voxel_radius[scale]
        # print([iter,radius,scale])
        # print("3-1. Downsample with a voxel size %.2f" % radius)
        source_down = py3d.voxel_down_sample(source, radius)
        target_down = py3d.voxel_down_sample(target, radius)
        # print("3-2. Estimate normal.")
        py3d.estimate_normals(source_down, py3d.KDTreeSearchParamHybrid(
                radius = radius * 2, max_nn = 30))
        py3d.estimate_normals(target_down, py3d.KDTreeSearchParamHybrid(
                radius = radius * 2, max_nn = 30))
        # print("3-3. Applying colored point cloud registration")
        result_icp = py3d.registration_colored_icp(source_down, target_down,
                radius, current_transformation,
                py3d.ICPConvergenceCriteria(relative_fitness = 1e-6,
                relative_rmse = 1e-6, max_iteration = iter))
        current_transformation = result_icp.transformation

    return result_icp

if __name__ == "__main__":
    # source, source_rgbd = get_pcd_from_rgbd_image_index(0)
    # source2, source_rgbd2 = get_smooth_pcd_from_rgbd_image_index(0)
    # py3d.write_point_cloud("test1.ply", source)
    # py3d.write_point_cloud("test2.ply", source2)
    # exit()
    trans_odometry = np.identity(4)
    pose_graph = py3d.PoseGraph()
    pose_graph.nodes.append(py3d.PoseGraphNode(trans_odometry))
    pinhole_camera_intrinsic = py3d.read_pinhole_camera_intrinsic(
        "E:\\Project\\LightFieldGiga\\SparseToDenseStereo\\camera.json")

    for i in range(5, 20):
        source, source_rgbd = get_pcd_from_rgbd_image_index(i)
        target, target_rgbd = get_pcd_from_rgbd_image_index(i + 1)
        result_icp = registrate_two_color_pcd(source, target) 

        information_icp = py3d.get_information_matrix_from_point_clouds(
                source, target, 1, result_icp.transformation)

        trans_odometry = np.dot(result_icp.transformation, trans_odometry)

        pose_graph.nodes.append(py3d.PoseGraphNode(trans_odometry))
        pose_graph.edges.append(py3d.PoseGraphEdge(i - 5, i + 1 - 5, 
                result_icp.transformation, information_icp, uncertain = False))


    # volume = py3d.ScalableTSDFVolume(voxel_length = 600.0 / 512.0,
    #         sdf_trunc = 12, with_color = True)

    # for i in range(5, 20):
    #     if i % 2 == 0:
    #         source, source_rgbd = get_smooth_pcd_from_rgbd_image_index(i)
    #         # source, source_rgbd = get_pcd_from_rgbd_image_index(i)
    #         pose = pose_graph.nodes[i - 5].pose
    #         volume.integrate(source_rgbd, pinhole_camera_intrinsic, np.linalg.inv(pose))

    # mesh = volume.extract_triangle_mesh()
    # mesh.compute_vertex_normals()
    
    py3d.write_triangle_mesh("test_mesh.ply", mesh)
    
    py3d.draw_geometries([mesh])

