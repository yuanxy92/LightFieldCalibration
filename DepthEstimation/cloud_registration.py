#conda install pillow matplotlib
import py3d as py3d
import copy
import numpy as np
import matplotlib.pyplot as plt
import util.camera_param_loader as loader

def get_pcd_from_rgbd_image_index(i):
    color_name = "E:\\Project\\LightFieldGiga\\data\\result\\%04d_left.jpg" % i
    depth_name = "E:\\Project\\LightFieldGiga\\data\\result\\%04d_depth_refine.png" % i
    print(color_name)
    print(depth_name)

    color_raw = py3d.read_image(color_name)
    depth_raw = py3d.read_image(depth_name)
    rgbd_image = py3d.create_rgbd_image_from_color_and_depth(
        color_raw, depth_raw, 100, 300, False);
    pinhole_camera_intrinsic = py3d.read_pinhole_camera_intrinsic(
        "E:\\Project\\LightFieldGiga\\SparseToDenseStereo\\camera.json")
    pcd = py3d.create_point_cloud_from_rgbd_image(rgbd_image,
            pinhole_camera_intrinsic)
    # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return pcd, rgbd_image

if __name__ == "__main__":
    filename = 'E:\\Project\\LightFieldGiga\\data\\CameraParams.txt'
    pinhole_camera_intrinsic = py3d.read_pinhole_camera_intrinsic(
        "E:\\Project\\LightFieldGiga\\SparseToDenseStereo\\camera.json")

    cameras = loader.CameraParamLoader(filename) 
    camera_num = len(cameras.cameras)
    rgbd_images = []

    volume = py3d.ScalableTSDFVolume(voxel_length = 300.0 / 512.0,
        sdf_trunc = 6, with_color = True)

    pcd = py3d.PointCloud()
    for ind in range(30):
        source, source_rgbd = get_pcd_from_rgbd_image_index(ind * 3)
        pose_inv = np.eye(4)
        pose_inv[0:3, 0:3] = np.linalg.inv(cameras.cameras[ind].R)
        pose = np.eye(4)
        pose[0:3, 0:3] = cameras.cameras[ind].R
        print(pose)
        # py3d.draw_geometries([source])
        source.transform(pose_inv)
        pcd = pcd + source
        volume.integrate(source_rgbd, pinhole_camera_intrinsic, pose)

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    py3d.write_triangle_mesh("test_mesh.ply", mesh)
    py3d.write_point_cloud("test_pcd.ply", pcd)