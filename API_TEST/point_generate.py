# Author：mystorm16
# Dataset:ScanNet
# Description：rgb+depth to point-cloud
# 点云是否在世界系下、是否对齐、depth_scale都对检测结果影响很大

import time
import numpy as np
import open3d as o3d
import cv2 as cv
from plyfile import PlyData, PlyElement

depth_scale = 300  # 重要参数，对检测结果影响很大


class point_cloud_generator:
    # 读取文件路径
    def __init__(self, depth_root, rgb_root, pose_root, ply_file, bin_file):
        self.depth = depth_root
        self.rgb = rgb_root
        self.pose = pose_root
        self.ply_file = ply_file
        self.bin_file = bin_file

        # rgb和深度相机内参
        self.fx_color = 1170.187988
        self.fx_depth = 577.870605
        self.fy_color = 1170.187988
        self.fy_depth = 577.870605
        self.mx_color = 647.750000
        self.mx_depth = 319.500000
        self.my_color = 483.750000
        self.my_depth = 239.500000
        self.colorHeight = 968
        self.colorWidth = 1296

        # rgb和深度相机外参
        self.colorToDepthExtrinsics = [0.999977, 0.004401, 0.005230, - 0.037931,
                                       -0.004314, 0.999852, - 0.016630, - 0.003321,
                                       -0.005303, 0.016607, 0.999848, - 0.021860,
                                       -0.000000, 0.000000, - 0.000000, 1.000000]
        # self.colorToDepthExtrinsics = np.eye(4,4)
        self.colorToDepthExtrinsics = np.array(self.colorToDepthExtrinsics).reshape(4, 4)

        # 点云对齐参数（和外参Twc一样使用）
        self.axisalignment = [-0.694658, 0.719340, 0.000000, 0.241760,
                              -0.719340, -0.694658, 0.000000, 3.999600,
                              0.000000, 0.000000, 1.000000, -0.088575,
                              0.000000, 0.000000, 0.000000, 1.000000]
        self.axisalignment = np.array(self.axisalignment).reshape(4, 4)

        # poses list，元素是4*4的numpy
        self.poses = []

        # 当前帧id
        self.cur_frame = 0

        # 融合帧标志位
        self.flag_fuse = False

    # 生成单帧地图点 frames_num为帧数 在depth相机坐标系下
    def per_generate(self, frames_num, depth2world):
        self.cur_frame = frames_num
        poses = self.readtxt()  # 读取所有位姿
        t0 = time.perf_counter()
        points = []
        depth_path = cv.imread(self.depth + 'frame-' + str(frames_num).zfill(6) + ".depth.pgm", -1)
        rgb_path = cv.imread(self.rgb + 'frame-' + str(frames_num).zfill(6) + ".color.jpg")
        k = 0  # 剔除没有深度值的点
        kk = 0  # 剔除不在rgb图像范围内的点
        # 遍历depth图的每个像素
        for v in range(depth_path.shape[1]):
            for u in range(depth_path.shape[0]):
                if depth_path[u, v] == 0:
                    k = k + 1
                    continue
                points_camd = np.zeros(4)  # depth相机系下的坐标
                points_camd[3] = 1
                points_camd[2] = depth_path[u, v] / depth_scale
                points_camd[0] = (v - self.mx_depth) * points_camd[2] / self.fx_depth  # 这里的v和u顺序要注意
                points_camd[1] = (u - self.my_depth) * points_camd[2] / self.fy_depth
                points_camc = np.dot(self.colorToDepthExtrinsics, points_camd)  # rgb相机下的坐标

                # uu为纵坐标，vv为横坐标
                vv = int(points_camc[0] * self.fx_color / points_camc[2] + self.mx_color)  # 对应rgb图像中的坐标
                uu = int(points_camc[1] * self.fy_color / points_camc[2] + self.my_color)

                if 0 < uu < self.colorHeight and 0 < vv < self.colorWidth:  # 剔除不在rgb图像范围内的点
                    if depth2world:  # 深度相机系 转 世界系
                        point_world = np.dot(poses[frames_num], points_camd)
                        point = np.append(point_world[0:3], rgb_path[uu, vv]).tolist()  # 合并点的位置&颜色
                    else:
                        point = np.append(points_camd[0:3], rgb_path[uu, vv]).tolist()  # 合并点的位置&颜色
                    points.append(point)
                else:
                    kk = kk + 1

        t1 = time.perf_counter()
        print("总输入点：", eval('640*480'), '\n'
                                        "剔除没有深度的点：", k, '\n'
                                                        "剔除投影不在图像内的点：", kk, '\n'
                                                                            '最终数量：', len(points), "\n"
                                                                                                  '处理时间', t1 - t0, "\n")
        points = np.array(points)  # ndarray
        return points

    # 融合多帧地图点，融合前frames_num帧 得到世界坐标系下的坐标
    def fuse_frames(self, frames_num):
        self.flag_fuse = True  # 开启fuse标志位
        self.cur_frame = frames_num  # 融合前n帧

        poses = self.readtxt()
        multi_points_world = []
        for i in range(frames_num):
            per_points_cam = self.per_generate(i, False)
            per_points_cam = np.array(per_points_cam)
            for point in range(per_points_cam.shape[0]):
                xx = np.ones(4)
                xx[0:3] = per_points_cam[point][0:3]
                per_points_world = np.dot(poses[i], xx)
                per_points_world = np.append(per_points_world[0:3], per_points_cam[point][3:6])
                multi_points_world.append(per_points_world)
        multi_points_world = np.array(multi_points_world)
        np.random.shuffle(multi_points_world)
        multi_points_world = multi_points_world[0:40000]
        print("-----------------------")
        print("融合后总点数：", multi_points_world.shape[0])
        return multi_points_world  # ndarray  shape == (点数,6)

    # 点云对齐，对world系下的点使用
    def axisAlignment(self, world_points):
        axis_points = []
        for point in range(world_points.shape[0]):
            xx = np.ones(4)
            xx[0:3] = world_points[point][0:3]
            per_axis_point = np.dot(self.axisalignment, xx)
            per_axis_point = np.append(per_axis_point[0:3], world_points[point][3:6])
            axis_points.append(per_axis_point)
        axis_points = np.array(axis_points)
        np.random.shuffle(axis_points)
        axis_points = axis_points[0:40000]
        print("-----------------------")
        print("对齐后的点数", axis_points.shape[0])
        return axis_points

    # 从点云生成bin文件
    def generate_bin(self, points):
        if self.flag_fuse:
            points.astype(np.float32).tofile(
                self.bin_file + 'fuse_' + 'frame_' + str(self.cur_frame) + '_scale_' + str(depth_scale) + '.bin')
        else:
            points.astype(np.float32).tofile(
                self.bin_file + 'frame_' + str(self.cur_frame) + '_scale_' + str(depth_scale) + '.bin')

    # 从点云生成ply文件
    def generate_ply(self, points):
        points = np.array(points)  # list2numpy
        points = [(points[i, 0], points[i, 1], points[i, 2], points[i, 3], points[i, 4],
                   points[i, 5]) for i in range(points.shape[0])]
        vertex = np.array(points,
                          dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('blue', 'u1'), ('green', 'u1'),
                                 ('red', 'u1')])  # png图像是BGR
        pc = PlyElement.describe(vertex, 'vertex')
        if self.flag_fuse:
            PlyData([pc]).write(
                self.ply_file + 'fuse_' + 'frame_' + str(self.cur_frame) + '_scale_' + str(depth_scale) + '.ply')
        else:
            PlyData([pc]).write(self.ply_file + 'frame_' + str(self.cur_frame) + '_scale_' + str(depth_scale) + '.ply')

    # 读取位姿 txt文件
    def readtxt(self):
        for i in range(2513):
            pose = []
            raw_name = str(i).zfill(6)
            with open(
                    self.pose + "frame-" + raw_name + ".pose.txt",
                    "r") as f:
                for line in f.readlines():
                    data = line.split('\n\t')
                    for strr in data:
                        sub_str = strr.split(' ')
                    if sub_str:
                        pose.append(sub_str)
            self.poses.append(np.array(pose, dtype=float))  # poses list，元素是4*4的numpy
        return self.poses


a = point_cloud_generator(
    '/home/robot413/mmdetection3d/data/scannet/scans/scene0010_00/export/',
    '/home/robot413/mmdetection3d/data/scannet/scans/scene0010_00/export/',
    "/home/robot413/mmdetection3d/data/scannet/scans/scene0010_00/export/",
    '/home/robot413/mmdetection3d/API_TEST/point_generate/',
    '/home/robot413/mmdetection3d/API_TEST/point_generate/')

# pp = a.fuse_frames(5)
pp = a.per_generate(1500, True)
ppp = a.axisAlignment(pp)
a.generate_ply(ppp)
a.generate_bin(ppp)
