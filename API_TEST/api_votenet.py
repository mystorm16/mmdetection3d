from mmdet3d.apis import init_model, inference_detector
import time

# Kitti
# config_file = '/home/robot413/mmdetection3d/configs/second/hv_second_secfpn_6x8_80e_kitti-3d-car.py'
# checkpoints_file ='/home/robot413/mmdetection3d/checkpoints/hv_second_secfpn_6x8_80e_kitti-3d-car_20200620_230238-393f000c.pth'

# VoteNet ScanNet
# 采样点数目修改：/home/robot413/mmdetection3d/configs/_base_/datasets/scannet-3d-18class.py
votenet_config_file = '/home/robot413/mmdetection3d/configs/votenet/votenet_8x8_scannet-3d-18class.py'
votenet_checkpoints_file = '/home/robot413/mmdetection3d/checkpoints/votenet_8x8_scannet-3d-18class_20210823_234503-cf8134fa.pth'

# VoteNet SUNRGBD
# config_file = '/home/robot413/mmdetection3d/configs/votenet/votenet_16x8_sunrgbd-3d-10class.py'
# checkpoints_file = '/home/robot413/mmdetection3d/checkpoints/votenet_16x8_sunrgbd-3d-10class_20200620_230238-4483c0c0out.pth'

# H3DNet
H3DNet_config_file = '/home/robot413/mmdetection3d/configs/h3dnet/h3dnet_3x8_scannet-3d-18class.py'
H3DNet_checkpoints_file = '/home/robot413/mmdetection3d/checkpoints/h3dnet_scannet-3d-18class_20200830_000136-02e36246cc.pth'

# Group-Free 3D
Group_config_file = '/home/robot413/mmdetection3d/configs/groupfree3d/groupfree3d_8x4_scannet-3d-18class-w2x-L12-O512.py'
Group_checkpoints_file = '/home/robot413/mmdetection3d/checkpoints/groupfree3d_8x4_scannet-3d-18class-w2x-L12-O512_20210702_220204-187b71c7.pth'


# 从配置文件和预训练的模型文件中构建模型
def run_detection(config, checkpoints):
    model = init_model(config, checkpoints, device='cuda:0')
    # point_cloud = '/home/robot413/mmdetection3d/API_TEST/scan10.bin'
    point_cloud = '/home/robot413/mmdetection3d/API_TEST/point_generate/frame_1500_scale_1200.bin'

    t0 = time.perf_counter()
    result, data = inference_detector(model, point_cloud)
    print(time.perf_counter() - t0, "seconds process time")

    model.show_results(data, result, out_dir='results', show=True)


# run_detection(Group_config_file, Group_checkpoints_file)
run_detection(votenet_config_file, votenet_checkpoints_file)
