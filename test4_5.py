from mmdet3d.apis import init_model, inference_detector

config_file = 'configs/votenet/votenet_8x8_scannet-3d-18class.py'
checkpoint_file = 'checkpoints/votenet_8x8_scannet-3d-18class_20200620_230238-2cea9c3a0.pth'

# 从配置文件和预训练的模型文件中构建模型
model = init_model(config_file, checkpoint_file, device='cuda:0')

# 测试单个文件并可视化结果
point_cloud = '/home/robot413/mmdetection3d/data/sunrgbd/points/000001.bin'
result, data = inference_detector(model, point_cloud)
# 可视化结果并且将结果保存到 'results' 文件夹
model.show_results(data, result, out_dir='results')