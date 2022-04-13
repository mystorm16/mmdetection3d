import numpy as np
import pandas as pd
from plyfile import PlyData


def convert_ply(input_path, output_path):
    plydata = PlyData.read(input_path)  # read file
    data = plydata.elements[0].data  # read data
    data_pd = pd.DataFrame(data)  # convert to DataFrame
    data_np = np.zeros(data_pd.shape, dtype=np.float)  # initialize array to store data
    property_names = data[0].dtype.names  # read names of properties
    for i, name in enumerate(
            property_names):  # read data by property
        data_np[:, i] = data_pd[name]

    data_np = data_np[..., 0:6]  # 把7维的数据除去最后一维 rgba->rgb

    data_np.astype(np.float32).tofile(output_path)


convert_ply('/home/robot413/mmdetection3d/API_TEST/test_diff_scale.bin.ply',
            '/home/robot413/mmdetection3d/API_TEST/test_diff_scale.bin')

# scene0010_00_vh_clean_2.ply是7个元素表示一个点，3 location + 4 rgba
# convert_ply('/home/robot413/mmdetection3d/data/scannet/scans/scene0010_00/scene0010_00_vh_clean_2.ply',
#             './test.bin')
# convert_ply('/home/robot413/mmdetection3d/data/scannet/scans/scene0010_00/scene0010_00_vh_clean_2.labels.ply',
# './test.bin')
