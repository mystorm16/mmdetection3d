# show_pkl.py

import pickle

path = '/home/robot413/mmdetection3d/data/sunrgbd/sunrgbd_infos_train.pkl'  # path='/root/……/aus_openface.pkl'   pkl文件所在路径

f = open(path, 'rb')
data = pickle.load(f)

print(data)
print(len(data))