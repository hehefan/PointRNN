import os
import numpy as np

input_dir = 'sweeps/LIDAR_TOP'
output_dir = 'trainval' # or 'test'

for f_name in os.listdir(input_dir):
  arr = f_name.split('__')
  dir_name = arr[0]
  pc_name = arr[2].split('.')[0]
  dir_path = os.path.join(output_dir, dir_name)
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)
  pc_path = os.path.join(dir_path, pc_name)

  scan = np.fromfile(os.path.join(input_dir, f_name), dtype=np.float32)
  points = scan.reshape((-1, 5))[:, :4]
  points = points[:,:-1]
  np.save(pc_path, points)
#'sweeps/LIDAR_TOP/n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243147470.pcd.bin'
