iimport os
import numpy as np

max_r = 5

output = str(max_r)+'m'

for split in ['train', 'val', 'test']: # Argoverse
#for split in ['trainval', 'test']: # nuScenes
    print(split)
    num_logs = 0
    num_frames = 0
    num_points1 = 0.0
    num_points2 = 0.0

    for log in os.listdir(split):
        num_logs += 1

        log_path = os.path.join(split, log)
        output_log_path = os.path.join(output, log_path)
        if not os.path.exists(output_log_path):
            os.makedirs(output_log_path)
        for pc_name in os.listdir(log_path):
            num_frames += 1

            pc_file = os.path.join(log_path, pc_name)
            pc = np.load(pc_file)
            new_pc = []
            for i in range(pc.shape[0]):
                if abs(pc[i][0]) <= max_r and abs(pc[i][1]) <= max_r and abs(pc[i][2]) <= max_r:
                    new_pc.append(pc[i])
            new_pc = np.stack(new_pc, 0)
            num_points1 += pc.shape[0]
            num_points2 += new_pc.shape[0]
            np.save(os.path.join(output_log_path, pc_name), new_pc)
    print('\t#logs:\t%d'%num_logs)
    print('\t#frames:\t%d'%num_frames)
    print('\t#original points:\t%d'%(num_points1))
    print('\t#cropped points:\t%d'%(num_points2))
    print('\t#original average points:\t%f'%(num_points1/num_frames))
    print('\t#cropped average points:\t%f'%(num_points2/num_frames))
