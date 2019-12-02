import os
import numpy as np

class Argoverse(object):
    def __init__(self, root='data/argo', seq_length=10, num_points=1024, train=True):
        self.seq_length = seq_length
        self.num_points = num_points
        self.data = []

        if train:
            splits = ['train', 'val']
        else:
            splits = ['test']

        for split in splits:
            split_path = os.path.join(root, split)
            for log in os.listdir(split_path):
                log_path = os.path.join(split_path, log)
                log_data = []
                for npy in sorted(os.listdir(log_path)):
                    npy_file = os.path.join(log_path, npy)
                    npy_data = np.load(npy_file)
                    log_data.append(npy_data)
                self.data.append(log_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, _):

        log_data = self.data[np.random.randint(len(self.data))]
        start = np.random.randint(len(log_data)-self.seq_length)

        cloud_sequence = []
        for i in range(start, start+self.seq_length):
            pc = log_data[i]
            npoints = pc.shape[0]
            sample_idx = np.random.choice(npoints, self.num_points, replace=False)
            cloud_sequence.append(pc[sample_idx, :])

        return np.stack(cloud_sequence, axis=0)

class nuScenes(object):
    def __init__(self, root='data/nu', seq_length=10, num_points=1024, train=True):
        self.seq_length = seq_length
        self.num_points = num_points
        self.data = []

        if train:
            splits = ['trainval']
        else:
            splits = ['test']

        for split in splits:
            split_path = os.path.join(root, split)
            for log in os.listdir(split_path):
                log_path = os.path.join(split_path, log)
                log_data = []
                for npy in sorted(os.listdir(log_path)):
                    npy_file = os.path.join(log_path, npy)
                    npy_data = np.load(npy_file)
                    log_data.append(npy_data)
                self.data.append(log_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, _):

        log_data = self.data[np.random.randint(len(self.data))]
        start = np.random.randint(len(log_data)-self.seq_length)

        cloud_sequence = []
        for i in range(start, start+self.seq_length):
            pc = log_data[i]
            npoints = pc.shape[0]
            sample_idx = np.random.choice(npoints, self.num_points, replace=False)
            cloud_sequence.append(pc[sample_idx, :])

        return np.stack(cloud_sequence, axis=0)
