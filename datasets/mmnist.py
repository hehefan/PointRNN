"""
Moving MNIST (MMNIST) Point Cloud
"""

import os
import numpy as np
import gzip

# the pixel value in MNIST ranges from 0 to 255 and the digit size is 28
# if pixel_threshold = 1 (non-zero), the max number of points for a digit is 351 and the min number of points is 34


def random_trajectory(image_size, seq_length, step_length):
    canvas_size = image_size - 28

    # Initial position uniform random inside the box.
    y = np.random.rand()
    x = np.random.rand()

    # Choose a random velocity.
    theta = np.random.rand() * 2 * np.pi
    v_y = np.sin(theta)
    v_x = np.cos(theta)

    start_y = np.zeros(seq_length)
    start_x = np.zeros(seq_length)
    for i in range(seq_length):
        # Take a step along velocity.
        y += v_y * step_length
        x += v_x * step_length

        # Bounce off edges.
        if x <= 0:
          x = 0
          v_x = -v_x
        if x >= 1.0:
          x = 1.0
          v_x = -v_x
        if y <= 0:
          y = 0
          v_y = -v_y
        if y >= 1.0:
          y = 1.0
          v_y = -v_y
        start_y[i] = y
        start_x[i] = x

    # Scale to the size of the canvas.
    start_y = (canvas_size * start_y).astype(np.int32)
    start_x = (canvas_size * start_x).astype(np.int32)
    return start_y, start_x

class MMNIST(object):
    def __init__(self, root='data/mnist', seq_length=20, num_digits=1, image_size=64, step_length=0.1, num_points=128, pixel_threshold=16, train=True):
        self.seq_length = seq_length
        self.num_digits = num_digits
        self.image_size = image_size
        self.num_points = num_points
        self.step_length = step_length
        self.pixel_threshold = pixel_threshold

        if train:
            file_path = os.path.join(root, 'train-images-idx3-ubyte.gz')
        else:
            file_path = os.path.join(root, 't10k-images-idx3-ubyte.gz')
        with gzip.open(file_path, 'r') as f:
            self.data = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)


    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, _):

        fail = True
        while fail:
            video = np.zeros([self.seq_length, self.image_size, self.image_size])
            cloud_sequence = []
            for i in range(self.num_digits):
                ty, tx = random_trajectory(self.image_size, self.seq_length, self.step_length)
                digit = self.data[np.random.randint(self.data.shape[0])]
                for j in range(self.seq_length):
                    top = ty[j]
                    left = tx[j]
                    bottom = top  + 28
                    right  = left + 28
                    video[j,  top:bottom, left:right] += digit

            for i in range(self.seq_length):
                image = video[i]
                cloud = np.column_stack(np.where(image >= self.pixel_threshold))
                if cloud.shape[0] < self.num_points:
                    fail = True
                    break
                else:
                    fail = False
                random_selection = np.random.choice(cloud.shape[0], size=self.num_points, replace=False)
                cloud = cloud[random_selection]
                cloud_sequence.append(cloud)

        cloud_sequence2D = np.stack(cloud_sequence, axis=0)
        cloud_sequence3D = np.concatenate((cloud_sequence2D, np.zeros((self.seq_length, self.num_points, 1), dtype=cloud_sequence2D.dtype)),2)

        return cloud_sequence3D
