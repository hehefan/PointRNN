import os
import sys
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))
sys.path.append(os.path.join(ROOT_DIR, 'modules/tf_ops/nn_distance'))
sys.path.append(os.path.join(ROOT_DIR, 'modules/tf_ops/approxmatch'))

import tf_nndistance
import tf_approxmatch

class Model(object):
    def __init__(self, seq_length, num_points=128):

        self.ground_truth = tf.placeholder(tf.float32, [1, seq_length, num_points, 3])
        self.prediction = tf.placeholder(tf.float32, [1, seq_length, num_points, 3])

        gt_frames = tf.split(value=self.ground_truth, num_or_size_splits=seq_length, axis=1)
        gt_frames = [tf.squeeze(input=frame, axis=[1]) for frame in gt_frames]

        pd_frames = tf.split(value=self.prediction, num_or_size_splits=seq_length, axis=1)
        pd_frames = [tf.squeeze(input=frame, axis=[1]) for frame in pd_frames]

        cds, emds = [], []

        for i in range(seq_length):
            match = tf_approxmatch.approx_match(gt_frames[i], pd_frames[i])
            emd_distance = tf.reduce_mean(tf_approxmatch.match_cost(gt_frames[i], pd_frames[i], match))
            emds.append(emd_distance)

            dists_forward, _, dists_backward, _ = tf_nndistance.nn_distance(pd_frames[i], gt_frames[i])
            cd_distance = tf.reduce_mean(dists_forward+dists_backward)
            cds.append(cd_distance)

        self.cds = tf.stack(cds, 0)
        self.emds = tf.stack(emds, 0)
