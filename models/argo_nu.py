import os
import sys
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))
sys.path.append(os.path.join(ROOT_DIR, 'modules/tf_ops/nn_distance'))
sys.path.append(os.path.join(ROOT_DIR, 'modules/tf_ops/approxmatch'))

from pointnet2 import *
from pointrnn_cell_impl import *
import tf_nndistance
import tf_approxmatch

class PointRNN(object):
    def __init__(self, batch_size, seq_length, num_points=1024, num_samples=8, knn=False, alpha=1.0, beta=1.0, learning_rate=0.001, max_gradient_norm=5.0, is_training=False):

        self.global_step = tf.Variable(0, trainable=False)

        self.inputs = tf.placeholder(tf.float32, [batch_size, seq_length, num_points, 3])
        frames = tf.split(value=self.inputs, num_or_size_splits=seq_length, axis=1)
        frames = [tf.squeeze(input=frame, axis=[1]) for frame in frames]

        cell1 = PointRNNCell(radius=1.0+1e-6, nsample=3*num_samples, out_channels=128, knn=knn, pooling='max')
        cell2 = PointRNNCell(radius=2.0+1e-6, nsample=2*num_samples, out_channels=256, knn=knn, pooling='max')
        cell3 = PointRNNCell(radius=4.0+1e-6, nsample=1*num_samples, out_channels=512, knn=knn, pooling='max')

        # context
        states1 = None
        states2 = None
        states3 = None
        for i in range(int(seq_length/2)):
            # 512
            xyz1, _, _, _ = sample_and_group(int(num_points/2), radius=0.5+1e-6, nsample=num_samples, xyz=frames[i], points=None, knn=False, use_xyz=False)
            with tf.variable_scope('encoder_1', reuse=tf.AUTO_REUSE) as scope:
                states1 = cell1((xyz1, None), states1)
                s_xyz1, s_feat1 = states1
            # 256
            xyz2, feat2, _, _ = sample_and_group(int(num_points/2/2), radius=1.0+1e-6, nsample=num_samples, xyz=s_xyz1, points=s_feat1, knn=False, use_xyz=False)
            feat2 = tf.reduce_max(feat2, axis=[2], keepdims=False, name='maxpool')
            with tf.variable_scope('encoder_2', reuse=tf.AUTO_REUSE) as scope:
                states2 = cell2((xyz2, feat2), states2)
                s_xyz2, s_feat2 = states2
            # 128
            xyz3, feat3, _, _ = sample_and_group(int(num_points/2/2/2), radius=2.0+1e-6, nsample=num_samples, xyz=s_xyz2, points=s_feat2, knn=False, use_xyz=False)
            feat3 = tf.reduce_max(feat3, axis=[2], keepdims=False, name='maxpool')
            with tf.variable_scope('encoder_3', reuse=tf.AUTO_REUSE) as scope:
                states3 = cell3((xyz3, feat3), states3)

        # prediction
        predicted_motions = []
        predicted_frames = []
        input_frame = frames[int(seq_length/2)-1]
        for i in range(int(seq_length/2), seq_length):
            # 512
            xyz1, _, _, _ = sample_and_group(int(num_points/2), radius=0.5+1e-6, nsample=num_samples, xyz=input_frame, points=None, knn=False, use_xyz=False)
            with tf.variable_scope('decoder_1', reuse=tf.AUTO_REUSE) as scope:
                states1 = cell1((xyz1, None), states1)
                s_xyz1, s_feat1 = states1
            # 256
            xyz2, feat2, _, _ = sample_and_group(int(num_points/2/2), radius=1.0+1e-6, nsample=num_samples, xyz=s_xyz1, points=s_feat1, knn=False, use_xyz=False)
            feat2 = tf.reduce_max(feat2, axis=[2], keepdims=False, name='maxpool')
            with tf.variable_scope('decoder_2', reuse=tf.AUTO_REUSE) as scope:
                states2 = cell2((xyz2, feat2), states2)
                s_xyz2, s_feat2 = states2
            # 128
            xyz3, feat3, _, _ = sample_and_group(int(num_points/2/2/2), radius=2.0+1e-6, nsample=num_samples, xyz=s_xyz2, points=s_feat2, knn=False, use_xyz=False)
            feat3 = tf.reduce_max(feat3, axis=[2], keepdims=False, name='maxpool')
            with tf.variable_scope('decoder_3', reuse=tf.AUTO_REUSE) as scope:
                states3 = cell3((xyz3, feat3), states3)
                s_xyz3, s_feat3 = states3


            with tf.variable_scope('fp', reuse=tf.AUTO_REUSE) as scope:
                l2_feat = pointnet_fp_module(xyz2,
                                             xyz3,
                                             s_feat2,
                                             s_feat3,
                                             mlp=[256],
                                             last_mlp_activation=True,
                                             scope='fp2')
                l1_feat = pointnet_fp_module(xyz1,
                                             xyz2,
                                             s_feat1,
                                             l2_feat,
                                             mlp=[256],
                                             last_mlp_activation=True,
                                             scope='fp1')
                l0_feat = pointnet_fp_module(input_frame,
                                             xyz1,
                                             None,
                                             l1_feat,
                                             mlp=[256],
                                             last_mlp_activation=True,
                                             scope='fp0')

            with tf.variable_scope('fc', reuse=tf.AUTO_REUSE) as scope:
                predicted_motion = tf.layers.conv1d(inputs=l0_feat, filters=128, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=tf.nn.relu, name='fc1')
                predicted_motion = tf.layers.conv1d(inputs=predicted_motion, filters=3, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=None, name='fc2')

            predicted_motions.append(predicted_motion)
            input_frame += predicted_motion
            predicted_frames.append(input_frame)

        # loss
        if is_training:
            self.loss = self.emd = self.cd = 0
            for i in range(int(seq_length/2)):
                match = tf_approxmatch.approx_match(frames[i+int(seq_length/2)], predicted_frames[i])
                emd_distance = tf.reduce_mean(tf_approxmatch.match_cost(frames[i+int(seq_length/2)], predicted_frames[i], match))
                loss_emd = emd_distance
                self.emd += loss_emd

                dists_forward, _, dists_backward, _ = tf_nndistance.nn_distance(predicted_frames[i], frames[i+int(seq_length/2)])
                loss_cd = tf.reduce_mean(dists_forward+dists_backward)
                self.cd += loss_cd

                self.loss += (alpha*loss_cd + beta*loss_emd)

            self.cd /= int(seq_length/2)
            self.emd /= (int(seq_length/2)*num_points)

            self.loss /= int(seq_length/2)

            params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, params)
            clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
            self.train_op = tf.train.AdamOptimizer(learning_rate).apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

        self.predicted_motions = tf.stack(values=predicted_motions, axis=1)
        self.predicted_frames = tf.stack(values=predicted_frames, axis=1)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)


class PointGRU(object):
    def __init__(self, batch_size, seq_length, num_points=1024, num_samples=8, knn=False, alpha=1.0, beta=1.0, learning_rate=0.001, max_gradient_norm=5.0, is_training=False):

        self.global_step = tf.Variable(0, trainable=False)

        self.inputs = tf.placeholder(tf.float32, [batch_size, seq_length, num_points, 3])
        frames = tf.split(value=self.inputs, num_or_size_splits=seq_length, axis=1)
        frames = [tf.squeeze(input=frame, axis=[1]) for frame in frames]

        cell1 = PointGRUCell(radius=1.0+1e-6, nsample=3*num_samples, out_channels=128, knn=knn, pooling='max')
        cell2 = PointGRUCell(radius=2.0+1e-6, nsample=2*num_samples, out_channels=256, knn=knn, pooling='max')
        cell3 = PointGRUCell(radius=4.0+1e-6, nsample=1*num_samples, out_channels=512, knn=knn, pooling='max')

        # context
        states1 = None
        states2 = None
        states3 = None
        for i in range(int(seq_length/2)):
            # 512
            xyz1, _, _, _ = sample_and_group(int(num_points/2), radius=0.5+1e-6, nsample=num_samples, xyz=frames[i], points=None, knn=False, use_xyz=False)
            with tf.variable_scope('encoder_1', reuse=tf.AUTO_REUSE) as scope:
                states1 = cell1((xyz1, None), states1)
                s_xyz1, s_feat1 = states1
            # 256
            xyz2, feat2, _, _ = sample_and_group(int(num_points/2/2), radius=1.0+1e-6, nsample=num_samples, xyz=s_xyz1, points=s_feat1, knn=False, use_xyz=False)
            feat2 = tf.reduce_max(feat2, axis=[2], keepdims=False, name='maxpool')
            with tf.variable_scope('encoder_2', reuse=tf.AUTO_REUSE) as scope:
                states2 = cell2((xyz2, feat2), states2)
                s_xyz2, s_feat2 = states2
            # 128
            xyz3, feat3, _, _ = sample_and_group(int(num_points/2/2/2), radius=2.0+1e-6, nsample=num_samples, xyz=s_xyz2, points=s_feat2, knn=False, use_xyz=False)
            feat3 = tf.reduce_max(feat3, axis=[2], keepdims=False, name='maxpool')
            with tf.variable_scope('encoder_3', reuse=tf.AUTO_REUSE) as scope:
                states3 = cell3((xyz3, feat3), states3)

        # prediction
        predicted_motions = []
        predicted_frames = []
        input_frame = frames[int(seq_length/2)-1]
        for i in range(int(seq_length/2), seq_length):
            # 512
            xyz1, _, _, _ = sample_and_group(int(num_points/2), radius=0.5+1e-6, nsample=num_samples, xyz=input_frame, points=None, knn=False, use_xyz=False)
            with tf.variable_scope('decoder_1', reuse=tf.AUTO_REUSE) as scope:
                states1 = cell1((xyz1, None), states1)
                s_xyz1, s_feat1 = states1
            # 256
            xyz2, feat2, _, _ = sample_and_group(int(num_points/2/2), radius=1.0+1e-6, nsample=num_samples, xyz=s_xyz1, points=s_feat1, knn=False, use_xyz=False)
            feat2 = tf.reduce_max(feat2, axis=[2], keepdims=False, name='maxpool')
            with tf.variable_scope('decoder_2', reuse=tf.AUTO_REUSE) as scope:
                states2 = cell2((xyz2, feat2), states2)
                s_xyz2, s_feat2 = states2
            # 128
            xyz3, feat3, _, _ = sample_and_group(int(num_points/2/2/2), radius=2.0+1e-6, nsample=num_samples, xyz=s_xyz2, points=s_feat2, knn=False, use_xyz=False)
            feat3 = tf.reduce_max(feat3, axis=[2], keepdims=False, name='maxpool')
            with tf.variable_scope('decoder_3', reuse=tf.AUTO_REUSE) as scope:
                states3 = cell3((xyz3, feat3), states3)
                s_xyz3, s_feat3 = states3


            with tf.variable_scope('fp', reuse=tf.AUTO_REUSE) as scope:
                l2_feat = pointnet_fp_module(xyz2,
                                             xyz3,
                                             s_feat2,
                                             s_feat3,
                                             mlp=[256],
                                             last_mlp_activation=True,
                                             scope='fp2')
                l1_feat = pointnet_fp_module(xyz1,
                                             xyz2,
                                             s_feat1,
                                             l2_feat,
                                             mlp=[256],
                                             last_mlp_activation=True,
                                             scope='fp1')
                l0_feat = pointnet_fp_module(input_frame,
                                             xyz1,
                                             None,
                                             l1_feat,
                                             mlp=[256],
                                             last_mlp_activation=True,
                                             scope='fp0')

            with tf.variable_scope('fc', reuse=tf.AUTO_REUSE) as scope:
                predicted_motion = tf.layers.conv1d(inputs=l0_feat, filters=128, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=tf.nn.relu, name='fc1')
                predicted_motion = tf.layers.conv1d(inputs=predicted_motion, filters=3, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=None, name='fc2')

            predicted_motions.append(predicted_motion)
            input_frame += predicted_motion
            predicted_frames.append(input_frame)

        # loss
        if is_training:
            self.loss = self.emd = self.cd = 0
            for i in range(int(seq_length/2)):
                match = tf_approxmatch.approx_match(frames[i+int(seq_length/2)], predicted_frames[i])
                emd_distance = tf.reduce_mean(tf_approxmatch.match_cost(frames[i+int(seq_length/2)], predicted_frames[i], match))
                loss_emd = emd_distance
                self.emd += loss_emd

                dists_forward, _, dists_backward, _ = tf_nndistance.nn_distance(predicted_frames[i], frames[i+int(seq_length/2)])
                loss_cd = tf.reduce_mean(dists_forward+dists_backward)
                self.cd += loss_cd

                self.loss += (alpha*loss_cd + beta*loss_emd)

            self.cd /= int(seq_length/2)
            self.emd /= (int(seq_length/2)*num_points)

            self.loss /= int(seq_length/2)

            params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, params)
            clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
            self.train_op = tf.train.AdamOptimizer(learning_rate).apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

        self.predicted_motions = tf.stack(values=predicted_motions, axis=1)
        self.predicted_frames = tf.stack(values=predicted_frames, axis=1)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

class PointLSTM(object):
    def __init__(self, batch_size, seq_length, num_points=1024, num_samples=8, knn=False, alpha=1.0, beta=1.0, learning_rate=0.001, max_gradient_norm=5.0, is_training=False):

        self.global_step = tf.Variable(0, trainable=False)

        self.inputs = tf.placeholder(tf.float32, [batch_size, seq_length, num_points, 3])
        frames = tf.split(value=self.inputs, num_or_size_splits=seq_length, axis=1)
        frames = [tf.squeeze(input=frame, axis=[1]) for frame in frames]

        cell1 = PointLSTMCell(radius=1.0+1e-6, nsample=3*num_samples, out_channels=128, knn=knn, pooling='max')
        cell2 = PointLSTMCell(radius=2.0+1e-6, nsample=2*num_samples, out_channels=256, knn=knn, pooling='max')
        cell3 = PointLSTMCell(radius=4.0+1e-6, nsample=1*num_samples, out_channels=512, knn=knn, pooling='max')

        # context
        states1 = None
        states2 = None
        states3 = None
        for i in range(int(seq_length/2)):
            # 512
            xyz1, _, _, _ = sample_and_group(int(num_points/2), radius=0.5+1e-6, nsample=num_samples, xyz=frames[i], points=None, knn=False, use_xyz=False)
            with tf.variable_scope('encoder_1', reuse=tf.AUTO_REUSE) as scope:
                states1 = cell1((xyz1, None), states1)
                s_xyz1, h_feat1, _ = states1
            # 256
            xyz2, feat2, _, _ = sample_and_group(int(num_points/2/2), radius=1.0+1e-6, nsample=num_samples, xyz=s_xyz1, points=h_feat1, knn=False, use_xyz=False)
            feat2 = tf.reduce_max(feat2, axis=[2], keepdims=False, name='maxpool')
            with tf.variable_scope('encoder_2', reuse=tf.AUTO_REUSE) as scope:
                states2 = cell2((xyz2, feat2), states2)
                s_xyz2, h_feat2, _ = states2
            # 128
            xyz3, feat3, _, _ = sample_and_group(int(num_points/2/2/2), radius=2.0+1e-6, nsample=num_samples, xyz=s_xyz2, points=h_feat2, knn=False, use_xyz=False)
            feat3 = tf.reduce_max(feat3, axis=[2], keepdims=False, name='maxpool')
            with tf.variable_scope('encoder_3', reuse=tf.AUTO_REUSE) as scope:
                states3 = cell3((xyz3, feat3), states3)

        # prediction
        predicted_motions = []
        predicted_frames = []
        input_frame = frames[int(seq_length/2)-1]
        for i in range(int(seq_length/2), seq_length):
            # 512
            xyz1, _, _, _ = sample_and_group(int(num_points/2), radius=0.5+1e-6, nsample=num_samples, xyz=input_frame, points=None, knn=False, use_xyz=False)
            with tf.variable_scope('decoder_1', reuse=tf.AUTO_REUSE) as scope:
                states1 = cell1((xyz1, None), states1)
                s_xyz1, h_feat1, _ = states1
            # 256
            xyz2, feat2, _, _ = sample_and_group(int(num_points/2/2), radius=1.0+1e-6, nsample=num_samples, xyz=s_xyz1, points=h_feat1, knn=False, use_xyz=False)
            feat2 = tf.reduce_max(feat2, axis=[2], keepdims=False, name='maxpool')
            with tf.variable_scope('decoder_2', reuse=tf.AUTO_REUSE) as scope:
                states2 = cell2((xyz2, feat2), states2)
                s_xyz2, h_feat2, _ = states2
            # 128
            xyz3, feat3, _, _ = sample_and_group(int(num_points/2/2/2), radius=2.0+1e-6, nsample=num_samples, xyz=s_xyz2, points=h_feat2, knn=False, use_xyz=False)
            feat3 = tf.reduce_max(feat3, axis=[2], keepdims=False, name='maxpool')
            with tf.variable_scope('decoder_3', reuse=tf.AUTO_REUSE) as scope:
                states3 = cell3((xyz3, feat3), states3)
                s_xyz3, h_feat3, _ = states3


            with tf.variable_scope('fp', reuse=tf.AUTO_REUSE) as scope:
                l2_feat = pointnet_fp_module(xyz2,
                                             xyz3,
                                             h_feat2,
                                             h_feat3,
                                             mlp=[256],
                                             last_mlp_activation=True,
                                             scope='fp2')
                l1_feat = pointnet_fp_module(xyz1,
                                             xyz2,
                                             h_feat1,
                                             l2_feat,
                                             mlp=[256],
                                             last_mlp_activation=True,
                                             scope='fp1')
                l0_feat = pointnet_fp_module(input_frame,
                                             xyz1,
                                             None,
                                             l1_feat,
                                             mlp=[256],
                                             last_mlp_activation=True,
                                             scope='fp0')

            with tf.variable_scope('fc', reuse=tf.AUTO_REUSE) as scope:
                predicted_motion = tf.layers.conv1d(inputs=l0_feat, filters=128, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=tf.nn.relu, name='fc1')
                predicted_motion = tf.layers.conv1d(inputs=predicted_motion, filters=3, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=None, name='fc2')

            predicted_motions.append(predicted_motion)
            input_frame += predicted_motion
            predicted_frames.append(input_frame)

        # loss
        if is_training:
            self.loss = self.emd = self.cd = 0
            for i in range(int(seq_length/2)):
                match = tf_approxmatch.approx_match(frames[i+int(seq_length/2)], predicted_frames[i])
                emd_distance = tf.reduce_mean(tf_approxmatch.match_cost(frames[i+int(seq_length/2)], predicted_frames[i], match))
                loss_emd = emd_distance
                self.emd += loss_emd

                dists_forward, _, dists_backward, _ = tf_nndistance.nn_distance(predicted_frames[i], frames[i+int(seq_length/2)])
                loss_cd = tf.reduce_mean(dists_forward+dists_backward)
                self.cd += loss_cd

                self.loss += (alpha*loss_cd + beta*loss_emd)

            self.cd /= int(seq_length/2)
            self.emd /= (int(seq_length/2)*num_points)

            self.loss /= int(seq_length/2)

            params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, params)
            clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
            self.train_op = tf.train.AdamOptimizer(learning_rate).apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

        self.predicted_motions = tf.stack(values=predicted_motions, axis=1)
        self.predicted_frames = tf.stack(values=predicted_frames, axis=1)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
