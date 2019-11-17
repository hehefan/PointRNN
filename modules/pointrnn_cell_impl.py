import os
import sys
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules/tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR,'modules/tf_ops/grouping'))
sys.path.append(os.path.join(ROOT_DIR,'modules/tf_ops/3d_interpolation'))
from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import query_ball_point, group_point, knn_point
from tf_interpolate import three_nn, three_interpolate

def point_rnn(P1,
              P2,
              X1,
              S2,
              radius,
              nsample,
              out_channels,
              knn=False,
              pooling='max',
              scope='point_rnn'):
    """
    Input:
        P1:     (batch_size, npoint, 3)
        P2:     (batch_size, npoint, 3)
        X1:     (batch_size, npoint, feat_channels)
        S2:     (batch_size, npoint, out_channels)
    Output:
        S1:     (batch_size, npoint, out_channels)
    """
    # 1. Sample points
    if knn:
        _, idx = knn_point(nsample, P2, P1)
    else:
        idx, cnt = query_ball_point(radius, nsample, P2, P1)
        _, idx_knn = knn_point(nsample, P2, P1)
        cnt = tf.tile(tf.expand_dims(cnt, -1), [1, 1, nsample])
        idx = tf.where(cnt > (nsample-1), idx, idx_knn)

    # 2.1 Group P2 points
    P2_grouped = group_point(P2, idx)                       # batch_size, npoint, nsample, 3
    # 2.2 Group P2 states
    S2_grouped = group_point(S2, idx)                       # batch_size, npoint, nsample, out_channels

    # 3. Calcaulate displacements
    P1_expanded = tf.expand_dims(P1, 2)                     # batch_size, npoint, 1,       3
    displacement = P2_grouped - P1_expanded                 # batch_size, npoint, nsample, 3

    # 4. Concatenate X1, S2 and displacement
    if X1 is not None:
        X1_expanded = tf.tile(tf.expand_dims(X1, 2), [1, 1, nsample, 1])                # batch_size, npoint, sample,  feat_channels
        correlation = tf.concat([S2_grouped, X1_expanded], axis=3)                      # batch_size, npoint, nsample, feat_channels+out_channels
        correlation = tf.concat([correlation, displacement], axis=3)                    # batch_size, npoint, nsample, feat_channels+out_channels+3
    else:
        correlation = tf.concat([S2_grouped, displacement], axis=3)                     # batch_size, npoint, nsample, out_channels+3

    # 5. Fully-connected layer (the only parameters)
    with tf.variable_scope(scope) as sc:
        S1 = tf.layers.conv2d(inputs=correlation, filters=out_channels, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=None, name='fc')

    # 6. Pooling
    if pooling=='max':
        return tf.reduce_max(S1, axis=[2], keepdims=False)
    elif pooling=='avg':
        return tf.reduce_mean(S1, axis=[2], keepdims=False)

class PointRNNCell(object):
    def __init__(self,
                 radius,
                 nsample,
                 out_channels,
                 knn=False,
                 pooling='max'):

        self.radius = radius
        self.nsample = nsample
        self.out_channels = out_channels
        self.knn = knn
        self.pooling = pooling

    def init_state(self, inputs, state_initializer=tf.zeros_initializer(), dtype=tf.float32):
        """Helper function to create an initial state given inputs.
        Args:
            inputs: tube of (P, X). the first dimension P or X being batch_size
            state_initializer: Initializer(shape, dtype) for state Tensor.
            dtype: Optional dtype, needed when inputs is None.
        Returns:
            A tube of tensors representing the initial states.
        """
        # Handle both the dynamic shape as well as the inferred shape.
        P, X = inputs

        # inferred_batch_size = tf.shape(P)[0]
        inferred_batch_size = P.get_shape().with_rank_at_least(1)[0]
        inferred_npoints = P.get_shape().with_rank_at_least(1)[1]
        inferred_xyz_dimensions = P.get_shape().with_rank_at_least(1)[2]

        P = state_initializer([inferred_batch_size, inferred_npoints, inferred_xyz_dimensions], dtype=P.dtype)
        S = state_initializer([inferred_batch_size, inferred_npoints, self.out_channels], dtype=dtype)

        return (P, S)

    def __call__(self, inputs, states):
        if states is None:
            states = self.init_state(inputs)

        P1, X1 = inputs
        P2, S2 = states

        S1 = point_rnn(P1, P2, X1, S2, radius=self.radius, nsample=self.nsample, out_channels=self.out_channels, knn=self.knn, pooling=self.pooling)

        return (P1, S1)

class PointGRUCell(PointRNNCell):
    def __init__(self,
                 radius,
                 nsample,
                 out_channels,
                 knn=False,
                 pooling='max'):
        super().__init__(radius, nsample, out_channels, knn, pooling)

    def __call__(self, inputs, states):
        if states is None:
            states = self.init_state(inputs)

        P1, X1 = inputs
        P2, S2 = states

        Z = point_rnn(P1, P2, X1, S2, radius=self.radius, nsample=self.nsample, out_channels=self.out_channels, knn=self.knn, pooling=self.pooling, scope='update_gate')
        R = point_rnn(P1, P2, X1, S2, radius=self.radius, nsample=self.nsample, out_channels=self.out_channels, knn=self.knn, pooling=self.pooling, scope='reset_gate')
        Z = tf.sigmoid(Z)
        R = tf.sigmoid(R)

        S_old = point_rnn(P1, P2, None, S2, radius=self.radius, nsample=self.nsample, out_channels=self.out_channels, knn=self.knn, pooling=self.pooling, scope='old_state')

        if X1 is None:
            S_new = R*S_old
        else:
            S_new = tf.concat([X1, R*S_old], axis=2)
        S_new = tf.layers.conv1d(inputs=S_new, filters=self.out_channels, kernel_size=1, strides=1, padding='valid', data_format='channels_last', activation=None, name='new_state')
        S_new = tf.tanh(S_new)

        S1 = Z * S_old + (1 - Z) * S_new

        return (P1, S1)

class PointLSTMCell(PointRNNCell):
    def __init__(self,
                 radius,
                 nsample,
                 out_channels,
                 knn=False,
                 pooling='max'):
        super().__init__(radius, nsample, out_channels, knn, pooling)

    def init_state(self, inputs, state_initializer=tf.zeros_initializer(), dtype=tf.float32):
        """Helper function to create an initial state given inputs.
        Args:
            inputs: tube of (P, X). the first dimension P or X being batch_size
            state_initializer: Initializer(shape, dtype) for state Tensor.
            dtype: Optional dtype, needed when inputs is None.
        Returns:
            A tube of tensors representing the initial states.
        """
        # Handle both the dynamic shape as well as the inferred shape.
        P, X = inputs

        # inferred_batch_size = tf.shape(P)[0]
        inferred_batch_size = P.get_shape().with_rank_at_least(1)[0]
        inferred_npoints = P.get_shape().with_rank_at_least(1)[1]
        inferred_xyz_dimensions = P.get_shape().with_rank_at_least(1)[2]

        P = state_initializer([inferred_batch_size, inferred_npoints, inferred_xyz_dimensions], dtype=P.dtype)
        H = state_initializer([inferred_batch_size, inferred_npoints, self.out_channels], dtype=dtype)
        C = state_initializer([inferred_batch_size, inferred_npoints, self.out_channels], dtype=dtype)

        return (P, H, C)

    def __call__(self, inputs, states):
        if states is None:
            states = self.init_state(inputs)

        P1, X1 = inputs
        P2, H2, C2 = states

        I = point_rnn(P1, P2, X1, H2, radius=self.radius, nsample=self.nsample, out_channels=self.out_channels, knn=self.knn, pooling=self.pooling, scope='input_gate')
        F = point_rnn(P1, P2, X1, H2, radius=self.radius, nsample=self.nsample, out_channels=self.out_channels, knn=self.knn, pooling=self.pooling, scope='forget_gate')
        O = point_rnn(P1, P2, X1, H2, radius=self.radius, nsample=self.nsample, out_channels=self.out_channels, knn=self.knn, pooling=self.pooling, scope='output_gate')

        C_new = point_rnn(P1, P2, X1, H2, radius=self.radius, nsample=self.nsample, out_channels=self.out_channels, knn=self.knn, pooling=self.pooling, scope='new_cell')
        C_old = point_rnn(P1, P2, None, C2, radius=self.radius, nsample=self.nsample, out_channels=self.out_channels, knn=self.knn, pooling=self.pooling, scope='old_cell')

        I = tf.sigmoid(I)
        F = tf.sigmoid(F)
        O = tf.sigmoid(O)
        C_new = tf.tanh(C_new)

        C1 = F * C_old + I * C_new
        H1 = O * tf.tanh(C1)

        return (P1, H1, C1)
