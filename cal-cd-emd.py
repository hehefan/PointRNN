import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from models.eval import Model

parser = argparse.ArgumentParser()
parser.add_argument('--gt', type=str, default='data/test-2mnist-64-256point-20step.npy', help='Ground truth npy file.')
parser.add_argument('--pd', type=str, default='outputs/mmnist-2digit-advanced-pointlstm/test-predictions.npy', help='Prediction npy file.')

args = parser.parse_args()
gt_data = np.load(args.gt)
n_pcs, seq_len, n_pts, dim = gt_data.shape
seq_len = int(seq_len/2)

pd_data = np.load(args.pd)
assert pd_data.shape[1] == seq_len

gt_data = gt_data[:, seq_len:, :, :]
if dim == 2:
    gt_data = np.concatenate((gt_data, np.zeros((n_pcs, seq_len, n_pts, 1), dtype=gt_data.dtype)),3)

model = Model(num_points=n_pts, seq_length=seq_len)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    CDs = 0
    EMDs = 0
    for i in range(gt_data.shape[0]):

        gt_seq = np.expand_dims(gt_data[i], axis=0)
        pd_seq = np.expand_dims(pd_data[i], axis=0)

        feed_dict = {model.ground_truth: gt_seq, model.prediction: pd_seq}
        cds, emds = sess.run([model.cds, model.emds], feed_dict=feed_dict)

        CDs += cds
        EMDs += emds

    CDs /= float(n_pcs)
    EMDs /= float(n_pcs*n_pts)

    avg_CD = np.mean(CDs)
    avg_EMD = np.mean(EMDs)

    print('CD:\t%f'%(avg_CD))
    print(CDs)

    print('EMD:\t%f'%(avg_EMD))
    print(EMDs)
