import os
import sys
import io
from datetime import datetime
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf
from models.argo_nu import PointRNN as Model

parser = argparse.ArgumentParser()
parser.add_argument('--data-path', default='data/test-argo-5m-1024point-10step.npy', help='Data path [default: data/test-argo-5m-1024point-10step.npy]')
parser.add_argument('--ckpt-step', type=int, default=200000, help='Checkpoint step [default: 200000]')
parser.add_argument('--num-points', type=int, default=1024, help='Number of points [default: 1024]')
parser.add_argument('--num-samples', type=int, default=8, help='Number of samples [default: 8]')
parser.add_argument('--seq-length', type=int, default=10, help='Length of sequence [default: 10]')
parser.add_argument('--log-dir', default='outputs/argo-pointrnn', help='Log dir [outputs/argo-pointrnn]')
args = parser.parse_args()

data = np.load(args.data_path)

model = Model(batch_size=1,
              seq_length=args.seq_length,
              num_points=args.num_points,
              num_samples=args.num_samples,
              knn=True,
              is_training=False)

checkpoint_dir = os.path.join(args.log_dir, 'checkpoints')
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
checkpoint_path = os.path.join(checkpoint_dir, 'ckpt-%d'%args.ckpt_step)

with tf.Session() as sess:
    model.saver.restore(sess, checkpoint_path)

    flops = tf.profiler.profile(sess.graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    parameters = tf.profiler.profile(sess.graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print ('total flops: {}'.format(flops.total_float_ops))
    print ('total parameters: {}'.format(parameters.total_parameters))

    outputs = []
    for i in range(data.shape[0]):
        batch_data = np.expand_dims(data[i], axis=0)
        feed_dict = {model.inputs: batch_data}
        predicted_frames = sess.run([model.predicted_frames], feed_dict=feed_dict)[0]
        outputs.append(predicted_frames)

    outputs = np.concatenate(outputs, 0)
    np.save(os.path.join(args.log_dir, 'test-predictions'), frame_outputs)
