import os
import sys
import io
from datetime import datetime
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf

parser = argparse.ArgumentParser()
from models.mmnist import AdvancedPointLSTM as Model

# Advanced, 1 digit 
"""
parser.add_argument('--data-path', default='data/test-1mnist-64-128point-20step.npy', help='Data path [default: data/test-1mnist-64-128point-20step.npy]')
parser.add_argument('--ckpt-step', type=int, default=200000, help='Checkpoint step [default: 200000]')
parser.add_argument('--num-points', type=int, default=128, help='Number of points [default: 128]')
parser.add_argument('--num-samples', type=int, default=4, help='Number of samples [default: 4]')
parser.add_argument('--seq-length', type=int, default=20, help='Length of sequence [default: 20]')
parser.add_argument('--num-digits', type=int, default=1, help='Number of moving digits [default: 1]')
"""
# Advanced, 2 digits 
parser.add_argument('--data-path', default='data/test-2mnist-64-256point-20step.npy', help='Data path [default: data/test-2mnist-64-256point-20step.npy]')
parser.add_argument('--ckpt-step', type=int, default=200000, help='Checkpoint step [default: 200000]')
parser.add_argument('--num-points', type=int, default=256, help='Number of points [default: 256]')
parser.add_argument('--num-samples', type=int, default=4, help='Number of samples[default: 4]')
parser.add_argument('--seq-length', type=int, default=20, help='Length of sequence [default: 20]')
parser.add_argument('--num-digits', type=int, default=2, help='Number of moving digits [default: 1]')

parser.add_argument('--image-size', type=int, default=64, help='Image size [default: 64]')
parser.add_argument('--log-dir', default='outputs/mmnist-2digit-advanced-pointlstm', help='Log dir [default: outputs/mmnist-2digit-advanced-pointlstm]')

args = parser.parse_args()

data = np.load(args.data_path)
data = np.concatenate((data, np.zeros((data.shape[0], args.seq_length, args.num_points, 1), dtype=data.dtype)),3)

model = Model(1,
              num_points=args.num_points,
              num_samples=args.num_samples,
              seq_length=args.seq_length,
              knn=False,
              is_training=False)

checkpoint_dir = os.path.join(args.log_dir, 'checkpoints')
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
checkpoint_path = os.path.join(checkpoint_dir, 'ckpt-%d'%args.ckpt_step)

example_dir = os.path.join(args.log_dir, 'test-examples')
if not os.path.exists(example_dir):
    os.makedirs(example_dir)

with tf.Session() as sess:
    model.saver.restore(sess, checkpoint_path)

    flops = tf.profiler.profile(sess.graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    parameters = tf.profiler.profile(sess.graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print ('total flops: {}'.format(flops.total_float_ops))
    print ('total parameters: {}'.format(parameters.total_parameters))

    outputs = []
    for i in range(data.shape[0]):

        curr_dir = os.path.join(example_dir, '%04d'%(i+1))
        if not os.path.exists(curr_dir):
            os.makedirs(curr_dir)

        batch_data = np.expand_dims(data[i], axis=0)

        feed_dict = {model.inputs: batch_data}
        predictions = sess.run([model.predicted_frames], feed_dict=feed_dict)[0]
        outputs.append(predictions)

        pc_context = batch_data[0, :int(args.seq_length/2),:]
        pc_ground_truth = batch_data[0, int(args.seq_length/2):,:]
        pc_prediction = predictions[0]                                                  # [int(args.seq_length/2), num_digits, 3]

        context = np.zeros(shape=(int(args.seq_length/2), args.image_size, args.image_size))
        ground_truth = np.zeros(shape=(int(args.seq_length/2), args.image_size, args.image_size))
        prediction = np.zeros(shape=(int(args.seq_length/2), args.image_size, args.image_size))

        pc_context = np.ceil(pc_context).astype(np.uint8)
        pc_ground_truth = np.ceil(pc_ground_truth).astype(np.uint8)
        pc_prediction = np.ceil(pc_prediction).astype(np.uint8)

        pc_prediction = np.clip(pc_prediction, a_min=0, a_max=args.image_size-1)

        for j in range(int(args.seq_length/2)):
            for k in range(args.num_points):
                context[j, pc_context[j,k,0], pc_context[j,k,1]] = 255
                ground_truth[j, pc_ground_truth[j,k,0], pc_ground_truth[j,k,1]] = 255
                prediction[j, pc_prediction[j,k,0], pc_prediction[j,k,1]] = 255
        context = np.swapaxes(context.astype(np.uint8), 0, 1)
        ground_truth = np.swapaxes(ground_truth.astype(np.uint8), 0, 1)
        prediction = np.swapaxes(prediction.astype(np.uint8), 0, 1)

        context = np.reshape(context, (args.image_size, -1))
        ground_truth = np.reshape(ground_truth, (args.image_size, -1))
        prediction = np.reshape(prediction, (args.image_size, -1))

        for j in range(1, int(args.seq_length/2)):
            context[:, j*args.image_size] = 255
            ground_truth[:, j*args.image_size] = 255
            prediction[:, j*args.image_size] = 255

        context = Image.fromarray(context, 'L')
        ground_truth = Image.fromarray(ground_truth, 'L')
        prediction = Image.fromarray(prediction, 'L')

        context.save(os.path.join(curr_dir, 'ctx.png'))
        ground_truth.save(os.path.join(curr_dir, 'gth.png'))
        prediction.save(os.path.join(curr_dir, 'pdt.png'))
    outputs = np.concatenate(outputs, 0)
    np.save(os.path.join(args.log_dir, 'test-predictions'), outputs)
