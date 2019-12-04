import os
import sys
import io
from datetime import datetime
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from PIL import Image
import models.argo_nu as models

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='/data/argo-5m', help='Dataset directory [default: data/argo-5m]')
parser.add_argument('--dataset', default='argo', help='Dataset. argo or nu [default: argo]')
parser.add_argument('--batch-size', type=int, default=4, help='Batch Size during training [default: 4]')
parser.add_argument('--num-iters', type=int, default=200000, help='Iterations to run [default: 200000]')
parser.add_argument('--save-iters', type=int, default=1000, help='Iterations to save checkpoints [default: 1000]')
parser.add_argument('--learning-rate', type=float, default=1e-5, help='Learning rate [default: 1e-5]')
parser.add_argument('--max-gradient-norm', type=float, default=5.0, help='Clip gradients to this norm [default: 5.0].')
parser.add_argument('--seq-length', type=int, default=10, help='Length of sequence [default: 10]')
parser.add_argument('--num-points', type=int, default=1024, help='Number of points [default: 1024]')
parser.add_argument('--num-samples', type=int, default=8, help='Number of samples [default: 8]')
parser.add_argument('--unit', type=str, default='pointrnn', help='Unit. pointrnn, pointgru or pointlstm [default: pointrnn]')
parser.add_argument('--alpha', type=float, default=1.0, help='Weigh on CD loss [default: 1.0]')
parser.add_argument('--beta', type=float, default=1.0, help='Weigh on EMD loss [default: 1.0]')
parser.add_argument('--log-dir', default='outputs', help='Log dir [default: outputs]')

args = parser.parse_args()
np.random.seed(999)
tf.set_random_seed(999)

args.log_dir += '/%s-%s'%(args.dataset, args.unit)

if args.dataset == 'argo':
    from datasets.argo_nu import Argoverse as Dataset
if args.dataset == 'nu':
    from datasets.argo_nu import nuScenes as Dataset

train_dataset = Dataset(root=args.data_dir,
                        seq_length=args.seq_length,
                        num_points=args.num_points,
                        train=True)

point_size = 5
axes_limits = [[-5, 5], [-5, 5], [-5, 5]]  # X axis range  # Y axis range  # Z axis range
axes_str = ["X", "Y", "Z"]
axes = [1, 0, 2]

def draw_point_cloud(cloud_sequence1, cloud_sequence2, output):
    if not os.path.exists(output):
        os.makedirs(output)
    for i in range(cloud_sequence1.shape[0]):
        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(111, projection='3d')
        pc = cloud_sequence1[i]
        ax.scatter(*np.transpose(pc[:, axes]), s=point_size, c=pc[:, 2], cmap="gray")
        ax.set_xlabel("{} axis".format(axes_str[axes[0]]))
        ax.set_ylabel("{} axis".format(axes_str[axes[1]]))
        ax.set_xlim3d(*axes_limits[axes[0]])
        ax.set_ylim3d(*axes_limits[axes[1]])
        ax.set_zlim3d(*axes_limits[axes[2]])
        ax.set_zlabel("{} axis".format(axes_str[axes[2]]))
        ax.axis('off')
        plt.savefig(os.path.join(output, '1-%02d.png'%(i+1)))
        plt.close()
    for i in range(cloud_sequence2.shape[0]):
        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(111, projection='3d')
        pc = cloud_sequence2[i]
        ax.scatter(*np.transpose(pc[:, axes]), s=point_size, c=pc[:, 2], cmap="gray")
        ax.set_xlabel("{} axis".format(axes_str[axes[0]]))
        ax.set_ylabel("{} axis".format(axes_str[axes[1]]))
        ax.set_xlim3d(*axes_limits[axes[0]])
        ax.set_ylim3d(*axes_limits[axes[1]])
        ax.set_zlim3d(*axes_limits[axes[2]])
        ax.set_zlabel("{} axis".format(axes_str[axes[2]]))
        ax.axis('off')
        plt.savefig(os.path.join(output, '2-%02d.png'%(i+1)))
        plt.close()

def get_batch(dataset, batch_size):
    batch_data = []
    for i in range(batch_size):
        sample = dataset[0]
        batch_data.append(sample)
    return np.stack(batch_data, axis=0)

model_name = 'Point' + args.unit[5:].upper()
Model = getattr(models, model_name)
model = Model(batch_size=args.batch_size,
              seq_length=args.seq_length,
              num_points=args.num_points,
              num_samples=args.num_samples,
              knn=True,
              alpha=args.alpha,
              beta=args.beta,
              learning_rate=args.learning_rate,
              max_gradient_norm=args.max_gradient_norm,
              is_training=True)

tf.summary.scalar('cd', model.cd)
tf.summary.scalar('emd', model.emd)

summary_op = tf.summary.merge_all()

checkpoint_dir = os.path.join(args.log_dir, 'checkpoints')
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
checkpoint_path = os.path.join(checkpoint_dir, 'ckpt')
example_dir = os.path.join(args.log_dir, 'examples')
if not os.path.exists(example_dir):
    os.makedirs(example_dir)

log = open(os.path.join(args.log_dir, 'train.log'), 'w')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter(os.path.join(args.log_dir, 'summary'), sess.graph)
    for i in range(args.num_iters):
        batch_data = get_batch(dataset=train_dataset, batch_size=args.batch_size)
        feed_dict = {model.inputs: batch_data}
        cd, emd, step, summary, predictions, _ = sess.run([model.cd, model.emd, model.global_step, summary_op, model.predicted_frames, model.train_op], feed_dict=feed_dict)
        log.write('[%s]\t[%10d:]\t%.12f\t%.12f\n'%(str(datetime.now()), i+1, cd, emd))
        log.flush()
        summary_writer.add_summary(summary, step)
        if (i+1) % args.save_iters == 0:
            ckpt = os.path.join(checkpoint_path, )
            model.saver.save(sess, checkpoint_path, global_step=model.global_step)

            pc_ground_truth = batch_data[0, int(args.seq_length/2):,:]
            pc_prediction = predictions[0]          # [int(args.seq_length/2), num_points, 3]
            draw_point_cloud(pc_ground_truth, pc_prediction, os.path.join(example_dir, str(i+1)))
log.close()
