import os
import sys
import io
from datetime import datetime
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf
from datasets.mmnist import MMNIST
import models.mmnist as models

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='data/mnist', help='Dataset directory [default: data/mnist]')
parser.add_argument('--batch-size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--num-iters', type=int, default=200000, help='Iterations to run [default: 200000]')
parser.add_argument('--save-iters', type=int, default=1000, help='Iterations to save checkpoints [default: 1000]')
parser.add_argument('--learning-rate', type=float, default=1e-5, help='Learning rate [default: 1e-5]')
parser.add_argument('--max-gradient-norm', type=float, default=5.0, help='Clip gradients to this norm [default: 5.0].')
parser.add_argument('--num-points', type=int, default=128, help='Number of points [default: 128]')
parser.add_argument('--num-samples', type=int, default=4, help='Number of samples [default: 4]')
parser.add_argument('--seq-length', type=int, default=20, help='Length of sequence [default: 20]')
parser.add_argument('--num-digits', type=int, default=1, help='Number of moving digits [default: 1]')
parser.add_argument('--image-size', type=int, default=64, help='Image size [default: 64]')
parser.add_argument('--mode', type=str, default='advanced', help='Basic model or advanced model [default: advanced]')
parser.add_argument('--unit', type=str, default='pointlstm', help='Unit. pointrnn, pointgru or pointlstm [default: pointlstm]')
parser.add_argument('--step-length', type=float, default=0.1, help='Step length [default: 0.1]')
parser.add_argument('--alpha', type=float, default=1.0, help='Weigh on CD loss [default: 1.0]')
parser.add_argument('--beta', type=float, default=1.0, help='Weigh on EMD loss [default: 1.0]')
parser.add_argument('--log-dir', default='outputs/mmnist', help='Log dir [default: outputs/mminst]')

args = parser.parse_args()
np.random.seed(999)
tf.set_random_seed(999)

args.log_dir += '-%ddigit-%s-%s'%(args.num_digits, args.mode, args.unit)

train_dataset = MMNIST(root=args.data_dir,
                       seq_length=args.seq_length,
                       num_digits=args.num_digits,
                       image_size=args.image_size,
                       step_length=args.step_length,
                       num_points=args.num_points,
                       train=True)

def get_batch(dataset, batch_size):
    batch_data = []
    for i in range(batch_size):
        sample = dataset[0]
        batch_data.append(sample)
    return np.stack(batch_data, axis=0)

model_name = args.mode.capitalize() + 'Point' + args.unit[5:].upper()
Model = getattr(models, model_name)
model = Model(batch_size=args.batch_size,
              num_points=args.num_points,
              num_samples=args.num_samples,
              knn=False,
              alpha=args.alpha,
              beta=args.beta,
              seq_length=args.seq_length,
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
            pc_prediction = predictions[0]                                      # [int(args.seq_length/2), num_digits, 3]

            ground_truth = np.zeros(shape=(int(args.seq_length/2), args.image_size, args.image_size))
            prediction = np.zeros(shape=(int(args.seq_length/2), args.image_size, args.image_size))

            pc_ground_truth = np.ceil(pc_ground_truth).astype(np.uint8)
            pc_prediction = np.ceil(pc_prediction).astype(np.uint8)

            pc_prediction = np.clip(pc_prediction, a_min=0, a_max=args.image_size-1)

            for j in range(int(args.seq_length/2)):
                for k in range(args.num_points):
                    ground_truth[j, pc_ground_truth[j,k,0], pc_ground_truth[j,k,1]] = 255
                    prediction[j, pc_prediction[j,k,0], pc_prediction[j,k,1]] = 255
            ground_truth = np.swapaxes(ground_truth.astype(np.uint8), 0, 1)
            prediction = np.swapaxes(prediction.astype(np.uint8), 0, 1)
            ground_truth = np.reshape(ground_truth, (args.image_size, -1))
            prediction = np.reshape(prediction, (args.image_size, -1))
            image = np.concatenate((ground_truth, prediction), 0)               # [3*args.image_size, int(args.seq_length/2)*args.image_size]
            image[args.image_size, :] = 255
            for j in range(1, int(args.seq_length/2)):
                image[:, j*args.image_size] = 255
            image = Image.fromarray(image, 'L')
            image_path = os.path.join(example_dir, '%d.png'%step)
            image.save(image_path)

            buff = io.BytesIO()
            image.save(buff, format='PNG')
            image_string = buff.getvalue()
            buff.close()
            example = tf.Summary.Image(height=3*args.image_size, width=int(args.seq_length/2)*args.image_size, colorspace=1, encoded_image_string=image_string)
            summary = tf.Summary(value=[tf.Summary.Value(tag='image', image=example)])
            summary_writer.add_summary(summary, step)
log.close()
