import tensorflow as tf
import os
import cv2
import random
import numpy as np
from models.homographynet import HomographyNet as HomoNet
import shutil

iter_max = 90000
batch_size = 64
pairs_per_img = 4
lr_base = 0.005
lr_decay_iter = 30000

dir_train = 'data/generated_data/train/'  # dir of train2014
label_train = 'data/generated_data/label/train_homography_ab.txt'
dir_val = 'data/generated_data/val'  # dir of val2014
label_val = 'data/generated_data/label/val_homography_ab.txt'

dir_model = 'model_2/20170320_2'  # dir of model to be saved
log_train = 'log_2/train_0320_2'  # dir of train loss to be saved
log_val = 'log_2/val_0320_2'  # dir of val loss to be saved

if os.path.exists(dir_model):
    shutil.rmtree(dir_model)
if os.path.exists(log_train):
    shutil.rmtree(log_train)
if os.path.exists(log_val):
    shutil.rmtree(log_val)

os.mkdir(dir_model)
os.mkdir(log_train)
os.mkdir(log_val)


def load_data(raw_data_path):
    dir_list_out = []
    dir_list = os.listdir(raw_data_path)
    if '.' in dir_list:
        dir_list.remove('.')
    if '..' in dir_list:
        dir_list.remove('.')
    if '.DS_Store' in dir_list:
        dir_list.remove('.DS_Store')
    if '.directory' in dir_list:
        dir_list.remove('.directory')
    dir_list.sort()
    for i in range(len(dir_list)):
        dir_list_out.append(os.path.join(raw_data_path, dir_list[i]))

    return dir_list_out


def load_label(label_path):

    with open(label_path) as f:

        i = 0
        label = []
        label_re = []
        for line in f.readlines():
            label.append(float(line))
            i += 1
            if i == 8:
                label_re.append(label)
                i = 0
                label = []

    return label_re


class DataSet(object):
    def __init__(self, img_path_list, labels):
        self.img_path_list = img_path_list
        self.labels = labels
        self.index_in_epoch = 0
        self.count = 0
        self.number = len(img_path_list)

    def next_batch(self):
        data_batch = []
        label_batch = []
        self.count += 1
        start = self.index_in_epoch
        self.index_in_epoch += batch_size * 2
        if self.index_in_epoch > self.number:
            self.index_in_epoch = 0
            start = self.index_in_epoch
            self.index_in_epoch += batch_size * 2
        end = self.index_in_epoch

        for i in range(start, end, 2):
            data = []
            img1 = cv2.imread(self.img_path_list[i], 0)
            img2 = cv2.imread(self.img_path_list[i+1], 0)
            data.append(img1)
            data.append(img2)
            data_batch.append(data)
            label_batch.append(self.labels[i/2])

        data_batch = np.array(data_batch).transpose([0,2,3,1])
        # cv2.imshow('win1', data_batch[0,:,:,0])
        # cv2.imshow('win2', data_batch[0,:,:,1])
        # cv2.waitKey()
        label_batch = np.array(label_batch).squeeze()
        return data_batch, label_batch


def main(_):

    # with tf.device('/gpu:1'):
    train_img_list = load_data(dir_train)
    train_labels = load_label(label_train)
    val_img_list = load_data(dir_val)
    val_labels = load_label(label_val)

    x1 = tf.placeholder(tf.float32, [None, 128, 128, 2])
    x2 = tf.placeholder(tf.float32, [None, 8])
    x3 = tf.placeholder(tf.float32, [])
    net = HomoNet({'data': x1})
    net_out = net.layers['fc2']

    loss = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(net_out, x2))) / 2 / batch_size)
    train_op = tf.train.MomentumOptimizer(learning_rate=x3, momentum=0.9).minimize(loss)

    # tensor board
    tf.scalar_summary('loss',loss)  # record loss
    merged = tf.merge_all_summaries()

    # gpu configuration
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    # gpu_opinions = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

    init = tf.initialize_all_variables()
    saver = tf.train.Saver()

    with tf.Session(config=tf_config) as sess:
        sess.run(init)
        writer_train = tf.train.SummaryWriter(log_train, sess.graph)  # use writer1 to record loss when train
        writer_val = tf.train.SummaryWriter(log_val, sess.graph)  # use writer2 to record loss when val

        train_model = DataSet(train_img_list, train_labels)
        val_model = DataSet(val_img_list, val_labels)
        x_batch_val, y_batch_val = val_model.next_batch()  # fix the val data

        for i in range(iter_max):
            lr_decay = 0.1 ** (i/lr_decay_iter)
            lr = lr_base * lr_decay
            x_batch_train, y_batch_train = train_model.next_batch()
            sess.run(train_op, feed_dict={x1: x_batch_train, x2: y_batch_train, x3: lr})

            # display
            if not (i+1) % 50:
                result1, loss_train = sess.run([merged, loss], feed_dict={x1: x_batch_train, x2: y_batch_train})
                print ('iter %05d, lr = %.5f, train loss = %.5f' % ((i+1), lr, loss_train))
                writer_train.add_summary(result1, i+1)

            if not (i+1) % 200:
                result2, loss_val = sess.run([merged, loss], feed_dict={x1: x_batch_val, x2: y_batch_val})
                print ('iter %05d, val loss = %.5f' % ((i+1), loss_val))
                print "============================"
                writer_val.add_summary(result2, i+1)

            # save model
            if not (i+1) % 5000:
                saver.save(sess, (dir_model + "/model_%d.ckpt") % (i+1))


if __name__ == "__main__":
    tf.app.run()
