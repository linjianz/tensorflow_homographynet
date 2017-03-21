import tensorflow as tf
import os
import cv2
import random
import numpy as np
from models.homographynet import HomographyNet as HomoNet
import shutil

iter_max = 90000
save_iter = 2000
batch_size = 64
pairs_per_img = 1
lr_base = 5e-5
lr_decay_iter = 20000

dir_train = '/media/csc105/Data/dataset/ms-coco/train2014'  # dir of train2014
dir_val = '/media/csc105/Data/dataset/ms-coco/val2014'  # dir of val2014

dir_model = 'model/20170321_4'  # dir of model to be saved
log_train = 'log/train_0321_4'  # dir of train loss to be saved
log_val = 'log/val_0321_4'  # dir of val loss to be saved

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
    dir_list.sort()
    for i in range(len(dir_list)):
        dir_list_out.append(os.path.join(raw_data_path, dir_list[i]))
    return dir_list_out


def generate_data(img_path):
    data_re = []
    label_re = []
    random_list = []
    img = cv2.resize(cv2.imread(img_path, 0), (320, 240))
    i = 1
    while i < pairs_per_img + 1:
        data = []
        label = []
        y_start = random.randint(32, 80)
        y_end = y_start + 128
        x_start = random.randint(32, 160)
        x_end = x_start + 128

        y_1 = y_start
        x_1 = x_start
        y_2 = y_end
        x_2 = x_start
        y_3 = y_end
        x_3 = x_end
        y_4 = y_start
        x_4 = x_end

        img_patch = img[y_start:y_end, x_start:x_end]  # patch 1

        y_1_offset = random.randint(-32, 32)
        x_1_offset = random.randint(-32, 32)
        y_2_offset = random.randint(-32, 32)
        x_2_offset = random.randint(-32, 32)
        y_3_offset = random.randint(-32, 32)
        x_3_offset = random.randint(-32, 32)
        y_4_offset = random.randint(-32, 32)
        x_4_offset = random.randint(-32, 32)

        y_1_p = y_1 + y_1_offset
        x_1_p = x_1 + x_1_offset
        y_2_p = y_2 + y_2_offset
        x_2_p = x_2 + x_2_offset
        y_3_p = y_3 + y_3_offset
        x_3_p = x_3 + x_3_offset
        y_4_p = y_4 + y_4_offset
        x_4_p = x_4 + x_4_offset

        pts_img_patch = np.array([[y_1,x_1],[y_2,x_2],[y_3,x_3],[y_4,x_4]]).astype(np.float32)
        pts_img_patch_perturb = np.array([[y_1_p,x_1_p],[y_2_p,x_2_p],[y_3_p,x_3_p],[y_4_p,x_4_p]]).astype(np.float32)
        h,status = cv2.findHomography(pts_img_patch, pts_img_patch_perturb, cv2.RANSAC)

        img_perburb = cv2.warpPerspective(img, h, (320, 240))
        img_perburb_patch = img_perburb[y_start:y_end, x_start:x_end]  # patch 2
        if not [y_1,x_1,y_2,x_2,y_3,x_3,y_4,x_4] in random_list:
            data.append(img_patch)
            data.append(img_perburb_patch)  # [2, 128, 128]
            random_list.append([y_1,x_1,y_2,x_2,y_3,x_3,y_4,x_4])
            h_4pt = np.array([y_1_offset,x_1_offset,y_2_offset,x_2_offset,y_3_offset,x_3_offset,y_4_offset,x_4_offset])
            # h_4pt = np.array([y_1_p,x_1_p,y_2_p,x_2_p,y_3_p,x_3_p,y_4_p,x_4_p])  # labels
            label.append(h_4pt)  # [1, 8]
            i += 1
        data_re.append(data)  # [4, 2, 128, 128]
        label_re.append(label)  # [4, 1, 8]

    return data_re, label_re


class DataSet(object):
    def __init__(self, img_path_list):
        self.img_path_list = img_path_list
        self.index_in_epoch = 0
        self.count = 0
        self.number = len(img_path_list)

    def next_batch(self):
        self.count += 1
        # print self.count
        start = self.index_in_epoch
        self.index_in_epoch += batch_size / pairs_per_img
        if self.index_in_epoch > self.number:
            self.index_in_epoch = 0
            start = self.index_in_epoch
            self.index_in_epoch += batch_size / pairs_per_img
        end = self.index_in_epoch

        data_batch, label_batch = generate_data(self.img_path_list[start])
        for i in range(start+1, end):
            data, label = generate_data(self.img_path_list[i])  # [4, 2, 128, 128], [4, 1, 8]
            data_batch = np.concatenate((data_batch, data))  # [64, 2, 128, 128]
            label_batch = np.concatenate((label_batch, label))  # [64, 1, 8]

        data_batch = np.array(data_batch).transpose([0, 2, 3, 1])  # (64, 128, 128, 2)
        # cv2.imshow('window2', data_batch[1,:,:,1].squeeze())
        # cv2.waitKey()
        label_batch = np.array(label_batch).squeeze()  # (64, 1, 8)

        return data_batch, label_batch


def main(_):

    train_img_list = load_data(dir_train)
    val_img_list = load_data(dir_val)

    x1 = tf.placeholder(tf.float32, [None, 128, 128, 2])
    x2 = tf.placeholder(tf.float32, [None, 8])
    x3 = tf.placeholder(tf.float32, [])
    x4 = tf.placeholder(tf.float32, [])  # 1.0: use dropout; 0.0: turn off dropout
    net = HomoNet({'data': x1, 'use_dropout': x4})
    net_out = net.layers['fc2']

    loss = tf.reduce_sum(tf.square(tf.sub(net_out, x2))) / 2 / batch_size
    tvars = tf.trainable_variables()
    # grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5.0)
    # grads = tf.gradients(loss, tvars)
    # optimizer = tf.train.GradientDescentOptimizer(x3)
    # train_op = optimizer.apply_gradients(zip(grads, tvars))

    # train_op = tf.train.AdamOptimizer(x3).minimize(loss)
    train_op = tf.train.MomentumOptimizer(learning_rate=x3, momentum=0.9).minimize(loss)

    # tensor board
    tf.scalar_summary('loss',loss)  # record loss
    tf.histogram_summary('', )
    merged = tf.merge_all_summaries()

    # gpu configuration
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    # gpu_opinions = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

    init = tf.initialize_all_variables()
    saver = tf.train.Saver(max_to_keep=None)

    with tf.Session(config=tf_config) as sess:

        sess.run(init)

        writer_train = tf.train.SummaryWriter(log_train, sess.graph)  # use writer1 to record loss when train
        writer_val = tf.train.SummaryWriter(log_val, sess.graph)  # use writer2 to record loss when val

        train_model = DataSet(train_img_list)
        val_model = DataSet(val_img_list)
        x_batch_val, y_batch_val = val_model.next_batch()  # fix the val data

        for i in range(iter_max):
            lr_decay = 0.1 ** (i/lr_decay_iter)
            lr = lr_base * lr_decay
            x_batch_train, y_batch_train = train_model.next_batch()
            sess.run(train_op, feed_dict={x1: x_batch_train, x2: y_batch_train, x3: lr, x4: 1.0})

            # display
            if not (i+1) % 1:
                # print sess.run(tvars[35])
                result1, loss_train = sess.run([merged, loss], feed_dict={x1: x_batch_train, x2: y_batch_train, x4: 0.0})
                print ('iter %05d, lr = %.5f, train loss = %.5f' % ((i+1), lr, loss_train))
                writer_train.add_summary(result1, i+1)

            if not (i+1) % 1:
                result2, loss_val = sess.run([merged, loss], feed_dict={x1: x_batch_val, x2: y_batch_val, x4: 0.0})
                print ('iter %05d, lr = %.5f, val   loss = %.5f' % ((i+1), lr, loss_val))
                print "============================"
                writer_val.add_summary(result2, i+1)

            # save model
            if not (i+1) % save_iter:
                saver.save(sess, (dir_model + "/model_%d.ckpt") % (i+1))


if __name__ == "__main__":
    tf.app.run()
