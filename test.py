import tensorflow as tf
import os
import cv2
import random
import numpy as np
from models.homographynet import HomographyNet as HomoNet


iter_max = 1
batch_size = 1
pairs_per_img = 1

dir_test = '/media/csc105/Data/dataset/ms-coco/test2014'
dir_model = 'model/20170317/model_5000.ckpt'


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
    data = []
    label = []
    random_list = []
    img = cv2.resize(cv2.imread(img_path, 0), (640, 480))
    i = 1
    while i < pairs_per_img + 1:
        y_start = random.randint(64, 160)
        y_end = y_start + 256
        x_start = random.randint(64, 320)
        x_end = x_start + 256

        y_1 = y_start
        x_1 = x_start
        y_2 = y_end
        x_2 = x_start
        y_3 = y_end
        x_3 = x_end
        y_4 = y_start
        x_4 = x_end

        img_patch = img[y_start:y_end, x_start:x_end]  # patch 1

        y_1_offset = random.randint(-64, 64)
        x_1_offset = random.randint(-64, 64)
        y_2_offset = random.randint(-64, 64)
        x_2_offset = random.randint(-64, 64)
        y_3_offset = random.randint(-64, 64)
        x_3_offset = random.randint(-64, 64)
        y_4_offset = random.randint(-64, 64)
        x_4_offset = random.randint(-64, 64)

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

        img_perburb = cv2.warpPerspective(img, h, (640, 480))
        img_perburb_patch = img_perburb[y_start:y_end, x_start:x_end]  # patch 2
        if not [y_1,x_1,y_2,x_2,y_3,x_3,y_4,x_4] in random_list:
            data.append(img_patch)
            data.append(img_perburb_patch)
            random_list.append([y_1,x_1,y_2,x_2,y_3,x_3,y_4,x_4])
            # h_4pt = np.array([y_1_offset,x_1_offset,y_2_offset,x_2_offset,y_3_offset,x_3_offset,y_4_offset,x_4_offset])  # labels
            h_4pt = np.array([y_1_p,x_1_p,y_2_p,x_2_p,y_3_p,x_3_p,y_4_p,x_4_p])  # labels
            label.append(h_4pt)
            i += 1
    return data, label


class DataSet(object):
    def __init__(self, img_path_list):
        self.img_path_list = img_path_list
        self.index_in_epoch = 0

    def next_batch(self):
        data_batch = []
        label_batch = []
        start = self.index_in_epoch
        self.index_in_epoch += batch_size / pairs_per_img
        if self.index_in_epoch > 50000:
            self.index_in_epoch = 0
            start = self.index_in_epoch
            self.index_in_epoch += batch_size / pairs_per_img
        end = self.index_in_epoch

        for i in range(start, end):
            data, label = generate_data(self.img_path_list[i])
            data_batch.append(data)
            label_batch.append(label)

        return np.reshape(data_batch, [batch_size, 128, 128, 2]), np.reshape(label_batch, [batch_size, 8])


def main(_):

    test_img_list = load_data(dir_test)

    x1 = tf.placeholder(tf.float32, [None, 256, 256, 2])
    x2 = tf.placeholder(tf.float32, [None, 8])
    net = HomoNet({'data': x1})
    net_out = net.layers['fc2']

    loss = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(net_out, x2))) / 2 / batch_size)

    # tensor board
    tf.scalar_summary('loss', loss)  # record loss
    merged = tf.merge_all_summaries()

    # gpu configuration
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    # gpu_opinions = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

    init = tf.initialize_all_variables()
    saver = tf.train.Saver()
    with tf.Session(config=tf_config) as sess:
        saver.restore(sess, dir_model)

        writer_test = tf.train.SummaryWriter("log/test/", sess.graph)  # use writer to record loss when test

        test_model = DataSet(test_img_list)
        sess.run(init)
        for i in range(iter_max):
            x_batch_test, y_batch_test = test_model.next_batch()
            result, loss_test = sess.run([merged, loss], feed_dict={x1: x_batch_test, x2: y_batch_test})
            print ('iter %05d, test loss = %.5f' % ((i+1), loss_test))
            writer_test.add_summary(result, i+1)


if __name__ == "__main__":
    tf.app.run()
