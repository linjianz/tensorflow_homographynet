import tensorflow as tf
import os
import cv2
import random
import numpy as np
import shutil

iter_max = 100
batch_size = 1

dir_test = '/media/csc105/Data/dataset/ms-coco/test2014'  # dir of test2014
# dir_load = 'model_mycnn/20170322_4/model_54000.ckpt'
dir_load = 'model_mycnn/20170322_2/model_80000.ckpt'
dir_save = 'test/20170223_2'

if os.path.exists(dir_save):
    shutil.rmtree(dir_save)

os.mkdir(dir_save)


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


def generate_data_test(img_path):
    data = []
    img = cv2.resize(cv2.imread(img_path, 0), (640, 480))

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
    h, status = cv2.findHomography(pts_img_patch, pts_img_patch_perturb, cv2.RANSAC)

    img_perburb = cv2.warpPerspective(img, h, (640, 480))
    img_perburb_patch = img_perburb[y_start:y_end, x_start:x_end]  # patch 2

    data.append(img_patch)
    data.append(img_perburb_patch)

    h_4pt = np.array([y_1_offset,x_1_offset,y_2_offset,x_2_offset,y_3_offset,x_3_offset,y_4_offset,x_4_offset])
    h1 = np.array([y_1,x_1,y_2,x_2,y_3,x_3,y_4,x_4])
    # h2 = np.array([y_1_p,x_1_p,y_2_p,x_2_p,y_3_p,x_3_p,y_4_p,x_4_p])

    return data, h_4pt, h1, img, img_perburb


def generate_data_train(img_path):
    data = []
    img = cv2.resize(cv2.imread(img_path, 0), (320, 240))

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
    h, status = cv2.findHomography(pts_img_patch, pts_img_patch_perturb, cv2.RANSAC)

    img_perburb = cv2.warpPerspective(img, h, (320, 240))
    img_perburb_patch = img_perburb[y_start:y_end, x_start:x_end]  # patch 2

    data.append(img_patch)
    data.append(img_perburb_patch)

    h_4pt = np.array([y_1_offset,x_1_offset,y_2_offset,x_2_offset,y_3_offset,x_3_offset,y_4_offset,x_4_offset])
    h1 = np.array([y_1,x_1,y_2,x_2,y_3,x_3,y_4,x_4])
    # h2 = np.array([y_1_p,x_1_p,y_2_p,x_2_p,y_3_p,x_3_p,y_4_p,x_4_p])

    return data, h_4pt, h1, img, img_perburb


class DataSet(object):
    def __init__(self, img_path_list):
        self.img_path_list = img_path_list
        self.count = 0

    def next_batch(self):
        data = []
        label = []
        h1_batch = []
        data_batch, label_batch, h1, img1, img2 = generate_data_train(self.img_path_list[self.count])
        data.append(data_batch)
        data_batch = np.array(data).transpose([0, 2, 3, 1])  # (1, 128, 128, 2)
        # cv2.imshow('window2', data_batch[1,:,:,1].squeeze())
        # cv2.waitKey()
        label.append(label_batch)
        label_batch = np.array(label)  # (1, 8)
        h1_batch.append(h1)
        self.count += 1
        return data_batch, label_batch, h1_batch, img1, img2


def main(_):
    test_img_list = load_data(dir_test)

    x1 = tf.placeholder(tf.float32, [None, 128, 128, 2])  # data
    x2 = tf.placeholder(tf.float32, [None, 8])  # label

    # conv1 conv2 maxpooling
    w1 = tf.Variable(tf.truncated_normal([3,3,2,64], stddev=0.1))
    b1 = tf.Variable(tf.constant(0.1, shape=[64]))
    conv1 = tf.nn.conv2d(x1, w1, strides=[1,1,1,1], padding='SAME') + b1
    mean1, var1 = tf.nn.moments(conv1, axes=[0, 1, 2])
    offset1 = tf.Variable(tf.constant(0.0, shape=[64]))
    scale1 = tf.Variable(tf.constant(1.0, shape=[64]))
    bn1 = tf.nn.batch_normalization(conv1, mean=mean1, variance=var1, offset=offset1, scale=scale1, variance_epsilon=1e-5)
    relu1 = tf.nn.relu(bn1)

    w2 = tf.Variable(tf.truncated_normal([3,3,64,64], stddev=0.1))
    b2 = tf.Variable(tf.constant(0.1, shape=[64]))
    conv2 = tf.nn.conv2d(relu1, w2, strides=[1,1,1,1], padding='SAME') + b2
    mean2, var2 = tf.nn.moments(conv2, axes=[0, 1, 2])
    offset2 = tf.Variable(tf.constant(0.0, shape=[64]))
    scale2 = tf.Variable(tf.constant(1.0, shape=[64]))
    bn2 = tf.nn.batch_normalization(conv2, mean=mean2, variance=var2, offset=offset2, scale=scale2, variance_epsilon=1e-5)
    relu2 = tf.nn.relu(bn2)
    maxpool1 = tf.nn.max_pool(relu2, ksize=[1,2,2,1], strides=[1,2,2,1],padding='VALID')

    # conv3 conv4 maxpooling
    w3 = tf.Variable(tf.truncated_normal([3,3,64,64], stddev=0.1))
    b3 = tf.Variable(tf.constant(0.1, shape=[64]))
    conv3 = tf.nn.conv2d(maxpool1, w3, strides=[1,1,1,1], padding='SAME') + b3
    mean3, var3 = tf.nn.moments(conv3, axes=[0, 1, 2])
    offset3 = tf.Variable(tf.constant(0.0, shape=[64]))
    scale3 = tf.Variable(tf.constant(1.0, shape=[64]))
    bn3 = tf.nn.batch_normalization(conv3, mean=mean3, variance=var3, offset=offset3, scale=scale3, variance_epsilon=1e-5)
    relu3 = tf.nn.relu(bn3)

    w4 = tf.Variable(tf.truncated_normal([3,3,64,64], stddev=0.1))
    b4 = tf.Variable(tf.constant(0.1, shape=[64]))
    conv4 = tf.nn.conv2d(relu3, w4, strides=[1,1,1,1], padding='SAME') + b4
    mean4, var4 = tf.nn.moments(conv4, axes=[0, 1, 2])
    offset4 = tf.Variable(tf.constant(0.0, shape=[64]))
    scale4 = tf.Variable(tf.constant(1.0, shape=[64]))
    bn4 = tf.nn.batch_normalization(conv4, mean=mean4, variance=var4, offset=offset4, scale=scale4, variance_epsilon=1e-5)
    relu4 = tf.nn.relu(bn4)
    maxpool2 = tf.nn.max_pool(relu4, ksize=[1,2,2,1], strides=[1,2,2,1],padding='VALID')

    # conv5 conv6 maxpooling
    w5 = tf.Variable(tf.truncated_normal([3,3,64,128], stddev=0.1))
    b5 = tf.Variable(tf.constant(0.1, shape=[128]))
    conv5 = tf.nn.conv2d(maxpool2, w5, strides=[1,1,1,1], padding='SAME') + b5
    mean5, var5 = tf.nn.moments(conv5, axes=[0, 1, 2])
    offset5 = tf.Variable(tf.constant(0.0, shape=[128]))
    scale5 = tf.Variable(tf.constant(1.0, shape=[128]))
    bn5 = tf.nn.batch_normalization(conv5, mean=mean5, variance=var5, offset=offset5, scale=scale5, variance_epsilon=1e-5)
    relu5 = tf.nn.relu(bn5)

    w6 = tf.Variable(tf.truncated_normal([3,3,128,128], stddev=0.1))
    b6 = tf.Variable(tf.constant(0.1, shape=[128]))
    conv6 = tf.nn.conv2d(relu5, w6, strides=[1,1,1,1], padding='SAME') + b6
    mean6, var6 = tf.nn.moments(conv6, axes=[0, 1, 2])
    offset6 = tf.Variable(tf.constant(0.0, shape=[128]))
    scale6 = tf.Variable(tf.constant(1.0, shape=[128]))
    bn6 = tf.nn.batch_normalization(conv6, mean=mean6, variance=var6, offset=offset6, scale=scale6, variance_epsilon=1e-5)
    relu6 = tf.nn.relu(bn6)
    maxpool3 = tf.nn.max_pool(relu6, ksize=[1,2,2,1], strides=[1,2,2,1],padding='VALID')

    # conv7 conv8 maxpooling
    w7 = tf.Variable(tf.truncated_normal([3,3,128,128], stddev=0.1))
    b7 = tf.Variable(tf.constant(0.1, shape=[128]))
    conv7 = tf.nn.conv2d(maxpool3, w7, strides=[1,1,1,1], padding='SAME') + b7
    mean7, var7 = tf.nn.moments(conv7, axes=[0, 1, 2])
    offset7 = tf.Variable(tf.constant(0.0, shape=[128]))
    scale7 = tf.Variable(tf.constant(1.0, shape=[128]))
    bn7 = tf.nn.batch_normalization(conv7, mean=mean7, variance=var7, offset=offset7, scale=scale7, variance_epsilon=1e-5)
    relu7 = tf.nn.relu(bn7)

    w8 = tf.Variable(tf.truncated_normal([3,3,128,128], stddev=0.1))
    b8 = tf.Variable(tf.constant(0.1, shape=[128]))
    conv8 = tf.nn.conv2d(relu7, w8, strides=[1,1,1,1], padding='SAME') + b8
    mean8, var8 = tf.nn.moments(conv6, axes=[0, 1, 2])
    offset8 = tf.Variable(tf.constant(0.0, shape=[128]))
    scale8 = tf.Variable(tf.constant(1.0, shape=[128]))
    bn8 = tf.nn.batch_normalization(conv8, mean=mean8, variance=var8, offset=offset8, scale=scale8, variance_epsilon=1e-5)
    relu8 = tf.nn.relu(bn8)

    # maxpool4 = tf.nn.max_pool(relu8, ksize=[1,2,2,1], strides=[1,2,2,1],padding='VALID')

    reshape1 = tf.reshape(relu8, [-1, 32768])
    w_fc1 = tf.Variable(tf.truncated_normal([32768,1024], stddev=0.1))
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]))
    fc1 = tf.matmul(reshape1, w_fc1) + b_fc1

    w_fc2 = tf.Variable(tf.truncated_normal([1024,8], stddev=0.1))
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[8]))
    fc2 = tf.matmul(fc1, w_fc2) + b_fc2

    loss = tf.reduce_sum(tf.square(tf.sub(fc2, x2))) / 2 / batch_size

    # gpu configuration
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    # gpu_opinions = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

    saver = tf.train.Saver(max_to_keep=None)

    with tf.Session(config=tf_config) as sess:
        saver.restore(sess, dir_load)
        test_model = DataSet(test_img_list)
        loss_total = []
        for i in range(iter_max):
            x_batch_test, y_batch_test, h1_test, img1, img2 = test_model.next_batch()
            np.savetxt(((dir_save + '/h1_%d.txt') % i), h1_test)
            np.savetxt(((dir_save + '/label_%d.txt') % i), y_batch_test)
            cv2.imwrite(((dir_save + '/image_%d_1.jpg') % i), img1)
            cv2.imwrite(((dir_save + '/image_%d_2.jpg') % i), img2)
            pre, average_loss = sess.run([fc2, loss], feed_dict={x1: x_batch_test, x2: y_batch_test})
            np.savetxt(((dir_save + '/predict_%d.txt') % i), pre)
            loss_total.append(average_loss)
            print ('iter %05d, test loss = %.5f' % ((i+1), average_loss))
        np.savetxt((dir_save + '/loss.txt'), loss_total)


if __name__ == "__main__":
    tf.app.run()
