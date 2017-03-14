import tensorflow as tf
from models.homographynet import HomographyNet as HomoNet


iter_max = 90000
batch_size = 65
lr_base = 0.005


def load_data():
    labels = []
    images = []
    return images, labels


class DataSet(object):
    def __init__(self, images, labels):
        self.data1 = images
        self.data2 = labels
        self.index_in_epoch = 0

    def next_batch(self):
        start = self.index_in_epoch
        self.index_in_epoch += batch_size
        if self.index_in_epoch > 500000:
            self.index_in_epoch = 0
            start = self.index_in_epoch
            self.index_in_epoch += batch_size
        end = self.index_in_epoch
        return self.data1[start:end], self.data2[start:end]


def main(_):
    raw_images, raw_labels = load_data()
    x1 = tf.placeholder(tf.float32, [None, 128, 128, 2])
    x2 = tf.placeholder(tf.float32, [None, 8])
    x3 = tf.placeholder(tf.float32, [1])
    net = HomoNet({'data': x1})
    net_out = net.layers['fc2']

    loss = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(net_out, x2)))) / 2 / batch_size
    train_op = tf.train.MomentumOptimizer(learning_rate=x3, momentum=0.9).minimize(loss)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    init = tf.initialize_all_variables()
    saver = tf.train.Saver()
    with tf.Session(config=tf_config) as sess:
        data_model = DataSet(raw_images, raw_labels)
        sess.run(init)
        for i in range(iter_max):
            lr_decay = 0.1 ** (i/30000)
            lr = lr_base * lr_decay
            x_batch, y_batch = data_model.next_batch()
            sess.run(train_op, feed_dict={x1: x_batch, x2: y_batch, x3: lr})


if __name__ == "__main__":
    tf.app.run()
