from network import Network


class HomographyNet(Network):
    def setup(self):
        (self.feed('data')
             .conv(3, 3, 64, 1, 1, relu='false', name='conv1')
             .batch_normalization(64, relu='true')
             .conv(3, 3, 64, 1, 1, relu='false', name='conv2')
             .batch_normalization(64, relu='true')
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool1')
             .conv(3, 3, 64, 1, 1, relu='false', name='conv3')
             .batch_normalization(64, relu='true')
             .conv(3, 3, 64, 1, 1, relu='false', name='conv4')
             .batch_normalization(64, relu='true')
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool2')
             .conv(3, 3, 128, 1, 1, relu='false', name='conv5')
             .batch_normalization(128, relu='true')
             .conv(3, 3, 128, 1, 1, relu='false', name='conv6')
             .batch_normalization(128, relu='true')
             .max_pool(2, 2, 2, 2, padding='VALID', name='pool3')
             .conv(3, 3, 128, 1, 1, relu='false', name='conv7')
             .batch_normalization(128, relu='true')
             .conv(3, 3, 128, 1, 1, relu='false', name='conv8')
             .batch_normalization(128, relu='true')
             .dropout(0.5, name='dropout1')
             .fc(1024, name='fc1')
             .dropout(0.5, name='dropout2')
             .fc(8, name='fc2'))
