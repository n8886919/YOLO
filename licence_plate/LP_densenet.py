import mxnet
from mxnet.gluon import nn
from gluoncv.model_zoo.densenet import _make_dense_block, _make_transition


class LPDenseNet(mxnet.gluon.HybridBlock):
    # https://github.com/dmlc/gluon-cv/blob/3658339acbdfc78c2191c687e4430e3a673
    # 66b7d/gluoncv/model_zoo/densenet.py#L620
    # Densely Connected Convolutional Networks
    # <https://arxiv.org/pdf/1608.06993.pdf>
    def __init__(self, num_init_features, growth_rate, block_config,
                 bn_size=4, dropout=0, classes=1, **kwargs):
        super(LPDenseNet, self).__init__(**kwargs)
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(nn.Conv2D(num_init_features, kernel_size=7,
                                        strides=2, padding=3, use_bias=False))
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))
            self.features.add(nn.MaxPool2D(pool_size=3, strides=2, padding=1))
            # Add dense blocks
            num_features = num_init_features
            for i, num_layers in enumerate(block_config):
                self.features.add(
                    _make_dense_block(
                        num_layers, bn_size, growth_rate, dropout, i+1))

                num_features = num_features + num_layers * growth_rate
                if i != len(block_config) - 1:
                    self.features.add(_make_transition(num_features // 2))
                    num_features = num_features // 2
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))

            self.features.add(nn.Conv2D(512, (3, 3), padding=1))
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))

            self.features.add(nn.Conv2D(7+classes, (1, 1)))

    def hybrid_forward(self, F, x):
        x = self.features(x)
        return x
