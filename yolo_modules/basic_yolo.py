from mxnet import gluon
from gluoncv.model_zoo.yolo.darknet import DarknetBasicBlockV3, _conv2d
from gluoncv.model_zoo.yolo.yolo3 import YOLODetectionBlockV3, _upsample


class BasicYOLONet(gluon.HybridBlock):
    def __init__(self, spec, num_sync_bn_devices, **kwargs):
        super(BasicYOLONet, self).__init__(**kwargs)

        layers = spec['layers']
        channels = spec['channels']

        assert len(layers) == len(channels) - 1, (
            "len(channels) should equal to len(layers) + 1, given {} vs {}".format(
                len(channels), len(layers)))

        with self.name_scope():
            self.stages = gluon.nn.HybridSequential()
            self.stages.add(_conv2d(channels[0], 3, 1, 1))

            for nlayer, channel in zip(layers[0:], channels[1:]):
                stage = gluon.nn.HybridSequential()
                stage.add(_conv2d(channel, 3, 1, 2))
                for _ in range(nlayer):
                    stage.add(DarknetBasicBlockV3(channel // 2, num_sync_bn_devices))
                self.stages.add(stage)

        anchors = spec['all_anchors']
        self.slice_point = spec['slice_point']

        self.transitions, self.yolo_blocks, self.yolo_outputs = YOLOPyrmaid(
            channels[-len(anchors):],
            anchors,
            self.slice_point[-1],
            num_sync_bn_devices
        )

        self.num_pyrmaid_layers = len(anchors)

    def hybrid_forward(self, F, x, *args):
        routes = []
        all_output = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i >= len(self.stages) - self.num_pyrmaid_layers:
                routes.append(x)

        # the YOLO output layers are used in reverse order, i.e., from very deep layers to shallow
        for i, block, output in zip(range(len(routes)), self.yolo_blocks, self.yolo_outputs):
            x, tip = block(x)
            all_output.append(output(tip))

            if i >= len(routes) - 1:
                break

            # add transition layers
            x = self.transitions[i](x)

            # upsample feature map reverse to shallow layers
            upsample = _upsample(x, stride=2)
            x = F.concat(upsample, routes[::-1][i + 1], dim=1)

        return self.merge_and_slice(F, all_output, self.slice_point)
        # score, yxhw, cls_pred

    def merge_and_slice(self, F, all_output, points):
        output = F.concat(*all_output, dim=1)
        i = 0
        x = []
        for pt in points:
            x.append(output.slice_axis(begin=i, end=pt, axis=-1))
            i = pt

        return x

    def test(self, h, w):
        from mxnet import init, nd
        import time

        self.initialize(init=init.Xavier())
        self.hybridize()
        fake_img = nd.zeros((1, 3, h, w))
        t = time.time()
        result = self.__call__(fake_img)
        print(t - time.time())
        for r in result:
            print(r)


class YOLOOutput(gluon.HybridBlock):
    def __init__(self, channel, num_anchors, **kwargs):
        super(YOLOOutput, self).__init__(**kwargs)
        # channel = num_lass + 5
        self.channel = channel
        self.num_anchors = num_anchors

        self.yolooutput = gluon.nn.Conv2D(channel * num_anchors, kernel_size=1)

    def hybrid_forward(self, F, x, *args):
        x = self.yolooutput(x)
        x = x.transpose((0, 2, 3, 1))
        x = x.reshape((0, -1, self.num_anchors, self.channel))

        return x


def YOLOPyrmaid(pyrmaid_channels, anchors, total_channels_per_anchor, num_sync_bn_devices):
    transitions = gluon.nn.HybridSequential()
    yolo_blocks = gluon.nn.HybridSequential()
    yolo_outputs = gluon.nn.HybridSequential()

    # note that anchors and strides should be used in reverse order
    for i, channel, anchor in zip(range(len(anchors)), pyrmaid_channels[::-1], anchors[::-1]):

        output = YOLOOutput(total_channels_per_anchor, len(anchor))
        yolo_outputs.add(output)
        block = YOLODetectionBlockV3(channel, num_sync_bn_devices)
        yolo_blocks.add(block)
        if i > 0:
            transitions.add(_conv2d(channel, 1, 0, 1, num_sync_bn_devices))

    return transitions, yolo_blocks, yolo_outputs


if __name__ == '__main__':
    import yaml

    with open('test.yaml') as f:
            spec = yaml.load(f)

    basic_yolo = BasicYOLONet(spec, num_sync_bn_devices=2)
    basic_yolo.test(64*3, 64*4)
