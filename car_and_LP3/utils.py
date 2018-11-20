import argparse

import mxnet
from gluoncv.model_zoo.yolo.yolo3 import YOLODetectionBlockV3
from gluoncv.model_zoo.yolo.yolo3 import _upsample

from yolo_modules import basic_yolo


def yolo_Parser():
    parser = argparse.ArgumentParser(prog="python YOLO.py")

    parser.add_argument("version", help="v1")
    parser.add_argument("mode", help="train or valid")

    # -------------------- select options -------------------- #
    parser.add_argument("--gpu", help="gpu index", dest="gpu", default="0")
    parser.add_argument(
        "--record",
        dest="record", default=1, type=int,
        help="record to tensorboard or not")

    parser.add_argument(
        "--weight",
        dest="weight", default=None,
        help="pretrain weight file")

    parser.parse_args().record = bool(parser.parse_args().record)
    return parser.parse_args()


def video_Parser():
    parser = argparse.ArgumentParser(prog="python YOLO.py")

    parser.add_argument("version", help="v1")

    # -------------------- select options -------------------- #
    parser.add_argument("--gpu", help="gpu index", dest="gpu", default="0")

    parser.add_argument(
        "--topic", "-t",
        dest="topic", default="",
        help="ros topic to subscribe")

    parser.add_argument(
        "--radar",
        dest="radar", default=0, type=int,
        help="show radar plot")

    parser.add_argument(
        "--show",
        dest="show", default=1, type=int,
        help="show processed image")

    parser.add_argument(
        "--LP",
        dest="show", default=1, type=int,
        help="show affined licence plate, if show, add LP box")

    parser.add_argument(
        "--weight",
        dest="weight", default=None,
        help="pretrain weight path")

    parser.parse_args().show = bool(parser.parse_args().show)
    parser.parse_args().radar = bool(parser.parse_args().radar)
    parser.parse_args().LP = bool(parser.parse_args().LP)
    return parser.parse_args()


class CarLPNet(basic_yolo.BasicYOLONet):
    def __init__(self, spec, num_sync_bn_devices=1, **kwargs):
        super(CarLPNet, self).__init__(spec, num_sync_bn_devices, **kwargs)

        LP_channel = spec['channels'][-3]
        self.LP_slice_point = spec['LP_slice_point']

        self.LP_branch = mxnet.gluon.nn.HybridSequential()
        self.LP_branch.add(YOLODetectionBlockV3(LP_channel, num_sync_bn_devices))
        self.LP_branch.add(basic_yolo.YOLOOutput(self.LP_slice_point[-1], 1))

    def hybrid_forward(self, F, x, *args):
        routes = []
        all_output = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i >= len(self.stages) - self.num_pyrmaid_layers:
                routes.append(x)

        # the YOLO output layers are used in reverse order, i.e., from very deep layers to shallow
        end = False
        for i, block, output in zip(range(len(routes)), self.yolo_blocks, self.yolo_outputs):
            if i >= len(routes) - 1:
                _, LP_output = self.LP_branch[0](x)
                LP_output = self.LP_branch[1](LP_output)
                end = True

            x, tip = block(x)
            all_output.append(output(tip))

            if end:
                break
            # add transition layers
            x = self.transitions[i](x)

            # upsample feature map reverse to shallow layers
            upsample = _upsample(x, stride=2)
            x = F.concat(upsample, routes[::-1][i + 1], dim=1)

        out = [self.merge_and_slice(F, all_output[::-1], self.slice_point)] + \
            [self.merge_and_slice(F, LP_output, self.LP_slice_point)]
        return out  # [(score, yx, hw, cls_pred), (score, xy, z, r)]


if __name__ == '__main__':
    args = Parser()

    with open(args.version+'/spec.yaml') as f:
        spec = yaml.load(f)

    net = CarLPNet(spec, num_sync_bn_devices=2)
    net.test(64*5, 64*8)

    from mxboard import SummaryWriter
    sw = SummaryWriter(logdir=args.version+'/logs', flush_secs=60)
    sw.add_graph(net)
