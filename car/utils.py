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
        "--tensorrt",
        dest="tensorrt", default=0, type=int,
        help="use Tensor_RT or not")

    parser.add_argument(
        "--weight",
        dest="weight", default=None,
        help="pretrain weight file")

    parser.parse_args().record = bool(parser.parse_args().record)
    return parser.parse_args()


def video_Parser():
    parser = argparse.ArgumentParser(prog="python video_node.py")
    parser.add_argument("version", help="v1")

    # -------------------- select options -------------------- #
    parser.add_argument("--mode", help="ignore this", dest="mode", default="video")
    parser.add_argument("--gpu", help="gpu index", dest="gpu", default="0")

    parser.add_argument(
        "--topic",
        dest="topic", default="/usb_cam/image_raw",
        help="ros topic to subscribe")

    parser.add_argument(
        "--dev",
        dest="dev", default="ros",
        help="ros or tx2 or video path or  camera index")

    parser.add_argument(
        "--radar",
        dest="radar", default=0, type=int,
        help="show radar plot")

    parser.add_argument(
        "--show",
        dest="show", default=1, type=int,
        help="show processed image")
    '''
    parser.add_argument(
        "--LP",
        dest="LP", default=1, type=int,
        help="show affined licence plate, if show, add LP box")

    parser.add_argument(
        "--car",
        dest="car", default=1, type=int,
        help="add car box")

    parser.add_argument(
        "--weight",
        dest="weight", default=None,
        help="pretrain weight path")
    '''
    parser.add_argument(
        "--flip",
        dest="flip", default=3, type=int,
        help="1: left-right, 0: top-down, -1: 0&&1")

    parser.add_argument(
        "--clip_h",
        dest="clip_h", default=1., type=float,
        help="height_ratio")

    parser.add_argument(
        "--clip_w",
        dest="clip_w", default=1., type=float,
        help="width_ratio")

    parser.add_argument(
        "--tensorrt",
        dest="tensorrt", default=0, type=int,
        help="use Tensor_RT or not")

    parser.add_argument(
        "--record",
        dest="record", default=0, type=int,
        help="record or not")

    parser.parse_args().radar = bool(parser.parse_args().radar)
    parser.parse_args().show = bool(parser.parse_args().show)
    parser.parse_args().LP = bool(parser.parse_args().LP)

    return parser.parse_args()


class CarNet(basic_yolo.BasicYOLONet):
    def __init__(self, spec, num_sync_bn_devices=-1, **kwargs):
        super(CarNet, self).__init__(spec, num_sync_bn_devices, **kwargs)

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

        return all_output[::-1]


if __name__ == '__main__':
    args = Parser()

    with open(args.version+'/spec.yaml') as f:
        spec = yaml.load(f)

    net = CarNet(spec, num_sync_bn_devices=-1)
    net.test(64*5, 64*8)

    from mxboard import SummaryWriter
    sw = SummaryWriter(logdir=args.version+'/logs', flush_secs=60)
    sw.add_graph(net)
