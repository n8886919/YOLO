from __future__ import print_function
import argparse
import cv2
import numpy as np
import os
import time

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import mxnet
from mxnet import autograd
from mxnet import gluon
from mxnet import gpu
from mxnet import init
from mxnet import nd
from mxnet.gluon import nn
from mxboard import SummaryWriter
from gluoncv.model_zoo.densenet import _make_dense_block, _make_transition

from yolo_modules import yolo_cv
from yolo_modules import yolo_gluon
from yolo_modules import global_variable
from yolo_modules import licence_plate_render
import matplotlib.pyplot as plt

alphabet = 'ABCDEFGHJKLMNPQRSTUVWXYZ'
number = '012356789'
save_counter = 0


class OCRDenseNet(gluon.HybridBlock):
    # https://github.com/dmlc/gluon-cv/blob/3658339acbdfc78c2191c687e4430e3a673
    # 66b7d/gluoncv/model_zoo/densenet.py#L620
    # Densely Connected Convolutional Networks
    # <https://arxiv.org/pdf/1608.06993.pdf>

    def __init__(self, num_init_features, growth_rate, block_config,
                 bn_size=4, dropout=0, classes=1000, **kwargs):

        super(OCRDenseNet, self).__init__(**kwargs)
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
                self.features.add(_make_dense_block(num_layers, bn_size, growth_rate, dropout, i+1))
                num_features = num_features + num_layers * growth_rate

                if i != len(block_config) - 1:
                    self.features.add(_make_transition(num_features // 2))
                    num_features = num_features // 2
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))

            self.features.add(nn.Conv2D(512, (10, 1)))
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))

            self.features.add(nn.Conv2D(classes+1, (1, 1)))

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = x.transpose((0, 2, 3, 1))
        score_x = x.slice_axis(begin=0, end=1, axis=-1)
        class_x = x.slice_axis(begin=1, end=35, axis=-1)
        return [score_x, class_x]


def loss_mask(labels, gpu_index):
    bs = labels.shape[0]
    score_y = nd.zeros((bs, 1, area, 1), ctx=ctx[gpu_index])
    class_y = nd.ones((bs, 1, area), ctx=ctx[gpu_index]) * (-1)

    for b in range(bs):
        label = labels[b]
        nd.random.shuffle(label)

        for L in label:  # all object in the image
            L = L.asnumpy()
            left = int(round(L[1]*area))
            right = int(round(L[2]*area))

            for i in range(left, right):
                text_cent = (L[1] + L[2]) / 2.
                box_cent = (i + 0.5) / float(area)
                score = 1 - abs(box_cent-text_cent) / float(L[2]-L[1])

                score_y[b, 0, i, 0] = score  # others are zero
                # if score.asnumpy()[0] > 0.7:
                class_y[b, 0, i] = L[0]

    return score_y, class_y


def train_the(batch_xs, batch_ys):
    with autograd.record():
        for gpu_i in range(len(batch_xs)):
            batch_x = batch_xs[gpu_i]
            batch_y = batch_ys[gpu_i]

            score_x, class_x = net(batch_x)
            with autograd.pause():
                score_y, class_y = loss_mask(batch_y, gpu_i)

            score_loss = LG_loss(score_x, score_y) * score_weight
            class_loss = CE_loss(class_x, class_y, score_y) * class_weight
            # Use score_y as mask
            (score_loss + class_loss).backward()

    trainer.step(batch_size)
    record_to_tensorboard([score_loss, class_loss])


def record_to_tensorboard(loss):
    global backward_counter
    n = ['score', 'class']
    yolo_gluon.record_loss(loss, n, summary_writer, step=backward_counter)

    backward_counter += 1
    if backward_counter % record_step == 0:
        save_path = os.path.join(backup_dir, args.version + '_%d')

        net.collect_params().save(save_path)


def _image_callback(img):
    img = bridge.imgmsg_to_cv2(img, "bgr8")
    img = cv2.resize(img, tuple(size[::-1]))

    nd_img = yolo_gluon.cv_img_2_ndarray(img, ctx[0])

    score, text = predict(nd_img)
    cv2_show_OCR_result(img, score, text)
    pub.publish(text)


saver = {}
def cv2_show_OCR_result(img, score, text):
    global save_counter, saver
    # image/text/left-top/font type/size/color/width
    cv2.imshow('img', img)
    if len(text) == 7 and \
       text[0] in alphabet and \
       text[1] in alphabet and \
       text[2] in alphabet and \
       text[3] in number and \
       text[4] in number and \
       text[5] in number and \
       text[6] in number:

        x = np.arange(8, 384, 16).reshape(-1, 1)
        y = ((1-score) * 160).reshape(-1, 1)
        points = np.concatenate((x, y), axis=-1)
        points = np.expand_dims(points, axis=0).astype(np.int32)

        cv2.polylines(img, points, 0, (255, 0, 0), 2)
        cv2.putText(img, text, (0, 60), 2, 2, (0, 0, 255), 2)

        folder = '/media/nolan/SSD1/YOLO_backup/ocr/saved_LP'
        path = os.path.join(folder, 'ocr_out_%d.jpg' % save_counter)
        #cv2.imwrite(path, img)
        save_counter += 1

        if text not in saver:
            saver[text] = 0
        else:
            saver[text] += 1
        print(text)
    cv2.waitKey(1)


def predict(nd_img):
    score_x, class_x = executor.forward(is_train=False, data=nd_img)

    score_x = nd.sigmoid(score_x[0].reshape(-1)).asnumpy()
    class_x = nd.softmax(class_x[0][0], axis=-1).asnumpy()

    #p = np.argmax(class_x, axis=-1)[0]

    score_x2 = np.concatenate(([0], score_x, [0]))
    # zero-dimensional arrays cannot be concatenated
    # Find peaks
    text = ''
    for i in range(24):
        if score_x2[i+1] > 0.6 and \
           score_x2[i+1] > score_x2[i+2] and score_x2[i+1] > score_x2[i]:

            c = np.argmax(class_x[i])
            #print('\033[1;32m%s\033[0m(%.2f),' % (
            #      cls_names[c], np.max(class_x[i])), end='')
            text = text + cls_names[c]
    #print()
    return score_x, text


# -------------------- Parser -------------------- #
parser = argparse.ArgumentParser(prog="python YOLO.py")
parser.add_argument("version", help="v1")
parser.add_argument("mode", help="train or valid or video")

parser.add_argument("--gpu", help="gpu index", dest="gpu", default="0")
parser.add_argument(
    "--tensorrt",
    dest="tensorrt", default=0, type=int,
    help="use Tensor_RT or not")

args = parser.parse_args()

ctx = yolo_gluon.get_ctx(args.gpu)

size = [160, 384]
cls_names = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
    'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R',
    'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

LR = 0.001
batch_size = 100
record_step = 1000

num_init_features = 32
growth_rate = 12
block_config = [6, 12, 24]

area = size[1] / 2 ** (len(block_config)+1)
score_weight = 0.1
class_weight = 1.0

export_file = args.version + '/export/'

if args.mode == 'video':
    executor = yolo_gluon.init_executor(export_file, size, ctx[0])
    bridge = CvBridge()
    rospy.init_node("OCR_node", anonymous=True)
    pub = rospy.Publisher('YOLO/OCR', String, queue_size=0)
    rospy.Subscriber('/YOLO/clipped_LP', Image, _image_callback)

    print('Image Topic: /YOLO/clipped_LP')
    print('checkpoint file: %s' % export_file)

    r = rospy.Rate(30)
    while not rospy.is_shutdown():
        r.sleep()

else:
    net = OCRDenseNet(num_init_features, growth_rate, block_config, classes=34)
    backup_dir = os.path.join(args.version, 'backup')
    weight = yolo_gluon.get_latest_weight_from(backup_dir)
    yolo_gluon.init_NN(net, weight, ctx)

if args.mode == 'train':
    batch_size = 100 * len(ctx)
    backward_counter = 0

    CE_loss = gluon.loss.SoftmaxCrossEntropyLoss(from_logits=False)
    LG_loss = gluon.loss.LogisticLoss(label_format='binary')
    L2_loss = gluon.loss.L2Loss()

    trainer = gluon.Trainer(
        net.collect_params(),
        'adam',
        {'learning_rate': LR})

    log_dir = args.version + '/logs'
    summary_writer = SummaryWriter(logdir=log_dir, verbose=True)
    # prob, score  = net(nd.zeros((1,3,160,384), ctx=ctx[0]))
    # summary_writer.add_graph(get_feature)
    # print(prob.shape, score.shape)
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)

    print(global_variable.cyan)
    print('OCR Render And Train')

    h, w = size
    bg_iter_train = yolo_gluon.load_background('train', batch_size, h, w)
    generator = licence_plate_render.LPGenerator(*size)

    while True:
        if (backward_counter % 10 == 0 or 'bg' not in locals()):
            bg = yolo_gluon.ImageIter_next_batch(bg_iter_train)
            bg = bg.as_in_context(ctx[0])

        imgs, labels = generator.render(bg)

        batch_xs = yolo_gluon.split_render_data(imgs, ctx)
        batch_ys = yolo_gluon.split_render_data(labels, ctx)

        train_the(batch_xs, batch_ys)
        #raw_input('next')

elif args.mode == 'valid':
    print(global_variable.cyan)
    print('Render And Valid')

    bs = 16
    h, w = size
    bg_iter = yolo_gluon.load_background('val', bs, *size)
    generator = licence_plate_render.LPGenerator(*size)

    plt.ion()
    fig = plt.figure()
    axs = []
    for i in range(bs):
        axs.append(fig.add_subplot(4, 4, i+1))

    for bg in bg_iter:
        bg = bg.data[0].as_in_context(ctx[0])
        imgs, labels = generator.render(bg)
        score_x, class_x = net(imgs)
        print(score_x.shape)
        print(class_x.shape)
        imgs = yolo_gluon.batch_ndimg_2_cv2img(imgs)
        for i in range(bs):
            ax = axs[i]
            s = score_x[i]
            s = nd.sigmoid(s.reshape(-1)).asnumpy()
            p = class_x[i, 0].asnumpy()
            p = np.argmax(p, axis=-1)
            yolo_cv.matplotlib_show_img(ax, imgs[i])
            ax.plot(range(8, 384, 16), (1-s)*160)
            ax.axis('off')

            s = np.concatenate(([0], s, [0]))
            # zero-dimensional arrays cannot be concatenated
            # Find peaks
            text = ''
            for i in range(24):
                if s[i+1] > 0.2 and s[i+1] > s[i+2] and s[i+1] > s[i]:
                    c = int(p[i])
                    text = text + cls_names[c]
            print(text)

        raw_input('press Enter to next batch....')

elif args.mode == 'export':
    yolo_gluon.export(net,
                      (1, 3, size[0], size[1]),
                      ctx[0],
                      export_file)
