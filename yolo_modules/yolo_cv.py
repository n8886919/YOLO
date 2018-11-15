import cv2
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import PIL

_color = [
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (255, 255, 255),
    (0, 0, 0)]


class PILImageEnhance():
    def __init__(self, M=0, N=0, R=0, G=1, noise_var=50):
        self.M = M
        self.N = N
        self.R = R
        self.G = G
        self.noise_var = noise_var

    def __call__(self, img, **kwargs):
        r = 0
        if self.M > 0 or self.N > 0 or \
            ('M' in kwargs and kwargs['M'] != 0) or \
                ('N' in kwargs and kwargs['N'] != 0):
            img = self.random_shearing(img, **kwargs)
        if self.R != 0 or ('R' in kwargs and kwargs['R'] != 0):
            img, r = self.random_rotate(img, **kwargs)
        if self.G != 0 or ('G' in kwargs and kwargs['G'] != 0):
            img = self.random_blur(img, **kwargs)
        if self.noise_var != 0 or \
            ('self.noise_var' in kwargs and kwargs['self.noise_var'] != 0):
            img = self.random_noise(img, **kwargs)
        return img, r

    def random_shearing(self, img, **kwargs):
        #https://stackoverflow.com/questions/14177744/
        #how-does-perspective-transformation-work-in-pil
        M = kwargs['M'] if 'M' in kwargs else self.M
        N = kwargs['N'] if 'N' in kwargs else self.N
        w, h = img.size

        m, n = np.random.random()*M*2-M, np.random.random()*N*2-N # +-M or N
        xshift, yshift = abs(m)*h, abs(n)*w

        w, h = w + int(round(xshift)), h + int(round(yshift))
        img = img.transform((w, h), PIL.Image.AFFINE,
            (1, m, -xshift if m > 0 else 0, n, 1, -yshift if n > 0 else 0),
            PIL.Image.BILINEAR)

        return img

    def random_noise(self, img, **kwargs):
        noise_var = kwargs['noise_var'] if 'noise_var' in kwargs else self.noise_var
        np_img = np.array(img)
        noise = np.random.normal(0., noise_var, np_img.shape)
        np_img = np_img + noise
        np_img = np.clip(np_img, 0, 255)
        img = PIL.Image.fromarray(np.uint8(np_img))
        return img

    def random_rotate(self, img, **kwargs):
        R = kwargs['R'] if 'R' in kwargs else self.R

        r = np.random.uniform(low=-R, high=R)
        img = img.rotate(r, PIL.Image.BILINEAR, expand=1)
        r = float(r*np.pi)/180
        return img, r

    def random_blur(self, img,  **kwargs):
        G = kwargs['G'] if 'G' in kwargs else self.G
        img = img.filter(PIL.ImageFilter.GaussianBlur(radius=np.random.rand()*G))
        return img


def init_matplotlib_figure():
    plt.ion()
    fig = plt.figure()
    return fig.add_subplot(1, 1, 1)


def matplotlib_show_img(ax, img):
    ax.clear()
    ax.imshow(img)
    ax.axis('off')


def _numpy_softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)


def cv2_add_bbox(im, b, color_idx):
    r = -b[5]
    im_w = im.shape[1]
    im_h = im.shape[0]
    h = b[3] * im_h
    w = b[4] * im_w
    a = np.array([[
        [ w*math.cos(r)/2 - h*math.sin(r)/2,  w*math.sin(r)/2 + h*math.cos(r)/2],
        [-w*math.cos(r)/2 - h*math.sin(r)/2, -w*math.sin(r)/2 + h*math.cos(r)/2],
        [-w*math.cos(r)/2 + h*math.sin(r)/2, -w*math.sin(r)/2 - h*math.cos(r)/2],
        [ w*math.cos(r)/2 + h*math.sin(r)/2,  w*math.sin(r)/2 - h*math.cos(r)/2]]])
    s = np.array([b[2], b[1]])*[im_w, im_h]
    a = (a + s).astype(int)
    c = np.array(_color[color_idx]) / 255.
    cv2.polylines(im, a, 1, c, 2)
    return im


def cv2_add_bbox_text(img, p, text, c):
    size = img.shape
    c = _color[c % len(_color)]
    l = min(max(int(p[1] * size[1]), 0), size[1])
    t = min(max(int(p[2] * size[0]), 0), size[0])
    r = min(max(int(p[3] * size[1]), 0), size[1])
    b = min(max(int(p[4] * size[0]), 0), size[0])
    cv2.rectangle(img, (l, t), (r, b), c, 2)
    cv2.putText(img, '%s %.3f' % (text, p[0]),
                (l, t-10), 2, 1, c, 2)


class RadarProb():
    def __init__(self, num_cls, classes=None):
        s = 360/num_cls
        self.cos_offset = np.array([math.cos(x*math.pi/180) for x in range(0, 360, s)])
        self.sin_offset = np.array([math.sin(x*math.pi/180) for x in range(0, 360, s)])

        plt.ion()
        fig = plt.figure()
        self.ax = fig.add_subplot(111, polar=True)
        self.ax.grid(False)
        self.ax.set_ylim(0, 1)
        if classes is not None:
            classes = np.array(classes) * np.pi / 180.
            x = np.expand_dims(np.cos(classes[:, 1]) * np.cos(classes[:, 0]), axis=1)
            y = np.expand_dims(np.cos(classes[:, 1]) * np.sin(classes[:, 0]), axis=1)
            z = np.expand_dims(np.sin(classes[:, 1]), axis=1)
            self.classes_xyz = np.concatenate((x, y, z), axis=1)

    def plot3d(self, confidence, prob):
        prob = _numpy_softmax(prob)
        prob = prob * confidence / max(prob)
        vecs = self.classes_xyz * np.expand_dims(prob, axis=1)
        print(np.sum(vecs, axis=0))

        num_angs = [24, 21, 17, 12]
        c = 0
        self.ax.clear()
        for ele, num_ang in enumerate(num_angs):
            ang = np.linspace(0, 2*np.pi, num_ang, endpoint=False)
            width = np.pi * 2 / num_ang + 0.02
            # add 0.02 to avoid white edges between patches
            top = np.array([1.0 - float(ele)/len(num_angs)]*len(ang))
            bottom = top - 1./len(num_angs)

            bars = self.ax.bar(ang, top, width=width, bottom=bottom, linewidth=0)

            for p, bar in zip(prob[c:c+num_ang], bars):
                bar.set_facecolor((p, p, p))
                #bar.set_facecolor(matplotlib.cm.jet(p))
                #bar.set_alpha(0.5)

            c += num_ang
        self.ax.set_title(str(confidence), bbox=dict(facecolor='g', alpha=0.2))
        self.ax.grid(False)

        plt.pause(0.001)

    def plot(self, vec_ang, vec_rad, prob):
        cls_num = len(prob)
        ang = np.linspace(0, 2*np.pi, cls_num, endpoint=False)
        ang = np.concatenate((ang, [ang[0]]))

        prob = np.concatenate((prob, [prob[0]]))

        self.ax.clear()
        self.ax.plot([0, vec_ang], [0, vec_rad], 'r-', linewidth=3)
        self.ax.plot(ang, prob, 'b-', linewidth=1)
        self.ax.set_ylim(0, 1)
        self.ax.set_thetagrids(ang*180/np.pi)
        plt.pause(0.001)

    def cls2ang(self, confidence, prob):
        prob = _numpy_softmax(prob)

        c = sum(self.cos_offset*prob)
        s = sum(self.sin_offset*prob)
        vec_ang = math.atan2(s, c)
        vec_rad = confidence*(s**2+c**2)**0.5

        prob = confidence * prob
        return vec_ang, vec_rad, prob


def open_tx2_onboard_camera(width, height):
    # On versions of L4T previous to L4T 28.1, flip-method=2
    # Use Jetson onboard camera
    gst_str = ("nvcamerasrc ! "
               "video/x-raw(memory:NVMM), width=(int)2592, height=(int)1458, format=(string)I420, framerate=(fraction)30/1 ! "
               "nvvidconv ! video/x-raw, width=(int){}, height=(int){}, format=(string)BGRx ! "
               "videoconvert ! appsink").format(width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
