[![](https://i.imgur.com/V9L34qE.png)](http://ncrl.nctu.edu.tw)
# YOLOv3 For ALPR And VOR
ALPR: Automatic License Plate  / VOR: Vehicle Orientation Recognition Recognition
## Contents
+ [Demo](#Demo)
+ [Installation](#Installation)
+ [Projects](#Projects)
  + [Licence-Plate-Detection](#Licence-Plate-Detection)
  + [Licence-Plate-Recognition](#Licence-Plate-Recognition)
  + [Vehicle-Orientation-Recognition](#Vehicle-Orientation-Recognition)
  + [VOR+LPD](#VOR+LPD)
## Demo

**[ALPR室內測試影片連結](https://www.youtube.com/watch?v=fkOfiv5M6co)**
<img src="https://i.imgur.com/7vC1mX4.png" width=50% height="50%" />

---
**[ALPR室外測試影片連結](https://youtu.be/6XFVttX3pAU?t=10)**
<img src="https://i.imgur.com/RcfgStm.png" width=50% height="50%" />

---
**[模型車姿態辨識影片連結](https://www.youtube.com/watch?v=cGhPUM9HWag&t=10s)**
<img src="https://i.imgur.com/JHLEKpp.png" width=50% height="50%" />

---
**[驗證姿態辨識影片連結](https://www.youtube.com/watch?v=RME7ldMSddQ&t=3)**
<img src="https://i.imgur.com/OJuFdih.png" width=50% height="50%" />

---
**[ALPR+IBVS+VOR](https://youtu.be/uX_UBp0ZFNk)**
<img src="https://i.imgur.com/baNtXRU.png" width=50% height="50%" />

---
## Installation
+ [Install ROS](http://wiki.ros.org/ROS/Installation)(Not necessary for train/valid)
+ Install ROS camera package(Not necessary for train/valid)
```sh
(sudo) apt-get install ros-$version-usb-cam
(sudo) apt-get install ros-$version-cv-bridge
```
+ Download Source Code
```sh
git clone https://github.com/n8886919/YOLO_ALPR
```
+ Install Dependencies

如果不是使用CUDA10,請將requirements.txt 中mxnet-cu100改成對應版本
(ex: CUDA9.2->mxnet-cu92)
```sh
# use Conda(optional)
conda create --name yolo_test python=2.7 pip
conda activate yolo_test
# then
cd <$git clone path>
pip install -r requirements.txt 
```
+ [Install pycuda](https://wiki.tiker.net/PyCuda/Installation/Linux/Ubuntu)(或是把用到他的地方都註解掉)
+ [Install tensorrt](https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html#installing-tar)(或是把用到他的地方都註解掉)

+ [Download Test  Video](https://drive.google.com/file/d/1dYkultUic8WBqNL02yzqRZjyujPuWGA1/view?usp=sharing)
## Projects
### [Licence-Plate-Detection](https://github.com/n8886919/YOLO/tree/master/licence_plate)
預測車牌在空間中的位置與姿態,並將車牌以此姿態投影至相機平面,找出邊界框,最後將邊界框變形回長方形,以利後續辨識文字。由於需要知道車牌姿態來訓練,因此訓練資料完全以合成方式產生。
可匯出網路成ONNX格式,以TensorRT進行推斷,在[Jetson Xavier](https://www.nvidia.com/zh-tw/autonomous-machines/jetson-agx-xavier)約可達50FPS
### [Licence-Plate-Recognition](https://github.com/n8886919/YOLO/tree/master/OCR)
用於辨識車牌偵測後變形回長方形車牌的文字。
### [Vehicle-Orientation-Recognition](https://github.com/n8886919/YOLO/tree/master/car)
以[Blender合成](https://github.com/n8886919/RenderForCar)車輛訓練圖片,用於預測車輛方位角、俯角與邊界框。
### [VOR+LPD](https://github.com/n8886919/YOLO/tree/master/car_and_LP)
<!--
## Troubleshooting
if:
:::danger
[ERROR] [1552911199.414362]: bad callback: <bound method LicencePlateDetectioin._image_callback of <__main__.LicencePlateDetectioin instance at 0x7fd1aa0e1200>>
Traceback (most recent call last):
  File "/opt/ros/melodic/lib/python2.7/dist-packages/rospy/topics.py", line 750, in _invoke_callback
    cb(msg)
  File "LP_detection.py", line 460, in _image_callback
    self.img = self.bridge.imgmsg_to_cv2(img, "bgr8")
  File "/opt/ros/melodic/lib/python2.7/dist-packages/cv_bridge/core.py", line 163, in imgmsg_to_cv2
    dtype, n_channels = self.encoding_to_dtype_with_channels(img_msg.encoding)
  File "/opt/ros/melodic/lib/python2.7/dist-packages/cv_bridge/core.py", line 99, in encoding_to_dtype_with_channels
    return self.cvtype2_to_dtype_with_channels(self.encoding_to_cvtype2(encoding))
  File "/opt/ros/melodic/lib/python2.7/dist-packages/cv_bridge/core.py", line 91, in encoding_to_cvtype2
    from cv_bridge.boost.cv_bridge_boost import getCvType
ImportError: /usr/lib/x86_64-linux-gnu/libblas.so.3: undefined symbol: sgemm_thread_nn
:::
try:
```sh
sudo apt-get remove libopenblas-base
```
-->
