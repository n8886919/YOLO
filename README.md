# YOLO
## requirment
```sh
pip install mxnet-cu92 mxboard tensorflow
pip install gluoncv --pre --upgrade
pip install pyyaml opencv-python rospkg numpy==1.14.0 cython
conda install matplotlib
sudo apt-get install ros-<$ROS_VERSION>-cv-bridge
```
https://askubuntu.com/questions/904561/how-to-change-hard-drive-name
## Demo
### car_and_LP
```sh
python HP_31.py HP_34 {train, valid, video} 
```
if select video mode, set [here](https://github.com/n8886919/YOLO/blob/master/car_and_LP/HP_31.py#L953) to subscribe specific rostopic
if no 'rostopic' args in run, use '/dev/video0' as default video 

### car_and_LP2
(https://github.com/n8886919/YOLO/blob/master/car_and_LP2/YOLO.py)
```sh
python YOLO.py v2 {train, valid, video} --topic rostopic --gpu 01 --radar {0, 1} --show {0, 1}
```

pretrain weight [Download](https://drive.google.com/open?id=1nYwdq2NUFbPKEoJoROa4dllTroVCVRpz)
### car_and_LP3
```sh
python YOLO.py v1 {train, valid, video} --topic rostopic --gpu 0123 --radar {0, 1} --show {0, 1}
```
## 可能遇到的問題
No package 'yaml-cpp' found
```sh
sudo apt-get install libyaml-cpp-dev
```
ImportError: No module named em
```sh
pip install empy
```
GPU 散熱不好
```sh
nvidia-xconfig --enable-all-gpus
nvidia-xconfig --cool-bits=4
```
忘記幹麻的了
```sh
wget -q -O /tmp/libpng12.deb http://mirrors.kernel.org/ubuntu/pool/main/libp/libpng/libpng12-0_1.2.54-1ubuntu1_amd64.deb \
  && dpkg -i /tmp/libpng12.deb \
  && rm /tmp/libpng12.deb
```
