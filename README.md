# [My YOLO README.md](https://github.com/n8886919/YOLO/blob/master/README.md)

# requirment
```sh
cd YOLO
conda env create -f environment.yml
export PYTHONPATH=$PYTHONPATH:YOLO_PATH
source activate cu100
sudo apt-get install ros-<$ROS_VERSION>-cv-bridge
```
# Demo
<iframe width="560" height="315" src="https://www.youtube.com/embed/6XFVttX3pAU?start=10" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
<iframe width="560" height="315" src="https://www.youtube.com/embed/uX_UBp0ZFNk" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

# Car + Lincese Plate
## 訓練/驗證
```sh
python YOLO.py v1/v2/v3 train [--gpu 0] [--weight path] [--record 1]
python YOLO.py v1/v2/v3 valid [--gpu 0] [--weight path] [--tensorrt 0]
python YOLO.py v1/v2/v3 export [--gpu 0] [--weight path]
```
positional arguments:
- version:
  + v1: darknet53, 方位角+俯角
  + v2: darknet53, 方位角
  + v3: small darknet, 方位角
  + v4: middle darknet, 方位角
- mode:
  + train
  + valid: use executor
  + export
  + render_and_train
  + kmean: get default anchor size
  + valid_Nima
  + valid_Nima_plot
- (gpu): 預設 "gpu 0", 多GPU訓練或gpu0忙碌時時可以"gpu 1,2"
- (weight): 權重路徑
- (record): 測試用, 訓練時是否紀錄權重與loss
## 影片測試
需要先export!!!
```sh
python video_node.py v1/v2/v3 train [--gpu 0] [--weight path] [--record 1]
```
- (tensorrt): xxx
- (dev): 預設使用ros topic, 可改video_path(mp4, avi, m2ts)或/dev/video*的*或'jetson'(開啟tx2相機）
- (topic): 如果沒有dev,可設定ros topic
- (radar): bool,畫方位角雷達圖
- (show): 顯示包含邊界框的圖片
- (flip):
  + flip = 1: left-right
  + flip = 0: top-down
  + flip = -1: flip=1 && flip=0
- (clip_h)
- (clip_w)
- (record): xxx
# Car
- version:
  + v1: darknet53, 方位角
  + v2: small darknet, 方位角+俯角
  + v3: small darknet, 方位角, FP16
<iframe width="560" height="315" src="https://www.youtube.com/embed/cGhPUM9HWag" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

<iframe width="560" height="315" src="https://www.youtube.com/embed/RME7ldMSddQ" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

# Lincese Plate

# OCR
```sh
python OCR.py v1 train/valid/video/export [--gpu 0]
```
[權重連結](https://drive.google.com/open?id=1YbSsDs8FMpEPOYzTW8iPQY_6ESKIhvhr)
- (gpu)

# Agent_controller
# yolo_ws
