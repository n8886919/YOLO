# Vehicle Orientation Recognition(VOR)
<!--[VOR Weight Link v1](https://drive.google.com/file/d/15N1DZMfx1FsYSp-y597U-pQ3UqhQDZF4/view?usp=sharing)
[VOR Weight Link v2](https://drive.google.com/file/d/1MjkZuel-bEtuY6qEKOAreLbYO27H6rtG/view?usp=sharing)-->
+ [VOR Weight Link v4](https://drive.google.com/file/d/1q5gFMpZopVaN77bGGO0z9-MSFz1Dux_7/view?usp=sharing)
+ [Download Test  Video](https://drive.google.com/file/d/1dYkultUic8WBqNL02yzqRZjyujPuWGA1/view?usp=sharing)
## Video Demo
```sh
cd <$git clone path>
export PYTHONPATH=$PYTHONPATH:$(pwd)
cd car
python YOLO.py v4 export --weight <$LPD_weight_path>
python video_node.py v4 video --dev <$video_path>
```

### optional arguments:
- \-\-dev: 預設使用ros topic, 可改video_path(mp4, avi, m2ts)或/dev/video*的*或'jetson'(開啟tx2相機）
- \-\-topic: 如果沒有dev,可設定ROS topic
- \-\-radar: bool,畫方位角雷達圖
- \-\-show: 顯示包含邊界框的圖片
- \-\-flip:
  + flip = 1: left-right
  + flip = 0: top-down
  + flip = -1: flip=1 && flip=0
- \-\-clip_h:[0,1]
- \-\-clip_w:[0,1]

- \-\-gpu: 預設 "gpu 0", 多GPU訓練或gpu0忙碌時時可以"gpu 1,2"

## Train/Valid/Export
```sh
python YOLO.py <version> <mode>
```
### positional arguments:
- version:
  + v1: darknet53, 方位角
  + v2: small darknet, 方位角+俯角
  + v3: small darknet, 方位角, FP16
  + v4: middle darknet, 方位角

- mode:
  + train
  + valid: use executor
  + export
  + render_and_train
  + kmean: get default anchor size
  + valid_Nima
  + valid_Nima_plot
### optional arguments:
- \-\-gpu: 預設 "gpu 0", 多GPU訓練或gpu0忙碌時時可以"gpu 1,2"
- \-\-weight: 權重路徑
- \-\-record: 測試用, **訓練時**是否紀錄權重與loss
