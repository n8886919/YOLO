# Licence Plate Detection
+ [LPD Weight Link](https://drive.google.com/file/d/1KbWPrcSqCCn3XZ3wqlEP45U13wHm2mZS/view?usp=sharing)
+ [Download Test  Video](https://drive.google.com/file/d/1dYkultUic8WBqNL02yzqRZjyujPuWGA1/view?usp=sharing)
```sh
cd <$git clone path>
export PYTHONPATH=$PYTHONPATH:$(pwd)
cd licence_plate
python LP_detection.py v2 export --weight <$LPD_weight_path>
```
## Video Demo
```sh
python LPD_video_node.py v2 video --dev <$video_path>
```
## Camera Demo
```sh
rosrun usb_cam usb_cam_node (or another camera_node)
py LPD_video_node.py v2 video (--topic your_video_topic)
```
## Train/Valid/Export
- [ ] TODO
