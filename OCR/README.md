# Licence Plate Recognition(OCR)
+ [OCR Weight Link](https://drive.google.com/open?id=1YbSsDs8FMpEPOYzTW8iPQY_6ESKIhvhr)
## Demo
先跑[Licence Plate Detection](https://github.com/n8886919/YOLO/tree/master/licence_plate)的Demo
```sh
cd <$git clone path>
export PYTHONPATH=$PYTHONPATH:$(pwd)
cd OCR
python OCR v1 export <$OCR_weight_path>
python OCR.py v1 video
```
## Train/Valid/Export
```sh 
python OCR.py v1 train/valid/video/export [--gpu 0]
```
