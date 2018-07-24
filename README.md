# yolov3-mxnet
A yolov3 simple implementation in MXNet, based on version 1.2.0 and cuda 9.0(optional), python3.
Works on Windows and Ubuntu 16.04.

<br>
<br>
**new: hybridized, speed up.**
<br>
**new: train demo.**
<br>
<br>
**Detect Part Completed.**
<br>
=======

## Table of Contents

- [yolov3-mxnet](#yolov3-mxnet)
  * [Paper](#paper)
  * [Installation](#installation)
  * [Detection](#detection)
  * [Train](#train)
  * [Credit](#credit)

## Paper

### YOLOv3: An Incremental Improvement

_Joseph Redmon, Ali Farhadi_
<br \>

**Abstract**
<br \>
We present some updates to YOLO! We made a bunch
of little design changes to make it better. We also trained
this new network that’s pretty swell. It’s a little bigger than
last time but more accurate. It’s still fast though, don’t
worry. At 320 × 320 YOLOv3 runs in 22 ms at 28.2 mAP,
as accurate as SSD but three times faster. When we look
at the old .5 IOU mAP detection metric YOLOv3 is quite
good. It achieves 57.9 AP50 in 51 ms on a Titan X, compared
to 57.5 AP50 in 198 ms by RetinaNet, similar performance
but 3.8× faster. As always, all the code is online at
https://pjreddie.com/yolo/.

[[Paper]](https://pjreddie.com/media/files/papers/YOLOv3.pdf) [[Original Implementation]](https://github.com/pjreddie/darknet)

## Installation

    $ git clone https://github.com/eriklindernoren/PyTorch-YOLOv3
    $ cd PyTorch-YOLOv3/
    $ sudo pip3 install opencv-python mxnet-cu90==1.2.0

## Detection

put your images in ./images, and

    $ python detect.py [--gpu GPU ID]

you will get the results in ./results
or you can detect a video file, this need your opencv is compiled with ffmpeg.

    $ python detect.py --video VIDEO_FILE

you will get result.avi in ./results

## Train (for reference only)

The IMAGE_FOLDER should contains two directorys, "train" and "train_label", (or four, "train", "train_label", "val", "val_label") label should be xml file like VOC's format.
PS: You can use voc_label.py in https://pjreddie.com/darknet/yolo/ to get train.txt and val.txt, set path to --train and --val instead of --images.
```
    train.py [-h] [--epochs EPOCHS] [--images IMAGES FOLDER]
                [--batch_size BATCH_SIZE]
                [--params WEIGHTS_PATH] [--classes CLASS_PATH]
                [--confidence CONF_THRES] [--nms_thresh NMS_THRES]
                [--gpu GPU ID] [--prefix PARAMS FILE NAME PREFIX]
```

## Credit

```
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
```
