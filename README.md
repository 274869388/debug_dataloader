# BUG复现

用于复现单GPU环境下使用mmdet多线程读取数据卡死的问题。

## 一、环境准备

### 1. pytorch

```shell
pip install torch==1.8.0+cu101 torchvision==0.9.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

### 2. mmcv

```shell
pip install mmcv-full==1.4.2 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.8.0/index.html
```

### 3. others

```shell
pip install -r requirements.txt
```

## 二、测试本地读取数据单线程与多线程

### 1. 单线程

```shell
python tools/train.py configs/faster_rcnn_r50_fpn_1x_coco_single_thread.py --work-dir work_dirs/single_thread
```

### 2. 多线程

```shell
python tools/train.py configs/faster_rcnn_r50_fpn_1x_coco_multi_threads.py --work-dir work_dirs/single_multi_threads
```