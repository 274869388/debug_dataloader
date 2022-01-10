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

## 二、测试本地读取数据

### 1. 单线程

```shell
PYTHONPATH='.' python tools/train.py configs/faster_rcnn_r50_fpn_1x_coco_single_thread.py --work-dir work_dirs/single_thread
```

### 2. 多线程

```shell
PYTHONPATH='.' python tools/train.py configs/faster_rcnn_r50_fpn_1x_coco_multi_threads.py --work-dir work_dirs/multi_threads
```

## 三、测试aws后端读取数据

使用自定义的aws后端读取数据需要安装awscli，在requirements.txt中已经指定安装了，这里只需配置一下认证账户即可，需设定AK和SK，如需测试，可以邮件我获取。

```shell
aws configure
```

### 1. 单线程

```shell
PYTHONPATH='.' python tools/train.py configs/faster_rcnn_r50_fpn_1x_coco_single_thread_with_aws.py --work-dir work_dirs/single_thread_with_aws
```

### 2. 多线程

```shell
PYTHONPATH='.' python tools/train.py configs/faster_rcnn_r50_fpn_1x_coco_multi_threads_with_aws.py --work-dir work_dirs/multi_threads_with_aws
```