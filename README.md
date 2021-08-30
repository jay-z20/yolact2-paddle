# YOLACT++ paddle

论文：YOLACT++: Better Real-time Instance Segmentation
使用 Paddle 复现


## 一、内容简介

本项目基于paddlepaddle框架复现YOLACT++，YOLACT++是一种新的实例分割网络，由于没有使用两阶段方法中的pooling操作使得可以获得无损失的特征信息，并且在大目标的分割场景下性能表现更优秀

论文地址：

https://arxiv.org/pdf/1912.06218.pdf

参考项目：

https://github.com/dbolya/yolact

https://github.com/PaddlePaddle/PaddleDetection

https://github.com/PaddlePaddle/PaddleSeg

## 二、复现精度
**COCO test-dev2017**

新建 `weights` 复制下面模型参数到 `weights` 目录

```
mkdir  weights
```

`Resnet50` 预训练模型 [百度网盘](https://pan.baidu.com/s/1IsfWKrPhlLTf6d6iDXNVsw) 提取码：cttr

| Image Size | Backbone      | mAP  |download|
|:----------:|:-------------:|:----:|:----:|
| 550        | Resnet50-FPN | 34.1 |[百度网盘](https://pan.baidu.com/s/1jfsLnUrz_2ck4vZoahiu7Q) 提取码: rmxj |

## 三、数据集

COCO2017-完整数据集:

https://aistudio.baidu.com/aistudio/datasetdetail/97273



## 四、环境依赖

> pip install -r requirments.txt

- 硬件：GPU、CPU

- 框架：
  
  aistudio 默认 2.1 版本
  
  Name: paddlepaddle-gpu
  
  Version: 2.1.2.post101

## 五、快速开始

**修改训练数据配置文件**

修改配置文件 `data/config.py`

```
dataset_base = Config({
    'name': 'Base Dataset',

    # Training images and annotations
    'train_images': '/home/aistudio/train2017/', # 修改为训练数据文件夹
    'train_info':   'path_to_annotation_file',

    # Validation images and annotations.
    'valid_images': '/home/aistudio/val2017/',    # 修改为验证数据文件夹
    'valid_info':   'path_to_annotation_file',

    # Whether or not to load GT. If this is False, eval.py quantitative evaluation won't work.
    'has_gt': True,

    # A list of names for each of you classes.
    'class_names': COCO_CLASSES,

    # COCO class ids aren't sequential, so this is a bandage fix. If your ids aren't sequential,
    # provide a map from category_id -> index in class_names + 1 (the +1 is there because it's 1-indexed).
    # If not specified, this just assumes category ids start at 1 and increase sequentially.
    'label_map': None
})

coco2017_dataset = dataset_base.copy({
    'name': 'COCO 2017',
    
    'train_info': '/home/aistudio/annotations/instances_train2017.json', # 修改为 train 据标注
    'valid_info': '/home/aistudio/annotations/instances_val2017.json',   # 修改为 val 数据标注

    'label_map': COCO_LABEL_MAP
})

coco2017_testdev_dataset = dataset_base.copy({
    'name': 'COCO 2017 Test-Dev',
    'valid_images': '/home/aistudio/test2017/',                                  # coco test-dev2017 测试数据集文件夹
    'valid_info': '/home/aistudio/annotations/image_info_test-dev2017.json',     # image_info_test-dev2017.json 文件
    'has_gt': False,

    'label_map': COCO_LABEL_MAP
})
```

**预测**
> 参考 `run.sh`

**训练**
> 参考 `run.sh`


## 六、代码结构与详细说明

### 6.1 代码结构

```
├─data                            # 数据加载和配置
   |--config.py                   # 配置文件
├─layers                          # 中间处理过程和 loss
   |--modules
      |--multibox_loss.py         # 训练的 loss
|--weights                        # 保存模型参数
├─logs                            # 训练日志
├─utils                           # 工具包（计时、日志记录、数据增强）
│--backbone.py                    # backbone(resnet 实现)
│--eval.py                        # 预测和评估
│--yolact.py                      # model 实现
│--README.md                      # 中文readme
│--requirement.txt                # 依赖
│--train.py                       # 训练
```

### 6.2 评估流程
> 参考 `run.sh`

训练中数据记录：日志中可以找到

>"mask": {"all": 34.1, "50": 53.35, "55": 50.95, "60": 48.02, "65": 44.9, "70": 41.01, "75": 36.17, "80": 30.07, "85": 21.89, "90": 12.05, "95": 2.62}


