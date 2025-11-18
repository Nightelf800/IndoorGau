

# Symphonies使用说明

### [Symphonize 3D Semantic Scene Completion with Contextual Instance Queries](https://arxiv.org/abs/2306.15670)



## 环境配置

### 环境安装

- python 3.8
- cuda 11.3
- pytorch 1.12.0

```bash
pip install -U openmim

mim install mmengine
mim install "mmcv==2.1.0"
mim install mmdet

pip install -r requirements.txt
```

### 数据准备

#### 下载数据

**honor采集数据**
1. 解压honor_1519.zip、honor_1540.zip

```
20240711_1519_finterval1
|-- color
    |-- 000000.jpg
    |-- 000001.jpg
    ...
|-- depth
    |-- 000000.png
    |-- 000001.png
    ...
|-- instrinsic
    |-- 000000.txt
    |-- 000001.txt
    ...
|-- voxels
    |-- 000000.pkl
    |-- 000001.pkl
    ...
20240711_1540_finterval1
|-- color
|-- depth
|-- instrinsic
|-- voxels
```
2. 分别软链接到代码`./data/honor_data_0920/`路径下。 `ln -s 解压路径 ./data/honor_data_0920/20240711_1519_finterval1`
3. 预处理数据。 
方案一: 复制train_file_xxx.txt和test_file_xxx.txt到`./data/honor_data_0920/`路径下。 
方案二: 执行`tools/preproces_honor_from_scenes.py`，生成 test_files_split_9_1.txt 和 train_files_split_9_1.txt。 
拆分规则: 每连续的10帧，前9帧为训练帧，后1帧为测试帧。

#### 预训练权重

1. Encoder MaskDINO的预训练权重

MaskDINO 预训练权重下载 [here](https://github.com/hustvl/Symphonies/releases/download/v1.0/maskdino_r50_50e_300q_panoptic_pq53.0.pth)，同样保存到`./checkpoints/`路径下。

2. 支持只输入RGB图像，深度图使用深度估计模型

**DepthAnything:** 模型中已经内嵌DepthAnything推理模型，需下载两个权重文件，保存到`./checkpoints/`路径下。

- [depth_indoor](https://huggingface.co/spaces/LiheYoung/Depth-Anything/tree/main/checkpoints_metric_depth)
- [vitl14](https://huggingface.co/spaces/LiheYoung/Depth-Anything/blob/main/checkpoints/depth_anything_vitl14.pth)


## 训练

### 训练参数调整：

`configs/config_syndata_8cm_442_11class_8k.yaml`:

- trainer
  - device: 1-8  # 管理使用的显卡数，推荐使用2卡复现

`configs/datasets/syndata_8cm_442_11class_8k.yaml`：

- data_root：honor数据集采集路径
- depth_root（label）：honor数据集采集路径
- label_root：honor数据集采集路径
- depth_eval：true / false    # true表示使用depthAnything深度估计，false表示使用深度label
- use_crop: true / false   # true表示使用crop，false表示使用scale
- voxel_size: 0.04 / 0.08   # dataloader中使用分辨率0.04或者0.08
- num_classes: 11           # 类别数量

### 运行

1. 有RGB图像对应的深度图

```
[CUDA_VISIBLE_DEVICES=0,1,2...] python train.py --config-name config_syndata_8cm_442_11class_8k
```

2. 使用深度估计模型预测深度

```
[CUDA_VISIBLE_DEVICES=0,1,2...] python train.py --config-name syndata_depth_eval_8cm_442_11class_8k
```


## 测试

加载训练得到的模型，执行推理，得到推理指标、fps、动态物体可视化结果。

```
python test.py --config-name config_syndata_8cm_442_11class_8k +ckpt_path=...
```

## 可视化

使用可视化mlab工具，推荐使用python3.8,推荐安装下列版本的第三方包：
```pip
pip install vtk==9.0.1
pip install mayavi==4.7.3
```

1. 可视化测试集occ预测结果

```
python visualize.py --config-name config_syndata_8cm_442_11class_8k +ckpt_path=...
```

2. 可视化动态物体occ预测结果

```
python visualize.py --config-name config_syndata_people_8cm_442_11class_8k +ckpt_path=...
```

3. 可视化镜面occ预测结果

```
python visualize.py --config-name config_syndata_mirror_8cm_442_11class_8k +ckpt_path=...
```



## 结果

|         Method          |  train  |  test   |          划分        |  类别数  |     depth      |
| :---------------------: |  :----: |  :----: | :-----------------:  |  :--:   | :------------: |
|       Symphonies        |   654   |   163   |   前80%训练后20%推理   |  128   |      label     |
|       Symphonies        |   1264  |   315   |   前80%训练后20%推理   |  128   |      label     |
|       Symphonies        |   6960  |   776   |   前80%训练后20%推理   |  128   |      label     |
|       Symphonies        |   6960  |   776   | 每10帧前9帧训练后1帧推理 |  128   |      label     |
|       Symphonies        |   6960  |   776   | 每10帧前9帧训练后1帧推理 |  128   |      label     |

