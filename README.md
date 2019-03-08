# 经典论文

## MODEL ARCHITECTURE
### Alexnet
[ImageNet Classification with Deep Convolutional Neural Networks.pdf](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)

深度学习兴起的引子

### VGG
[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

使用 3*3 卷积减少参数量的深层网络

### GoogleNet，Inception系列
* Inception-V1: [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)
* Inception-V2: [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
* Inception-V3: [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567)
* Inception-V4: [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261), 即 inception-resnet
* Xception: [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/abs/1610.02357)

### Densenet
[Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)


### 參差系列
何恺明个人主页[http://kaiminghe.com/](http://kaiminghe.com/)
* ResNet: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) 2015

  [论文翻译](http://noahsnail.com/2017/07/31/2017-7-31-ResNet%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E2%80%94%E2%80%94%E4%B8%AD%E6%96%87%E7%89%88/)
* ResNeXt: [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431) 2017

  [pytorch](https://github.com/miraclewkf/ResNeXt-PyTorch)
* SE-ResNet & SE-ResNeXt: [Squeeze-And-Excitation Networks](https://arxiv.org/abs/1709.01507) 2018



### 轻量级网络
* MobileNet-V1: [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861)
  * MobileNet: 可以牺牲少量性能来大幅降低网络的计算量
  * 将标准卷积转化为深度可分离卷积(depthwise conv + pointwise conv)，depthwise conv只在单一通道上进行卷积，提取特征，pointwise conv 为1*1*N的卷积，将不同通道的特征连接起来，产生新的特征; 定义了两个超参 \alpha 和 \rho 用于进一步缩减网络，分别是宽度乘数和分辨率乘数，用于减少通道数和输入图像的大小，范围为(0,1);
  * MobileNet 可以用于分类、检测、识别等各个领域，适用于移动端和嵌入式端
* MobileNet-V2: [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
* ShuffleNet-V1: [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083)
* ShuffleNet-V2: [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164)

## LOSS FUNCTION
* AM-Softmax
  * [Additive Margin Softmax for Face Verification](https://arxiv.org/abs/1801.05599)
  * [github](https://github.com/happynear/AMSoftmax) caffe tensorflow

* A-Softmax
  * [SphereFace: Deep Hypersphere Embedding for Face Recognition](https://arxiv.org/abs/1704.08063)
  * [github](https://github.com/wy1iu/sphereface) caffe

* L-Softmax
  * [Large-Margin Softmax Loss for Convolutional Neural Networks](http://proceedings.mlr.press/v48/liud16.pdf)
  * [github](https://github.com/wy1iu/LargeMargin_Softmax_Loss) caffe

* Triplet loss
  * [In Defense of the Triplet Loss for Person Re-Identification](https://arxiv.org/abs/1703.07737)
  * [github](https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py)

## Object Detection
* RCNN
  [Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/abs/1311.2524)
* Fast-RCNN
  [Fast R-CNN](https://arxiv.org/abs/1504.08083)
* Faster R-CNN
  [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)
* RetinaNet-Focal Loss
  [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
* Mask-RCNN
  [Mask R-CNN](https://arxiv.org/abs/1703.06870)
* YOLO
  * [YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767): YOLO-V3, 检测速度极大提升
* [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325):
  * SSD，目标检测
  * github [pytorch](https://github.com/amdegroot/ssd.pytorch) [caffe](https://github.com/weiliu89/caffe/tree/ssd)
* [FaceBoxes: A CPU Real-time Face Detector with High Accuracy](https://arxiv.org/abs/1708.05234) 
  * 人脸检测的小网络，可以做到CPU实时，用到了SSD的思想
  * [github](https://github.com/XiaXuehai/faceboxes)
* [Finding Tiny Faces](https://arxiv.org/abs/1612.04402): 小目边检测最好的方法，但是速度极慢
* [An Analysis of Scale Invariance in Object Detection – SNIP](https://arxiv.org/abs/1711.08189): 小目标检测相关
* [Generalized Intersection over Union: A Metric and A Loss for Bounding BoxRegression](https://arxiv.org/abs/1902.09630): 提出一种新的IOU计算方法

## Semantic Segmentation
* FCNN
  * [Fully Convolutional Networks for Semantic Segmentation](chrome-extension://gfbliohnnapiefjpjlpjnehglfpaknnc/pages/pdf_viewer.html?r=https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)：
    用于语义分割的全卷积神经网络，开启使用卷积神经网络进行语义分割的先河;

## OTHERS
### [A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285)
上采样的优点

### [Escaping From Saddle Points –Online Stochastic Gradient for Tensor Decomposition](https://arxiv.org/abs/1503.02101)
batch的优点

### [Data Distillation: Towards Omni-Supervised Learning](https://arxiv.org/abs/1712.04440)
  数据蒸馏，一种使用无标签数据训练的全方位学习方法，在Kaggle等大数据竞赛中非常有用


## IQA
No-reference Image Quality Assessment 相关论文，包括人脸姿态估计

## DeepLearning-500-questions
四川大学深度学习500问，包含了深度学习数学基础、经典框架、常见问题等
[github](https://github.com/scutan90/DeepLearning-500-questions)

