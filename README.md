# 经典论文

## MODEL ARCHITECTURE
### Alexnet
[ImageNet Classification with Deep Convolutional Neural Networks.pdf]()
深度学习兴起的引子

### VGG

### GoogleNet
* Inception-V1: Going Deeper with Convolutions
* Inception-V2: Batch Normalization: Accelerating Deep Network Training by
Reducing Internal Covariate Shift
* Inception-V3: Rethinking the Inception Architecture for Computer Vision



### Densenet


### Res-Net
[Deep Residual Learning for Image Recognition](chrome-extension://gfbliohnnapiefjpjlpjnehglfpaknnc/pages/pdf_viewer.html?r=https://arxiv.org/pdf/1512.03385.pdf)

论文[翻译](http://noahsnail.com/2017/07/31/2017-7-31-ResNet%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E2%80%94%E2%80%94%E4%B8%AD%E6%96%87%E7%89%88/)

## LOSS FUNCTION
### AM-Softmax
* [Additive Margin Softmax for Face Verification](https://arxiv.org/abs/1801.05599)
* [github](https://github.com/happynear/AMSoftmax) caffe tensorflow

### A-Softmax
* [SphereFace: Deep Hypersphere Embedding for Face Recognition](https://arxiv.org/abs/1704.08063)
* [github](https://github.com/wy1iu/sphereface) caffe

### L-Softmax
* [Large-Margin Softmax Loss for Convolutional Neural Networks](http://proceedings.mlr.press/v48/liud16.pdf)
* [github](https://github.com/wy1iu/LargeMargin_Softmax_Loss) caffe

## OTHERS
### [A guide to convolution arithmetic for deep learning]()
上采样的优点

### [Escaping From Saddle Points –Online Stochastic Gradient for Tensor Decomposition]()
batch的优点

### [Fine-Grained Head Pose Estimation Without Keypoints](https://github.com/natanielruiz/deep-head-pose/)
细粒度头部姿态估计，主干网络为 ResNet, 先训练角度分类的网络，然后在后面加了回归的网络，反向传播的损失为分类的softmax损失与回归L2损失的加权和;

### [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications]()
MobileNet

### [RankIQA: Learning from Rankings for No-reference Image Quality Assessment](https://github.com/xialeiliu/RankIQA)
RankIQA: 分类网络+回归网络，将清晰图像降级生成不同等级的模糊图像，将之作为训练集先训练一个图像质量分级的网络，然后使用开源的图像质量数据集训练一个回归网络输出图像质量评分;

### [Learning a Similarity Metric Discriminatively, with Application to Face Verification]()
Siamese网络：孪生网络，有两个完全一样的分支，用于邮编识别、人脸识别等需要对比识别的领域，早RankIQA中使用该模型训练将图像质量分级的网络;
