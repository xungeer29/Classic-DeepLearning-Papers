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
* Inception-V4: [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](chrome-extension://gfbliohnnapiefjpjlpjnehglfpaknnc/pages/pdf_viewer.html?r=https://arxiv.org/pdf/1602.07261.pdf), 即 inception-resnet



### Densenet


### Res-Net
[Deep Residual Learning for Image Recognition](chrome-extension://gfbliohnnapiefjpjlpjnehglfpaknnc/pages/pdf_viewer.html?r=https://arxiv.org/pdf/1512.03385.pdf)

论文[翻译](http://noahsnail.com/2017/07/31/2017-7-31-ResNet%E8%AE%BA%E6%96%87%E7%BF%BB%E8%AF%91%E2%80%94%E2%80%94%E4%B8%AD%E6%96%87%E7%89%88/)

### [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications]()
* MobileNet: 可以牺牲少量性能来大幅降低网络的计算量
* 将标准卷积转化为深度可分离卷积(depthwise conv + pointwise conv)，depthwise conv只在单一通道上进行卷积，提取特征，pointwise conv 为1*1*N的卷积，将不同通道的特征连接起来，产生新的特征; 定义了两个超参 \alpha 和 \rho 用于进一步缩减网络，分别是宽度乘数和分辨率乘数，用于减少通道数和输入图像的大小，范围为(0,1);
* MobileNet 可以用于分类、检测、识别等各个领域，适用于移动端和嵌入式端

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




## IQA
No-reference Image Quality Assessment 相关论文，包括人脸姿态估计

## DeepLearning-500-questions
四川大学深度学习500问，包含了深度学习数学基础、经典框架、常见问题等
[github](https://github.com/scutan90/DeepLearning-500-questions)

