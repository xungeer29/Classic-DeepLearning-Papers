# No-Reference Image Quality Assessment
[RankIQA: Learning from Rankings for No-reference Image Quality Assessment](https://arxiv.org/abs/1707.08347): 分类+回归进行图像质量评价，与图像内容无关，Siamese+VGG，分类网络+回归网络，将清晰图像降级生成不同等级的模糊图像，将之作为训练集先训练一个图像质量分级的网络，然后使用开源的图像质量数据集训练一个回归网络输出图像质量评分;
[github](https://github.com/xialeiliu/RankIQA)

[Fine-Grained Head Pose Estimation Without Keypoints](https://arxiv.org/abs/1710.00925)
细粒度头部姿态估计，主干网络为 ResNet, 先训练角度分类的网络，然后在后面加了回归的网络，反向传播的损失为分类的softmax损失与回归L2损失的加权和; 采用端到端的方式，不依赖人脸关键点，因为人脸质量也包括人脸姿态，所以放在IQA文件夹下;
[github](https://github.com/natanielruiz/deep-head-pose/)

[NIMA: Neural Image Assessment](https://arxiv.org/abs/1709.05424): Google推出的图像质量检测，不光检测质量，还从美学方面检测图像质量

[Som: Semantic obviousness metric for image quality assessment](https://ieeexplore.ieee.org/document/7298853): RankIQA 比较的论文，对图像进行语意分割，然后对分割的物体进行质量检测，符合人眼的质量评判，而且该方法可以与任何方法结合使用;

[Learning a Similarity Metric Discriminatively, with Application to Face Verification](chrome-extension://gfbliohnnapiefjpjlpjnehglfpaknnc/pages/pdf_viewer.html?r=http://yann.lecun.com/exdb/publis/pdf/chopra-05.pdf)
Siamese网络：孪生网络，有两个完全一样的分支，用于邮编识别、人脸识别、跟踪等需要对比识别的领域，在RankIQA中使用该模型训练将图像质量分级的网络;

[A DEEP NEURAL NETWORK FOR IMAGE QUALITY ASSESSMENT](chrome-extension://gfbliohnnapiefjpjlpjnehglfpaknnc/pages/pdf_viewer.html?r=http://iphome.hhi.de/samek/pdf/BosICIP16.pdf)
DNN 

[RAN4IQA: Restorative Adversarial Netsfor No-Reference Image Quality Assessment](http://arxiv-export-lb.library.cornell.edu/abs/1708.02237)
使用GAN做无参考图像质量评估

[iQIYI-VID: A Large Dataset for Multi-modal Person Identification](https://arxiv.org/abs/1811.07548)
爱奇异多模态视频人物识别数据集制作说明

[Predicting Face Recognition Performance Using Image Quality](http://arxiv-export-lb.library.cornell.edu/abs/1510.07119)