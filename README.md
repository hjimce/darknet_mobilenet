# darknet mobilenet

Implement  the paper:MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications  based on darknet framework

1、git clone https://github.com/hjimce/darknet_mobilenet.git

2、open Makefile ,set GPU=1 、CUDNN=1 and make compile

3、network example:cfg/mobilenet_imagenet.cfg 

4、main implement :depthwise_convolutional_kernels.cu  depthwise_convolutional_layer.c depthwise_convolutional_layer.h