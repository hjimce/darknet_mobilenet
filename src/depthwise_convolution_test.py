import  tensorflow as tf
import  numpy as np

batch_size = 1
input_channel = 3
input_height = 5
input_width = 5
kenel_height = 3
kenel_widht = 3

bottom=tf.constant([1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3],shape=(batch_size,input_channel,input_height,input_width),dtype=tf.float32)

print (tf.__version__)
weightsnp=np.asarray([0.01*i for i in range(input_channel*kenel_height*kenel_widht)],dtype=float).reshape(input_channel,kenel_height,kenel_widht)
weights=tf.constant(np.transpose(weightsnp, (1, 2, 0)),shape=(kenel_height,kenel_widht,input_channel,1),dtype=tf.float32)
#bais=tf.constant([i*0.2 for i in range(output_channel)],shape=[output_channel],dtype=tf.float32)
label=tf.constant(np.asarray([1,0,1]).reshape(batch_size,-1),dtype=tf.float32)

conv1=tf.nn.depthwise_conv2d(bottom,weights,strides=[1,1,1,1],padding='VALID',data_format='NCHW')
globals_pool=tf.nn.avg_pool(conv1,[1,1,3,3],strides=[1,1,1,1],padding='VALID',data_format='NCHW')
sotfmax=tf.nn.softmax(tf.reshape(globals_pool,[batch_size,-1]))
cost=tf.nn.l2_loss(sotfmax-label)*2#这边乘以2的目的，是因为l2_loss函数，计算的时候除以2
#loss =-tf.reduce_mean(one_hot * tf.log(predicts))

#打印相关变量，梯度等，验证是否与c++结果相同
#d_output,d_inputs,d_weights,d_bais=tf.gradients(loss,[relu_output,inputs,weights,bais])

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	print (sess.run([cost]))


