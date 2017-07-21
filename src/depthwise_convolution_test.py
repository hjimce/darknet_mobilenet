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
weights=tf.constant([0.01*i for i in range(input_channel*kenel_height*kenel_widht)],shape=(input_channel,kenel_height,kenel_widht),dtype=tf.float32)
weights_T=tf.reshape(tf.transpose(weights, (1, 2, 0)),(kenel_height,kenel_widht,input_channel,1))

one_hot=tf.constant([[0,0,1]],dtype=tf.float32)

conv1=tf.nn.depthwise_conv2d(bottom,weights_T,strides=[1,1,1,1],padding='VALID',data_format='NCHW')
globals_pool=tf.nn.avg_pool(conv1,[1,1,3,3],strides=[1,1,1,1],padding='VALID',data_format='NCHW')
sotfmax=tf.nn.softmax(tf.reshape(globals_pool,[batch_size,-1]))
print (sotfmax)
print (one_hot)
loss =tf.reduce_sum(one_hot * tf.log(sotfmax))
gradient=tf.gradients(loss,bottom)

print (gradient)
#cost_forward=tf.nn.l2_loss(sotfmax-label)*2#这边乘以2的目的，是因为l2_loss函数，计算的时候除以2,前向传导的时候，需要和darknet匹配，所以乘以2
#cost_backward=-tf.nn.l2_loss(sotfmax-label)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	#print (sess.run(one_hot))
	#print(sess.run([sotfmax]))

	print (sess.run([gradient]))

'''import tensorflow as tf

inputs = tf.constant([[0.00365879,  0.02649967,  0.9698416]], dtype=tf.float32)
label = tf.constant([1])
one_hot = tf.one_hot(label, 3)
predicts = tf.nn.softmax(inputs)
loss = tf.reduce_sum(one_hot * tf.log(predicts))
gradient = tf.gradients(loss, inputs)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print(sess.run(one_hot))
	print (sess.run(predicts))
	print(sess.run(gradient))'''