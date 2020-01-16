#该文件的目的是构造循环神经网络的整体结构，并进行训练和测试（评估）过程
import tensorflow as tf
import numpy as np
import time
import math
import Cifar10_data
import re

max_steps=20000
batch_size=100
num_examples_for_eval=10000
data_dir='E:/python程序/learning/tf_learn/CIFAR10_learn/cifar-10-binary'
TOWER_NAME = 'tower'
global_step=0
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 500.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.001       # Initial learning rate.

#创建一个variable_with_weight_loss()函数，该函数的作用是：
#   1.使用参数w1控制L2 loss的大小
#   2.使用函数tf.nn.l2_loss()计算权重L2 loss
#   3.使用函数tf.multiply()计算权重L2 loss与w1的乘积，并赋值给weights_loss
#   4.使用函数tf.add_to_collection()将最终的结果放在名为losses的集合里面，方便后面计算神经网络的总体loss，
def variable_with_weight_loss(shape,stddev,w1):
    var=tf.Variable(tf.truncated_normal(shape,stddev=stddev))
    if w1 is not None:
        weights_loss=tf.multiply(tf.nn.l2_loss(var),w1,name="weights_loss")
        tf.add_to_collection("losses",weights_loss)
    return var

def _activation_summary(x):
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)      #替换name中的tower_[0-9]*为''
    tf.summary.histogram(tensor_name + '/activations', x)               #将变量以直方图的形式展示在tensorboard
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))#将输入的Tensor中0元素在所有元素中所占的比例计算并返回

#使用上一个文件里面已经定义好的文件序列读取函数读取训练数据文件和测试数据从文件，其中训练数据文件进行数据增强处理，测试数据文件不进行数据增强处理
images_train,labels_train=Cifar10_data.inputs(data_dir=data_dir,batch_size=batch_size,distorted=None)
images_test,labels_test=Cifar10_data.inputs(data_dir=data_dir,batch_size=batch_size,distorted=None)

#创建x和y_两个placeholder，用于在训练或评估时提供输入的数据和对应的标签值。要注意的是，由于以后定义全连接网络的时候用到了batch_size，所以x中，第一个参数不应该是None，而应该是batch_size
x=tf.placeholder(tf.float32,[batch_size,24,24,3])
y_=tf.placeholder(tf.int64,[batch_size])

#创建第一个卷积层
kernel1=variable_with_weight_loss(shape=[5,5,3,64],stddev=5e-2,w1=0.0)
conv1=tf.nn.conv2d(x,kernel1,[1,1,1,1],padding="SAME")
bias1=tf.Variable(tf.constant(0.0,shape=[64]))
relu1=tf.nn.relu(tf.nn.bias_add(conv1,bias1))
_activation_summary(relu1)
norm1 = tf.nn.lrn(relu1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)             #局部响应归一化
pool1=tf.nn.max_pool(norm1,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME")

#创建第二个卷积层
kernel2=variable_with_weight_loss(shape=[5,5,64,64],stddev=5e-2,w1=0.0)
conv2=tf.nn.conv2d(pool1,kernel2,[1,1,1,1],padding="SAME")
bias2=tf.Variable(tf.constant(0.1,shape=[64]))
relu2=tf.nn.relu(tf.nn.bias_add(conv2,bias2))
_activation_summary(relu2)
norm2 = tf.nn.lrn(relu2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
pool2=tf.nn.max_pool(norm2,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME")

#因为要进行全连接层的操作，所以这里使用tf.reshape()函数将pool2输出变成一维向量，并使用get_shape()函数获取扁平化之后的长度
reshape=tf.reshape(pool2,[batch_size,-1])    #这里面的-1代表将pool2的三维结构拉直为一维结构
dim=reshape.get_shape()[1].value             #get_shape()[1].value表示获取reshape之后的第二个维度的值

keep_prob=tf.placeholder(tf.float32)

#建立第一个全连接层
weight1=variable_with_weight_loss(shape=[dim,768],stddev=0.04,w1=0)
fc_bias1=tf.Variable(tf.constant(0.1,shape=[768]))
local3=tf.nn.relu(tf.matmul(reshape,weight1)+fc_bias1)
drop3=tf.nn.dropout(local3,keep_prob)
_activation_summary(drop3)

#建立第二个全连接层
weight2=variable_with_weight_loss(shape=[768,384],stddev=0.04,w1=0)
fc_bias2=tf.Variable(tf.constant(0.1,shape=[384]))
local4=tf.nn.relu(tf.matmul(local3,weight2)+fc_bias2)
drop4=tf.nn.dropout(local4,keep_prob)
_activation_summary(drop4)

#建立第三个全连接层
weight3=variable_with_weight_loss(shape=[384,10],stddev=1 / 192.0,w1=0.0)
fc_bias3=tf.Variable(tf.constant(0.1,shape=[10]))
result=tf.add(tf.matmul(local4,weight3),fc_bias3)
_activation_summary(result)

#计算损失，包括全中参数的正则化损失和交叉熵损失
cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result,labels=tf.cast(y_,tf.int64))

weights_with_l2_loss=tf.add_n(tf.get_collection("losses"))
loss=tf.reduce_mean(cross_entropy)+weights_with_l2_loss
tf.summary.scalar('loss', loss)

num_batches_per_epoch = Cifar10_data.num_examples_pre_epoch_for_train / batch_size  #一轮epoch所包含的batch数
decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)         #迭代多少次后改变学习速率
tmp_step=tf.placeholder(tf.int64)
lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                tmp_step,
                                decay_steps,
                                LEARNING_RATE_DECAY_FACTOR,
                                staircase=True)
tf.summary.scalar('learning_rate', lr)
train_op=tf.train.AdamOptimizer(lr).minimize(loss)

#函数tf.nn.in_top_k()用来计算输出结果中top k的准确率，函数默认的k值是1，即top 1的准确率，也就是输出分类准确率最高时的数值
top_k_op=tf.nn.in_top_k(result,y_,1)

pred = tf.argmax(result,1)
corr = tf.equal(pred,y_)
accuracy = tf.reduce_mean(tf.cast(corr,tf.float64))
tf.summary.scalar('accuracy', accuracy)

init_op=tf.global_variables_initializer()

saver=tf.train.Saver()
save_model_dir='E:/python程序/learning/tf_learn/CIFAR10_learn/save_model'

with tf.Session() as sess:
    sess.run(init_op)
    merged_summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(r'E:\python程序\learning\tf_learn\CIFAR10_learn\cifar10_logs\004', sess.graph)
    tf.train.start_queue_runners()      #启动线程操作，这是因为之前数据增强的时候使用train.shuffle_batch()函数的时候通过参数num_threads()配置了16个线程用于组织batch的操作

    #每隔100step会计算并展示当前的loss、每秒钟能训练的样本数量、以及训练一个batch数据所花费的时间
    for step in range (max_steps):
        start_time=time.time()
        image_batch,label_batch=sess.run([images_train,labels_train])
        _,loss_value=sess.run([train_op,loss],feed_dict={x:image_batch,y_:label_batch,keep_prob:0.5,tmp_step:global_step})
        summary_str = sess.run(merged_summary_op,feed_dict={x:image_batch,y_:label_batch,keep_prob:0.5,tmp_step:global_step})
        summary_writer.add_summary(summary_str, step)
        global_step+=100
        duration=time.time() - start_time

        if step % 100 == 0:
            examples_per_sec=batch_size / duration
            sec_per_batch=float(duration)
            print("step %d,loss=%.2f(%.1f examples/sec;%.3f sec/batch)"%(step,loss_value,examples_per_sec,sec_per_batch))

    #计算最终的正确率
    num_batch=int(math.ceil(num_examples_for_eval/batch_size))  #math.ceil()函数用于求整
    true_count=0
    total_sample_count=num_batch * batch_size

    #在一个for循环里面统计所有预测正确的样例个数
    for j in range(num_batch):
        image_batch,label_batch=sess.run([images_test,labels_test])
        predictions=sess.run([top_k_op],feed_dict={x:image_batch,y_:label_batch})
        true_count += np.sum(predictions)
    saver.save(sess,save_model_dir+'/model.ckpt',global_step)
    #打印正确率信息
    print("accuracy = %.3f%%"%((true_count/total_sample_count) * 100))