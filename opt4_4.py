#设损失函数loss=(w+1)^2 令w初值是常数5.
#反向传播就是求最优w，即求最小loss对于的w值
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#定义待优化参数w初值赋为5
w=tf.Variable(tf.constant(5,dtype=tf.float32))

#定义损失函数loss
loss=tf.square(w+1)

#定义反向传播方法(学习率)
train_step=tf.train.GradientDescentOptimizer(0.2).minimize(loss)

#生成会话，训练40轮
with tf.Session() as sess:
	init_op=tf.global_variables_initializer()
	sess.run(init_op)
	for i in range(40):
		sess.run(train_step)
		w_val=sess.run(w)
		loss_val=sess.run(loss)
		print('After %s steps : w is %f ,loss is %f.'%(i,w_val,loss_val))
