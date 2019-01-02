import tensorflow as tf
import pickle

path = 'path'
parameter_list = pickle.load(open('cnn99.23%SCE.p'.format(path), 'rb'))

inputs = tf.placeholder(tf.float32, [28, 28])
new_inputs =  tf.reshape(inputs, [-1, 28, 28, 1])

weights1 = tf.Variable(parameter_list[0])
biases1 = tf.Variable(parameter_list[1])
relu1 = tf.nn.relu(tf.add(tf.nn.conv2d(new_inputs, weights1, strides = [1,1,1,1], padding = "SAME"), biases1))
weights2 = tf.Variable(parameter_list[2])
biases2 = tf.Variable(parameter_list[3])
relu2 = tf.nn.relu(tf.add(tf.nn.conv2d(relu1, weights2, strides = [1,1,1,1], padding = "SAME"), biases2))
weights3 = tf.Variable(parameter_list[4])
biases3 = tf.Variable(parameter_list[5])
relu3 = tf.nn.relu(tf.add(tf.nn.conv2d(relu2, weights3, strides = [1,1,1,1], padding = "SAME"), biases3))
weights4 = tf.Variable(parameter_list[6])
biases4 = tf.Variable(parameter_list[7])
relu4 = tf.reshape(tf.nn.relu(tf.add(tf.nn.conv2d(relu3, weights4, strides = [1,1,1,1], padding = "SAME"), biases4)), [-1,15680])
weights5 = tf.Variable(parameter_list[8])
biases5 = tf.Variable(parameter_list[9])
relu5 =  tf.nn.relu(tf.matmul(relu4, weights5) + biases5)
weights6 = tf.Variable(parameter_list[10])
biases6 = tf.Variable(parameter_list[11])
relu6 = tf.nn.relu(tf.matmul(relu5, weights6) + biases6)
weights7 = tf.Variable(parameter_list[12])
biases7 = tf.Variable(parameter_list[13])
output = tf.matmul(relu6, weights7) + biases7