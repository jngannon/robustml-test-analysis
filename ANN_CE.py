import tensorflow as tf
import pickle

path = 'path'
parameter_list = pickle.load(open('ANN_Softmax-CE.p'.format(path), 'rb'))

inputs = tf.placeholder(tf.float32, [28, 28])
new_inputs = tf.reshape(inputs, [-1, 784])

weights1 = tf.Variable(parameter_list[0], name='Weights1')
biases1 = tf.Variable(parameter_list[1], name='Weights1')
relu1 = tf.nn.relu(tf.matmul(new_inputs, weights1) + biases1)
weights2 = tf.Variable(parameter_list[2], name='Weights2')
biases2 = tf.Variable(parameter_list[3], name='Weights2')
relu2 = tf.nn.relu(tf.matmul(relu1, weights2) + biases2)
weights3 = tf.Variable(parameter_list[4], name='Weights3')
biases3 = tf.Variable(parameter_list[5], name='Weights3')
relu3 = tf.nn.relu(tf.matmul(relu2, weights3) + biases3)
output_weights = tf.Variable(parameter_list[6], name='output_weights')
output_biases = tf.Variable(parameter_list[7], name='output_biases')
output = tf.matmul(relu3, output_weights) + output_biases
