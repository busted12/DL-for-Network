import tensorflow as tf
import numpy as np

tf.set_random_seed(1)
num_of_neurons_hl1 = 100
num_of_neurons_hl2 = 100
split_ratio = 0.67

# placeholder
x = tf.placeholder(dtype=tf.float32, shape=[None, 5])
y = tf.placeholder(dtype=tf.float32, shape=[None, 9])

# load dataset
x_data = np.loadtxt('x_data_set')
y_data = np.loadtxt('y_data_set')

num_of_data = np.shape(x_data)[0]


# split data into train and test
x_data_train = x_data[0:int(num_of_data*split_ratio), ]
y_data_train = y_data[0:int(num_of_data*split_ratio), ]

x_data_test = x_data[int(num_of_data*split_ratio):, ]
y_data_test = y_data[int(num_of_data*split_ratio):, ]

# weights and biases
W1 = tf.Variable(tf.random_normal(shape=[5, num_of_neurons_hl1], stddev=0.1))
W2 = tf.Variable(tf.random_normal(shape=[num_of_neurons_hl1, num_of_neurons_hl2], stddev=0.1))
W3 = tf.Variable(tf.random_normal(shape=[num_of_neurons_hl1, 9], stddev=0.1))
b1 = tf.Variable(tf.random_normal(shape=[num_of_neurons_hl1]))
b2 = tf.Variable(tf.random_normal(shape=[num_of_neurons_hl2]))
b3 = tf.Variable(tf.random_normal(shape=[9]))

l1 = tf.add(tf.matmul(x, W1), b1)
l1 = tf.nn.relu(l1)

l2 = tf.add(tf.matmul(l1, W2), b2)
l2 = tf.nn.relu(l2)

output = tf.add(tf.matmul(l2, W3), b3)
output = tf.nn.relu(output)

square_delta = tf.square(output-y)
loss = tf.reduce_mean(square_delta)

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for _ in range(500):
    _, train_loss = sess.run([train, loss], feed_dict={x:x_data_train, y:y_data_train})
    print(train_loss)



# test
y_ceil =  tf.ceil(output)
y_floor = tf.floor(output)
square_delta2 = tf.square(y_ceil-y)
loss2 = tf.reduce_mean(square_delta2)

square_delta3 = tf.square(y_floor-y)
loss3 = tf.reduce_mean(square_delta3)

print('test loss = ', sess.run(loss,feed_dict={x:x_data_test, y:y_data_test}))
print('ceil_test_loss = ', sess.run(loss2,feed_dict={x:x_data_test, y:y_data_test}))
print('floor_test_loss = ', sess.run(loss3,feed_dict={x:x_data_test, y:y_data_test}))


