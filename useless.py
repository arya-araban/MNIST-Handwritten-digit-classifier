import tensorflow.compat.v1 as tf
from matplotlib import pyplot

tf.compat.v1.disable_eager_execution()

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# summarize loaded dataset
print('Train: X=%s, y=%s' % (x_train.shape, y_train.shape))
print('Test: X=%s, y=%s' % (x_test.shape, y_test.shape))

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])  # real label

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

y = tf.nn.softmax(tf.matmul(x, W) + b)  # initial prediction

loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.AdamOptimizer(0.5).minimize(loss)
# What TensorFlow actually did in this single line was to add new operations to the computation graph.
# These operations included ones to compute gradients, compute parameter update steps, and apply update steps to prms..
# The returned operation train_step, when run, will apply the gradient descent updates to the parameters.
# Training the model can therefore be accomplished by repeatedly running train_step

for i in range(1000):
  batch = x_train.train.next_batch(100)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})