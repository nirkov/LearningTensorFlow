import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime

""" ----------------------------------------------------------------------------------------------"""
#                                     Construction Phase
""" ----------------------------------------------------------------------------------------------"""

def time_stamp_for_TensorBoard():
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_log_directory = "tf_logs_dnn"
    log_directory = "{}/run-{}/".format(root_log_directory, now)
    return log_directory


# Load MNIST data
mnist = input_data.read_data_sets("/tmp/data/")

input_vec = 28 * 28
hidden_layer_1 = 300
hidden_layer_2 = 100
output_vec = 10
learning_rate = 0.01
batch_size = 50
n_epochs = 10  # 1 epochs = one forward pass and backward pass over all the training data
number_of_batch = mnist.train.num_examples // batch_size


# the first dimnation is None because in the moment we dont know what the size of each batch
X = tf.placeholder(tf.float32, shape=(None, input_vec), name='X')
Y = tf.placeholder(tf.int64, shape=(None), name='Y')


def neurons_layer_output(input, num_neurons, name, activation=None):
    with tf.name_scope(name):
        # we get the number of inputs by looking up the input matrix’s
        # shape and getting the size of the second dimension
        num_input = int(input.get_shape()[1])

        # make the W matrix which represent all the connection between the input and the neurons in
        # current layer. Hence, its shape will be (num_input, num_neurons)
        # In addition we calculate the standard deviation (stddev) which help the algorithm converge faster.
        stddev = 2 / np.sqrt(num_input)
        init_W = tf.truncated_normal((num_input, num_neurons), stddev=stddev)
        W = tf.Variable(init_W, name="whight")
        b = tf.Variable(tf.zeros([num_neurons]), name="bias")
        z = tf.matmul(input, W) + b
        if activation == "relu":
            return tf.nn.relu(z)
        elif activation == "sigmoid":
            return tf.nn.sigmoid(z)
        else:
            return z


with tf.name_scope("DeepNeuralNetwork"):
    hidden1 = neurons_layer_output(input=X, num_neurons=10, name="hidden1", activation="relu")
    hidden2 = neurons_layer_output(input=hidden1, num_neurons=20, name="hidden2", activation="sigmoid")
    logits = neurons_layer_output(input=hidden2, num_neurons=10, name="output")


# sparse_softmax_cross_entropy_with_logits() -
# function is equivalent to applying the softmax activation function and then
# computing the cross entropy, but it is more efficient, and it properly takes
# care of corner cases like logits equal to 0.
# This is why we did not apply the softmax activation function earlier.

with tf.name_scope("loss"):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=logits)
    loss = tf.reduce_mean(cross_entropy, name="loss")   # reduce_mean - only compute the mean of the tensor

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, Y, 2) # Says whether the targets are in the top `K` predictions.
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


""" ----------------------------------------------------------------------------------------------"""
#                                     Execution Phase
""" ----------------------------------------------------------------------------------------------"""

init = tf.global_variables_initializer()
saver = tf.train.Saver()
file_writer = tf.summary.FileWriter(time_stamp_for_TensorBoard(), tf.get_default_graph())

with tf.Session() as sess:
    init.run()
    for epoch in range (n_epochs):
        for iter in range (number_of_batch):
            X_batch, Y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, Y: Y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, Y: Y_batch})
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images, Y: mnist.test.labels})

        print(epoch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)
    save_path = saver.save(sess, "./save DNN model/final_model.ckpt")
    file_writer.close()



""" ----------------------------------------------------------------------------------------------"""
#                                     Using the Neural Network
""" ----------------------------------------------------------------------------------------------"""

# When we want to reusing the NN we trained, we can read the trained weights.
with tf.Session() as reuse:
    saver.restore(reuse,"./save DNN model/final_model.ckpt")
    X_batch, Y_batch = mnist.train.next_batch(batch_size)
    Z = logits.eval(feed_dict={X: X_batch})
    # Z include all the values vector which represent the probabilities that a given
    # image (for all the batch's image) belong to a number {0,1...9}
    y_pred = np.argmax(Z, axis=1)
    # Hence, we use np.argmax to take all the highest probabilities to y_pred


end = 1











