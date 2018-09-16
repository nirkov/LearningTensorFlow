import tensorflow as tf
from sklearn import datasets
import numpy as np
import numpy.random as rand
from sklearn.preprocessing import StandardScaler
from datetime import datetime


def fetch_batch(iter, batch_index, batch_size):
    """
    we need to use rand.seed function because in this way we
    promise that every each value give as a different indices in
    every batch.

    :param iter: number of iteration
    :param batch_index:
    :param batch_size:
    :return: X_batch, Y_batch for the next training
    """
    rand.seed(batch_index + iter * batch_size)
    indices = rand.randint(m, size=batch_size)
    X_batch = scaled_data_with_bias[indices]
    Y_batch = houses.target.reshape(-1, 1)[indices]
    return X_batch, Y_batch


def time_stamp_for_TensorBoard():
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_log_directory = "tf_logs_for_TensorBoard"
    log_directory = "{}/run-{}/".format(root_log_directory, now)
    return log_directory


houses = datasets.fetch_california_housing()
iteration = 700
learningRate = 0.01
m, n = houses.data.shape
scaler = StandardScaler()
scaled_data = scaler.fit_transform(houses.data)
scaled_data_with_bias = np.c_[np.ones((m, 1)), scaled_data]

# ----------------------------------------------------------------------------------------------
#                                 Mini - Batch Gradient Descent
# ----------------------------------------------------------------------------------------------

#                                 --- Construction Phase ---
"""
placeholder nodes -  are special because they don’t actually perform any
computation, they just output the data you tell them to output at runtime.
We need to change the definition of X and Y and make them to be a placeholder
nodes.
"""

X_mini = tf.placeholder(tf.float32, shape=(None, n+1), name="X_mini_batch")
Y_mini = tf.placeholder(tf.float32, shape=(None, 1), name="Y_mini_batch")
bach_size = 100
# numpy ceil function ceil the argument value to the smallest integer wich bigger then the argument
num_of_batches = int(np.ceil(m / bach_size))
theta = tf.Variable(tf.random_uniform([n+1, 1], -1.0, 1.0), name="theta")

# Error function
y_pred = tf.matmul(X_mini, theta, name="predictions")
with tf.name_scope("loss_node") as scope:
    error = y_pred - Y_mini
    mse = tf.reduce_mean(tf.square(error), name="mse")

# Gradient calculation - with optimization of TensorFlow
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningRate)
training_node_optimizer = optimizer.minimize(mse)

# init node and saver node
init_node_mini = tf.global_variables_initializer()
saver = tf.train.Saver()

# some nodes for TensorBoard uses
mse_summary = tf.summary.scalar('MSE', mse)  # a node that evaluate the MSE value and write it to a TensorBoard
file_writer = tf.summary.FileWriter(time_stamp_for_TensorBoard(), tf.get_default_graph())


#                                 --- Execution Phase ---

# Run TensorFlow session
# todo: need to run "tensorboard --logdir tf_logs_for_TensorBoard/" in python shell to run tensorboard

with tf.Session() as sess_mini:
    sess_mini.run(init_node_mini)
    for iter in range (0,iteration):
        if iter % 100 == 0:
            print("Iteration : ", iter)
            Save_to_path = saver.save(sess_mini, "C:/Users/nirkov/PycharmProjects/Learning/Save the model/miniBatchGradientDescent.ckpt")
        for batch_index in range(0, num_of_batches):
            X_batch, Y_batch = fetch_batch(iter, batch_index, bach_size)
            if batch_index % 10 == 0 :
                summary_str = mse_summary.eval(feed_dict={X_mini: X_batch, Y_mini: Y_batch})
                step = iter * num_of_batches + batch_index
                file_writer.add_summary(summary_str, step)
            sess_mini.run(training_node_optimizer, feed_dict={X_mini: X_batch, Y_mini: Y_batch})

    best_teta = theta.eval()
    Save_to_path = saver.save(sess_mini,"C:/Users/nirkov/PycharmProjects/Learning/Save the model/miniBatchGradientDescent_final.ckpt")
    file_writer.close()


end = 0