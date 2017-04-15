
import tensorflow as tf
import math
import numpy as np


class NNClassifier:

    def __init__(self, n_classes=3):
        self.sess = tf.Session()
        self.n_classes = n_classes

    def get_batch(self, n, i, x, y):
        batch_size = int(math.ceil((x.shape[0]+1)/n))

        # idx = np.random.randint(x.shape[0], size=batch_size)
        # batch_x = x[idx, :]
        # batch_y = y[idx, :]

        start = i*batch_size
        stop = min((i+1)*batch_size, x.shape[0])
        batch_x = x[start:stop:1]
        batch_y = y[start:stop:1]
        return batch_x, batch_y

    def fit(self, x_train, y_train, x_test):
        # For each training, discard old data
        self.sess.close()
        self.sess = tf.Session()

        def multilayer_perceptron(x, weights, biases):
            layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
            layer_1 = tf.nn.sigmoid(layer_1)
            layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
            layer_2 = tf.nn.sigmoid(layer_2)
            layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
            layer_3 = tf.nn.sigmoid(layer_3)
            out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
            return tf.nn.sigmoid(out_layer)

        learning_rate = 0.0001
        training_epochs = 1000

        n_hidden_1 = 30  # 1st layer number of features
        n_hidden_2 = 45  # 2nd layer number of features
        n_hidden_3 = 30  # 2nd layer number of features
        n_input = x_train.shape[1]  # input number of dimensions
        n_batches = 5

        x = tf.placeholder("float", [None, n_input])
        y = tf.placeholder("float", [None, self.n_classes])

        weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
            'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
            'out': tf.Variable(tf.random_normal([n_hidden_3, self.n_classes]))
        }
        biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([n_hidden_2])),
            'b3': tf.Variable(tf.random_normal([n_hidden_3])),
            'out': tf.Variable(tf.random_normal([self.n_classes]))
        }

        pred = multilayer_perceptron(x, weights, biases)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        init = tf.global_variables_initializer()

        self.sess.run(init)

        for epoch in range(training_epochs):
            for i in range(n_batches):
                batch_x, batch_y = self.get_batch(n_batches, i, x_train, y_train)

                _, c = self.sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})

        print("Optimization Finished!")

        return self.sess.run(pred, feed_dict={x: x_test})
