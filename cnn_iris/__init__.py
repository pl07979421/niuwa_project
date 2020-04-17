import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

model_base_dir = "/home/xuzs/Project/python_project/niuwa_project/model/cnn_iris/"
model_name = "cnn_iris"


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 把iris的label变成one-hot向量
def get_one_hot(labels, nb_classes):
    res = np.eye(nb_classes)[np.array(labels).reshape(-1)]
    return res.reshape(list(labels.shape) + [nb_classes])


def expand_column_data(data, count):
    # 默认填充为1
    expand = np.ones(np.shape(data)[0])
    for i in range(count):
        data = np.column_stack((data, np.transpose(expand)))
    return data


def load_iris_data():
    iris = datasets.load_iris()
    train_data, test_data, train_labels, test_label = train_test_split(iris.data, iris.target, test_size=0.3,
                                                                       random_state=0)
    expand_count = 780  # iris的feature数量少的可怜只有4个,需要增加多780个feature以适应28*28的单通道reshape模拟图片输入
    return expand_column_data(train_data, expand_count), expand_column_data(test_data, expand_count), get_one_hot(
        train_labels, 3), get_one_hot(test_label, 3), iris.target_names


def train_cnn_model(train_data, train_labels, test_data, test_label, model_path, epoch):
    x = tf.placeholder(tf.float32, [None, np.shape(train_data)[1]], name='x')
    y_true = tf.placeholder(tf.float32, [None, 3], name='y_true')

    input_image = tf.reshape(x, [-1, 28, 28, 1])

    with tf.variable_scope("cnn_model"):
        w1 = weight_variable([5, 5, 1, 32])
        b1 = bias_variable([32])
        h1 = tf.nn.relu(conv2d(input_image, w1) + b1)
        h_pool1 = max_pool_2x2(h1)

        w2 = weight_variable([5, 5, 32, 64])
        b2 = bias_variable([64])
        h2 = tf.nn.relu(conv2d(h_pool1, w2) + b2)
        h_pool2 = max_pool_2x2(h2)

        w_fc1 = weight_variable([7 * 7 * 64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

        keep_prob = tf.placeholder(tf.float32, name="dropout")
        h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob)

        w_fc2 = weight_variable([1024, 3])
        b_fc2 = bias_variable([3])
        predict_cls = tf.add(tf.matmul(h_fc1_dropout, w_fc2), b_fc2, name='predict')

    with tf.variable_scope("cnn_loss"):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=predict_cls))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_true, 1), tf.argmax(predict_cls, 1)), tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epoch):
            _, loss = sess.run([train_step, cross_entropy],
                               feed_dict={x: train_data, y_true: train_labels, keep_prob: 0.5})
            if i % 100 == 0:
                print("训练了%d步" % i, "loss:%f" % loss)
        test_accuracy = accuracy.eval(feed_dict={x: test_data, y_true: test_label, keep_prob: 1.})
        print('测试集准确率:{}'.format(test_accuracy))
        saver = tf.train.Saver()
        saver.save(sess, model_path)


def load_model_and_predict(test_data, label_names):
    sess = tf.Session()
    saver = tf.train.import_meta_graph(model_base_dir + model_name + '.meta')
    saver.restore(sess, tf.train.latest_checkpoint(model_base_dir))

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    keep_prob = graph.get_tensor_by_name("cnn_model/dropout:0")
    predict = tf.argmax(graph.get_tensor_by_name("cnn_model/predict:0"), 1)
    print("测试集上iris预测结果: %s" % label_names[sess.run(predict, feed_dict={x: test_data, keep_prob: 1.})], " 靓仔,吃猪脚饭了")


if __name__ == "__main__":
    train_data, test_data, train_labels, test_label, target_names = load_iris_data()
    train_cnn_model(train_data, train_labels, test_data, test_label, model_path=model_base_dir + model_name, epoch=3000)
    load_model_and_predict(test_data, label_names=target_names)
