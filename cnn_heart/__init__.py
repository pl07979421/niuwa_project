import tensorflow as tf
import pandas as pd

"""
CNN神经网络
"""

model_base_dir = "/home/xuzs/Project/python_project/niuwa_project/model/cnn_heart/"
model_name = "cnn_heart"

def create_heart_data():
    """
    获取心脏病数据集
    数据处理
    :return:
    """
    name_colum = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
                  "exang", "oldpeak", "slope", "ca", "thal", "num"]
    data = pd.read_csv("/home/xuzs/Project/python_project/niuwa_project/data/processed.cleveland.data",
                       names=name_colum)
    # 去除数据中的?
    data = data.iloc[:10, :]
    data = data.replace('?', 0)
    label = data["num"].astype("float32")
    data1 = data.iloc[:, :-1].astype("float32")
    print("data1===", data1, data1.shape)
    print("label===", label, label.shape)

    return data1.values, label.values


def full_connection():
    """
    全连接单层神经网咯
    :return:
    """
    # 获取真实的数据
    test_x, test_y = create_heart_data()
    test_y = test_y.reshape(1, -1)
    # 1.建立数据占位符
    x = tf.placeholder(tf.float32, [None, 13], name='x')
    y_true = tf.placeholder(tf.float32, [None, 10], name='y_true')

    # 2.建立一个全连接的神经网络 w[784, 10] b [10]
    with tf.variable_scope("tf_model"):
        # 随机初始化权重和偏执
        weight = tf.Variable(tf.random_normal([13, 10], mean=0.0, stddev=1.0), name="weight")

        bias = tf.Variable(tf.constant(0.0, shape=[10]), name="bias")
    # 预测结果
    y_predict = tf.add(tf.matmul(x, weight), bias, name="predict")

    # 3.求出所有样本的损失，然后求平均值
    with tf.variable_scope("soft_cross"):
        # 求平均交差熵损失
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict)

    # 4.梯度下降
    with tf.variable_scope("op_minimize"):
        train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    # 5.计算准确率
    with tf.variable_scope("acc"):
        # 求准求率 真实值和与测试比较
        equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))
        # 求平均值 equal_list  None个样本   [1, 0, 1, 0, 1, 1,..........]
        accracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    saver = tf.train.Saver()
    # 6 开启会话
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(60):
            # 取出真实存在的特征值和目标值
            # mnist_x, mnist_y = mnist.train.next_batch(50)
            # print("mnist_x====", mnist_x.shape)
            # print("mnist_y====", mnist_y.shape)
            # 运行train_op训练
            sess.run(train_op, feed_dict={x: test_x, y_true: test_y})
            print("训练第%d步,准确率为:%f" % (i, sess.run(accracy, feed_dict={x: test_x, y_true: test_y})))

        saver.save(sess, model_base_dir + model_name)

    return None


def load_model_and_predict():
    test_x, test_y = create_heart_data()

    sess = tf.Session()
    saver = tf.train.import_meta_graph(model_base_dir + model_name+'.meta')
    saver.restore(sess, tf.train.latest_checkpoint(model_base_dir))

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    predict = tf.argmax(graph.get_tensor_by_name("predict:0"),1)
    print(sess.run(predict, {x: test_x}))


if __name__ == "__main__":
    full_connection()
    load_model_and_predict()
