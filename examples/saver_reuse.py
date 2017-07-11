import tensorflow as tf

v1 = tf.Variable(tf.constant(2.0, shape=[1]), name='v1')
v2 = tf.Variable(tf.constant(3.0, shape=[1]), name='v2')

result = v1 + v2

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "./saver_data.ckpt")
    print(sess.run(result))

