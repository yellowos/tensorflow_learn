import tensorflow as tf

saver = tf.train.import_meta_graph("./saver_data.ckpt.meta")

with tf.Session() as sess:
    saver.restore(sess, "./saver_data.ckpt")
    print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))
