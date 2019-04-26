import tensorflow as tf

def NSTOPIC(emb_topic_x,num_topics,embedding_size,weight_init,bias_init,is_training):
    with tf.variable_scope("NSTOPIC"):
        # prior variable
        ############ hidden prior ######################
        # https://lirnli.wordpress.com/2017/09/27/variational-recurrent-neural-network-vrnn-with-pytorch/
        # topic variable
        w_e_theta = tf.get_variable('w_e_theta', shape=[embedding_size, embedding_size],
                                    initializer=weight_init)
        b_e_theta = tf.get_variable('b_e_theta', shape=[embedding_size],
                                    initializer=bias_init)

        w_theta_mu = tf.get_variable('w_theta_mu', shape=[embedding_size, num_topics], initializer=weight_init)
        b_theta_mu = tf.get_variable('b_tehta_mu', shape=[num_topics], initializer=bias_init)
        w_theta_std = tf.get_variable('w_theta_std', shape=[embedding_size, num_topics], initializer=weight_init)
        b_theta_std = tf.get_variable('b_theta_std', shape=[num_topics], initializer=bias_init)

        ############################################ Topic Generation ############################################
        encoder_for_theta = tf.nn.elu(tf.matmul(emb_topic_x, w_e_theta) + b_e_theta)
        mu_theta = tf.matmul(encoder_for_theta, w_theta_mu) + b_theta_mu

        def train_theta():
            std_theta = 1e-6 + tf.nn.softplus(tf.matmul(encoder_for_theta, w_theta_std) + b_theta_std)
            pre_theta = mu_theta + tf.multiply(std_theta, tf.random_normal(tf.shape(mu_theta), 0, 1, dtype=tf.float32))
            return std_theta, pre_theta
        def evaluate_theta():
            std_theta = tf.zeros_like(mu_theta)
            return std_theta, mu_theta

        ############################################ Prior Generation ############################################
        std_theta, pre_theta = tf.cond(is_training, lambda: train_theta(), lambda: evaluate_theta())
        theta = tf.nn.softmax(pre_theta)
    return theta,mu_theta,std_theta
