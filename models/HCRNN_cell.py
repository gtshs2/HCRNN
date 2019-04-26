import tensorflow as tf
from models.NSTOPIC import NSTOPIC
from tensorflow.python.ops.rnn_cell import RNNCell
from utils import kl_normal_reg_loss,compute_alpha_mat,compute_alpha
import numpy as np

class CustomGRUCell(RNNCell):
    def __init__(self, rnn_hidden_size,embedding_size):
        self.rnn_hidden_size = rnn_hidden_size
        self.embedding_size = embedding_size

    @property
    def state_size(self):
        return self.rnn_hidden_size

    @property
    def output_size(self):
        return (self.rnn_hidden_size, self.rnn_hidden_size)

    def __call__(self, embedded_rnn_x, rnn_hidden, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            # update gate
            self.w = tf.get_variable('w_xi', shape=[self.embedding_size + self.rnn_hidden_size, 2 * self.rnn_hidden_size], initializer=None)
            self.b = tf.get_variable('b_z', shape=[2 * self.rnn_hidden_size], initializer=None)

            # output gate variable
            self.w_xh = tf.get_variable('w_xh', shape=[self.embedding_size, self.rnn_hidden_size], initializer=None)
            self.w_hh = tf.get_variable('w_hh', shape=[self.rnn_hidden_size, self.rnn_hidden_size], initializer=None)
            self.b_h = tf.get_variable('b_h', shape=[self.rnn_hidden_size], initializer=None)

        xh = tf.concat([embedded_rnn_x, rnn_hidden], axis=1)
        value = tf.nn.sigmoid(tf.matmul(xh, self.w) + self.b)
        r_g, z_g = tf.split(value=value, num_or_size_splits=2, axis=1)

        new_rnn_hidden_tilda = tf.nn.tanh(tf.matmul(embedded_rnn_x,self.w_xh) + tf.matmul(tf.multiply(r_g,rnn_hidden),self.w_hh) + self.b_h)
        new_rnn_hidden = tf.multiply(z_g,rnn_hidden) + tf.multiply(1-z_g , new_rnn_hidden_tilda)

        return (new_rnn_hidden, r_g), new_rnn_hidden

class HCRNN_cell_v1(RNNCell):
    def __init__(self, rnn_hidden_size,embedding_size,num_topics,theta,W_thetatv,weight_init=None, bias_init=None, gate_bias_init=None):
        self.rnn_hidden_size = rnn_hidden_size
        self.embedding_size = embedding_size
        self.theta = theta
        self.num_topics = num_topics
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.gate_bias_init = gate_bias_init
        self.W_thetatv = W_thetatv

    @property
    def state_size(self):
        return (self.rnn_hidden_size,self.embedding_size)

    @property
    def output_size(self):
        return (self.rnn_hidden_size,self.embedding_size, self.rnn_hidden_size)

    def __call__(self, rnn_x, state, scope=None):
        rnn_hidden = state[0]
        rnn_state = state[1]

        with tf.variable_scope("HCRNN"):
            ############ topic attention ###################
            W_theta_alpha = tf.get_variable('W_theta_alpha', (self.embedding_size, self.rnn_hidden_size), tf.float32,
                                    initializer=self.weight_init)
            W_h_alpha = tf.get_variable('W_h_alpha', (self.rnn_hidden_size, self.rnn_hidden_size), tf.float32,
                                    initializer=self.weight_init)
            Bi_alpha_vector = tf.get_variable('Bi_alpha_vector', [1, self.rnn_hidden_size], initializer=self.weight_init)

            ############ T1 (Topic) gate ###################
            W_xht = tf.get_variable('W_xht', (self.embedding_size + self.rnn_hidden_size, self.embedding_size), tf.float32,
                                    initializer=self.weight_init)
            W_ct1 = tf.get_variable('W_ct1', (self.embedding_size, self.embedding_size), tf.float32,
                                       initializer=self.weight_init)
            b_t1 = tf.get_variable('b_t1', self.embedding_size, tf.float32, initializer=self.bias_init)

            ############### reset gate ################
            W_xhr = tf.get_variable('W_xhr', (self.embedding_size + self.rnn_hidden_size, self.rnn_hidden_size), tf.float32,
                                    initializer=self.weight_init)
            W_cr = tf.get_variable('W_cr', (self.embedding_size, self.rnn_hidden_size), tf.float32,
                                       initializer=self.weight_init)
            b_r = tf.get_variable('b_r', self.rnn_hidden_size, tf.float32, initializer=self.gate_bias_init)

            ################ h_tilda ##################
            Whxh = tf.get_variable('Whxh', (self.embedding_size + self.rnn_hidden_size, self.rnn_hidden_size), tf.float32,
                                   initializer=self.weight_init)
            b_h = tf.get_variable('b_h', self.rnn_hidden_size, tf.float32, initializer=self.bias_init)

            ############## z gate #############
            W_xhz = tf.get_variable('W_xhz', (self.embedding_size + self.rnn_hidden_size, self.rnn_hidden_size), tf.float32,
                                    initializer=self.weight_init)
            W_cz = tf.get_variable('W_cz', (self.embedding_size, self.rnn_hidden_size), tf.float32,
                                       initializer=self.weight_init)
            b_z = tf.get_variable('b_z', self.rnn_hidden_size, tf.float32, initializer=self.gate_bias_init)

            ############################################ Attention (rnn_state_tilda) ############################################
            theta_embedding_stack = tf.einsum('ij,jk->ijk',self.theta,self.W_thetatv) # batch * topic * hidden
            theta_embedding_stack = tf.transpose(theta_embedding_stack, perm=[1, 0, 2]) # topic * batch * hidden

            squares = compute_alpha_mat(theta_embedding_stack, rnn_hidden, W_theta_alpha, W_h_alpha, Bi_alpha_vector, self.embedding_size, self.num_topics)
            # squares : num_topic x batch_size

            self.weight = tf.nn.softmax(tf.transpose(squares), axis=1)  # batch_size * num_topic
            new_rnn_state_tilda = tf.einsum('ij,jk->ik',self.weight,self.W_thetatv)

            ############################################ Recurrent Operation ############################################
            xh = tf.concat([rnn_x, rnn_hidden], axis=1)
            T1 = tf.nn.sigmoid(tf.matmul(xh, W_xht) + tf.matmul(rnn_state, W_ct1) + b_t1)
            new_rnn_state = tf.multiply((1 - T1), rnn_state) + tf.multiply(T1, new_rnn_state_tilda)

            r = tf.nn.sigmoid(tf.matmul(xh, W_xhr) + tf.matmul(rnn_state,W_cr) + b_r)
            rh_1 = tf.multiply(r, rnn_hidden)
            xrh_1 = tf.concat([rnn_x, rh_1], axis=1)
            h_tilda = tf.matmul(xrh_1, Whxh) + b_h
            z = tf.nn.sigmoid(tf.matmul(xh, W_xhz) + tf.matmul(new_rnn_state, W_cz) + b_z)
            new_rnn_hidden = tf.multiply((1 - z), rnn_hidden) + tf.multiply(z, tf.nn.tanh(h_tilda))

        return (new_rnn_hidden, new_rnn_state,r), (new_rnn_hidden,new_rnn_state)

class HCRNN_cell_v2(RNNCell):
    def __init__(self, rnn_hidden_size,embedding_size,num_topics,theta,W_thetatv,weight_init=None, bias_init=None, gate_bias_init=None):
        self.rnn_hidden_size = rnn_hidden_size
        self.embedding_size = embedding_size
        self.theta = theta
        self.num_topics = num_topics
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.gate_bias_init = gate_bias_init
        self.W_thetatv = W_thetatv

    @property
    def state_size(self):
        return (self.rnn_hidden_size,self.embedding_size)

    @property
    def output_size(self):
        return (self.rnn_hidden_size,self.embedding_size, self.rnn_hidden_size)

    def __call__(self, rnn_x, state, scope=None):
        rnn_hidden = state[0]
        rnn_state = state[1]

        with tf.variable_scope("HCRNN"):
            ############ topic attention ###################
            W_theta_alpha = tf.get_variable('W_theta_alpha', (self.embedding_size, self.rnn_hidden_size), tf.float32,
                                    initializer=self.weight_init)
            W_h_alpha = tf.get_variable('W_h_alpha', (self.rnn_hidden_size, self.rnn_hidden_size), tf.float32,
                                    initializer=self.weight_init)
            Bi_alpha_vector = tf.get_variable('Bi_alpha_vector', [1, self.rnn_hidden_size], initializer=self.weight_init)

            ############ T1 (Topic) gate ###################
            W_xht = tf.get_variable('W_xht', (self.embedding_size + self.rnn_hidden_size, self.embedding_size), tf.float32,
                                    initializer=self.weight_init)
            W_ct1 = tf.get_variable('W_ct1', (self.embedding_size, self.embedding_size), tf.float32,
                                       initializer=self.weight_init)
            b_t1 = tf.get_variable('b_t1', self.embedding_size, tf.float32, initializer=self.bias_init)

            ############### reset gate ################
            W_xhr = tf.get_variable('W_xhr', (self.embedding_size + self.rnn_hidden_size, self.rnn_hidden_size), tf.float32,
                                    initializer=self.weight_init)
            W_ct2 = tf.get_variable('W_ct2', (self.embedding_size, self.rnn_hidden_size), tf.float32,
                                    initializer=self.weight_init,
                                    constraint=lambda x: tf.clip_by_value(x, 1e-12, np.infty))
            b_r = tf.get_variable('b_r', self.rnn_hidden_size, tf.float32, initializer=self.gate_bias_init)

            ################ h_tilda ##################
            Whxh = tf.get_variable('Whxh', (self.embedding_size + self.rnn_hidden_size, self.rnn_hidden_size), tf.float32,
                                   initializer=self.weight_init)
            b_h = tf.get_variable('b_h', self.rnn_hidden_size, tf.float32, initializer=self.bias_init)

            ############## z gate #############
            W_xhz = tf.get_variable('W_xhz', (self.embedding_size + self.rnn_hidden_size, self.rnn_hidden_size), tf.float32,
                                    initializer=self.weight_init)
            W_cz = tf.get_variable('W_cz', (self.embedding_size, self.rnn_hidden_size), tf.float32,
                                       initializer=self.weight_init)
            b_z = tf.get_variable('b_z', self.rnn_hidden_size, tf.float32, initializer=self.gate_bias_init)

            ############################################ Attention (rnn_state_tilda) ############################################
            theta_embedding_stack = tf.einsum('ij,jk->ijk',self.theta,self.W_thetatv) # batch * topic * hidden
            theta_embedding_stack = tf.transpose(theta_embedding_stack, perm=[1, 0, 2]) # topic * batch * hidden

            squares = compute_alpha_mat(theta_embedding_stack, rnn_hidden, W_theta_alpha, W_h_alpha, Bi_alpha_vector, self.embedding_size, self.num_topics)
            # squares : num_topic x batch_size

            self.weight = tf.nn.softmax(tf.transpose(squares), axis=1)  # batch_size * num_topic
            new_rnn_state_tilda = tf.einsum('ij,jk->ik',self.weight,self.W_thetatv)

            ############################################ Recurrent Operation ############################################
            xh = tf.concat([rnn_x, rnn_hidden], axis=1)
            T1 = tf.nn.sigmoid(tf.matmul(xh, W_xht) + tf.matmul(rnn_state, W_ct1) + b_t1)
            new_rnn_state = tf.multiply((1 - T1), rnn_state) + tf.multiply(T1, new_rnn_state_tilda)

            x_state_sim = tf.multiply(rnn_x,new_rnn_state)
            r = tf.nn.sigmoid(tf.matmul(xh, W_xhr) + tf.matmul(x_state_sim,W_ct2) + b_r)
            rh_1 = tf.multiply(r, rnn_hidden)
            xrh_1 = tf.concat([rnn_x, rh_1], axis=1)
            h_tilda = tf.matmul(xrh_1, Whxh) + b_h
            z = tf.nn.sigmoid(tf.matmul(xh, W_xhz) + tf.matmul(new_rnn_state, W_cz) + b_z)
            new_rnn_hidden = tf.multiply((1 - z), rnn_hidden) + tf.multiply(z, tf.nn.tanh(h_tilda))

        return (new_rnn_hidden, new_rnn_state,r), (new_rnn_hidden,new_rnn_state)

class HCRNN_cell_v3(RNNCell):
    def __init__(self, rnn_hidden_size,embedding_size,num_topics,theta,W_thetatv,weight_init=None, bias_init=None, gate_bias_init=None):
        self.rnn_hidden_size = rnn_hidden_size
        self.embedding_size = embedding_size
        self.theta = theta
        self.num_topics = num_topics
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.gate_bias_init = gate_bias_init
        self.W_thetatv = W_thetatv

    @property
    def state_size(self):
        return (self.rnn_hidden_size,self.embedding_size)

    @property
    def output_size(self):
        return (self.rnn_hidden_size,self.embedding_size, self.rnn_hidden_size)

    def __call__(self, rnn_x, state, scope=None):
        rnn_hidden = state[0]
        rnn_state = state[1]

        with tf.variable_scope("HCRNN"):
            ############ topic attention ###################
            W_theta_alpha = tf.get_variable('W_theta_alpha', (self.embedding_size, self.rnn_hidden_size), tf.float32,
                                    initializer=self.weight_init)
            W_h_alpha = tf.get_variable('W_h_alpha', (self.rnn_hidden_size, self.rnn_hidden_size), tf.float32,
                                    initializer=self.weight_init)
            Bi_alpha_vector = tf.get_variable('Bi_alpha_vector', [1, self.rnn_hidden_size], initializer=self.weight_init)

            ############ T1 (Topic) gate ###################
            W_xht = tf.get_variable('W_xht', (self.embedding_size + self.rnn_hidden_size, self.embedding_size), tf.float32,
                                    initializer=self.weight_init)
            W_ct1 = tf.get_variable('W_ct1', (self.embedding_size, self.embedding_size), tf.float32,
                                       initializer=self.weight_init)
            b_t1 = tf.get_variable('b_t1', self.embedding_size, tf.float32, initializer=self.bias_init)

            ############## T2 gate ####################
            W_ct2 = tf.get_variable('W_ct2', (self.embedding_size, self.rnn_hidden_size), tf.float32,
                                    initializer=self.weight_init,
                                    constraint=lambda x: tf.clip_by_value(x, 1e-12, np.infty))
            b_t2 = tf.get_variable('b_t2', self.rnn_hidden_size, tf.float32, initializer=self.gate_bias_init)

            ############### reset gate ################
            W_xhr = tf.get_variable('W_xhr', (self.embedding_size + self.rnn_hidden_size, self.rnn_hidden_size), tf.float32,
                                    initializer=self.weight_init)
            b_r = tf.get_variable('b_r', self.rnn_hidden_size, tf.float32, initializer=self.gate_bias_init)

            ################ h_tilda ##################
            Whxh = tf.get_variable('Whxh', (self.embedding_size + self.rnn_hidden_size, self.rnn_hidden_size), tf.float32,
                                   initializer=self.weight_init)
            b_h = tf.get_variable('b_h', self.rnn_hidden_size, tf.float32, initializer=self.bias_init)

            ############## z gate #############
            W_xhz = tf.get_variable('W_xhz', (self.embedding_size + self.rnn_hidden_size, self.rnn_hidden_size), tf.float32,
                                    initializer=self.weight_init)
            W_cz = tf.get_variable('W_cz', (self.embedding_size, self.rnn_hidden_size), tf.float32,
                                       initializer=self.weight_init)
            b_z = tf.get_variable('b_z', self.rnn_hidden_size, tf.float32, initializer=self.gate_bias_init)

            ############################################ Attention (rnn_state_tilda) ############################################
            theta_embedding_stack = tf.einsum('ij,jk->ijk',self.theta,self.W_thetatv) # batch * topic * hidden
            theta_embedding_stack = tf.transpose(theta_embedding_stack, perm=[1, 0, 2]) # topic * batch * hidden

            squares = compute_alpha_mat(theta_embedding_stack, rnn_hidden, W_theta_alpha, W_h_alpha, Bi_alpha_vector, self.embedding_size, self.num_topics)
            # squares : num_topic x batch_size

            self.topic_att = tf.nn.softmax(tf.transpose(squares), axis=1)  # batch_size * num_topic
            new_rnn_state_tilda = tf.einsum('ij,jk->ik',self.topic_att,self.W_thetatv)

            ############################################ Recurrent Operation ############################################
            xh = tf.concat([rnn_x, rnn_hidden], axis=1)
            T1 = tf.nn.sigmoid(tf.matmul(xh, W_xht) + tf.matmul(rnn_state, W_ct1) + b_t1)
            new_rnn_state = tf.multiply((1 - T1), rnn_state) + tf.multiply(T1, new_rnn_state_tilda)

            x_state_sim = tf.multiply(rnn_x,new_rnn_state)
            T2 = tf.nn.sigmoid(tf.matmul(x_state_sim, W_ct2) + b_t2)

            r = tf.nn.sigmoid(tf.matmul(xh, W_xhr) + b_r)
            rh_1 = tf.multiply(r, tf.multiply(T2,rnn_hidden))
            xrh_1 = tf.concat([rnn_x, rh_1], axis=1)
            h_tilda = tf.matmul(xrh_1, Whxh) + b_h
            z = tf.nn.sigmoid(tf.matmul(xh, W_xhz) + tf.matmul(new_rnn_state, W_cz) + b_z)
            new_rnn_hidden = tf.multiply((1 - z), rnn_hidden) + tf.multiply(z, tf.nn.tanh(h_tilda))

        return (new_rnn_hidden, new_rnn_state,r), (new_rnn_hidden,new_rnn_state)