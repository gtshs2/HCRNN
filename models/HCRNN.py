from utils import compute_alpha,write_log,EarlyStopping,evaluation,convert_batch_data_HCRNN,kl_normal_reg_loss,variable_parser,compute_global_alpha,compute_global_alpha_norm
from models.NSTOPIC import NSTOPIC
import time
import numpy as np
import tensorflow as tf
import math
from models.HCRNN_cell import HCRNN_cell_v1,HCRNN_cell_v2,HCRNN_cell_v3

class HCRNN:
    def __init__(self, sess, k, configs,tr_x,tr_y,val_x,val_y,te_x,te_y,num_items,init_way,logger):
        self.sess = sess
        self.configs = configs
        self.tr_x = tr_x
        self.tr_y = tr_y
        self.val_x = val_x
        self.val_y = val_y
        self.te_x = te_x
        self.te_y = te_y
        self.num_items = num_items
        self.logger =logger

        self.rnn_hidden_size = configs.rnn_hidden_size
        self.batch_size = configs.batch_size
        self.num_layers = configs.num_layers

        # Initialize the optimizer
        self.optimizer_type = configs.optimizer_type
        self.weight_decay = configs.weight_decay
        self.momentum = configs.momentum
        self.lr = configs.lr
        self.eps = configs.eps

        self.clip_grad = configs.clip_grad
        self.clip_grad_threshold = configs.clip_grad_threshold
        self.lr_decay_step = configs.lr_decay_step
        self.lr_decay = configs.lr_decay
        self.lr_decay_rate = configs.lr_decay_rate
        self.drop_prob_ho = configs.drop_prob_ho
        self.drop_prob_input = configs.drop_prob_input
        self.drop_prob_recurrent = configs.drop_prob_recurrent
        self.reg_lambda = configs.reg_lambda
        self.att_type = configs.att_type
        # etc
        self.k = k
        self.time_sort = configs.time_sort
        self.loss_type = configs.loss_type
        self.n_epochs = configs.n_epochs
        self.is_shuffle = configs.is_shuffle
        self.embedding_size = configs.embedding_size
        self.num_topics = configs.num_topics
        self.early_stop = EarlyStopping(configs.max_patience)

        # batch_iterator
        self.tr_sess_idx = np.arange(len(self.tr_y))
        self.val_sess_idx = np.arange(len(self.val_y))
        self.te_sess_idx = np.arange(len(self.te_y))

        # record best epoch
        self.max_val_recall = [0 for _ in range(len(self.k))]
        self.max_te_recall = [0 for _ in range(len(self.k))]
        self.best_epoch = 0
        self.two_phase_learning = self.configs.two_phase_learning
        tr_lengths = [len(s) for s in self.tr_x]; val_lengths = [len(s) for s in self.val_x]; te_lengths = [len(s) for s in self.te_x]
        tr_maxlen = np.max(tr_lengths); val_maxlen = np.max(val_lengths); te_maxlen = np.max(te_lengths)
        #self.maxlen = np.max([tr_maxlen,val_maxlen,te_maxlen])
        self.maxlen = None
        self.embed_init,self.weight_init,self.bias_init,self.gate_bias_init,self.kern_init = init_way

        self.save_path = configs.save_path
        self.prepare_model()
        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver(tf.trainable_variables())
        print("End of model prepare")

    def run(self):
        for epoch in range(self.n_epochs):
            start_time = time.time()
            tr_pred_loss = self.train_model()
            val_pred_loss, val_recall_list, val_mrr_list = self.pred_evaluation(mode="valid")
            te_pred_loss, te_recall_list, te_mrr_list = self.pred_evaluation(mode="test")

            self.best_epoch, best_check = write_log(self.logger, epoch, tr_pred_loss, val_pred_loss, te_pred_loss, self.k, val_recall_list, val_mrr_list,
                      te_recall_list, te_mrr_list, self.max_val_recall, self.max_te_recall, self.best_epoch,start_time)
            if best_check:
                if (self.configs.model_name == "HCRNN_v3") and (self.configs.random_seed == 10):
                    self.saver.save(self.sess, self.save_path + '/model')
            if self.early_stop.validate(val_recall_list[3]):
                self.logger.info("Training process is stopped early")
                break

    def prepare_model(self):
        self.rnn_x = tf.placeholder(tf.int32, [None, self.maxlen], name='input')
        self.rnn_y = tf.placeholder(tf.int32, [None, self.num_items], name='output')
        self.topic_x = tf.placeholder(tf.float32,[None,self.num_items], name='topic_x')
        self.mask = tf.placeholder(tf.float32, [None, None], name='mask')
        self.keep_prob_input = tf.placeholder(tf.float32, name='keep_prob_input')
        self.keep_prob_ho = tf.placeholder(tf.float32, name='keep_prob_ho')
        self.batch_var_length = tf.placeholder(tf.int32, name="variable_length")
        self.is_training = tf.placeholder_with_default(True, shape=())
        real_batch_size = tf.shape(self.rnn_x)[0]
        real_maxlen = tf.shape(self.rnn_x)[1]
        with tf.variable_scope("HCRNN"):
            Wemb = tf.get_variable('Wemb', [self.num_items, self.embedding_size], initializer=self.embed_init)
            self.W_thetatv = tf.get_variable('W_thetatv', (self.num_topics, self.embedding_size), tf.float32,
                                        initializer=self.weight_init)
            if self.att_type == "normal_att":
                W_encoder = tf.get_variable('W_encoder', [self.rnn_hidden_size, self.rnn_hidden_size], initializer=self.weight_init)
                W_decoder = tf.get_variable('W_decoder', [self.rnn_hidden_size, self.rnn_hidden_size], initializer=self.weight_init)
                Bi_vector = tf.get_variable('Bi_vector', [1, self.rnn_hidden_size], initializer=self.weight_init)
                bili = tf.get_variable('bili', [self.embedding_size, 2 * self.rnn_hidden_size], initializer=self.weight_init)
            elif "gl_att" in self.att_type:
                W_g1 = tf.get_variable('W_g1', [self.rnn_hidden_size, self.embedding_size], initializer=self.weight_init)
                W_g2 = tf.get_variable('W_g2', [self.rnn_hidden_size, self.embedding_size], initializer=self.weight_init)
                W_l1 = tf.get_variable('W_l1', [self.rnn_hidden_size, self.rnn_hidden_size], initializer=self.weight_init)
                W_l2 = tf.get_variable('W_l2', [self.rnn_hidden_size, self.rnn_hidden_size], initializer=self.weight_init)
                Bi_l_vector = tf.get_variable('Bi_l_vector', [1, self.rnn_hidden_size], initializer=self.weight_init)
                Bi_g_vector = tf.get_variable('Bi_g_vector', [1, self.rnn_hidden_size], initializer=self.weight_init)
            if self.att_type == "gl_att_concat":
                bili = tf.get_variable('bili', [self.embedding_size, 3 * self.rnn_hidden_size],
                                       initializer=self.weight_init)
            elif self.att_type == "gl_att_h_dot":
                bili = tf.get_variable('bili', [self.embedding_size, 2 * self.rnn_hidden_size],
                                       initializer=self.weight_init)
            elif self.att_type == "gl_att_theta_dot":
                bili = tf.get_variable('bili', [self.embedding_size, 2 * self.rnn_hidden_size],
                                       initializer=self.weight_init)
            elif self.att_type == "gl_att_theta_dot_concat":
                bili = tf.get_variable('bili', [self.embedding_size, 3 * self.rnn_hidden_size],
                                       initializer=self.weight_init)
            elif self.att_type == "gl_att_theta_dot_norm":
                bili = tf.get_variable('bili', [self.embedding_size, 2 * self.rnn_hidden_size],
                                       initializer=self.weight_init)
            elif self.att_type == "gl_att_theta_dot_concat_norm":
                bili = tf.get_variable('bili', [self.embedding_size, 3 * self.rnn_hidden_size],
                                       initializer=self.weight_init)

        ############## Topic Model #########################
        emb_rnn_x = tf.nn.embedding_lookup(Wemb, self.rnn_x)
        emb_topic_x = tf.matmul(self.topic_x, Wemb)
        emb_rnn_x = tf.nn.dropout(emb_rnn_x, self.keep_prob_input) # batch_size * maxlen * hidden
        emb_topic_x = tf.nn.dropout(emb_topic_x, self.keep_prob_input) # batch_size * hidden
        self.theta, mu_theta, std_theta = NSTOPIC(emb_topic_x, self.num_topics, self.embedding_size, self.weight_init, self.bias_init, self.is_training)

        if self.configs.model_name == "HCRNN_v1":
            custom_cell = HCRNN_cell_v1(self.rnn_hidden_size, self.embedding_size,self.num_topics,self.theta,self.W_thetatv,self.weight_init, self.bias_init,
                                            self.gate_bias_init)
        elif self.configs.model_name == "HCRNN_v2":
            custom_cell = HCRNN_cell_v2(self.rnn_hidden_size, self.embedding_size,self.num_topics,self.theta,self.W_thetatv,self.weight_init, self.bias_init,
                                            self.gate_bias_init)
        elif self.configs.model_name == "HCRNN_v3":
            custom_cell = HCRNN_cell_v3(self.rnn_hidden_size, self.embedding_size,self.num_topics,self.theta,self.W_thetatv,self.weight_init, self.bias_init,
                                            self.gate_bias_init)
        outputs, states = tf.nn.dynamic_rnn(cell=custom_cell, inputs=emb_rnn_x, sequence_length=self.batch_var_length,dtype=tf.float32)
        self.all_hidden = outputs[0]
        self.all_state = outputs[1]
        self.reset = outputs[2]

        self.last_hidden = states[0]  # 512 x 100
        self.last_state = states[1]  # 512 x 100

        self.all_hidden = tf.transpose(self.all_hidden, perm=[1, 0, 2])  # 19x512x100
        self.all_state = tf.transpose(self.all_state, perm=[1, 0, 2])  # 19x512x100

        if self.att_type == "normal_att":
            squares = tf.map_fn(lambda x: compute_alpha(x, self.last_hidden, W_encoder, W_decoder, Bi_vector),
                                self.all_hidden)  # 19x512
            self.local_weight = tf.nn.softmax(tf.transpose(squares) + 100000000. * (self.mask - 1),
                                        axis=1)  # batch_size * max_len
            attention_proj = tf.reduce_sum(self.all_hidden * tf.transpose(self.local_weight)[:, :, None], axis=0)

        elif self.att_type == "bi_att":
            global_squares = tf.map_fn(lambda x: compute_global_alpha_norm(x, self.last_state, W_g1, W_g2), self.all_state)
            self.global_weight = tf.nn.softmax(tf.transpose(global_squares) + 100000000. * (self.mask - 1),
                                          axis=1)  # batch_size * max_len
            global_attention_proj = tf.reduce_sum(self.all_hidden * tf.transpose(self.global_weight)[:, :, None], axis=0)
            local_squares = tf.map_fn(lambda x: compute_alpha(x, self.last_hidden, W_l1, W_l2, Bi_l_vector), self.all_hidden)
            self.local_weight = tf.nn.softmax(tf.transpose(local_squares) + 100000000. * (self.mask - 1),
                                         axis=1)  # batch_size * max_len
            local_attention_proj = tf.reduce_sum(self.all_hidden * tf.transpose(self.local_weight)[:, :, None], axis=0)
            attention_proj = tf.concat([global_attention_proj, local_attention_proj], 1)
        # num_items x 2*100
        if self.loss_type == 'EMB':
            proj = tf.concat([attention_proj, self.last_hidden], 1)
            proj = tf.nn.dropout(proj, self.keep_prob_ho)
            ytem = tf.matmul(Wemb, bili)
            pred = tf.matmul(proj, tf.transpose(ytem))
            self.pred = tf.nn.softmax(pred)
            self.pred_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=self.rnn_y))

        self.reg_cost = tf.reduce_mean(tf.reshape(kl_normal_reg_loss(mu_theta, std_theta), [-1, 1]))
        self.cost = self.pred_cost + self.reg_lambda * self.reg_cost

        optimizer = tf.train.AdamOptimizer(self.lr)
        fullvars = tf.trainable_variables()
        topic_vars = variable_parser(fullvars, 'NSTOPIC')
        rnn_vars = variable_parser(fullvars, 'HCRNN')
        topic_grads = tf.gradients(self.cost, topic_vars)
        rnn_grads = tf.gradients(self.cost, rnn_vars)
        if self.two_phase_learning:
            self.optimizer_rnn = optimizer.apply_gradients(zip(rnn_grads, rnn_vars))
            self.optimizer_topic = optimizer.apply_gradients(zip(topic_grads, topic_vars))
        else:
            self.optimizer_total = optimizer.minimize(self.cost)


    def train_model(self):
        if self.configs.is_shuffle:
            self.tr_sess_idx = np.random.permutation(self.tr_sess_idx)
        batch_loss_list = []
        num_batch = math.ceil(np.float32(len(self.tr_sess_idx)) / self.batch_size)
        for batch_itr in range(int(num_batch)):
            start_itr = self.batch_size * batch_itr
            end_itr = np.minimum(self.batch_size * (batch_itr+1),len(self.tr_sess_idx))
            temp_batch_x = self.tr_x[self.tr_sess_idx[start_itr:end_itr]]
            temp_batch_y = self.tr_y[self.tr_sess_idx[start_itr:end_itr]]
            batch_x,batch_topic_x,batch_y,mask,labels,lengths = convert_batch_data_HCRNN(temp_batch_x, temp_batch_y, self.num_items,maxlen=self.maxlen)
            temp_keep_prob_ho = 1.0 - self.drop_prob_ho
            temp_keep_prob_input = 1.0 - self.drop_prob_input
            feed_dict = {self.rnn_x: batch_x, self.rnn_y: batch_y,self.topic_x:batch_topic_x, self.mask: mask,
                         self.keep_prob_input: temp_keep_prob_input, self.keep_prob_ho: temp_keep_prob_ho,
                         self.batch_var_length: lengths}
            if self.two_phase_learning:
                _, pred_loss_ = self.sess.run([self.optimizer_topic, self.cost], feed_dict=feed_dict)
                _, pred_loss_ = self.sess.run([self.optimizer_rnn, self.cost], feed_dict=feed_dict)
            else:
                _, pred_loss_ = self.sess.run([self.optimizer_total, self.cost], feed_dict=feed_dict)
            batch_loss_list.append(pred_loss_)

        return np.mean(batch_loss_list)

    def pred_evaluation(self, mode):
        if mode == "valid":
            sess_idx = self.val_sess_idx
            df_x = self.val_x
            df_y = self.val_y
        elif mode == "test":
            sess_idx = self.te_sess_idx
            df_x = self.te_x
            df_y = self.te_y

        batch_loss_list = []
        recalls = [];   mrrs = [];  evaluation_point_count = []
        for itr in range(len(self.k)):
            recalls.append(0); mrrs.append(0);  evaluation_point_count.append(0)
        num_batch = math.ceil(np.float32(len(sess_idx)) / self.batch_size)
        #argmax_dict = dict()
        for batch_itr in range(int(num_batch)):
            start_itr = self.batch_size * batch_itr
            end_itr = np.minimum(self.batch_size * (batch_itr+1),len(sess_idx))
            temp_batch_x = df_x[sess_idx[start_itr:end_itr]]
            temp_batch_y = df_y[sess_idx[start_itr:end_itr]]
            batch_x,batch_topic_x,batch_y,mask,labels,lengths = convert_batch_data_HCRNN(temp_batch_x, temp_batch_y, self.num_items,maxlen=self.maxlen)
            feed_dict = {self.rnn_x: batch_x,self.rnn_y: batch_y,self.topic_x:batch_topic_x, self.mask: mask,
                         self.keep_prob_input: 1.0, self.keep_prob_ho: 1.0,
                         self.batch_var_length: lengths, self.is_training: False}
            preds,pred_loss_ = self.sess.run([self.pred,self.cost],feed_dict=feed_dict)
            batch_loss_list.append(pred_loss_)

            recalls,mrrs,evaluation_point_count = evaluation(labels, preds, recalls, mrrs, evaluation_point_count, self.k)
            '''
            theta = self.sess.run([self.theta], feed_dict=feed_dict)
            theta = np.squeeze(np.asarray(theta), axis=0)
            
            for itr in range(len(theta)):
                current_real_last_topic_ = theta[itr,:]
                #current_real_last_topic_ = real_last_topic_[itr, :]
                max_idx = np.argmax(current_real_last_topic_)
                max_value = np.max(current_real_last_topic_)
                print("=" * 100)
                if not max_idx in argmax_dict:
                    argmax_dict[max_idx] = 0
                argmax_dict[max_idx] += 1
                print(max_idx,max_value)
            print(argmax_dict)
            '''
        recall_list = []
        mrr_list = []
        for itr in range(len(self.k)):
            recall = np.asarray(recalls[itr], dtype=np.float32) / evaluation_point_count[itr]
            mrr = np.asarray(mrrs[itr], dtype=np.float32) / evaluation_point_count[itr]
            if self.max_val_recall[itr] < recall and mode == "valid": self.max_val_recall[itr] = recall
            if self.max_te_recall[itr] < recall and mode == "test": self.max_te_recall[itr] = recall
            recall_list.append(recall)
            mrr_list.append(mrr)

        return np.mean(batch_loss_list),recall_list, mrr_list
