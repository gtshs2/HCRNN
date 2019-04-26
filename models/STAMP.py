from utils import compute_alpha_STAMP,write_log,convert_batch_data_stamp,EarlyStopping,evaluation,loss_fn
import time
import numpy as np
import tensorflow as tf
import math

class STAMP:
    def __init__(self, sess, k, configs,tr_x,tr_y,val_x,val_y,te_x,te_y,num_items,init_way,logger):
        self.sess = sess
        self.configs = configs
        self.tr_x = tr_x
        self.tr_y = tr_y
        self.val_x = val_x
        self.val_y = val_y
        self.te_x = te_x
        self.te_y = te_y
        #self.num_items = 37484 #num_items
        self.num_items = num_items  # num_items
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

        tr_lengths = [len(s) for s in self.tr_x]; val_lengths = [len(s) for s in self.val_x]; te_lengths = [len(s) for s in self.te_x]
        tr_maxlen = np.max(tr_lengths); val_maxlen = np.max(val_lengths); te_maxlen = np.max(te_lengths)
        self.maxlen = np.max([tr_maxlen,val_maxlen,te_maxlen])
        self.maxlen = None
        self.embed_init,self.weight_init,self.bias_init,self.gate_bias_init,self.kern_init = init_way

    def run(self):
        self.prepare_model()
        tf.global_variables_initializer().run()
        print("End of model prepare")
        for epoch in range(self.n_epochs):
            start_time = time.time()
            tr_pred_loss = self.train_model()
            val_pred_loss, val_recall_list, val_mrr_list = self.pred_evaluation(mode="valid")
            te_pred_loss, te_recall_list, te_mrr_list = self.pred_evaluation(mode="test")

            self.best_epoch, best_check = write_log(self.logger, epoch, tr_pred_loss, val_pred_loss, te_pred_loss, self.k, val_recall_list, val_mrr_list,
                      te_recall_list, te_mrr_list, self.max_val_recall, self.max_te_recall, self.best_epoch,start_time)
            if self.early_stop.validate(val_recall_list[3]):
                self.logger.info("Training process is stopped early")
                break

    def prepare_model(self):
        self.rnn_x1 = tf.placeholder(tf.int32, [None, self.maxlen], name='input1')
        self.rnn_x2 = tf.placeholder(tf.int32, [None, 1], name='input2')
        self.rnn_y = tf.placeholder(tf.int64, [None, self.num_items], name='output')
        self.mask_x1 = tf.placeholder(tf.float32, [None, self.maxlen], name='mask_x1') # batch_size * maxlen
        self.mask_x2 = tf.placeholder(tf.float32, [None, 1], name='mask_x2')
        self.keep_prob_input = tf.placeholder(tf.float32, name='keep_prob_input')
        self.keep_prob_ho = tf.placeholder(tf.float32, name='keep_prob_ho')
        self.batch_var_length = tf.placeholder(tf.float32, name="variable_length")

        Wemb = tf.get_variable('Wemb', [self.num_items, self.embedding_size], initializer=self.embed_init)
        w0 = tf.get_variable('w0', [self.embedding_size, 1], initializer=self.weight_init)
        w1 = tf.get_variable('w1', [self.embedding_size, self.embedding_size], initializer=self.weight_init)
        w2 = tf.get_variable('w2', [self.embedding_size, self.embedding_size], initializer=self.weight_init)
        w3 = tf.get_variable('w3', [self.embedding_size, self.embedding_size], initializer=self.weight_init)
        ba = tf.get_variable('ba', [self.embedding_size], initializer=self.bias_init)

        if self.loss_type == 'EMB':
            bili = tf.get_variable('bili', [self.embedding_size, 2 * self.rnn_hidden_size], initializer=self.weight_init)
        elif self.loss_type == "Trilinear":
            ws = tf.get_variable('ws', [self.embedding_size, self.embedding_size], initializer=self.weight_init)
            bs = tf.get_variable('bs', [self.embedding_size], initializer=self.bias_init)
            wt = tf.get_variable('wt', [self.embedding_size, self.embedding_size], initializer=self.weight_init)
            bt = tf.get_variable('bt', [self.embedding_size], initializer=self.bias_init)
        elif self.loss_type == "TOP1":
            W_top1 = tf.get_variable('W_top1', [2 * self.rnn_hidden_size, self.num_items], initializer=self.weight_init)
            b_top1 = tf.get_variable('b_top1', [1, self.num_items], initializer=self.bias_init)
        elif self.loss_type == "TOP1_variant":
            bili = tf.get_variable('bili', [self.embedding_size, 2 * self.rnn_hidden_size], initializer=self.weight_init)
            W_top1 = tf.get_variable('W_top1', [2 * self.rnn_hidden_size, self.num_items], initializer=self.weight_init)
            b_top1 = tf.get_variable('b_top1', [1, self.num_items], initializer=self.bias_init)

        emb_x1 = tf.nn.embedding_lookup(Wemb, self.rnn_x1) # xi (batch_size * maxlen * num_hidden)
        emb_x2 = tf.squeeze(tf.nn.embedding_lookup(Wemb, self.rnn_x2),axis=1) # xt (batch_size * num_hidden)
        tiled_mask = tf.tile(tf.expand_dims(self.mask_x1,2),[1,1,self.rnn_hidden_size]) # xt (batch_size * maxlen * num_hidden)
        ms = tf.reduce_sum(tf.multiply(emb_x1,tiled_mask),axis=1) # batch_size * num_hidden
        tiled_var_length = tf.tile(tf.reshape(self.batch_var_length,[-1,1]),[1,self.rnn_hidden_size]) # (batch_size * num_hidden)
        ms = tf.reshape(tf.div(ms,tiled_var_length),[-1,self.rnn_hidden_size]) # batch_size * num_hidden

        outputs1 = tf.transpose(emb_x1, perm=[1, 0, 2])  # maxlen * batch_size * num_hidden
        unnormalized_alpha = tf.map_fn(lambda x: compute_alpha_STAMP(x,emb_x2, ms,w0,w1,w2,w3,ba), outputs1)  # maxlen * batch_size
        unnormalized_alpha = tf.multiply(tf.transpose(unnormalized_alpha),self.mask_x1) # batch_size * maxlen
        self.unnormalized_alpha = unnormalized_alpha
        alpha = unnormalized_alpha # batch_size * maxlen
        #alpha = tf.nn.softmax(unnormalized_alpha + 100000000. * (self.mask_x1 - 1), dim=1)  # batch_size * max_len
        self.alpha = alpha
        tiled_alpha = tf.tile(tf.expand_dims(alpha,axis=2),[1,1,self.rnn_hidden_size]) # batch_size * maxlen * hidden_size
        self.tiled_alpha = tiled_alpha
        ma = tf.reduce_sum(tf.multiply(emb_x1,tiled_alpha),axis=1) # batch * hidden
        hs = tf.nn.tanh(tf.matmul(ma,ws)+bs) # batch * hidden
        ht = tf.nn.tanh(tf.matmul(emb_x2,wt)+bt) # batch * hidden

        if self.loss_type == 'EMB':
            proj = tf.concat([hs, ht], 1)
            proj = tf.nn.dropout(proj, self.keep_prob_ho)
            ytem = tf.matmul(Wemb, bili)
            pred = tf.matmul(proj, tf.transpose(ytem))
            self.pred = tf.nn.softmax(pred)
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=self.rnn_y))
        elif self.loss_type == "Trilinear":
            pred = tf.nn.sigmoid(tf.matmul(tf.multiply(ht, hs), tf.transpose(Wemb)))  # batch * n_item
            self.pred = tf.nn.softmax(pred)
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=self.rnn_y))
        elif self.loss_type == "TOP1":
            proj = tf.concat([hs, ht], 1)
            proj = tf.nn.dropout(proj, self.keep_prob_ho)
            pred = tf.matmul(proj, W_top1) + b_top1
            self.pred = tf.nn.tanh(pred)
            self.cost = loss_fn(self.rnn_y, self.pred, self.loss_type)
        elif self.loss_type == "TOP1_variant":
            pred = tf.nn.sigmoid(tf.matmul(tf.multiply(ht, hs), tf.transpose(Wemb)))  # batch * n_item
            self.pred = tf.nn.tanh(pred)
            self.cost = loss_fn(self.rnn_y, self.pred, self.loss_type)

        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.cost)

    def train_model(self):
        if self.configs.is_shuffle:
            self.tr_sess_idx = np.random.permutation(self.tr_sess_idx)
        batch_loss_list = []
        num_batch = math.ceil(np.float32(len(self.tr_sess_idx))/self.batch_size)
        for batch_itr in range(int(num_batch)):
            start_itr = self.batch_size * batch_itr
            end_itr = np.minimum(self.batch_size * (batch_itr+1),len(self.tr_sess_idx))
            temp_batch_x = self.tr_x[self.tr_sess_idx[start_itr:end_itr]] #
            temp_batch_y = self.tr_y[self.tr_sess_idx[start_itr:end_itr]] #
            batch_x1,batch_x2,batch_y,mask_x1,mask_x2,labels,lengths = convert_batch_data_stamp(temp_batch_x, temp_batch_y, self.num_items,maxlen=self.maxlen)

            temp_keep_prob_ho = 1.0 - self.drop_prob_ho
            temp_keep_prob_input = 1.0 - self.drop_prob_input
            feed_dict = {self.rnn_x1: batch_x1,self.rnn_x2: batch_x2, self.rnn_y: batch_y, self.mask_x1: mask_x1,self.mask_x2: mask_x2,
                         self.keep_prob_input: temp_keep_prob_input, self.keep_prob_ho: temp_keep_prob_ho,
                         self.batch_var_length: lengths}
            _, pred_loss_, preds2 = self.sess.run([self.optimizer, self.cost, self.pred],feed_dict=feed_dict)
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

        for batch_itr in range(int(num_batch)):
            start_itr = self.batch_size * batch_itr
            end_itr = np.minimum(self.batch_size * (batch_itr+1),len(sess_idx))
            temp_batch_x = df_x[sess_idx[start_itr:end_itr]]
            temp_batch_y = df_y[sess_idx[start_itr:end_itr]]
            batch_x1, batch_x2, batch_y, mask_x1, mask_x2, labels, lengths \
                = convert_batch_data_stamp(temp_batch_x,temp_batch_y,self.num_items,maxlen=self.maxlen)

            feed_dict = {self.rnn_x1: batch_x1,self.rnn_x2: batch_x2, self.rnn_y: batch_y, self.mask_x1: mask_x1,self.mask_x2: mask_x2,
                         self.keep_prob_input: 1.0, self.keep_prob_ho: 1.0,
                         self.batch_var_length: lengths}
            preds,pred_loss_ = self.sess.run([self.pred,self.cost],feed_dict=feed_dict)

            batch_loss_list.append(pred_loss_)

            recalls,mrrs,evaluation_point_count = evaluation(labels, preds, recalls, mrrs, evaluation_point_count, self.k)

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
