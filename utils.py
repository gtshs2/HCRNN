import numpy as np
import tensorflow as tf
import pandas as pd
import time
import math
from numpy import linalg as LA

def preprocessing_data(configs, item_key, sess_key):
    if ("NARM" in configs.model_name) or ("STAMP" in configs.model_name) or ("HCRNN" in configs.model_name) or ("GRU4REC" in configs.model_name):
        PATH_TRAIN = configs.data_dir + configs.data + '/NARM_train.txt'
        PATH_VALID = configs.data_dir + configs.data + '/NARM_valid.txt'
        PATH_TEST = configs.data_dir + configs.data + '/NARM_test.txt'

    df_tr = pd.read_csv(PATH_TRAIN, sep='\t')
    df_val = pd.read_csv(PATH_VALID, sep='\t')
    df_te = pd.read_csv(PATH_TEST, sep='\t')

    num_items = pd.concat([df_tr, df_val, df_te])[item_key].nunique() + 1
    # num_items = df_tr[item_key].nunique() + 1 # for zero padding (zero padding idx : 0)

    print(df_tr.head(n=5))
    print("=" * 10)
    print("# of Train sessions : {0}".format(len(df_tr[sess_key].unique())))
    print("# of Valid sessions : {0}".format(len(df_val[sess_key].unique())))
    print("# of Test sessions : {0}".format(len(df_te[sess_key].unique())))
    print("# of Train items : {0}".format(len(df_tr[item_key].unique())))
    print("# of Valid items : {0}".format(len(df_val[item_key].unique())))
    print("# of Test items : {0}".format(len(df_te[item_key].unique())))
    print("# of Train events : {0}".format(len(df_tr)))
    print("# of Valid events : {0}".format(len(df_val)))
    print("# of Test events : {0}".format(len(df_te)))
    print("=" * 10)

    print("!" * 10)
    print("# of Train sessions : {0}".format(len(df_tr[sess_key].unique())+len(df_val[sess_key].unique())))
    print("# of Test sessions : {0}".format(len(df_te[sess_key].unique())))
    print("# of Events : {0}".format(len(df_tr)+len(df_val)+len(df_te)))
    print("# of Items : {0}".format(len(df_tr[item_key].unique())))
    print("# of avg length : {0}".format((len(df_tr)+len(df_val) + len(df_te))/float(len(df_tr[sess_key].unique())+len(df_val[sess_key].unique())+len(df_te[sess_key].unique()))))
    print("!" * 10)

    tr_x, tr_y = df_to_array(df_tr, sess_key, item_key)
    val_x, val_y = df_to_array(df_val, sess_key, item_key)
    te_x, te_y = df_to_array(df_te, sess_key, item_key)
    del df_tr, df_val, df_te
    return tr_x, tr_y, val_x, val_y, te_x, te_y, num_items

def df_to_array(df, sess_key, item_key):
    temp = df.groupby(sess_key)[item_key].apply(list)
    temp_dict = temp.to_dict()
    x_array = []
    y_array = []
    for key in temp_dict.keys():
        x_array.append(temp_dict[key][:-1])
        y_array.append(temp_dict[key][-1])
    return np.asarray(x_array), np.asarray(y_array)

def compute_alpha(state1, state2, W_encoder, W_decoder, Bi_vector):
    tmp = tf.nn.sigmoid(tf.matmul(W_encoder, tf.transpose(state1)) + tf.matmul(W_decoder, tf.transpose(state2)))
    alpha = tf.matmul(Bi_vector, tmp)  # 1x512
    res = tf.reduce_sum(alpha, axis=0)  # 512,
    return res  # 512,

def compute_alpha_mat(outputs, real_last_hidden, W_encoder, W_decoder, Bi_vector, hidden_size, real_max_len):
    # outputs (batch * num_topic)
    outputs = tf.reshape(outputs, [-1, hidden_size])
    temp = tf.nn.sigmoid(tf.matmul(outputs, W_encoder) + tf.tile(tf.matmul(real_last_hidden, W_decoder), [real_max_len, 1]))
    alpha = tf.matmul(temp, tf.transpose(Bi_vector))
    res = tf.reshape(alpha, [real_max_len, -1])
    return res


def compute_local_alpha(state1, state2, W_encoder, W_decoder, Bi_vector):
    tmp = tf.nn.sigmoid(tf.matmul(W_encoder, tf.transpose(state1)) + tf.matmul(W_decoder, tf.transpose(state2)))
    alpha = tf.matmul(Bi_vector, tmp)
    res = tf.reduce_sum(alpha, axis=0)
    return res  # 512,

def compute_global_alpha(state1, state2, W_encoder, W_decoder):  # state : batch * num_topic
    tmp = tf.multiply(tf.matmul(W_encoder, tf.transpose(state1)), tf.matmul(W_decoder, tf.transpose(state2)))  # rnn_hidden * batch_size
    res = tf.reduce_sum(tmp, axis=0) #/ tf.sqrt(tf.cast(tf.shape(W_encoder)[1], tf.float32))
    return res  # 512,

def compute_global_alpha_norm(state1, state2, W_encoder, W_decoder):  # state : batch * num_topic
    tmp = tf.multiply(tf.matmul(W_encoder, tf.transpose(state1)), tf.matmul(W_decoder, tf.transpose(state2)))  # rnn_hidden * batch_size
    res = tf.reduce_sum(tmp, axis=0) / tf.sqrt(tf.cast(tf.shape(W_encoder)[1], tf.float32))
    return res  # 512,

def compute_alpha_STAMP(xi, xt, ms, w0, w1, w2, w3, ba):
    pre_alpha = tf.nn.sigmoid(tf.matmul(xi, w1) + tf.matmul(xt, w2) + tf.matmul(ms, w3) + ba)  # batch_size * rnn_hidden
    unnormalized_alpha = tf.matmul(pre_alpha, w0)  # batch_size * 1
    unnormalized_alpha = tf.squeeze(unnormalized_alpha, axis=1)  # batch_size
    return unnormalized_alpha  # 512

def compute_trilinear_STAMP(x, hs, ht):
    x = tf.expand_dims(x, 1)  # 100 * 1
    left = tf.expand_dims(tf.reduce_sum(tf.multiply(hs, ht), axis=1), axis=1)
    right = tf.matmul(hs, x)
    pre_z_hat = tf.multiply(left, right)
    # pre_z_hat = tf.reduce_sum(tf.multiply(hs,tf.multiply(ht,x)),axis=1)
    z_hat = tf.squeeze(tf.nn.sigmoid(pre_z_hat), axis=1)
    return z_hat

def compute_cov_theta(state):  # https://github.com/changun/CollMetric/blob/master/CML.py
    theta, mask = state
    num_topics = tf.cast(tf.shape(theta)[1], tf.int32)
    tiled_mask = tf.tile(tf.expand_dims(mask, 1), [1, num_topics])
    masked_theta = tf.transpose(tf.multiply(theta, tiled_mask))
    real_length = tf.reduce_sum(mask)
    row_mean = tf.reduce_sum(masked_theta, axis=0) / real_length

    n_rows = tf.cast(tf.shape(masked_theta)[0], tf.float32)
    post_theta = masked_theta - row_mean
    cov_theta = tf.matmul(post_theta, post_theta, transpose_a=True) / n_rows
    cov_theta_norm = tf.norm(cov_theta, ord='fro', axis=(0, 1))

    return (cov_theta_norm, cov_theta_norm)  # 1x512

def write_log(logger, epoch, tr_pred_loss, val_pred_loss, te_pred_loss, k, val_recall_list, val_mrr_list, te_recall_list, te_mrr_list, max_val_recall,
              max_te_recall, best_epoch, start_time):
    wall_clock = (time.time() - start_time) / 60
    best_check = False
    if max_val_recall[3] == val_recall_list[3]:
        best_epoch = epoch + 1
        best_check = True
    logger_str = "=" * 50
    logger_str += "\n"
    logger_str = logger_str + 'Epoch:{0:2d}| Tr Pred_Loss:{1:0.3f}|Val Pred_Loss:{2:0.3f}|' \
                              'Te Pred_Loss:{3:0.3f}|'.format(epoch + 1, tr_pred_loss, val_pred_loss, te_pred_loss, te_pred_loss)
    logger_str = logger_str + ' Best Valid Recall@20 Epoch: {0:2d}|'.format(best_epoch)
    logger_str = logger_str + ' Epoch Time: {0: 0.3f}(min)'.format(wall_clock)
    logger_str += "\n"
    for itr in range(len(k)):
        logger_str = logger_str + ' Recall @ {0: 1d}: {1: 0.4f} | '.format(k[itr], 100 * val_recall_list[itr])
        logger_str = logger_str + ' MRR @ {0: 1d}: {1: 0.4f} | '.format(k[itr], 100 * val_mrr_list[itr])
    logger_str += "\n"
    for itr in range(len(k)):
        logger_str = logger_str + ' Recall @ {0: 1d}: {1: 0.4f} | '.format(k[itr], 100 * te_recall_list[itr])
        logger_str = logger_str + ' MRR @ {0: 1d}: {1: 0.4f} | '.format(k[itr], 100 * te_mrr_list[itr])

    logger.info(logger_str)
    return best_epoch, best_check

def loss_fn(rnn_y, logits, loss_type):
    # rnn_y : batch_size * num_item
    # logits : batch_size * num_item
    y_label = tf.argmax(rnn_y, axis=1, output_type=tf.int32)
    if loss_type == "CE":
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=rnn_y, logits=logits))
    elif "TOP1" in loss_type:
        diag_true_logits = tf.diag_part(tf.gather(logits, y_label, axis=1))  # batch_size
        expand_true_logits = tf.tile(tf.expand_dims(diag_true_logits, 1), [1, tf.shape(rnn_y)[0]])  # batch_size * batch_size
        logits_for_loss = tf.gather(logits, y_label, axis=1)  # batch_size * batch_size
        top1_mask = tf.ones([tf.shape(rnn_y)[0], tf.shape(rnn_y)[0]]) - tf.eye(tf.shape(rnn_y)[0])  # batch_size * batch_size
        pre_loss1 = tf.multiply(tf.nn.sigmoid(logits_for_loss - expand_true_logits), top1_mask)  # batch_size * batch_size
        pre_loss2 = tf.multiply(tf.nn.sigmoid(tf.multiply(logits_for_loss, logits_for_loss)), top1_mask)  # batch_size * batch_size
        loss = tf.reduce_mean(pre_loss1) + tf.reduce_mean(pre_loss2)  # scalar

    return loss

def convert_batch_data(seqs, labels, n_items, maxlen=None):
    """Create the matrices from the datasets.

    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    lenght.

    This swap the axis!
    """
    # x: a list of sentences

    lengths = [len(s) for s in seqs]
    n_samples = len(seqs)
    if maxlen is None:
        maxlen = np.max(lengths)
    x = np.zeros((n_samples, maxlen)).astype('int32')
    x_mask = np.ones((n_samples, maxlen)).astype('float32')
    for idx, s in enumerate(seqs):
        x[idx, :lengths[idx]] = s
    x_mask *= (1 - (x == 0))

    y_ = []
    for i in labels:
        temp = np.zeros([n_items])
        temp[i] = 1
        y_.append(temp)

    return x, y_, x_mask, labels, lengths

def convert_batch_data_HCRNN(seqs, labels, n_items, maxlen=None):
    """Create the matrices from the datasets.

    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    lenght.

    This swap the axis!
    """
    # x: a list of sentences

    lengths = [len(s) for s in seqs]
    n_samples = len(seqs)
    if maxlen is None:
        maxlen = np.max(lengths)
    x = np.zeros((n_samples, maxlen)).astype('int32')
    topic_x = np.zeros((n_samples, n_items)).astype('int32')
    x_mask = np.ones((n_samples, maxlen)).astype('float32')
    for idx, s in enumerate(seqs):
        x[idx, :lengths[idx]] = s
        topic_x[idx,s] += 1
    x_mask *= (1 - (x == 0))

    y_ = []
    for i in labels:
        temp = np.zeros([n_items])
        temp[i] = 1
        y_.append(temp)

    return x,topic_x, y_, x_mask, labels, lengths

def convert_batch_data_stamp(seqs, labels, n_items, maxlen=None):
    """Create the matrices from the datasets.

    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    lenght.

    This swap the axis!
    """
    # x: a list of sentences

    lengths = [len(s) for s in seqs]
    n_samples = len(seqs)
    if maxlen is None:
        maxlen = np.max(lengths)
    x1 = np.zeros((n_samples, maxlen)).astype('int32')
    x1_mask = np.ones((n_samples, maxlen)).astype('float32')
    x2 = np.zeros((n_samples, 1)).astype('int32')
    x2_mask = np.ones((n_samples, 1)).astype('float32')

    for idx, s in enumerate(seqs):
        x1[idx, :lengths[idx]] = s
        x2[idx, 0] = s[-1]

    x1_mask *= (1 - (x1 == 0))
    x2_mask *= (1 - (x2 == 0))
    y_ = []
    for i in labels:
        temp = np.zeros([n_items])
        temp[i] = 1
        y_.append(temp)

    return x1, x2, y_, x1_mask, x2_mask, labels, lengths

def kl_normal_reg_loss(mu, std):
    return 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(std) - tf.log(1e-8 + tf.square(std)) - 1, axis=1)
    # return -0.5 * tf.reduce_sum(1+logvar - tf.square(mu) - tf.exp(logvar+(1e-6)),axis=1)


def tf_kl_gaussgauss(mu_1, sigma_1, mu_2, sigma_2):
    # https://github.com/phreeza/tensorflow-vrnn/blob/master/model_vrnn.py
    with tf.variable_scope("kl_gaussgauss"):
        return tf.reduce_sum(0.5 * (
                2 * tf.log(tf.maximum(1e-9, sigma_2), name='log_sigma_2')
                - 2 * tf.log(tf.maximum(1e-9, sigma_1), name='log_sigma_1')
                + (tf.square(sigma_1) + tf.square(mu_1 - mu_2)) / tf.maximum(1e-9, (tf.square(sigma_2))) - 1
        ), 1)

def rbf_kernel(x1, x2, k_sigma, k_length):
    return k_sigma * tf.exp(-1 * scaled_square_dist(x1, x2, k_length) / 2 + 1e-6)

def per_kernel(x1, x2, k_sigma, k_length, k_p):  # https://github.com/GPflow/GPflow/blob/master/gpflow/kernels.py
    # Introduce dummy dimension so we can use broadcasting
    # f = tf.expand_dims(x1, 1)  # now N x 1 x D
    # f2 = tf.expand_dims(x2, 0)  # now 1 x M x D
    f = x1
    f2 = x2
    r = np.pi * (f - f2) / k_p
    # r = tf.reduce_sum(tf.square(tf.sin(r) / k_length), 2)
    r = tf.square(tf.sin(r) / k_length)
    return k_sigma * tf.exp(-0.5 * r)

def scaled_square_dist(x1, x2, k_length):
    # X = X / self.lengthscales
    X1 = x1 / (k_length + 1e-6)

    # X2 = X2 / self.lengthscales
    X2 = x2 / (k_length + 1e-6)
    dist = tf.square(X1 - X2)
    return dist

def evaluation(labels, preds, recalls, mrrs, evaluation_point_count, k):
    targets = labels

    ranks = (preds.T > np.diag(preds.T[targets])).sum(axis=0) + 1
    for temp_itr in range(len(k)):
        temp_cut = k[temp_itr]
        rank_ok = (ranks <= temp_cut)
        recalls[temp_itr] += rank_ok.sum()
        mrrs[temp_itr] += (1.0 / ranks[rank_ok]).sum()
        evaluation_point_count[temp_itr] += len(ranks)

    return recalls, mrrs, evaluation_point_count

class EarlyStopping:
    def __init__(self, patience=0):
        self.patience = patience
        self.step = 0
        self.max_recall_20 = 0

    def validate(self, val_recall_20):
        if self.max_recall_20 > val_recall_20:
            self.step += 1
            if self.step == self.patience:
                return True
        else:
            self.step = 0
            self.max_recall_20 = val_recall_20
        return False

def variable_parser(var_list, prefix):
    """return a subset of the all_variables by prefix."""
    ret_list = []
    for var in var_list:
        varname = var.name
        splitted_varname = varname.split('/')
        if len(splitted_varname) == 3:
            varprefix = splitted_varname[1]
        else:
            varprefix = splitted_varname[0]

        if varprefix == prefix:
            ret_list.append(var)
    return ret_list

def get_all_data(model, idx2item,item2label):
    sess_idx = model.te_sess_idx
    df_x = model.te_x
    df_y = model.te_y
    recalls = []
    mrrs = []
    evaluation_point_count = []

    for itr in range(len(model.k)):
        recalls.append(0)
        mrrs.append(0)
        evaluation_point_count.append(0)

    num_batch = int(math.ceil(np.float32(len(sess_idx)) / model.batch_size))
    maxlen = 99
    ###### outputs ######
    inputs = np.empty((0, maxlen + 1), dtype=np.int32)
    NARM_att = np.empty((0, maxlen))
    NARM_reset = np.empty((0, maxlen, model.rnn_hidden_size))

    CSREC_topic_last = np.empty((0, model.num_topics))
    CSREC_topic_all = np.empty((0, maxlen, model.num_topics))
    CSREC_attention_topic = np.empty((0, maxlen))
    CSREC_t2 = np.empty((0, maxlen, model.rnn_hidden_size))
    CSREC_reset = np.empty((0, maxlen, model.rnn_hidden_size))
    CSREC_attention_normal = np.empty((0, maxlen))
    CSREC_attention_local = np.empty((0, maxlen))
    CSREC_attention_global = np.empty((0, maxlen))
    lengths_all = []
    for batch_itr in range(int(num_batch)):
        start_itr = model.batch_size * batch_itr
        end_itr = np.minimum(model.batch_size * (batch_itr + 1), len(sess_idx))
        temp_batch_x = df_x[sess_idx[start_itr:end_itr]]
        temp_batch_y = df_y[sess_idx[start_itr:end_itr]]
        batch_x, batch_y, mask, labels, lengths = convert_batch_data(temp_batch_x, temp_batch_y, model.num_items, maxlen)
        lengths_all = lengths_all + lengths
        batch_all = np.concatenate([batch_x, np.array([temp_batch_y]).T], axis=1)
        inputs = np.concatenate([inputs, idx2item[batch_all]], axis=0)

        if model.configs.model_name == "NARM":
            feed_dict = {model.rnn_x           : batch_x, model.rnn_y: batch_y, model.mask: mask, model.keep_prob_input: 1.0, model.keep_prob_ho: 1.0,
                         model.batch_var_length: lengths}
        elif "CSREC" in model.configs.model_name:
            lengths = np.array([np.arange(len(lengths)), np.array(lengths) - 1]).T
            feed_dict = {model.rnn_x           : batch_x, model.rnn_y: batch_y, model.mask: mask, model.keep_prob_input: 1.0, model.keep_prob_ho: 1.0,
                         model.batch_var_length: lengths, model.is_training: False}
        preds, pred_loss_ = model.sess.run([model.pred, model.cost], feed_dict=feed_dict)
        recalls, mrrs, evaluation_point_count = evaluation(labels, preds, recalls, mrrs, evaluation_point_count, model.k)

        if model.configs.model_name == "NARM":
            NARM_att_temp, NARM_reset_temp = model.sess.run([model.weight, model.reset], feed_dict=feed_dict)
            NARM_att = np.concatenate([NARM_att, NARM_att_temp], axis=0)
            NARM_reset = np.concatenate([NARM_reset, NARM_reset_temp], axis=0)

        elif "CSREC" in model.configs.model_name:
            topic_last, topic_all, attention_topic, t2, reset = model.sess.run(
                [model.real_last_topic, model.theta_ta_final, model.real_alpha_topic, model.t2_final, model.reset_final], feed_dict=feed_dict)
            CSREC_topic_last = np.concatenate([CSREC_topic_last, topic_last], axis=0)
            CSREC_topic_all = np.concatenate([CSREC_topic_all, topic_all], axis=0)
            CSREC_attention_topic = np.concatenate([CSREC_attention_topic, attention_topic], axis=0)
            CSREC_t2 = np.concatenate([CSREC_t2, t2], axis=0)
            CSREC_reset = np.concatenate([CSREC_reset, reset], axis=0)
            if model.configs.att_type == 'normal_att':
                normal_attention = model.sess.run(model.weight, feed_dict=feed_dict)
                CSREC_attention_normal = np.concatenate([CSREC_attention_normal, normal_attention], axis=0)
            elif model.configs.att_type == 'gl_att':
                global_attention, local_attention = model.sess.run([model.global_weight, model.local_weight], feed_dict=feed_dict)
                CSREC_attention_local = np.concatenate([CSREC_attention_local, local_attention], axis=0)
                CSREC_attention_global = np.concatenate([CSREC_attention_global, global_attention], axis=0)

    print("End of Test dataset feedforward")
    real_sequence_idx = []
    for i, leng in enumerate(lengths_all):
        if leng == 1 and i != 0:
            real_sequence_idx.append(i-1)
    real_sequence_idx.append(i)

    print(CSREC_topic_all[real_sequence_idx[0],0,0:7])
    print(CSREC_topic_all[real_sequence_idx[1],0,0:7])
    print(CSREC_topic_all[real_sequence_idx[2],0,0:7])
    print(CSREC_topic_all[real_sequence_idx[3],0,0:7])
    print(CSREC_topic_all[real_sequence_idx[4],0,0:7])
    print(CSREC_topic_all[real_sequence_idx[5],0,0:7])




    sess_id_to_value_dict = dict()
    num_sequence = len(real_sequence_idx)

    result_log_path = 'results/table/' + str(model.configs.model_name) + "_" + str(model.configs.att_type) + "_" + str(model.configs.data) + "_value.txt"
    result_summary_log_path = 'results/table/' + str(model.configs.model_name) + "_" + str(model.configs.att_type) + "_" + str(
        model.configs.data) + "_summary.txt"
    result_summary_dict = dict() # key : 1,2,3 / value : (att_
    if model.configs.model_name == "NARM":
        att_tensor = NARM_att
        reset_tensor = NARM_reset
    elif model.configs.model_name == "CSREC_v3":
        local_att_tensor = CSREC_attention_local
        global_att_tensor = CSREC_attention_global
        reset_tensor = CSREC_reset
    with open(result_log_path, 'w') as f:
        for temp_itr in range(num_sequence):
            result_summary_dict[temp_itr] = []
            sess_id_to_value_dict[temp_itr] = []
            seq_id = real_sequence_idx[temp_itr]
            current_seq = inputs[seq_id, :]
            current_seq_len = len(current_seq)
            f.write("Ssession id" + "\t" + str(temp_itr+1))
            f.write("\n")
            # write item_id
            for current_item_id_temp_idx in range(current_seq_len):
                current_item_id = current_seq[current_item_id_temp_idx]
                if int(current_item_id) != -1:
                    f.write(str(current_item_id_temp_idx+1))
                    f.write("\t")
            f.write("\n")
            for current_item_id in current_seq:
                if int(current_item_id) != -1:
                    f.write(str(current_item_id))
                    f.write("\t")
            f.write("\n")
            for current_item_id in current_seq:
                if int(current_item_id) != -1:
                    title = item2label[current_item_id][0]
                    f.write(str(title))
                    f.write("\t")
            f.write("\n")
            for current_item_id in current_seq:
                if int(current_item_id) != -1:
                    genre = item2label[current_item_id][1]
                    f.write(str(genre))
                    f.write("\t")
            f.write("\n")

            if model.configs.model_name == "CSREC_v3":
                f.write("\n")
                f.write("\n")
                f.write("\n")

            if model.configs.model_name == "NARM":
                for current_item_id_temp_idx in range(current_seq_len):
                    current_item_id = current_seq[current_item_id_temp_idx]
                    if int(current_item_id) != -1:
                        hidden_att = att_tensor[seq_id,current_item_id_temp_idx]
                        f.write(str(hidden_att))
                        f.write("\t")
                    if current_item_id_temp_idx == 98:
                        break
                f.write("\n")
            elif model.configs.model_name == "CSREC_v3":
                for current_item_id_temp_idx in range(current_seq_len):
                    current_item_id = current_seq[current_item_id_temp_idx]
                    if int(current_item_id) != -1:
                        hidden_att = local_att_tensor[seq_id,current_item_id_temp_idx]
                        f.write(str(hidden_att))
                        f.write("\t")
                    if current_item_id_temp_idx == 98:
                        break
                f.write("\n")
                for current_item_id_temp_idx in range(current_seq_len):
                    current_item_id = current_seq[current_item_id_temp_idx]
                    if int(current_item_id) != -1:
                        hidden_att = global_att_tensor[seq_id,current_item_id_temp_idx]
                        f.write(str(hidden_att))
                        f.write("\t")
                    if current_item_id_temp_idx == 98:
                        break
                f.write("\n")

            for current_item_id_temp_idx in range(current_seq_len):
                current_item_id = current_seq[current_item_id_temp_idx]
                if int(current_item_id) != -1:
                    reset_value = reset_tensor[seq_id,current_item_id_temp_idx,:]
                    reset_value_norm = LA.norm(reset_value,2)
                    f.write(str(reset_value_norm))
                    f.write("\t")
                if current_item_id_temp_idx == 98:
                    break
            f.write("\n")
            for current_item_id_temp_idx in range(current_seq_len):
                current_item_id = current_seq[current_item_id_temp_idx]
                if int(current_item_id) != -1:
                    reset_value = reset_tensor[seq_id,current_item_id_temp_idx,:]
                    reset_value_mean = np.mean(reset_value)
                    f.write(str(reset_value_mean))
                    f.write("\t")
                if current_item_id_temp_idx == 98:
                    break
            f.write("\n")

            if model.configs.model_name == "CSREC_v3":
                for current_item_id_temp_idx in range(current_seq_len):
                    current_item_id = current_seq[current_item_id_temp_idx]
                    if int(current_item_id) != -1:
                        t2_value = CSREC_t2[seq_id,current_item_id_temp_idx,:]
                        t2_value_norm = LA.norm(t2_value,2)
                        f.write(str(t2_value_norm))
                        f.write("\t")
                    if current_item_id_temp_idx == 98:
                        break
                f.write("\n")
                for current_item_id_temp_idx in range(current_seq_len):
                    current_item_id = current_seq[current_item_id_temp_idx]
                    if int(current_item_id) != -1:
                        t2_value = CSREC_t2[seq_id,current_item_id_temp_idx,:]
                        t2_value_mean = np.mean(t2_value)
                        f.write(str(t2_value_mean))
                        f.write("\t")
                    if current_item_id_temp_idx == 98:
                        break
                f.write("\n")


            f.write("\n")
            if model.configs.model_name == "NARM":
                f.write("\n")
                f.write("\n")
                f.write("\n")
                f.write("\n")
                f.write("\n")
                f.write("\n")

    '''
    print(inputs.shape) # (34682, 100)
    print(NARM_att[real_sequence_idx].shape) # (928, 99)
    print(NARM_reset.shape) # overall_seq * maxlen * hidden
    print(CSREC_topic_last.shape)
    print(CSREC_topic_all.shape)
    print(CSREC_attention_topic.shape)
    print(CSREC_t2.shape)
    print(CSREC_reset.shape)
    print(CSREC_attention_normal.shape)
    print(CSREC_attention_local.shape)
    print(CSREC_attention_global.shape)
    '''

    recall_list = []
    mrr_list = []
    for itr in range(len(model.k)):
        recall = np.asarray(recalls[itr], dtype=np.float32) / evaluation_point_count[itr]
        mrr = np.asarray(mrrs[itr], dtype=np.float32) / evaluation_point_count[itr]
        recall_list.append(recall)
        mrr_list.append(mrr)

    return recall_list, mrr_list

def get_topic_feature(model, idx2item,item2label,genre2idx, num_similar=50):
    from scipy import spatial
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    def eucllidean_distance(a, A):
        sq_dist = (A ** 2).sum(1) + a.dot(a) - 2 * A.dot(a)
        return sq_dist

    def cosine_similarity(a, A):
        cos_dist = []
        for item in A:
            cos_dist.append(1-spatial.distance.cosine(a, item))
        return np.array(cos_dist)

    def l2_normalize(mat):
        res = (mat - mat.mean(axis=0)) / mat.std(axis=0)
        return res
    num_topics = model.num_topics
    Wemb = model.sess.run("CSREC/Wemb:0") # (num_item+1) * embedding
    W_thetat2 = model.sess.run("CSREC/W_thetat2:0") # num_topic * embedding
    topic_onehot = np.eye(num_topics)
    topic_vector = np.matmul(topic_onehot, W_thetat2)
    cosine_similarity(topic_vector[0], Wemb)

    top_items = []
    for vector in topic_vector:
        #top_items.append(idx2item[np.argsort(cosine_similarity(l2_normalize(vector), l2_normalize(Wemb[1:,:])))[::-1][:num_similar]])
        top_items.append(idx2item[np.argsort(cosine_similarity(vector, Wemb[1:, :]))[::-1][:num_similar]])

    #for items in top_items:
    #    for item in items:
    #        print(item, end=',')
    #    print()

    X = np.concatenate([l2_normalize(topic_vector), l2_normalize(Wemb[1:,:])], axis=0)
    X_embedded = TSNE(n_components=2).fit_transform(X)

    genre_list = []
    genre_idx_list = []
    item_list = []
    for temp_itr in range(len(X_embedded[50:])): # start from 0
        item = idx2item[temp_itr+1] # ml data item idx
        title = item2label[item][0]
        genre = item2label[item][1]
        genre_idx = genre2idx[genre]
        item_list.append(item)
        genre_list.append(genre)
        genre_idx_list.append(genre_idx)

    item_x = X_embedded[50:][:,0]
    item_y = X_embedded[50:][:,1]
    topic_x = X_embedded[:50][:, 0]
    topic_y = X_embedded[:50][:, 1]
    unique = np.unique(genre_list)

    fig = plt.figure()
    ax = plt.subplot(111)

    colors = [plt.cm.jet(i / float(len(unique) - 1+1)) for i in range(len(unique)+1)]
    for i, u in enumerate(unique):
        xi = [item_x[j] for j in range(len(item_x)) if genre_list[j] == u]
        yi = [item_y[j] for j in range(len(item_x)) if genre_list[j] == u]
        ax.scatter(xi, yi, c=colors[i], label=str(u),marker='o',s=5)
    #plt.scatter(X_embedded[50:][:,0], X_embedded[50:][:,1], marker='o',s=5,c=np.asarray(genre_idx_list),label=np.asarray(genre_list))
    ax.scatter(topic_x, topic_y, marker='x',s=5, c=colors[-1],label="Global Context")
    # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])

    # Put a legend below current axis
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=5)


    #plt.legend(loc="upper right",fontsize =10) # https://matplotlib.org/api/legend_api.html
    plt.savefig("results/figs/Item_Topic_Embedding_2d(without_annot).pdf", bbox_inches='tight',dpi=1000)

    topic_list = []
    for itr in range(50):
        topic_name = "t" + str(itr+1)
        topic_list.append(topic_name)
    for i, txt in enumerate(item_list):
        ax.annotate(txt, (item_x[i], item_y[i]),fontsize=2)
    for i, txt in enumerate(topic_list):
        ax.annotate(txt, (topic_x[i], topic_y[i]),fontsize=2)
    plt.savefig("results/figs/Item_Topic_Embedding_2d(with_annot).pdf", bbox_inches='tight',dpi=1000)

    plt.show()
