import tensorflow as tf
import time
import logging
from utils import preprocessing_data
import numpy as np
from models.NARM import NARM
from models.STAMP import STAMP
from models.HCRNN import HCRNN
from models.GRU4REC import GRU4REC
from config import get_config
import os

# TODO

def main():
    current_time = time.time()
    configs, _ = get_config()
    tf.set_random_seed(configs.random_seed)
    np.random.seed(configs.random_seed)

    PATH_LOG = 'logs/' + configs.model_name + "_" + configs.data + "_"  + str(configs.rnn_hidden_size) + "_" + str(
        configs.random_seed) + "_" + str(configs.two_phase_learning) + "_" + str(configs.num_topics) + '_' \
               + str(configs.reg_lambda) + '_'+ str(configs.att_type) + "_" + str(configs.is_prior_reg) \
               +  '_'+ str(configs.loss_type) + '_' + str(current_time) + '.txt'
    os.makedirs("logs/") if not os.path.exists("logs/") else 1

    logger = logging.getLogger('Result_log')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(PATH_LOG)
    logger.addHandler(file_handler)
    for param in vars(configs).keys():
        logger.info('--{0} {1}'.format(param, vars(configs)[param]))
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)

    k = [3, 5, 10, 20, 50, 100]
    print("=" * 20)
    print('current_time :  {0}'.format(current_time))
    print('dataset :  {0}'.format(configs.data))
    print('model_name :  {0}'.format(configs.model_name))
    print('rnn_hidden_size :  {0}'.format(configs.rnn_hidden_size))
    print('embedding_size :  {0}'.format(configs.embedding_size))

    print('n_epochs :  {0}'.format(configs.n_epochs))
    print('lr :  {0}'.format(configs.lr))
    print('lr_decay :  {0}'.format(configs.lr_decay))
    print('clip_grad :  {0}'.format(configs.clip_grad))
    print('clip_grad_threshold :  {0}'.format(configs.clip_grad_threshold))
    print('drop_prob_input :  {0}'.format(configs.drop_prob_input))
    print('drop_prob_recurrent :  {0}'.format(configs.drop_prob_recurrent))
    print('drop_prob_ho :  {0}'.format(configs.drop_prob_ho))
    print('is_shuffle :  {0}'.format(configs.is_shuffle))
    print('loss_type :  {0}'.format(configs.loss_type))
    print('max_patience : {0}'.format(configs.max_patience))
    print('num_topics :  {0}'.format(configs.num_topics))
    print('two_phase_learning :  {0}'.format(configs.two_phase_learning))
    print('is_prior_reg :  {0}'.format(configs.is_prior_reg))
    print('att_type :  {0}'.format(configs.att_type))
    print('reg_lambda :  {0}'.format(configs.reg_lambda))
    print('random_seed :  {0}'.format(configs.random_seed))
    print("=" * 20)

    item_key = 'item_idx'
    sess_key = 'sess_idx'

    #  initializer = tf.random_normal_initializer(mean=0, stddev=0.1)
    embed_init = tf.random_normal_initializer(mean=0, stddev=0.1) # tf.random_uniform_initializer(minval=-1.0,maxval=1.0) #tf.random_normal_initializer(mean=0, stddev=0.1)
    weight_init = None
    bias_init = tf.zeros_initializer()  # tf.ones_initializer()
    if configs.zero_init:
        gate_bias_init = tf.zeros_initializer()
    else:
        gate_bias_init = tf.ones_initializer()
    kern_init = tf.ones_initializer()
    init_way = [embed_init,weight_init,bias_init,gate_bias_init,kern_init]

    tr_x, tr_y, val_x, val_y, te_x, te_y, num_items = preprocessing_data(configs,item_key,sess_key)
    print("End of data preprocessing")
    gpu_config = tf.ConfigProto()
    if (configs.gpu_fraction == 1.0):
        gpu_config.gpu_options.allow_growth = True
    else:
        gpu_config.gpu_options.per_process_gpu_memory_fraction = configs.gpu_fraction
    with tf.Session(config=gpu_config) as sess:
        if configs.model_name == "NARM":
            model = NARM(sess, k, configs,tr_x,tr_y,val_x,val_y,te_x,te_y,num_items,init_way,logger)
        if configs.model_name == "GRU4REC":
            model = GRU4REC(sess, k, configs,tr_x,tr_y,val_x,val_y,te_x,te_y,num_items,init_way,logger)
        elif configs.model_name == "STAMP":
            model = STAMP(sess, k, configs,tr_x,tr_y,val_x,val_y,te_x,te_y,num_items,init_way,logger)
        elif "HCRNN" in configs.model_name:
            model = HCRNN(sess, k, configs,tr_x,tr_y,val_x,val_y,te_x,te_y,num_items,init_way,logger)
        model.run()

if __name__ == '__main__':
    main()
