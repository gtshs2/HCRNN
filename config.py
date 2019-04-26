import argparse

parser = argparse.ArgumentParser(description='Configuration file')
arg_lists = []

def str2bool(v):
    return v.lower() in ('true', '1')

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# parse the nn arguments
# parser = argparse.ArgumentParser()
net_arg = add_argument_group('Training')
net_arg.add_argument('--model_file', default='IPSRS_TOP1_SGD_0.1_4000_5', type=str)

net_arg.add_argument('--rnn_hidden_size', default=100, type=int)
net_arg.add_argument('--embedding_size', default=100, type=int)
net_arg.add_argument('--num_layers', default=1, type=int)
net_arg.add_argument('--batch_size', default=512, type=int)  # 100
net_arg.add_argument('--drop_prob_input', default=0.25, type=float)
net_arg.add_argument('--drop_prob_ho', default=0.5, type=float)
net_arg.add_argument('--drop_prob_recurrent', default=0.0, type=float)
net_arg.add_argument('--num_topics', default=50, type=int)
net_arg.add_argument('--att_type', default='normal_att', choices=['normal_att', 'bi_att'])
net_arg.add_argument('--is_prior_reg', default=True, type=str2bool)
net_arg.add_argument('--is_shuffle', default=True, type=str2bool)
net_arg.add_argument('--max_patience', default=10, type=int)

# parse the optimizer arguments
net_arg.add_argument('--optimizer_type', default='Adam', type=str)  # SGD
net_arg.add_argument('--lr', default=0.005, type=float)  # 0.01
net_arg.add_argument('--lr_decay', default=False, type=str2bool)
net_arg.add_argument('--lr_decay_rate', default=0.96, type=float)
net_arg.add_argument('--lr_decay_step', default=10, type=int)
net_arg.add_argument('--weight_decay', default=0, type=float)
net_arg.add_argument('--reg_lambda', default=0.001, type=float)
net_arg.add_argument('--momentum', default=0.1, type=float)
net_arg.add_argument('--eps', default=1e-6, type=float)
net_arg.add_argument('--random_seed', default=10, type=int)
net_arg.add_argument('--zero_init', default=True, type=str2bool)
net_arg.add_argument('--two_phase_learning', default=True, type=str2bool)

# parse the loss type
net_arg.add_argument('--loss_type', default='TOP1', choices=['TOP1','CE','EMB', 'Trilinear', 'CE'])

# etc
net_arg.add_argument('--n_epochs', default=200, type=int)
net_arg.add_argument('--time_sort', default=False, type=str2bool)  # default of theano is true
net_arg.add_argument('--model_name', default='GRU4REC', choices=['NARM','STAMP','GRU4REC','HCRNN_v1','HCRNN_v2','HCRNN_v3'])

net_arg.add_argument('--gpu_fraction', default=1.0, type=float)

# data path
net_arg.add_argument('--data', default=r"citeulike", choices=['citeulike', 'lastfm', 'movielens','gowalla50','gowalla25','gowalla5','yoo8192'])
# net_arg.add_argument('--data', default=r"diginetica", choices=['yoo64_NARM_same','yoo64_NARM_oldfashion','yoo8192','yoo1024','yoo64','yoo4','diginetica'])
net_arg.add_argument('--clip_grad', default=False, type=str2bool)
net_arg.add_argument('--clip_grad_threshold', default=0.01, type=float)
net_arg.add_argument('--save', default=False, type=str2bool)
net_arg.add_argument('--data_dir', default=r'data/')

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
