#!/usr/bin/env bash

# GRU4REC
for itr0 in citeulike lastfm movielens
do
for itr1 in 10 20 30 40 50
do
python3 main.py --data ${itr0} --model_name GRU4REC --loss_type CE --lr 0.001 --random_seed ${itr1}
done
done

# LSTM4REC
for itr0 in citeulike lastfm movielens
do
for itr1 in 10 20 30 40 50
do
python3 main.py --data ${itr0} --model_name LSTM4REC --loss_type CE --lr 0.001 --random_seed ${itr1}
done
done

# NARM
for itr0 in citeulike lastfm movielens
do
for itr1 in 10 20 30 40 50
do
python3 main.py --data ${itr0} --model_name NARM --att_type normal_att --loss_type EMB --lr 0.001 --random_seed ${itr1}
done
done

# STAMP
for itr0 in citeulike lastfm movielens
do
for itr1 in 10 20 30 40 50
do
python3 main.py --data ${itr0} --model_name STAMP --att_type normal_att --loss_type Trilinear --lr 0.001 --random_seed ${itr1}
done
done

# HCRNN_v1
for itr0 in citeulike lastfm movielens
do
for itr1 in 10 20 30 40 50
do
python3 main.py --data ${itr0} --model_name HCRNN_v1 --att_type normal_att --loss_type EMB --reg_lambda 0.001 --lr 0.001 --random_seed ${itr1}
done
done

# HCRNN_v2
for itr0 in citeulike lastfm movielens
do
for itr1 in 10 20 30 40 50
do
python3 main.py --data ${itr0} --model_name HCRNN_v2 --att_type normal_att --loss_type EMB --reg_lambda 0.001 --lr 0.001 --random_seed ${itr1}
done
done


# HCRNN_v3
for itr0 in citeulike lastfm movielens
do
for itr1 in 10 20 30 40 50
do
python3 main.py --data ${itr0} --model_name HCRNN_v3 --att_type normal_att --loss_type EMB --reg_lambda 0.001 --lr 0.001 --random_seed ${itr1}
done
done

# HCRNN_v3 + Bi
for itr0 in citeulike lastfm movielens
do
for itr1 in 10 20 30 40 50
do
python3 main.py --data ${itr0} --model_name HCRNN_v3 --att_type bi_att --loss_type EMB --reg_lambda 0.001 --lr 0.001 --random_seed ${itr1}
done
done