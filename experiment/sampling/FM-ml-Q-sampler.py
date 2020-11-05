
import math
import os
import numpy as np
import tensorflow as tf
from scipy.stats import beta
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from time import time

import argparse
import LoadData as DATA
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
import logging

from samplers import *
import trfl
import random
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


# from importance_sampling.training import ImportanceTraining

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run FM.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--topk', nargs='?', default=10,
                        help='Topk recommendation list')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epoch', type=int, default=8000,
                        help='Number of epochs.')
    parser.add_argument('--pretrain', type=int, default=-1,
                        help='flag for pretrain. 1: initialize from pretrain; 0: randomly initialize; -1: save the model to pretrain file')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=64,
                        help='Number of hidden factors.')
    parser.add_argument('--lamda', type=float, default=0,
                        help='Regularizer for bilinear part.')
    parser.add_argument('--keep_prob', type=float, default=0.8,
                        help='Keep probility (1-dropout_ratio) for the Bi-Interaction layer. 1: no dropout')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Learning rate.')
    parser.add_argument('--loss_type', nargs='?', default='log_loss',
                        help='Specify a loss type (square_loss or log_loss).')
    parser.add_argument('--optimizer', nargs='?', default='AdagradOptimizer',
                        help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show the results per X epochs (0, 1 ... any positive integer)')
    parser.add_argument('--batch_norm', type=int, default=0,
                        help='Whether to perform batch normaization (0 or 1)')
    parser.add_argument('--neg', type=int, default=5,
                        help='number of negative samples in which to chose the largest score')

    parser.add_argument('--qlr', type=float, default=1e-3, 
                        help='The learning rate of QNetwork')
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--sampler_layers', default='[]')
    parser.add_argument('--sampler_dropout_rate', type=float, default=0.5)
    parser.add_argument('--max_q', type=float, default=1.0)
    parser.add_argument('--tau', type=float, default=0.001)
    parser.add_argument('--discount', type=float, default=0.9)
    parser.add_argument('--maxlen', type=int, default=10000)

    # 探索过程所占的episode数量
    parser.add_argument('--num_exploration_episodes', type=int, default=30)
    parser.add_argument('--initial_epsilon', type=float, default=0.9)   # 探索起始时的探索率
    parser.add_argument('--final_epsilon', type=float, default=0.01)    # 探索终止时的探索率

    parser.add_argument('--v', default='v')
    parser.add_argument('--valid_dimen', type=int, default=6,
                        help='frappe-10 ml-6 last-4')


    return parser.parse_args()


class FM(BaseEstimator, TransformerMixin):
    def __init__(self, user_field_M, item_field_M, pretrain_flag, save_file, hidden_factor, loss_type, epoch,
                 batch_size, learning_rate,
                 lamda_bilinear, keep, optimizer_type, batch_norm, verbose, args, random_seed=2020):
        # bind params to class
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_factor = hidden_factor
        self.save_file = save_file
        self.pretrain_flag = pretrain_flag
        self.user_field_M = user_field_M
        self.item_field_M = item_field_M
        self.lamda_bilinear = lamda_bilinear
        self.keep = keep
        self.epoch = epoch
        self.random_seed = random_seed
        self.optimizer_type = optimizer_type
        self.batch_norm = batch_norm
        self.verbose = verbose
        self.valid_ndcg=[]
        self.args = args

        # init all variables in a tensorflow graph
        self._init_graph()

    def _init_graph(self):
        '''
        Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
        '''
        self.graph = tf.Graph()
        with self.graph.as_default():  # , tf.device('/cpu:0'):
            # Set graph level random seed
            tf.set_random_seed(self.random_seed)

            # create sampler
            self.sampler = QNetwork(self.args, state_size=self.args.hidden_factor*self.args.valid_dimen, item_num=max(data.item_map.keys())+1, name='mainQ')
            self.target_sampler = QNetwork(self.args, state_size=self.args.hidden_factor*self.args.valid_dimen, item_num=max(data.item_map.keys())+1, name='targetQ')

            self.target_network_update_ops = trfl.update_target_variables(self.target_sampler.get_qnetwork_variables(), 
                self.sampler.get_qnetwork_variables(), tau=self.args.tau)
            self.copy_weight = trfl.update_target_variables(self.target_sampler.get_qnetwork_variables(), 
                self.sampler.get_qnetwork_variables(), tau=1.0)

            # Input data.
            self.user_features = tf.placeholder(tf.int32, shape=[None, None])
            self.positive_features = tf.placeholder(tf.int32, shape=[None, None])
            self.negative_features = tf.placeholder(tf.int32, shape=[None, None])
            self.dropout_keep = tf.placeholder(tf.float32)
            self.train_phase = tf.placeholder(tf.bool)

            # Variables.
            self.weights = self._initialize_weights()

            # Model.
            # _________ sum_square part for positive (u,i)_____________
            self.user_feature_embeddings = tf.nn.embedding_lookup(self.weights['user_feature_embeddings'],
                                                                  self.user_features)
            self.positive_feature_embeddings = tf.nn.embedding_lookup(self.weights['item_feature_embeddings'],
                                                                      self.positive_features)

            # [user emb, item emb]
            self.positive_embedding = tf.concat([self.user_feature_embeddings,self.positive_feature_embeddings],1)
            self.positive_embedding = tf.reshape(self.positive_embedding,[tf.shape(self.user_feature_embeddings)[0],self.hidden_factor*args.valid_dimen]) #ml 1536 i 13944  u12265

            self.summed_user_emb = tf.reduce_sum(self.user_feature_embeddings, 1)
            self.summed_item_positive_emb = tf.reduce_sum(self.positive_feature_embeddings, 1)
            self.summed_positive_emb = tf.add(self.summed_user_emb, self.summed_item_positive_emb)
            self.summed_positive_emb_square = tf.square(self.summed_positive_emb)

            self.squared_user_emb = tf.square(self.user_feature_embeddings)
            self.squared_item_positiv_emb = tf.square(self.positive_feature_embeddings)
            self.squared_user_emb_sum = tf.reduce_sum(self.squared_user_emb, 1)
            self.squared_item_positive_emb_sum = tf.reduce_sum(self.squared_item_positiv_emb, 1)
            self.squared_positive_emb_sum = tf.add(self.squared_user_emb_sum, self.squared_item_positive_emb_sum)

            # ________ FM part for positive (u,i)__________
            self.FM_positive = 0.5 * tf.subtract(self.summed_positive_emb_square,
                                                 self.squared_positive_emb_sum)  # None * K
            # if self.batch_norm:
            # FM = self.batch_norm_layer(FM, train_phase=self.train_phase, scope_bn='bn_fm')
            self.FM_positive = tf.nn.dropout(self.FM_positive, self.dropout_keep)  # dropout at the FM layer
            # _________positive_________
            self.Bilinear_positive = tf.reduce_sum(self.FM_positive, 1, keepdims=True)  # None * 1
            self.user_feature_bias = tf.reduce_sum(
                tf.nn.embedding_lookup(self.weights['user_feature_bias'], self.user_features),
                1)  # None * 1
            self.item_feature_bias_positive = tf.reduce_sum(
                tf.nn.embedding_lookup(self.weights['item_feature_bias'], self.positive_features),
                1)  # None * 1
            # Bias = self.weights['bias'] * tf.ones_like(self.train_labels)  # None * 1
            self.positive = tf.add_n(
                [self.Bilinear_positive, self.user_feature_bias, self.item_feature_bias_positive])  # None * 1

            # _________ sum_square part for negative (u,j)_____________
            self.negative_feature_embeddings = tf.nn.embedding_lookup(self.weights['item_feature_embeddings'],
                                                                      self.negative_features)
            self.summed_item_negative_emb = tf.reduce_sum(self.negative_feature_embeddings, 1)
            self.summed_negative_emb = tf.add(self.summed_user_emb, self.summed_item_negative_emb)
            self.summed_negative_emb_square = tf.square(self.summed_negative_emb)

            self.squared_item_negative_emb = tf.square(self.negative_feature_embeddings)
            self.squared_item_negative_emb_sum = tf.reduce_sum(self.squared_item_negative_emb, 1)
            self.squared_negative_emb_sum = tf.add(self.squared_user_emb_sum, self.squared_item_negative_emb_sum)

            # ________ FM part for negative (u,j)__________
            self.FM_negative = 0.5 * tf.subtract(self.summed_negative_emb_square,
                                                 self.squared_negative_emb_sum)  # None * K
            # if self.batch_norm:
            # FM = self.batch_norm_layer(FM, train_phase=self.train_phase, scope_bn='bn_fm')
            self.FM_negative = tf.nn.dropout(self.FM_negative, self.dropout_keep)  # dropout at the FM layer

            # _________negative_________
            self.Bilinear_negative = tf.reduce_sum(self.FM_negative, 1, keepdims=True)  # None * 1
            self.item_feature_bias_negative = tf.reduce_sum(
                tf.nn.embedding_lookup(self.weights['item_feature_bias'], self.negative_features),
                1)  # None * 1
            self.negative = tf.add_n(
                [self.Bilinear_negative, self.user_feature_bias, self.item_feature_bias_negative])  # None * 1

            # Compute the loss.
            self.loss = -tf.log(tf.sigmoid(self.positive - self.negative)+0.001)
            self.loss1 = tf.sigmoid(self.negative - self.positive )

            self.loss = tf.reduce_sum(self.loss)

            # Optimizer.
            if self.optimizer_type == 'AdamOptimizer':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                        epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'AdagradOptimizer':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'GradientDescentOptimizer':
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == 'MomentumOptimizer':
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
                    self.loss)

            # init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)
            self.sess.run(self.copy_weight)      # copy weights

            # number of params
            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape()  # shape is an array of tf.Dimension
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            if self.verbose > 0:
                logger.info("#params: %d" % total_parameters)

    def _initialize_weights(self):
        all_weights = dict()
        if self.pretrain_flag > 0:
            weight_saver = tf.train.import_meta_graph(self.save_file + '.meta')
            pretrain_graph = tf.get_default_graph()
            user_feature_embeddings = pretrain_graph.get_tensor_by_name('user_feature_embeddings:0')
            item_feature_embeddings = pretrain_graph.get_tensor_by_name('item_feature_embeddings:0')
            user_feature_bias = pretrain_graph.get_tensor_by_name('user_feature_bias:0')
            item_feature_bias = pretrain_graph.get_tensor_by_name('item_feature_bias:0')
            # bias = pretrain_graph.get_tensor_by_name('bias:0')
            with tf.Session() as sess:
                weight_saver.restore(sess, self.save_file)
                ue, ie, ub, ib = sess.run(
                    [user_feature_embeddings, item_feature_embeddings, user_feature_bias, item_feature_bias])
            all_weights['user_feature_embeddings'] = tf.Variable(ue, dtype=tf.float32)
            all_weights['item_feature_embeddings'] = tf.Variable(ie, dtype=tf.float32)
            all_weights['user_feature_bias'] = tf.Variable(ub, dtype=tf.float32)
            all_weights['item_feature_bias'] = tf.Variable(ib, dtype=tf.float32)
        else:
            all_weights['user_feature_embeddings'] = tf.Variable(
                tf.random_normal([self.user_field_M, self.hidden_factor], 0.0, 0.1),
                name='user_feature_embeddings')  # user_field_M * K
            all_weights['item_feature_embeddings'] = tf.Variable(
                tf.random_normal([self.item_field_M, self.hidden_factor], 0.0, 0.1),
                name='item_feature_embeddings')  # item_field_M * K
            all_weights['user_feature_bias'] = tf.Variable(
                tf.random_uniform([self.user_field_M, 1], 0.0, 0.1), name='user_feature_bias')  # user_field_M * 1
            all_weights['item_feature_bias'] = tf.Variable(
                tf.random_uniform([self.item_field_M, 1], 0.0, 0.1), name='item_feature_bias')  # item_field_M * 1
            # all_weights['bias'] = tf.Variable(tf.constant(0.0), name='bias')  # 1 * 1
        return all_weights

    def batch_norm_layer(self, x, train_phase, scope_bn):
        # Note: the decay parameter is tunable
        bn_train = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

    def partial_fit(self, data):  # fit a batch
        feed_dict = {self.user_features: data['X_user'], self.positive_features: data['X_positive'],
                     self.negative_features: data['X_negative'], self.dropout_keep: self.keep,
                     self.train_phase: True}
        loss, loss1, opt = self.sess.run((self.loss, self.loss1, self.optimizer), feed_dict=feed_dict)
        return loss,loss1

    def push_replay(self, batch_xs):
        # push transitions in replay buffer
        states, actions, rewards = batch_xs['states'], batch_xs['actions'], batch_xs['rewards']
        next_states = self.sess.run(self.positive_embedding, feed_dict=batch_xs['get_ui_emb']) # (batch, emb size)
        for s, a, s_, r in zip(states, actions, next_states, rewards):
            done = True if r <= 0.5 else False
            self.sampler.memory.push(s, a, s_, r, done)

    def update_sampler(self, discount):
        transitions = self.sampler.memory.sample(self.args.batch_size)
        batch = Transition(*zip(*transitions))
        state, actions, rewards, next_state, is_done = batch.state, batch.action, batch.reward, batch.next_state, batch.is_done

        # mainQN = self.sampler
        # target_QN = self.target_sampler
        pointer = np.random.randint(0, 2)
        if pointer == 0:
            mainQN = self.sampler
            target_QN = self.target_sampler
        else:
            mainQN = self.target_sampler
            target_QN = self.sampler

        target_Qs = self.sess.run(target_QN.output, feed_dict={
            target_QN.inputs: next_state,
            target_QN.is_training: False})
        target_Qs_selector = self.sess.run(mainQN.output, feed_dict={
            mainQN.inputs: next_state,
            mainQN.is_training: False})

        # Set target_Qs to 0 for states where episode ends
        # TODO how to end?
        for index in range(target_Qs.shape[0]):
            if is_done[index]:
                target_Qs[index] = np.zeros([self.sampler.item_num])

        loss, _ = self.sess.run([mainQN.loss, mainQN.opt],
           feed_dict={mainQN.inputs: state,
                      mainQN.targetQs_: target_Qs,
                      mainQN.rewards: rewards,
                      mainQN.discount: discount,
                      mainQN.actions: actions,
                      mainQN.targetQs_selector: target_Qs_selector,
                      mainQN.is_training: True})
        # self.sess.run(self.target_network_update_ops)        # update target net
        return loss


    def get_random_block_from_data(self, train_data, batch_size, epsilon):  # generate a random block of training data
        X_user, X_positive, X_negative = [], [], []
        all_items = data.binded_items

        pos_idxs, pos_item_list, user_list = [], [], []

        # get sample        
        while len(X_user) < batch_size:
            index = np.random.randint(0, len(train_data['X_user']))
            X_user.append(train_data['X_user'][index])
            X_positive.append(train_data['X_item'][index])
            # uniform sampler
            user_features = "-".join([str(item) for item in train_data['X_user'][index][0:]])
            user_id = data.binded_users[user_features]  # get userID
            pos = data.user_positive_list[user_id]  # get positive list for the userID
           
            #user1_item_id = data.binded_items["-".join([str(item) for item in train_data['X_item'][index][0:]])]  #

            # Q sampling
            pos_item = np.array(train_data['X_item'][index]).reshape(-1)
            user = np.array(train_data['X_user'][index]).reshape(-1)
            pos_item_list.append(pos_item)
            user_list.append(user)
            pos_idxs.append(list(pos.keys()))

        feed_dict = {self.positive_features:pos_item_list, self.user_features:user_list}
        inputs = self.sess.run(self.positive_embedding, feed_dict=feed_dict) # (batch, emb size)

        q_scores = self.sess.run(self.sampler.output,   # (batch, item num)
            feed_dict={
            self.sampler.inputs:inputs, 
            self.sampler.is_training:False})

        actions = []
        for idx, x in enumerate(pos_idxs):
            # explore
            if random.random() < epsilon:
                neg = np.random.randint(len(all_items))
                while (neg in pos):
                    neg = np.random.randint(len(all_items))
            else:
                q_scores[idx, x] = -(2*self.args.max_q)   # remove pos item
                neg = q_scores[idx].argmax()
            actions.append(neg)
            negative_feature = data.item_map[neg]  # get negative item feature
            X_negative.append([int(item) for item in negative_feature[0:]])
        return {'X_user': X_user, 'X_positive': X_positive, 'X_negative': X_negative, 'states':inputs, 'actions':actions, 'get_ui_emb':feed_dict}


    def train(self, Train_data):  # fit a dataset
        discount = [self.args.discount] * self.args.batch_size

        #add tensorBoard
        from datetime import datetime
        now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        # root_logdir='tf_logs'
        # tf_logdir5 = "{}/run5-FMucb5ml-{}/".format(root_logdir, now)

        # tf_logdir10 = "{}/run10-FMucb5ml-{}/".format(root_logdir, now)

        # summary_writer5 = tf.summary.FileWriter(tf_logdir5)
        # summary_writer10 = tf.summary.FileWriter(tf_logdir10)
        for epoch in range(self.epoch):
            epsilon = max(self.args.initial_epsilon * (self.args.num_exploration_episodes - epoch) / self.args.num_exploration_episodes, self.args.final_epsilon)
            total_loss, total_qloss, total_reward, step = 0, 0, 0, 0
            total_batch = int(len(Train_data['X_user']) / self.batch_size)
            t1 = time()
            for i in range(total_batch):
                # generate a batch
                batch_xs = self.get_random_block_from_data(Train_data, self.batch_size, epsilon)
                # Fit training
                loss, rewards = self.partial_fit(batch_xs)

                # RL
                batch_xs['rewards'] = rewards.reshape((-1))
                self.push_replay(batch_xs)
                # update sampler
                qloss = self.update_sampler(discount)
                total_qloss += qloss
                total_reward += rewards.sum()
                total_loss = total_loss + loss
                step += 1

            t2 = time()
            total_loss, total_qloss, total_reward = round(total_loss.item()/step, 5), round(total_qloss.item()/step, 5), round(total_reward.item()/step, 5)
            logger.info(f"EPOCH:{epoch}, loss:{total_loss}, Q-Loss:{total_qloss}, reward:{total_reward}, epsilon:{epsilon}, time[{t2-t1}]")
            if total_loss==0:
                break
            #model.evaluate() 
            #if epoch>1000:

            if (epoch + 1) % 10 == 0:
                hit_rate, mrr, ndcg=model.evaluate()
                self.valid_ndcg.append(ndcg)
                # merged_summary_op5 = tf.Summary(value=[
                #     tf.Summary.Value(tag="hit_rate", simple_value=hit_rate[0]),
                #     tf.Summary.Value(tag="mrr", simple_value=mrr[0]),
                #     tf.Summary.Value(tag="ndcg", simple_value=ndcg[0])
                # ])
                # summary_writer5.add_summary(merged_summary_op5, (epoch + 1) / 10)

                # merged_summary_op10 = tf.Summary(value=[
                #     tf.Summary.Value(tag="hit_rate", simple_value=hit_rate[1]),
                #     tf.Summary.Value(tag="mrr", simple_value=mrr[1]),
                #     tf.Summary.Value(tag="ndcg", simple_value=ndcg[1])
                # ])
                # summary_writer10.add_summary(merged_summary_op10, (epoch + 1) / 10)

                if self.eva_termination(self.valid_ndcg):
                    logger.info("Early stop at %d based on validation result." %(epoch+1))
                    break

        print("end train begin save")
        if self.pretrain_flag < 0:
            logger.info("Save model to file as pretrain.")
            self.saver.save(self.sess, self.save_file)

    def evaluate(self):
        self.graph.finalize()
        count = [0, 0, 0, 0]
        rank = [[], []]
        topK = [5, 10]
        each_scorce_len = len(data.binded_items)
        test_len = len(data.Test_data['X_user'])
        # for index in range(len(data.Test_data['X_user'])):
        index = 0
        while (index < test_len):
            user = []
            item = []

            for i in range(8):
                if index < test_len:
                    user_features = data.Test_data['X_user'][index]
                    item_features = data.Test_data['X_item'][index]
                    user.append(user_features)
                    item.append(item_features)
                    index = index + 1
            # t1=time()
            scores = model.get_scores_per_user(user)

            # t2=time()
            # print("time1 is [%f]" %(t2-t1))
            # get true item score
            for i in range(len(item)):
                item_features = item[i]
                user_features = user[i]
                true_item_id = data.binded_items["-".join([str(item) for item in item_features[0:]])]
                cur_item_scores = scores[i * each_scorce_len:(i + 1) * each_scorce_len]
                true_item_score = cur_item_scores[true_item_id]
                # delete visited scores
                user_id = data.binded_users["-".join([str(item) for item in user_features[0:]])]  # get userID
                # logger.info(user_id)
                visited = data.user_positive_list[user_id]  # get positive list for the userID
                cur_item_scores = np.delete(cur_item_scores, list(visited.values()))
                
                # whether hit
                # sorted_scores = sorted(cur_item_scores, reverse=True)
                # alter
                cur_item_scores = torch.tensor(cur_item_scores, dtype=torch.float64)
                sorted_scores, _ = cur_item_scores.topk(max(topK))
                sorted_scores = sorted_scores.numpy().tolist()

                label = []
                for i in range(len(topK)):
                    label.append(sorted_scores[topK[i] - 1])
                    if true_item_score >= label[i]:
                        count[i] = count[i] + 1
                        rank[i].append(sorted_scores.index(true_item_score) + 1)
            # t3=time()
            # print("time2 is [%f]" %(t3-t2))
        hit_rate = []
        mrr = []
        ndcg = []

        for i in range(len(topK)):
            hit_rate.append(0.0)
            mrr.append(0.0)
            ndcg.append(0.0)

            hit_rate[i] = float(count[i]) / len(data.Test_data['X_user'])
            for item in rank[i]:
                mrr[i] = mrr[i] + float(1.0) / item
                ndcg[i] = ndcg[i] + float(1.0) / np.log2(item + 1)
            mrr[i] = mrr[i] / len(data.Test_data['X_user'])
            ndcg[i] = ndcg[i] / len(data.Test_data['X_user'])
            k = (i + 1) * 5
            logger.info("---------------------------------------------")
            logger.info("top:%f" % k)
            logger.info("the Hit Rate is: %f" % hit_rate[i])
            logger.info("the MRR is: %f" % mrr[i])
            logger.info("the NDCG is: %f" % ndcg[i])
        return hit_rate, mrr, ndcg

    def eva_termination(self, valid):
        if len(valid) > 5:
            if valid[-1] <= valid[-2] and valid[-2] <= valid[-3]and valid[-3] <= valid[-4] and valid[-4] <= valid[-5]:
                return True
        return False

    def get_scores_per_user(self, user):  # evaluate the results for an user context, return scorelist
        # num_example = len(Testdata['Y'])
        # get score list for a userID, store in scorelist, indexed by itemID
        # scorelist=[]
        X_user, X_item = [], []
        # X_item = []
        # Y=[[1]]
        all_items = data.binded_items.values()
        # true_item_id=data.binded_items[item]
        # user_feature_embeddings = tf.nn.embedding_lookup(self.weights['user_feature_embeddings'],X_user)
        for user_feature in user:
            for itemID in range(len(all_items)):
                X_user.append(user_feature)
                item_feature = data.item_map[itemID]
                X_item.append(item_feature)
        feed_dict = {self.user_features: X_user, self.positive_features: X_item, self.train_phase: False,
                     self.dropout_keep: 1.0}
        scores = self.sess.run((self.positive), feed_dict=feed_dict)
        scores = scores.reshape(len(user) * len(all_items))
        return scores


if __name__ == '__main__':
    args = parse_args()

    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(f'fm-{args.dataset}-q-{args.v}.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info('------------------------------NEXT VERSION--------------------------------------')
    # Data loading
    data = DATA.LoadData(args.path, args.dataset)
    if args.verbose > 0:
        logger.info(
            "FM: dataset=%s, factors=%d,  #epoch=%d, batch=%d, lr=%.4f, lambda=%.1e,optimizer=%s, batch_norm=%d, keep=%.2f, neg=%d"
            % (args.dataset, args.hidden_factor, args.epoch, args.batch_size, args.lr, args.lamda, args.optimizer,
               args.batch_norm, args.keep_prob,args.neg))

    logger.info(f"qlr:{args.qlr}, weight_decay:{args.weight_decay}, Q-layers:{args.sampler_layers}, " + 
        f"Q-dropout:{args.sampler_dropout_rate}, max_q{args.max_q}, discount:{args.discount}, maxlen{args.maxlen}")
    save_file = './pretrain-FM-%s/%s_%d' % (args.dataset, args.dataset, args.hidden_factor)
    # Training
    t1 = time()


    model = FM(data.user_field_M, data.item_field_M, args.pretrain, save_file, args.hidden_factor, args.loss_type,
               args.epoch, args.batch_size, args.lr, args.lamda, args.keep_prob, 
               args.optimizer, args.batch_norm, args.verbose, args)
    # model.evaluate()
    print("begin train")
    model.train(data.Train_data)
    print("end train")
    #model.evaluate()
    print("finish")
