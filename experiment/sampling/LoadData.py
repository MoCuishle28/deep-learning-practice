'''
Utilities for Loading data.

@author:
Xin Xin
Bo Chen

@references:
'''
import numpy as np
import os
#import networkx as nx

class LoadData(object):
    '''given the path of data, return the data format for CFM for Top-N recommendation
    :param path
    return:
    Train_data: a dictionary, 'Y' refers to a list of y values; 'X_user' and 'X_item' refers to features for context-aware user and item
    Test_data: same as Train_data
    '''

    # Two files are needed in the path
    def __init__(self, path, dataset):
        self.postive_i_u = {}
        self.path = path + dataset + "/"
        self.trainfile = self.path + "train.csv"
        self.testfile = self.path + "test.csv"
        self.user_field_M, self.item_field_M = self.get_length()
        print("user_field_M", self.user_field_M)
        print("item_field_M", self.item_field_M)
        print("field_M", self.user_field_M + self.item_field_M)
        self.item_bind_M,self.train_item_nums = self.bind_item()  # assaign a userID for a specific user-context
        print('train_item_nums', self.train_item_nums)  # DEBUG 3704
        print('ID range:', min(self.item_map.keys()), max(self.item_map.keys()))    # ML-1M: id->0~3705
        self.user_bind_M,self.train_user_nums = self.bind_user()  # assaign a itemID for a specific item-feature
        print("item_bind_M", len(self.binded_items.values()))
        print("user_bind_M", len(self.binded_users.values()))
        self.user_positive_list = self.get_positive_list(self.trainfile)  # userID positive itemID
        #self.get_postivei_u_list()
        self.Train_data, self.Test_data = self.construct_data()

        #self.G = self.create_Graph()
        #self.emb = np.load('./sort_val.npy')

    def get_length(self):
        '''
        map the user fields in all files, kept in self.user_fields dictionary
        :return:
        '''
        length_user = 0
        length_item = 0
        f = open(self.trainfile)
        line = f.readline()
        while line:
            user_features = line.strip().split(',')[0].split('-')
            item_features = line.strip().split(',')[1].split('-')
            for user_feature in user_features:
                feature = int(user_feature)
                if feature > length_user:
                    length_user = feature
            for item_feature in item_features:
                feature = int(item_feature)
                if feature > length_item:
                    length_item = feature
            line = f.readline()
        f.close()
        return length_user + 1, length_item + 1

    def bind_item(self):
        '''
        Bind item and feature
        :return:
        '''
        self.binded_items = {}  # dic{feature: id}
        self.item_map = {}  # dic{id: feature}
        self.bind_i(self.trainfile)

        item_nums=len(self.item_map)
        self.bind_i(self.testfile)
        return len(self.binded_items),item_nums

    def bind_i(self, file):
        '''
        Read a feature file and bind
        :param file: feature file
        :return:
        '''
        f = open(file)
        line = f.readline()
        i = len(self.binded_items)
        while line:
            features = line.strip().split(',')
            item_features = features[1]
            item_features_int = features[1].split('-')
            item_features_int = [int(i) for i in item_features_int]
            if item_features not in self.binded_items:
                self.binded_items[item_features] = i
                self.item_map[i] = item_features_int
                i = i + 1
            line = f.readline()
        f.close()

    def bind_user(self):
        '''
        Map the item fields in all files, kept in self.item_fields dictionary
        :return:
        '''
        self.binded_users = {}
        self.user_map={}
        self.bind_u(self.trainfile)
        user_items=len(self.binded_users)
        self.bind_u(self.testfile)
        return len(self.binded_users),user_items

    def bind_u(self, file):
        '''
        Read a feature file and bind
        :param file:
        :return:
        '''
        f = open(file)
        line = f.readline()
        i = len(self.binded_users)
        while line:
            features = line.strip().split(',')
            user_features = features[0].split('-')
            user_features = [int(i) for i in user_features]
            
            # features = line.strip().split(',')
            # user_features = features[0]
            if features[0] not in self.binded_users:
                self.binded_users[features[0]] = i
                self.user_map[i]=user_features
                i = i + 1
            line = f.readline()
        f.close()

    def get_positive_list(self, file):
        '''
        Obtain positive item lists for each user
        :param file: train file
        :return:
        '''
        f = open(file)
        line = f.readline()
        user_positive_list = {}
        while line:
            features = line.strip().split(',')
            user_id = self.binded_users[features[0]]
            item_id = self.binded_items[features[1]]
            if user_id in user_positive_list:
                user_positive_list[user_id][item_id]=item_id#.append(item_id)
            else:
                user_positive_list[user_id] ={item_id:item_id} #[item_id]
            line = f.readline()
        f.close()
        return user_positive_list

    def construct_data(self):
        '''
        Construct train and test data
        :return:
        '''
        X_user, X_item = self.read_data(self.trainfile)
        Train_data = self.construct_dataset(X_user, X_item)
        print("# of training:", len(X_user))

        X_user, X_item = self.read_data(self.testfile)
        Test_data = self.construct_dataset(X_user, X_item)
        print("# of test:", len(X_user))
        return Train_data, Test_data

    # lists of user and item
    def read_data(self, file):
        '''
        read raw data
        :param file: data file
        :return: structured data
        '''
        # read a data file;
        f = open(file)
        X_user = []
        X_item = []
        line = f.readline()
        while line:
            features = line.strip().split(',')
            user_features = features[0].split('-')
            X_user.append([int(item) for item in user_features[0:]])
            item_features = features[1].split('-')
            X_item.append([int(item) for item in item_features[0:]])
            line = f.readline()
        f.close()
        return X_user, X_item

    def construct_dataset(self, X_user, X_item):
        '''
        Construct dataset
        :param X_user: user structured data
        :param X_item: item structured data
        :return:
        '''
        Data_Dic = {}
        indexs = range(len(X_user))
        Data_Dic['X_user'] = [X_user[i] for i in indexs]
        Data_Dic['X_item'] = [X_item[i] for i in indexs]
        return Data_Dic
    def get_postivei_u_list(self):#新加的 构成图的函数
    
        # if len(self.binded_users) > len(self.binded_items):
        #     size=len(self.binded_users)
        # else:size=len(self.binded_items)
    
        # G = nx.Graph()
        # G.add_nodes_from(range(size))
    
        # edge=[]
        # usernum=len(self.user_positive_list)
        self.postive_i_u={}
        for user_id in  range(len(self.user_positive_list)):
            for item_id in self.user_positive_list[user_id]:
                #edge.append((user_id,usernum+item_id))#29770+4011
                if item_id in self.postive_i_u:
                    self.postive_i_u[item_id].append(user_id) #存储 itemId到userId的关系
                else:
                    self.postive_i_u[item_id] = [user_id]
        #self.user_positive_list[user_id].append(item_id)
        #G.add_edges_from(edge)
        