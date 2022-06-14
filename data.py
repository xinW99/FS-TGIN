from __future__ import print_function
from torchtools import *
import torch.utils.data as data
import json
import os
import multiprocessing
import numpy as np
import random
from transformers import BertModel, BertTokenizer
import torch
from tqdm import tqdm

from transformers import BertTokenizer
from itertools import islice



class fewrelDataLoader(data.Dataset):

    def __init__(self, root, file_name, max_length, case_sensitive=False, partition = 'train'):
        super(fewrelDataLoader, self).__init__()
        # set dataset information
        self.file_name = file_name
        self.root = root
        self.max_length = max_length
        self.case_sensitive = case_sensitive
        self.partition = partition
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        dataset_path = os.path.join(self.root, 'fewrel', 'fewrel_cut_%s.json' % self.partition)

        print("Loading data file...")
        self.ori_data = json.load(open(dataset_path, "r"))
        self.rel2scope_train1 = json.load(open('data/processed_data/rel2scope_train.json','r'))
        self.rel2scope_test1 = json.load(open('data/processed_data/rel2scope_train.json', 'r'))
        if not case_sensitive:
            print("Elimiating case sensitive problem...")
            for relation in self.ori_data:
                for ins in self.ori_data[relation]:
                    for i in range(len(ins['tokens'])):
                        ins['tokens'][i] = ins['tokens'][i].lower()
            print("Finish eliminating")
        if self.partition =='train':
            self.entity_list_id_bert = np.load('data/processed_data/entity_list_id_bert_train.npy', allow_pickle=True).tolist()
            self.entity_list_id_bert_dic ={}
            self.label_list_bert_dic = {}
            self.label_list_bert = np.load('data/processed_data/label_list_train.npy',allow_pickle=True).tolist()
            self.rel2scope_train = dict([(key, self.rel2scope[key]) for key in  self.rel2scope_train1])
            for i, class_id in enumerate(self.rel2scope_train.keys()):
                scope = self.rel2scope_train[class_id]
                self.entity_list_id_bert_dic[class_id] = self.entity_list_id_bert[scope[0]:scope[-1]]
                self.label_list_bert_dic[class_id] = self.label_list_bert[scope[0]:scope[-1]]
        else:
            self.entity_list_id_bert = np.load('data/processed_data/entity_list_id_bert_test.npy',allow_pickle=True).tolist()
            self.entity_list_id_bert_dic = {}
            self.label_list_bert_dic = {}
            self.label_list_bert = np.load('data/processed_data/label_list_test.npy', allow_pickle=True).tolist()
            self.rel2scope_test = dict([(key, self.rel2scope[key]) for key in  self.rel2scope_test1])
            for i, class_id in enumerate(self.rel2scope_test.keys()):
                scope = [x-35000 for x in self.rel2scope_test[class_id]]
                self.entity_list_id_bert_dic[class_id] = self.entity_list_id_bert[scope[0]:scope[-1]]
                self.label_list_bert_dic[class_id] = self.label_list_bert[scope[0]:scope[-1]]
        print("Finish loading")



    def get_task_batch(self,
                       num_tasks=40,
                       num_ways=5,
                       num_shots=1,
                       num_queries=1,
                       seed=None):
        if seed is not None:
            random.seed(seed)

        # init task batch data
        support_data_r, support_data_h,support_data_t,support_label_r,support_label_h,support_label_t,support_data_attention_mask = [], [], [], [], [], [], []
        query_data, query_entity_label, query_entity,query_data_attention_mask = [], [], [], []
        for _ in range(num_ways * num_shots):
            data_r = np.zeros(shape=[num_tasks]+[self.max_length],dtype='int')
            attention_mask_support = np.zeros(shape=[num_tasks]+[self.max_length],dtype='int')
            data_h = np.zeros(shape=[num_tasks], dtype='int').tolist()
            data_t = np.zeros(shape=[num_tasks], dtype='int').tolist()
            label_r = np.zeros(shape=[num_tasks],dtype='int')
            label_h = np.zeros(shape=[num_tasks], dtype='int')
            label_t = np.zeros(shape=[num_tasks], dtype='int')
            support_data_r.append(data_r)
            support_data_attention_mask.append(attention_mask_support)
            support_data_h.append(data_h)
            support_data_t.append(data_t)
            support_label_r.append(label_r)
            support_label_h.append(label_h)
            support_label_t.append(label_t)
        for _ in range(num_ways * num_queries):
            data = np.zeros(shape=[num_tasks]+[self.max_length],dtype='int')
            entity = np.zeros(shape=[num_tasks],dtype='int').tolist()
            attention_mask_query = np.zeros(shape=[num_tasks] + [self.max_length], dtype='int')
            label = np.zeros(shape=[num_tasks],dtype='int').tolist()
            query_data.append(data)
            query_data_attention_mask.append( attention_mask_query)
            query_entity_label.append(label)
            query_entity.append(entity)

        # get full class list in dataset
        full_class_list = list(self.ori_data.keys())

        # for each task
        for t_idx in range(num_tasks): # define task by sampling classes (num_ways)
            task_class_list = random.sample(full_class_list, num_ways)

            # for each sampled class in task
            for c_idx in range(num_ways):# sample data for support and query (num_shots + num_queries)
                indices = np.random.choice(list(range(0, 700)), num_shots + num_queries, False).tolist()
                class_data_dict = list(range(len(indices)))
                class_entity_list = list(range(len(indices)))
                class_entity_label = list(range(len(indices)))
                for i,ind in enumerate(indices):
                    class_data_dict[i] = self.ori_data[task_class_list[c_idx]][ind]
                    print(t_idx,task_class_list[c_idx],ind)
                    class_entity_list[i] = self.entity_list_id_bert_dic[task_class_list[c_idx]][ind]
                    class_entity_label[i] = self.label_list_bert_dic[task_class_list[c_idx]][ind]
                class_data_list = list(range(len(class_data_dict)))
                attention_mask_list = list(range(len(class_data_dict)))
                for i,ins in enumerate(class_data_dict): # sentence: dict--->str
                    word = ins['tokens']
                    words = ' '.join(word)
                    token_results = self.tokenizer(words, max_length= 128, padding='max_length')
                    class_data_list[i] = token_results['input_ids']
                    attention_mask_list[i] = token_results['attention_mask']
                class_head_entity_list = []
                class_tail_entity_list = []
                class_query_entity_list =[]
                class_query_label_list =[]

                for j,lab in enumerate(class_entity_label):
                    if j < num_shots :
                        head_label_num = class_entity_label[j].index('01')

                        head_entity_id = class_entity_list[j][head_label_num]
                        tail_label_num = class_entity_label[j].index('02')
                        tail_entity_id = class_entity_list[j][tail_label_num]
                        class_head_entity_list.append(head_entity_id)
                        class_tail_entity_list.append(tail_entity_id)
                    else:
                        class_query_entity = class_entity_list[j]
                        class_query_label = class_entity_label[j]
                        class_query_entity_list.append(class_query_entity)
                        class_query_label_list.append(class_query_label)

                        # load sample for support set
                for i_idx in range(num_shots):
                    # set data
                    rel_num = str(c_idx)
                    support_data_r[i_idx + c_idx * num_shots][t_idx] = class_data_list[i_idx]
                    support_data_attention_mask[i_idx + c_idx * num_shots][t_idx] = attention_mask_list[i_idx]
                    support_data_h[i_idx + c_idx * num_shots][t_idx] = class_head_entity_list[i_idx]
                    support_data_t[i_idx + c_idx * num_shots][t_idx] = class_tail_entity_list[i_idx]  # read support data in turn
                    support_label_r[i_idx + c_idx * num_shots][t_idx] = int(rel_num+'0')
                    support_label_h[i_idx + c_idx * num_shots][t_idx] =  int(rel_num+'1')
                    support_label_t[i_idx + c_idx * num_shots][t_idx] =  int(rel_num+'2') #get relation label for head and tail entity

                # load sample for query set
                class_query_entity_label_list =list(range(num_queries)) #refactor query entity label with relations
                for i_idx in range(num_queries):
                    rel_num = str(c_idx)
                    class_list =[]
                    for iii in class_query_label_list[i_idx]:
                        class_query_label = rel_num+iii[1]
                        class_list.append(class_query_label)
                    class_query_entity_label_list[i_idx] =  class_list
                    query_data[i_idx + c_idx * num_queries][t_idx] = class_data_list[num_shots + i_idx]
                    query_data_attention_mask[i_idx + c_idx *num_queries][t_idx] = attention_mask_list[num_shots + i_idx]
                    query_entity[i_idx + c_idx * num_queries][t_idx] = class_query_entity_list[i_idx]
                    query_entity_label[i_idx + c_idx * num_queries][t_idx] =class_query_entity_label_list[i_idx]

                #only get one query sentence
            #perm = np.random.permutation(num_queries)
            #query_entity = query_entity[perm]
            #query_entity_label = query_entity_label [perm]

        # convert to tensor (num_tasks x (num_ways * (num_supports + num_queries)) x ...)
        support_data_r =torch.stack([torch.from_numpy(data_r).to(tt.arg.device) for data_r in support_data_r],1)
        support_data_attention_mask = torch.stack([torch.from_numpy(attention_mask_support).to(tt.arg.device) for  attention_mask_support in support_data_attention_mask],1)

        support_label_r = torch.stack([torch.from_numpy(label_r).to(tt.arg.device) for label_r in support_label_r], 1)
        support_label_h = torch.stack([torch.from_numpy(label_h).to(tt.arg.device) for label_h in support_label_h], 1)
        support_label_t = torch.stack([torch.from_numpy(label_t).to(tt.arg.device) for label_t in support_label_t], 1)

        query_data = torch.stack([torch.from_numpy(data).to(tt.arg.device) for data in query_data], 1)
        #query_entity = torch.stack([torch.from_numpy(data).float().to(tt.arg.device) for data in query_entity], 1)
        query_data_attention_mask = torch.stack([torch.from_numpy(attention_mask_query).to(tt.arg.device) for attention_mask_query in query_data_attention_mask], 1)
        #query_entity_label = torch.stack([torch.from_numpy(label).float().to(tt.arg.device) for label in query_entity_label], 1) #40*5

        return [support_data_r, support_data_h,support_data_t,support_label_r,
                support_label_h,support_label_t,support_data_attention_mask,query_data,
                query_entity,query_entity_label,query_data_attention_mask]

