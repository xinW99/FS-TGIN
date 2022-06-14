from torchtools import *
from data import fewrelDataLoader
from model import EmbeddingSentences, GraphNetwork,CnnForEntity
from torchtools.tt.logger import get_logger
import shutil
import os
import random
import itertools
from transformers import AdamW
from random import sample
import time
import logging

#import seaborn as sns

class ModelTrainer(object):
    def __init__(self,
                 enc_module,
                 gnn_module,
                 cnn_module,
                 data_loader):
        # set encoder and gnn
        self.enc_module = enc_module.to(tt.arg.device)
        self.gnn_module = gnn_module.to(tt.arg.device)
        self.cnn_module = cnn_module.to(tt.arg.device)

        #self.enc_module.load_state_dict(torch.load(os.path.join(configs.checkpoints_dir, 'best_model_fewrel.pkl')))

        if tt.arg.num_gpus > 1:
            print('Construct multi-gpu model ...')
            self.enc_module = nn.DataParallel(self.enc_module, device_ids=[0, 1, 2, 3], dim=1)
            self.gnn_module = nn.DataParallel(self.gnn_module, device_ids=[0, 1, 2, 3], dim=1)
            self.cnn_module = nn.DataParallel(self.cnn_module, device_ids=[0, 1, 2, 3], dim=1)

            print('done!\n')

        # get data loader
        self.data_loader = data_loader
        #set freeze layers
        unfreeze_layers = ['layer.11.output']
        for name, ppp in self.enc_module.named_parameters():
            ppp.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    ppp.requires_grad = True
                    break

        # set optimizer
        self.module_params =  list(self.enc_module.parameters())+list(self.gnn_module.parameters())+list(self.cnn_module.parameters())
        #self.module_params = list(self.cnn_module.parameters())

        # set optimizer
        self.optimizer = AdamW(filter(lambda p: p.requires_grad, self.module_params),
                                    lr=tt.arg.lr,
                                    weight_decay=tt.arg.weight_decay)

        # set loss
        self.edge_loss = nn.BCELoss(reduction='none')
        self.entity_loss = nn.BCELoss(reduction='none')

        self.node_loss = nn.CrossEntropyLoss(reduction='none')

        self.global_step = 0
        self.val_acc = 0
        self.test_acc = 0

    def __dist__(self, x, y, dim):
        return (torch.pow(x - y, 2)).sum(dim)

    start_time_epoch = time.time()
    def train(self):
        logger.info('mode:train')
        val_acc = self.val_acc

        # set edge mask (to distinguish support and query edges)
        num_supports_r = tt.arg.num_ways_train * tt.arg.num_shots_train
        num_supports_h = tt.arg.num_ways_train * tt.arg.num_shots_train
        num_supports_t= tt.arg.num_ways_train * tt.arg.num_shots_train
        num_supports =  num_supports_r+ num_supports_h+ num_supports_t
        num_queries_e =5
        num_queries =tt.arg.num_ways_train * 1
        num_samples = num_supports + num_queries_e
        support_edge_mask = torch.zeros(tt.arg.meta_batch_size, num_samples, num_samples).to(tt.arg.device)#
        support_edge_mask[:, :num_supports, :num_supports] = 1 #size = batchsize * num_support all *num support all
        query_edge_mask = 1 - support_edge_mask
        evaluation_mask = torch.ones(tt.arg.meta_batch_size, num_samples, num_samples).to(tt.arg.device)#size = batchsize * num sample all *num sample all
        evaluation_mask[:, :num_supports, :num_supports] = 0

        # for each iteration
        logger.info('Start training....')
        #start_time_epoch = time.time()
        t = 0
        for iter in range(self.global_step + 1, tt.arg.train_iteration + 1):
            # init grad
            self.optimizer.zero_grad()

            # set current step
            self.global_step = iter

            # load task data list
            [support_data_r, support_data_h,support_data_t,support_label_r,
                support_label_h,support_label_t,support_data_attention_mask,query_data,
                query_entity,query_entity_label,query_data_attention_mask] = self.data_loader['train'].get_task_batch(num_tasks=tt.arg.meta_batch_size,
                                                                     num_ways=tt.arg.num_ways_train,
                                                                     num_shots=tt.arg.num_shots_train,
                                                                     seed=iter + tt.arg.seed)

            #(1) encode data
            full_data_encoder = torch.cat([support_data_r, query_data], 1)
            full_attention_mask = torch.cat([support_data_attention_mask, query_data_attention_mask], 1)

            self.enc_module.train()
            full_data_encoder_result = self.enc_module(full_data_encoder, full_attention_mask)

            self.cnn_module.train()
            support_data_r_result, support_data_h_result, support_data_t_result, query_data_entity_result, query_data_entity_mask, query_label_e = self.cnn_module(full_data_encoder_result,support_data_h,support_data_t,query_entity,num_supports_r,query_entity_label,full_attention_mask)
            query_data_entity_result_reshaped = query_data_entity_result.view(tt.arg.meta_batch_size,num_queries * num_queries_e,-1)
            query_label_e_reshaped = query_label_e.view(tt.arg.meta_batch_size,num_queries * num_queries_e)
            full_data = torch.cat([support_data_r_result,support_data_h_result,support_data_t_result, query_data_entity_result_reshaped], 1).float()
            full_label = torch.cat([support_label_r,support_label_h,support_label_t, query_label_e_reshaped], 1).float()
            full_edge,edge_M= self.label2edge(full_label,num_supports_r,num_supports_h,num_supports_t,num_queries_e)

            # set init edge and init edge_C

            init_edge = full_edge.clone()  # batch_size x 2 x num_samples x num_samples
            init_edge_M = edge_M.clone()   # batch_size x num_samples x num_samples

            #init_edge_part_t, init_edge_part_h = self.Compute_P_dis(support_data_h_result, support_data_t_result, query_data_entity_result_reshaped,query_data_entity_mask,tt.arg.num_ways,tt.arg.num_shots,tt.arg.num_unkonw_entity)
            #init_edge[:, 0, num_supports_r+num_supports_h+num_supports_t:, :num_supports_r] = init_edge_part_t
            #init_edge[:, 0, :num_supports_r, num_supports_r + num_supports_h + num_supports_t:] = init_edge_part_t.transpose(1,2)
            #init_edge[:, 1, num_supports_r + num_supports_h + num_supports_t:, :num_supports_r] = init_edge_part_h
            #init_edge[:, 1, :num_supports_r,num_supports_r + num_supports_h + num_supports_t:] = init_edge_part_h.transpose(1, 2)

            init_edge[:, 0, num_supports_r + num_supports_h + num_supports_t:, :num_supports_r] = 0.5
            init_edge[:, 0, :num_supports_r, num_supports_r + num_supports_h + num_supports_t:] = 0.5
            init_edge[:, 1, num_supports_r + num_supports_h + num_supports_t:, :num_supports_r] = 0.5
            init_edge[:, 1, :num_supports_r, num_supports_r + num_supports_h + num_supports_t:] = 0.5

            # set as train mode
            self.gnn_module.train()
            # (2) predict edge logit (consider only the last layer logit, num_tasks x 2 x num_samples x num_samples)


                # input_node_feat: (batch_size x num_queries) x (num_support + 1) x featdim
                # input_edge_feat: (batch_size x num_queries) x 2 x (num_support + 1) x (num_support + 1)
            support_data = full_data[:, :num_supports_r + num_supports_h + num_supports_t] # batch_size x num_support x 768
            query_data = full_data[:, num_supports_r + num_supports_h + num_supports_t:] # batch_size x num_query x 768
            support_data_tiled = support_data.unsqueeze(1).repeat(1, num_queries, 1, 1) # batch_size x num_queries x num_support x 768
            support_data_tiled = support_data_tiled.view(tt.arg.meta_batch_size * num_queries, num_supports, -1) # (batch_size x num_queries) x num_support x 768
            query_data_reshaped = query_data.contiguous().view(tt.arg.meta_batch_size * num_queries, num_queries_e, -1) # (batch_size x num_queries) x 5 x 768
            input_node_feat = torch.cat([support_data_tiled, query_data_reshaped], 1) # (batch_size x num_queries) x (num_support + num_entities) x featdim
            init_edge_support_temp = init_edge.split([num_supports,init_edge.size(2) - num_supports],dim = 2)
            init_edge_support = init_edge_support_temp[0].split([num_supports,init_edge.size(2) - num_supports],dim =3)[0]
            init_edge_query = init_edge_support_temp[1].split([num_supports_r,init_edge.size(2) - num_supports_r],dim =3)[0]
            init_edge_support_tailed = init_edge_support.unsqueeze(1).repeat(1,num_queries,1,1,1)
            init_edge_support_tailed = init_edge_support_tailed.view(tt.arg.meta_batch_size * num_queries, 3, num_supports, num_supports)
            init_edge_query_reshaped = init_edge_query.transpose(1,2).reshape(tt.arg.meta_batch_size * num_queries,num_queries_e,3,num_supports_r).transpose(1,2)
            input_edge_feat = torch.zeros(tt.arg.meta_batch_size * num_queries, 3, num_supports + num_queries_e, num_supports + num_queries_e).to(tt.arg.device) # batch_size x 3 x (num_support + num_unknown_entities) x (num_support + num_unknown_entities)
            input_edge_feat[:, :, :num_supports, :num_supports] = init_edge_support_tailed
            input_edge_feat[:, :, num_supports:, :num_supports_r] = init_edge_query_reshaped
            input_edge_feat[:, :, :num_supports_r, num_supports:] = init_edge_query_reshaped.transpose(3,2)

            # logit: (batch_size x num_queries) x 3 x (num_support + 5) x (num_support + 5)
            logit_results = self.gnn_module(node_feat=input_node_feat, edge_feat=input_edge_feat, num_supports = num_supports, entity_mask = query_data_entity_mask)
            logit_layers = logit_results[0]
            logit_nodes = logit_results[1]
            logit_layers = [logit_layer.view(tt.arg.meta_batch_size, num_queries, 3, num_supports + num_queries_e, num_supports + num_queries_e) for logit_layer in logit_layers]
            logit_nodes = [logit_node.view(tt.arg.meta_batch_size, num_queries,num_supports + num_queries_e,768) for logit_node in logit_nodes]
            # logit --> full_logit (batch_size x 2 x num_samples x num_samples)
            full_logit_layers = []
            full_logit_nodes = []
            for l in range(tt.arg.num_layers):
                full_logit_layers.append(torch.zeros(tt.arg.meta_batch_size, 3, num_supports+num_queries_e*num_queries, num_supports+num_queries_e*num_queries).to(tt.arg.device))
                full_logit_nodes.append(torch.zeros(tt.arg.meta_batch_size,num_supports+num_queries_e*num_queries,768).to(tt.arg.device))
            for l in range(tt.arg.num_layers):
                full_logit_layers[l][:, :, :num_supports, :num_supports] = logit_layers[l][:, :, :, :num_supports, :num_supports].mean(1)
                full_logit_layers[l][:, :, :num_supports, num_supports:] = logit_layers[l][:, :, :, :num_supports, num_supports:].transpose(1, 2).transpose(2, 3).reshape(tt.arg.meta_batch_size,3,num_supports,-1)
                full_logit_layers[l][:, :, num_supports:, :num_supports] = logit_layers[l][:, :, :, num_supports:, :num_supports].transpose(1, 2).reshape(tt.arg.meta_batch_size,3,-1,num_supports)
                full_logit_nodes[l][:,:num_supports,:] = logit_nodes[l][:,:,:num_supports,:].mean(1)
                full_logit_nodes[l][:, num_supports:,:] = logit_nodes[l][:,:,num_supports:,:].reshape(tt.arg.meta_batch_size,num_queries*num_queries_e,768)
            # (4) compute loss
            #(4.1) edge loss

            full_t_edge_loss_layers = [self.edge_loss(1-(full_logit_layer[:, 0]), (1-full_edge[:, 0])) for full_logit_layer in full_logit_layers]
            full_h_edge_loss_layers = [self.edge_loss((full_logit_layer[:, 1]), (full_edge[:, 1])) for full_logit_layer in full_logit_layers]
            # weighted edge loss for truth entity, wrong entity and none entity
            exist_entity_mask = query_data_entity_mask[:,:,:,0].reshape(tt.arg.meta_batch_size,num_queries*num_queries_e,-1).repeat(1, 1, num_supports_r)
            entity_mask = torch.zeros(tt.arg.meta_batch_size,num_supports+num_queries_e*num_queries,num_supports+num_queries_e*num_queries).to(tt.arg.device)
            entity_mask[:, num_supports:, :num_supports_r] = exist_entity_mask
            exist_entity_mask_transpose = exist_entity_mask.transpose(1,2)
            entity_mask[:, :num_supports_r, num_supports:] = exist_entity_mask_transpose
            edge_M[:, 0, num_supports_r+num_supports_h:num_supports_r+num_supports_h+num_supports_t, :num_supports_r] = 0
            edge_M[:, 0, :num_supports_r, num_supports_r+num_supports_h:num_supports_r+num_supports_h+num_supports_t] = 0
            edge_M[:, 1, num_supports_r:num_supports_r+num_supports_h,:num_supports_r] = 0
            edge_M[:, 1, :num_supports_r,num_supports_r:num_supports_r+num_supports_h] = 0
            truth_t_entity_loss_layers = [torch.sum(full_t_edge_loss_layer * full_edge[:,0]*entity_mask)/torch.sum(full_edge[:,0]*entity_mask) for full_t_edge_loss_layer in full_t_edge_loss_layers ]
            wrong_t_entity_loss_layers = [torch.sum(full_t_edge_loss_layer *(1-full_edge[:,0])*entity_mask)/torch.sum((1-full_edge[:,0])*entity_mask) for full_t_edge_loss_layer in full_t_edge_loss_layers]
            none_t_entity_loss_layers = [torch.sum(full_t_edge_loss_layer *(1-entity_mask)*edge_M[:,0])/torch.sum((1-entity_mask)*edge_M[:,0]) for full_t_edge_loss_layer in full_t_edge_loss_layers]
            truth_h_entity_loss_layers = [torch.sum(full_h_edge_loss_layer*full_edge[:,1]*entity_mask)/torch.sum(full_edge[:,1]*entity_mask) for full_h_edge_loss_layer in full_h_edge_loss_layers]
            wrong_h_entity_loss_layers = [torch.sum(full_h_edge_loss_layer*(1-full_edge[:,1])*entity_mask)/torch.sum((1-full_edge[:,1])*entity_mask) for full_h_edge_loss_layer in full_h_edge_loss_layers]
            none_h_entity_loss_layers = [torch.sum(full_h_edge_loss_layer*(1-entity_mask)*edge_M[:,1])/torch.sum((1-entity_mask)*edge_M[:,1]) for full_h_edge_loss_layer in full_h_edge_loss_layers]
            edge_loss_for_all_query_layers = [truth_t_entity_loss_layer + wrong_t_entity_loss_layer + none_t_entity_loss_layer + truth_h_entity_loss_layer + wrong_h_entity_loss_layer + none_h_entity_loss_layer for truth_t_entity_loss_layer, wrong_t_entity_loss_layer, none_t_entity_loss_layer, truth_h_entity_loss_layer, wrong_h_entity_loss_layer, none_h_entity_loss_layer in zip(truth_t_entity_loss_layers, wrong_t_entity_loss_layers, none_t_entity_loss_layers, truth_h_entity_loss_layers, wrong_h_entity_loss_layers, none_h_entity_loss_layers)]

            #(4.2) calculate TransE loss

            loss_transE_layers = []
            for l in range(tt.arg.num_layers):
                # get node embedding
                label_r = full_label[:,:num_supports_r]
                label_h = full_label[:,num_supports_r:num_supports_r+num_supports_h]
                label_t = full_label[:,num_supports_r+num_supports_h:num_supports_r+num_supports_h+num_supports_t]
                label_e = full_label[:,num_supports_r+num_supports_h+num_supports_t:]
                node_feat_r = full_logit_nodes[l][:,:num_supports_r,:]
                node_feat_h = full_logit_nodes[l][:,num_supports_r:num_supports_r+num_supports_h,:]
                node_feat_t = full_logit_nodes[l][:,num_supports_r+num_supports_h:num_supports_r+num_supports_h+num_supports_t,:]
                node_feat_e = full_logit_nodes[l][:,num_supports_r+num_supports_h+num_supports_t:,:]
                # (4.2.1) get corrupted triple for support triples
                label_relation_type = ((label_r.reshape(tt.arg.meta_batch_size*num_supports_r,-1))%100/10).reshape(tt.arg.meta_batch_size,num_supports_r)
                copynum_for_h_and_t = (tt.arg.num_ways-1) * tt.arg.num_shots * tt.arg.num_shots
                transE_loss_sum = 0
                for q_i in range(node_feat_h.size(1)):
                    feat_h_one = node_feat_h[:,q_i,:].unsqueeze(1)
                    feat_t_one = node_feat_t[:,q_i,:].unsqueeze(1)
                    feat_r_reshape = node_feat_r.reshape(tt.arg.meta_batch_size,tt.arg.num_ways,tt.arg.num_shots,768)
                    way_idx = int(q_i/tt.arg.num_shots)
                    feat_h_expand = feat_h_one.repeat(1, copynum_for_h_and_t, 1)
                    feat_t_expand = feat_t_one.repeat(1,copynum_for_h_and_t, 1)

                    feat_r_for_right = feat_r_reshape[:,way_idx,:,:].unsqueeze(2).repeat(1, 1, (tt.arg.num_ways-1) * tt.arg.num_shots,1).reshape(tt.arg.meta_batch_size,(tt.arg.num_ways-1) * tt.arg.num_shots*tt.arg.num_shots,768)

                    feat_r_combine_1 = feat_r_reshape[:, 0:way_idx, :, :]
                    feat_r_combine_2 = feat_r_reshape[:, way_idx+1:, :, :]
                    feat_r_for_wrong = torch.cat((feat_r_combine_1,feat_r_combine_2), dim = 1).repeat(1, tt.arg.num_shots, 1, 1).reshape(tt.arg.meta_batch_size,(tt.arg.num_ways-1) * tt.arg.num_shots * tt.arg.num_shots,768)
                    #score
                    dis_right_triple = feat_h_expand + feat_r_for_right - feat_t_expand
                    dis_wrong_triple = feat_h_expand + feat_r_for_wrong - feat_t_expand
                    score_right_triple = torch.norm(dis_right_triple, p=2, dim = -1)
                    score_wrong_tripe = torch.norm(dis_wrong_triple, p=2, dim = -1)
                    transE_loss = (1.5 + score_right_triple - score_wrong_tripe).reshape(-1,1)
                    a = 0
                    transE_loss_for_one_pair = 0
                    for iii in range (transE_loss.size(0)):
                        if transE_loss[iii] > 0:
                            transE_loss_for_one_pair = transE_loss_for_one_pair + transE_loss[iii]
                            a = a + 1
                    if a!=0:
                        transE_loss_for_one_pair_mean = transE_loss_for_one_pair/a
                    else:
                        transE_loss_for_one_pair_mean = transE_loss_for_one_pair
                    transE_loss_sum = transE_loss_sum + transE_loss_for_one_pair_mean
                support_transE_loss_mean_all =transE_loss_sum/node_feat_h.size(1)
                # (4.2.2) get corrupted triple for query triples
                node_feat_e_reshaped = node_feat_e.reshape(tt.arg.meta_batch_size, tt.arg.num_ways, num_queries_e, 768)
                label_e_reshaped = label_e.reshape(tt.arg.meta_batch_size, tt.arg.num_ways, num_queries_e)
                query_transE_loss_sum = 0
                query_transE_replace_loss_sum = 0
                replace_num = 0
                for b_i in range(label_e_reshaped.size(0)):
                    for r_i in range(tt.arg.num_ways):
                        headEntity_pos = torch.where(label_e_reshaped[b_i,r_i,:] % 10 ==1)
                        tailEntity_pos = torch.where(label_e_reshaped[b_i,r_i,:] % 10 ==2)
                        headEntity_vector = node_feat_e_reshaped[b_i, r_i, headEntity_pos[0][0]]
                        tailEntity_vector = node_feat_e_reshaped[b_i, r_i, tailEntity_pos[0][0]]
                        relation_vector = feat_r_reshape[b_i,r_i,:,:]

                        relation_wrong_vector_1 = feat_r_reshape[b_i, 0:r_i, :, :]
                        relation_wrong_vector_2 = feat_r_reshape[b_i, r_i+1:, :, :]
                        relation_wrong_vector = torch.cat((relation_wrong_vector_1,relation_wrong_vector_2),dim =0).reshape(-1,768).repeat(tt.arg.num_shots,1)
                        headEntity_vector_expand = headEntity_vector.unsqueeze(0).repeat(relation_wrong_vector.size(0),1)
                        tailEntity_vector_expand = tailEntity_vector.unsqueeze(0).repeat(relation_wrong_vector.size(0),1)
                        relation_vector_expand = relation_vector.unsqueeze(1).repeat(1,((tt.arg.num_ways-1)*tt.arg.num_shots),1).reshape(-1,768)

                        dis_right_query_triple = headEntity_vector_expand + relation_vector_expand - tailEntity_vector_expand
                        dis_wrong_query_triple =  headEntity_vector_expand + relation_wrong_vector - tailEntity_vector_expand
                        score_right_query_triple = torch.norm(dis_right_query_triple , p=2, dim=-1)
                        score_wrong_query_triple = torch.norm(dis_wrong_query_triple, p=2, dim=-1)
                        transE_query_r_loss = (1.5 + score_right_query_triple -score_wrong_query_triple).reshape(-1, 1)
                        q = 0
                        query_transE_loss_for_one_pair = 0
                        for iii in range(transE_query_r_loss.size(0)):
                            if transE_query_r_loss[iii] > 0:
                                query_transE_loss_for_one_pair = query_transE_loss_for_one_pair + transE_query_r_loss[iii]
                                q = q + 1
                        if q != 0:
                            query_transE_loss_for_one_pair_mean = query_transE_loss_for_one_pair / q
                        else:
                            query_transE_loss_for_one_pair_mean = query_transE_loss_for_one_pair
                        relation_vector_for_exchangeHT = relation_vector.reshape(-1, 768)
                        headEntity_vector_exchange_expand = headEntity_vector.unsqueeze(0).repeat(relation_vector_for_exchangeHT.size(0), 1)
                        tailEntity_vector_exchange_expand = tailEntity_vector.unsqueeze(0).repeat(relation_vector_for_exchangeHT.size(0), 1)
                        #score
                        exchange_right_query_triple = headEntity_vector_exchange_expand + relation_vector_for_exchangeHT - tailEntity_vector_exchange_expand
                        exchange_wrong_query_triple = tailEntity_vector_exchange_expand + relation_vector_for_exchangeHT - headEntity_vector_exchange_expand
                        score_exchange_right_query_triple = torch.norm(exchange_right_query_triple, p=2, dim=-1)
                        score_exchange_wrong_query_triple = torch.norm(exchange_wrong_query_triple, p=2, dim=-1)
                        transE_query_exchangeHT_loss = (1.5 + score_exchange_right_query_triple - score_exchange_wrong_query_triple).reshape(-1, 1)
                        ex = 0
                        query_transE_loss_for_exchange_one = 0
                        for iii in range(transE_query_exchangeHT_loss.size(0)):
                            if transE_query_exchangeHT_loss[iii] > 0:
                                query_transE_loss_for_exchange_one = query_transE_loss_for_exchange_one + transE_query_exchangeHT_loss[iii]
                                ex = ex + 1
                        if ex != 0:
                            query_transE_loss_for_exchange_mean = query_transE_loss_for_exchange_one / ex
                        else:
                            query_transE_loss_for_exchange_mean = query_transE_loss_for_exchange_one

                        other_entity_pos = torch.where(label_e_reshaped[b_i, r_i, :] % 10 == 3)
                        if len(other_entity_pos[0]) != 0:
                            other_entity_vector_set = []
                            for o_i in range(len(other_entity_pos[0])):
                                other_entity_vector = node_feat_e_reshaped[
                                    b_i, r_i, other_entity_pos[0][o_i]].unsqueeze(0)
                                other_entity_vector_set.append(other_entity_vector)
                            other_entity_vector_all = torch.cat(other_entity_vector_set, dim=0)
                            headEntity_vector_replace_expand = headEntity_vector.unsqueeze(0).repeat(other_entity_vector_all.size(0) * relation_vector.size(0), 1)
                            tailEntity_vector_replace_expand = tailEntity_vector.unsqueeze(0).repeat(other_entity_vector_all.size(0) * relation_vector.size(0), 1)
                            relation_vector_replace_expand = relation_vector.repeat(other_entity_vector_all.size(0), 1)
                            other_entity_vector_replace_expand = other_entity_vector_all.unsqueeze(1).repeat(1,relation_vector.size(0),1).reshape(-1, 768)

                            replaceH_right_query_triple = headEntity_vector_replace_expand + relation_vector_replace_expand - tailEntity_vector_replace_expand
                            replaceH_wrong_query_triple = other_entity_vector_replace_expand + relation_vector_replace_expand - tailEntity_vector_replace_expand
                            score_replaceH_right_query_triple = torch.norm(replaceH_right_query_triple, p=2, dim=-1)
                            score_replaceH_wrong_query_triple = torch.norm(replaceH_wrong_query_triple, p=2, dim=-1)
                            transE_query_replaceH_loss = (1.5 + score_replaceH_right_query_triple - score_replaceH_wrong_query_triple).reshape(-1, 1)
                            reh = 0
                            query_transE_loss_for_replaceH_one = 0
                            for iii in range(transE_query_replaceH_loss.size(0)):
                                if transE_query_replaceH_loss[iii] > 0:
                                    query_transE_loss_for_replaceH_one = query_transE_loss_for_replaceH_one + transE_query_replaceH_loss[iii]
                                    reh = reh + 1
                            if reh != 0:
                                query_transE_loss_for_replaceH_mean = query_transE_loss_for_replaceH_one / reh
                            else:
                                query_transE_loss_for_replaceH_mean = query_transE_loss_for_replaceH_one

                            replaceT_right_query_triple = headEntity_vector_replace_expand + relation_vector_replace_expand - tailEntity_vector_replace_expand
                            replaceT_wrong_query_triple = headEntity_vector_replace_expand + relation_vector_replace_expand - other_entity_vector_replace_expand
                            score_replaceT_right_query_triple = torch.norm(replaceT_right_query_triple, p=2, dim=-1)
                            score_replaceT_wrong_query_triple = torch.norm(replaceT_wrong_query_triple, p=2, dim=-1)
                            transE_query_replaceT_loss = (1.5 + score_replaceT_right_query_triple - score_replaceT_wrong_query_triple).reshape(-1, 1)
                            ret = 0
                            query_transE_loss_for_replaceT_one = 0
                            for iii in range(transE_query_replaceT_loss.size(0)):
                                if transE_query_replaceT_loss[iii] > 0:
                                    query_transE_loss_for_replaceT_one = query_transE_loss_for_replaceT_one + transE_query_replaceT_loss[iii]
                                    ret = ret + 1
                            if ret != 0:
                                query_transE_loss_for_replaceT_mean = query_transE_loss_for_replaceT_one / ret  # 是不是应该加起来，而不是求平均？？
                            else:
                                query_transE_loss_for_replaceT_mean = query_transE_loss_for_replaceT_one
                            transE_query_replace_loss = query_transE_loss_for_replaceT_mean + query_transE_loss_for_replaceH_mean
                        else:
                            transE_query_replace_loss = 0
                        if transE_query_replace_loss != 0:
                            replace_num = replace_num + 1
                        else:
                            replace_num = replace_num

                        query_transE_loss_sum = query_transE_loss_sum + query_transE_loss_for_one_pair_mean + query_transE_loss_for_exchange_mean
                        query_transE_replace_loss_sum = query_transE_replace_loss_sum + transE_query_replace_loss
                if replace_num != 0:
                    query_transE_replace_loss_mean = query_transE_replace_loss_sum / replace_num
                else:
                    query_transE_replace_loss_mean = 0

                query_transE_loss_sum_replaceR_mean = (query_transE_loss_sum / (tt.arg.num_ways * tt.arg.meta_batch_size)) + query_transE_replace_loss_mean
                transE_loss_all = support_transE_loss_mean_all + query_transE_loss_sum_replaceR_mean
                loss_transE_layers.append(transE_loss_all)

            #(5)compute total loss
            total_loss_layers = [loss_transE_layer + edge_loss_for_all_query_layer for loss_transE_layer, edge_loss_for_all_query_layer in zip(loss_transE_layers, edge_loss_for_all_query_layers )]
            #total_loss_layers = [ edge_loss_for_all_query_layer for  edge_loss_for_all_query_layer in edge_loss_for_all_query_layers]
            #total_loss_layers = [loss_transE_layer for loss_transE_layer in loss_transE_layers]
            # update model
            total_loss = []
            for l in range(tt.arg.num_layers - 1):
                total_loss += [total_loss_layers[l].view(-1) * 0.5]
            if type(total_loss_layers[-1])!=torch.Tensor:
                total_loss = torch.mean(torch.cat(total_loss, 0))
            else:
                total_loss += [total_loss_layers[-1].view(-1) * 1.0]
                total_loss = torch.mean(torch.cat(total_loss, 0))

            if self.global_step % tt.arg.train_loss_everycheck==0:
                logger.info('Train_iter:{} --> BCE_Loss:{}'.format(self.global_step,edge_loss_for_all_query_layers))
                logger.info('Train_iter:{} --> TransE_Loss:{}'.format(self.global_step,loss_transE_layers))
                logger.info('Train_iter:{} --> Total_Loss:{}'.format(self.global_step,total_loss))

            total_loss.backward()
            self.optimizer.step()

            # adjust learning rate
            self.adjust_learning_rate(optimizers=[self.optimizer],lr=tt.arg.lr,iter=self.global_step)

            #pred

            #full_logit_layers_t_pred = [full_logit_layer[:,0,:,:]*entity_mask for full_logit_layer in full_logit_layers]
            #full_logit_layers_h_pred = [full_logit_layer[:,1,:,:]*entity_mask for full_logit_layer in full_logit_layers]
            #support_label_r_use = torch.floor_divide(support_label_r % 100, 10).float()
            #query_node_t_pred_layers = [torch.bmm(full_logit_layer_t_pred[:, :, :num_supports_r], self.one_hot_encode(tt.arg.num_ways_train, support_label_r_use.long())) for full_logit_layer_t_pred in full_logit_layers_t_pred]
            #query_node_h_pred_layers = [torch.bmm(full_logit_layer_h_pred[:, :, :num_supports_r],self.one_hot_encode(tt.arg.num_ways_train, support_label_r_use.long())) for full_logit_layer_h_pred in full_logit_layers_h_pred]
            #entity_pair_accr_all, triple_accr_without_order_all, triple_accr_all = self.pred_triple_test(query_node_t_pred_layers[-1], query_node_h_pred_layers[-1], num_supports,num_supports_r, full_label, num_queries_e)

            # evaluation
            if self.global_step % tt.arg.test_interval == 0:
                start_time_epoch = time.time()
                #time_span_epoch = (time.time() - start_time_epoch) / 60
                #logger.info('Training time consumption:%.2f(min)'% time_span_epoch)
                logger.info('epoch:{}'.format(self.global_step / tt.arg.test_interval))
                logger.info('Start testing...')
                val_acc = self.eval(partition='val')

                is_best = 0

                if val_acc >= self.val_acc:
                    self.val_acc = val_acc
                    is_best = 1
                    logger.info('overall best triple accuary is {} at {} epoch'.format(self.val_acc, self.global_step / tt.arg.test_interval))

                self.save_checkpoint({
                    'iteration': self.global_step,
                    'enc_module_state_dict': self.enc_module.state_dict(),
                    'gnn_module_state_dict': self.gnn_module.state_dict(),
                    'cnn_module_state_dict': self.cnn_module.state_dict(),
                    'val_acc': val_acc,
                    'optimizer': self.optimizer.state_dict(),
                    }, is_best)
                time_span_test = (time.time() - start_time_epoch) / 60
                logger.info('epoch time consumption:%.2f(min)' % time_span_test)
                logger.info('====================================================================================================================================')
            tt.log_step(global_step=self.global_step)
            #print(time_span_iter)

    def eval(self, partition='test', log_flag=True):
        best_acc = 0
        # set edge mask (to distinguish support and query edges)
        num_supports_r = tt.arg.num_ways_train * tt.arg.num_shots_train
        num_supports_h = tt.arg.num_ways_train * tt.arg.num_shots_train
        num_supports_t = tt.arg.num_ways_train * tt.arg.num_shots_train
        num_supports = num_supports_r + num_supports_h + num_supports_t
        num_queries_e = 5
        num_queries = tt.arg.num_ways_train * 1
        num_samples = num_supports + num_queries_e
        support_edge_mask = torch.zeros(tt.arg.meta_batch_size, num_samples, num_samples).to(tt.arg.device)
        support_edge_mask[:, :num_supports, :num_supports] = 1
        query_edge_mask = 1 - support_edge_mask
        evaluation_mask = torch.ones(tt.arg.meta_batch_size, num_samples, num_samples).to(tt.arg.device)
        evaluation_mask[:, :num_supports, :num_supports] = 0


        query_entity_pair_accrs =[]
        triple_without_order_accs =[]
        triple_only_relation_accrs = []
        query_triple_accrs =[]
        query_losses = []

        # for each iteration
        for iter in range(tt.arg.test_iteration//tt.arg.test_batch_size):

            #start_time_iter = time.time()

            # load task data list
            [support_data_r, support_data_h, support_data_t, support_label_r,
             support_label_h, support_label_t, support_data_attention_mask, query_data,
             query_entity, query_entity_label, query_data_attention_mask] = self.data_loader[partition].get_task_batch(
                num_tasks=tt.arg.meta_batch_size,
                num_ways=tt.arg.num_ways_train,
                num_shots=tt.arg.num_shots_train,
                seed=iter + tt.arg.seed)

            # (1) encode data
            full_data_encoder = torch.cat([support_data_r, query_data], 1)
            full_attention_mask = torch.cat([support_data_attention_mask, query_data_attention_mask], 1)

            self.enc_module.train()
            full_data_encoder_result = self.enc_module(full_data_encoder, full_attention_mask)

            self.cnn_module.train()
            support_data_r_result, support_data_h_result, support_data_t_result, query_data_entity_result, query_data_entity_mask, query_label_e = self.cnn_module(
                full_data_encoder_result, support_data_h, support_data_t, query_entity, num_supports_r,
                query_entity_label, full_attention_mask)
            query_data_entity_result_reshaped = query_data_entity_result.view(tt.arg.meta_batch_size,
                                                                              num_queries * num_queries_e, -1)
            query_label_e_reshaped = query_label_e.view(tt.arg.meta_batch_size, num_queries * num_queries_e)
            full_data = torch.cat([support_data_r_result, support_data_h_result, support_data_t_result,
                                   query_data_entity_result_reshaped], 1).float()
            full_label = torch.cat([support_label_r, support_label_h, support_label_t, query_label_e_reshaped],
                                   1).float()
            full_edge, edge_M = self.label2edge(full_label, num_supports_r, num_supports_h, num_supports_t,
                                                num_queries_e)

            # set init edge and init edge_C

            init_edge = full_edge.clone().float()  # batch_size x 2 x num_samples x num_samples
            init_edge_M = edge_M.clone().float()  # batch_size x num_samples x num_samples
            init_edge_part_t, init_edge_part_h = self.Compute_P_dis(support_data_h_result, support_data_t_result,query_data_entity_result_reshaped,query_data_entity_mask,tt.arg.num_ways,tt.arg.num_shots,tt.arg.num_unkonw_entity )
            init_edge[:, 0, num_supports_r + num_supports_h + num_supports_t:, :num_supports_r] = init_edge_part_t
            init_edge[:, 0, :num_supports_r,num_supports_r + num_supports_h + num_supports_t:] = init_edge_part_t.transpose(1, 2)
            init_edge[:, 1, num_supports_r + num_supports_h + num_supports_t:, :num_supports_r] = init_edge_part_h
            init_edge[:, 1, :num_supports_r,num_supports_r + num_supports_h + num_supports_t:] = init_edge_part_h.transpose(1, 2)

            # set as train mode
            self.gnn_module.train()

            # (2) predict edge logit (consider only the last layer logit, num_tasks x 2 x num_samples x num_samples)
            if tt.arg.train_transductive:
                full_logit_layers,edge_C_result_list,node_feat_result_list = self.gnn_module(node_feat=full_data, edge_feat=init_edge, edge_C =init_edge_M,num_supports = num_supports,entity_mask = query_data_entity_mask)
            else:
                #evaluation_mask[:, num_supports:, num_supports:] = 0 # ignore query-query edges, since it is non-transductive setting,non-transductive表示query是一个图只推断一个query
                # input_node_feat: (batch_size x num_queries) x (num_support + 1) x featdim
                # input_edge_feat: (batch_size x num_queries) x 2 x (num_support + 1) x (num_support + 1)
                support_data = full_data[:, :num_supports_r + num_supports_h + num_supports_t] # batch_size x num_support x featdim
                query_data = full_data[:, num_supports_r + num_supports_h + num_supports_t:] # batch_size x num_query x featdim
                support_data_tiled = support_data.unsqueeze(1).repeat(1, num_queries, 1, 1) # batch_size x num_queries x num_support x featdim
                support_data_tiled = support_data_tiled.view(tt.arg.meta_batch_size * num_queries, num_supports, -1) # (batch_size x num_queries) x num_support x featdim
                query_data_reshaped = query_data.contiguous().view(tt.arg.meta_batch_size * num_queries, num_queries_e, -1) # (batch_size x num_queries) x 5 x featdim
                input_node_feat = torch.cat([support_data_tiled, query_data_reshaped], 1) # (batch_size x num_queries) x (num_support + num_entities) x featdim
                init_edge_support_temp = init_edge.split([num_supports,init_edge.size(2) - num_supports],dim = 2)
                init_edge_support = init_edge_support_temp[0].split([num_supports,init_edge.size(2) - num_supports],dim =3)[0]
                init_edge_query = init_edge_support_temp[1].split([num_supports_r,init_edge.size(2) - num_supports_r],dim =3)[0]
                init_edge_support_tailed = init_edge_support.unsqueeze(1).repeat(1,num_queries,1,1,1)
                init_edge_support_tailed = init_edge_support_tailed.view(tt.arg.meta_batch_size * num_queries, 3, num_supports, num_supports)
                init_edge_query_reshaped = init_edge_query.transpose(1,2).reshape(tt.arg.meta_batch_size * num_queries,num_queries_e,3,num_supports_r).transpose(1,2)
                input_edge_feat = torch.zeros(tt.arg.meta_batch_size * num_queries, 3, num_supports + num_queries_e, num_supports + num_queries_e).to(tt.arg.device) # batch_size x 2 x (num_support + 1) x (num_support + 1)
                input_edge_feat[:, :, :num_supports, :num_supports] = init_edge_support_tailed
                input_edge_feat[:, :, num_supports:, :num_supports_r] = init_edge_query_reshaped
                input_edge_feat[:, :, :num_supports_r, num_supports:] = init_edge_query_reshaped.transpose(3,2)

                # logit: (batch_size x num_queries) x 3 x (num_support + 5) x (num_support + 5)
                with torch.no_grad():
                    logit_results = self.gnn_module(node_feat=input_node_feat, edge_feat=input_edge_feat, num_supports = num_supports, entity_mask = query_data_entity_mask)
                logit_layers = logit_results[0]
                logit_nodes = logit_results[1]
                logit_layers = [logit_layer.view(tt.arg.meta_batch_size, num_queries, 3, num_supports + num_queries_e, num_supports + num_queries_e) for logit_layer in logit_layers]
                logit_nodes = [logit_node.view(tt.arg.meta_batch_size, num_queries,num_supports + num_queries_e,768) for logit_node in logit_nodes]
                # logit --> full_logit (batch_size x 2 x num_samples x num_samples)
                full_logit_layers = []
                full_logit_nodes = []
                for l in range(tt.arg.num_layers):
                    full_logit_layers.append(torch.zeros(tt.arg.meta_batch_size, 3, num_supports+num_queries_e*num_queries, num_supports+num_queries_e*num_queries).to(tt.arg.device)) #8*3*22*22
                    full_logit_nodes.append(torch.zeros(tt.arg.meta_batch_size,num_supports+num_queries_e*num_queries,768).to(tt.arg.device))
                for l in range(tt.arg.num_layers):
                    full_logit_layers[l][:, :, :num_supports, :num_supports] = logit_layers[l][:, :, :, :num_supports, :num_supports].mean(1)
                    full_logit_layers[l][:, :, :num_supports, num_supports:] = logit_layers[l][:, :, :, :num_supports, num_supports:].transpose(1, 2).transpose(2, 3).reshape(tt.arg.meta_batch_size,3,num_supports,-1)
                    full_logit_layers[l][:, :, num_supports:, :num_supports] = logit_layers[l][:, :, :, num_supports:, :num_supports].transpose(1, 2).reshape(tt.arg.meta_batch_size,3,-1,num_supports)
                    full_logit_nodes[l][:,:num_supports,:] = logit_nodes[l][:,:,:num_supports,:].mean(1)
                    full_logit_nodes[l][:, num_supports:,:] = logit_nodes[l][:,:,num_supports:,:].reshape(tt.arg.meta_batch_size,num_queries*num_queries_e,768)

            exist_entity_mask = query_data_entity_mask[:, :, :, 0].reshape(tt.arg.meta_batch_size,num_queries * num_queries_e, -1).repeat(1, 1,num_supports_r)
            entity_mask = torch.zeros(tt.arg.meta_batch_size, num_supports + num_queries_e * num_queries,num_supports + num_queries_e * num_queries).to(tt.arg.device)
            entity_mask[:, num_supports:, :num_supports_r] = exist_entity_mask
            entity_mask[:, :num_supports_r, num_supports:] = exist_entity_mask.transpose(1, 2)
            '''
            # (4.1)compute entity loss
            # (4.1) edge loss
            full_t_edge_loss_layers = [self.edge_loss(1 - (full_logit_layer[:, 0]), (1 - full_edge[:, 0])) for full_logit_layer in full_logit_layers]
            full_h_edge_loss_layers = [self.edge_loss((full_logit_layer[:, 1]), (full_edge[:, 1])) for full_logit_layer in full_logit_layers]
            # weighted edge loss for truth entity, wrong entity and none entity
            exist_entity_mask = query_data_entity_mask[:, :, :, 0].reshape(tt.arg.meta_batch_size,num_queries * num_queries_e, -1).repeat(1, 1, num_supports_r)
            entity_mask = torch.zeros(tt.arg.meta_batch_size, num_supports + num_queries_e * num_queries, num_supports + num_queries_e * num_queries).to(tt.arg.device)
            entity_mask[:, num_supports:, :num_supports_r] = exist_entity_mask
            entity_mask[:, :num_supports_r, num_supports:] = exist_entity_mask.transpose(1, 2)
            edge_M[:, 0, num_supports_r + num_supports_h:num_supports_r + num_supports_h + num_supports_t,:num_supports_r] = 0
            edge_M[:, 0, :num_supports_r, num_supports_r + num_supports_h:num_supports_r + num_supports_h + num_supports_t] = 0
            edge_M[:, 1, num_supports_r:num_supports_r + num_supports_h, :num_supports_r] = 0
            edge_M[:, 1, :num_supports_r, num_supports_r:num_supports_r + num_supports_h] = 0
            truth_t_entity_loss_layers = [torch.sum(full_t_edge_loss_layer * full_edge[:, 0] * entity_mask) / torch.sum(full_edge[:, 0] * entity_mask) for full_t_edge_loss_layer in full_t_edge_loss_layers]
            wrong_t_entity_loss_layers = [torch.sum(full_t_edge_loss_layer * (1 - full_edge[:, 0]) * entity_mask) / torch.sum((1 - full_edge[:, 0]) * entity_mask) for full_t_edge_loss_layer in full_t_edge_loss_layers]
            none_t_entity_loss_layers = [torch.sum(full_t_edge_loss_layer * (1 - entity_mask) * edge_M[:, 0]) / torch.sum((1 - entity_mask) * edge_M[:, 0]) for full_t_edge_loss_layer in full_t_edge_loss_layers]
            truth_h_entity_loss_layers = [torch.sum(full_h_edge_loss_layer * full_edge[:, 1] * entity_mask) / torch.sum(full_edge[:, 1] * entity_mask) for full_h_edge_loss_layer in full_h_edge_loss_layers]
            wrong_h_entity_loss_layers = [torch.sum(full_h_edge_loss_layer * (1 - full_edge[:, 1]) * entity_mask) / torch.sum((1 - full_edge[:, 1]) * entity_mask) for full_h_edge_loss_layer in full_h_edge_loss_layers]
            none_h_entity_loss_layers = [torch.sum(full_h_edge_loss_layer * (1 - entity_mask) * edge_M[:, 1]) / torch.sum((1 - entity_mask) * edge_M[:, 1]) for full_h_edge_loss_layer in full_h_edge_loss_layers]
            edge_loss_for_all_query_layers = [truth_t_entity_loss_layer + wrong_t_entity_loss_layer + none_t_entity_loss_layer + truth_h_entity_loss_layer + wrong_h_entity_loss_layer + none_h_entity_loss_layer for truth_t_entity_loss_layer, wrong_t_entity_loss_layer, none_t_entity_loss_layer, truth_h_entity_loss_layer, wrong_h_entity_loss_layer, none_h_entity_loss_layer in zip(truth_t_entity_loss_layers, wrong_t_entity_loss_layers, none_t_entity_loss_layers,truth_h_entity_loss_layers, wrong_h_entity_loss_layers, none_h_entity_loss_layers)]

            # (4.2) calculate TransE loss
            loss_transE_layers = []
            for l in range(tt.arg.num_layers):
                # get node embedding
                label_r = full_label[:, :num_supports_r]
                label_h = full_label[:, num_supports_r:num_supports_r + num_supports_h]
                label_t = full_label[:, num_supports_r + num_supports_h:]
                node_feat_r = full_logit_nodes[l][:, :num_supports_r, :]
                node_feat_h = full_logit_nodes[l][:, num_supports_r:num_supports_r + num_supports_h, :]
                node_feat_t = full_logit_nodes[l][:, num_supports_r + num_supports_h:, :]
                # get corrupted triple for support triples
                label_relation_type = ((label_r.reshape(tt.arg.meta_batch_size * num_supports_r, -1)) % 100 / 10).reshape(tt.arg.meta_batch_size, num_supports_r)
                copynum_for_h_and_t = (tt.arg.num_ways - 1) * tt.arg.num_shots * tt.arg.num_shots
                transE_loss_mean = []
                for q_i in range(node_feat_h.size(1)):
                    feat_h_one = node_feat_h[:, q_i, :].unsqueeze(1)
                    feat_t_one = node_feat_t[:, q_i, :].unsqueeze(1)
                    feat_r_reshape = node_feat_r.reshape(tt.arg.meta_batch_size, tt.arg.num_ways,tt.arg.num_shots, 768)
                    way_idx = int(q_i / tt.arg.num_shots)
                    feat_h_expand = feat_h_one.repeat(1, copynum_for_h_and_t, 1)
                    feat_t_expand = feat_t_one.repeat(1, copynum_for_h_and_t, 1)
                    # 得到正确的r
                    feat_r_for_right = feat_r_reshape[:, way_idx, :, :].unsqueeze(2).repeat(1, 1, (tt.arg.num_ways - 1) * tt.arg.num_shots, 1).reshape(tt.arg.meta_batch_size, (tt.arg.num_ways - 1) * tt.arg.num_shots * tt.arg.num_shots, 768)
                    # 得到错误的r
                    feat_r_combine_1 = feat_r_reshape[:, 0:way_idx, :, :]
                    feat_r_combine_2 = feat_r_reshape[:, way_idx + 1:, :, :]
                    feat_r_for_wrong = torch.cat((feat_r_combine_1, feat_r_combine_2), dim=1).repeat(1, tt.arg.num_shots, 1, 1).reshape(tt.arg.meta_batch_size, -1, 768)
                    # 计算三元组transE得分
                    dis_right_triple = feat_h_expand + feat_r_for_right - feat_t_expand
                    dis_wrong_triple = feat_h_expand + feat_r_for_wrong - feat_t_expand
                    score_right_triple = torch.norm(dis_right_triple, p=2, dim=-1)
                    score_wrong_tripe = torch.norm(dis_wrong_triple, p=2, dim=-1)
                    transE_loss = (1.5 + score_right_triple - score_wrong_tripe).reshape(-1, 1)
                    a = 0
                    transE_loss_for_one_pair = 0
                    for iii in range(transE_loss.size(0)):
                        if transE_loss[iii] > 0:
                            transE_loss_for_one_pair += transE_loss[iii]
                            a += 1
                    if a != 0:
                        transE_loss_for_one_pair_mean = transE_loss_for_one_pair / a  # 是不是应该加起来，而不是求平均？？
                    else:
                        transE_loss_for_one_pair_mean = transE_loss_for_one_pair
                    transE_loss_mean.append(transE_loss_for_one_pair_mean)
                transE_loss_mean_all = torch.mean(torch.tensor(transE_loss_mean).float())
                loss_transE_layers.append(transE_loss_mean_all)
            # (5)compute total loss
            total_loss_layers = [loss_transE_layer + 0.3 * edge_loss_for_all_query_layer for loss_transE_layer, edge_loss_for_all_query_layer in zip(loss_transE_layers, edge_loss_for_all_query_layers)]
            # update model
            total_loss = []
            for l in range(tt.arg.num_layers - 1):
                total_loss += [total_loss_layers[l].view(-1) * 0.5]
            total_loss += [total_loss_layers[-1].view(-1) * 1.0]
            total_loss = torch.mean(torch.cat(total_loss, 0))
            query_losses += [total_loss.item()]
            #print(iter, total_loss)
            '''
            # compute accuracy
            full_logit_layers_t_pred = [full_logit_layer[:, 0, :, :] * entity_mask for full_logit_layer in full_logit_layers]
            full_logit_layers_h_pred = [full_logit_layer[:, 1, :, :] * entity_mask for full_logit_layer in full_logit_layers]
            support_label_r_use = torch.floor_divide(support_label_r % 100, 10).float()
            query_node_t_pred_layers = [torch.bmm(full_logit_layer_t_pred[:, :, :num_supports_r],self.one_hot_encode(tt.arg.num_ways_train, support_label_r_use.long())) for full_logit_layer_t_pred in full_logit_layers_t_pred]
            query_node_h_pred_layers = [torch.bmm(full_logit_layer_h_pred[:, :, :num_supports_r],self.one_hot_encode(tt.arg.num_ways_train, support_label_r_use.long())) for full_logit_layer_h_pred in full_logit_layers_h_pred]
            entity_pair_accr_all, triple_accr_only_relation_all, triple_accr_without_order_all, triple_accr_all = self.pred_triple_test (query_node_t_pred_layers[-1], query_node_h_pred_layers[-1], num_supports, num_supports_r, full_label,num_queries_e)

            query_entity_pair_accrs += [entity_pair_accr_all / (tt.arg.test_batch_size * num_queries)]
            triple_only_relation_accrs+=[triple_accr_only_relation_all/(tt.arg.test_batch_size * num_queries)]
            triple_without_order_accs +=[triple_accr_without_order_all / (tt.arg.test_batch_size * num_queries)]
            query_triple_accrs += [triple_accr_all / (tt.arg.test_batch_size * num_queries)]

        # logging
        if log_flag:
            tt.log('---------------------------------------------------------')
            #tt.log_scalar('{}/query_loss'.format(partition), np.array(query_losses).mean(), self.global_step)
            tt.log_scalar('{}/entity_pair_accr'.format(partition), np.array(query_entity_pair_accrs).mean(), self.global_step)
            tt.log_scalar('{}/triple_only_relation_accr'.format(partition), np.array(triple_only_relation_accrs).mean(),self.global_step)
            tt.log_scalar('{}/triple_without_order_acc'.format(partition), np.array(triple_without_order_accs).mean(),self.global_step)
            tt.log_scalar('{}/triple_accr'.format(partition), np.array(query_triple_accrs).mean(), self.global_step)
            logger.info('===============================================================================================================================================================')
            logger.info(
                'entity_pair_accuary: %.4f, triple_only_relation_accr: %.4f, triple_without_order_acc: %.4f, triple_accuary: %.4f\n' %
                (np.array(query_entity_pair_accrs).mean(), np.array(triple_only_relation_accrs).mean(),np.array(triple_without_order_accs).mean(), np.array(query_triple_accrs).mean()))
            logger.info(
                'entity_pair_accuary_std: %.2f, triple_only_relation_accr_std : % .2f, triple_without_order_acc_std: %.2f, triple_accuary_std: %.2f\n' %
                (np.array(query_entity_pair_accrs).std(), np.array(triple_only_relation_accrs).std(), np.array(triple_without_order_accs).std(), np.array(query_triple_accrs).std()))
            tt.log('----------------------------------------------------------')

        return np.array(query_triple_accrs).mean()


    def adjust_learning_rate(self, optimizers, lr, iter):
        new_lr = lr * (0.5 ** (int(iter / tt.arg.dec_lr)))

        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr



    def label2edge(self, label,num_supports_r,num_supports_h,num_supports_t,num_queries_e):
        # get size
        num_samples = label.size(1)
        batch_size = label.size(0)
        # reshape
        label_i = label.unsqueeze(-1).repeat(1, 1, num_samples)
        edge_ij = torch.eq(label_i,label_j).float()
        # init edge pos_t
        edge_pos_t = torch.zeros(batch_size, num_samples, num_samples)
        edge_neg_h = torch.zeros(batch_size, num_samples, num_samples)
        edge_r = torch.zeros(batch_size, num_samples, num_samples)
        edge_pos_t_mask = torch.zeros(batch_size, num_samples, num_samples)
        edge_neg_h_mask = torch.zeros(batch_size, num_samples, num_samples)
        edge_r_mask = torch.zeros(batch_size, num_samples, num_samples)
        #set mask matrix
        edge_pos_t_mask[:, num_supports_r+num_supports_h:, :num_supports_r] = 1
        edge_pos_t_mask[:, :num_supports_r, num_supports_r+num_supports_h:] = 1
        edge_neg_h_mask[:, num_supports_r:num_supports_r+num_supports_h, :num_supports_r] = 1
        edge_neg_h_mask[:,num_supports_r+num_supports_h+num_supports_t:, :num_supports_r] = 1
        edge_neg_h_mask[:, :num_supports_r, num_supports_r:num_supports_r+num_supports_h] = 1
        edge_neg_h_mask[:, :num_supports_r,num_supports_r+num_supports_h+num_supports_t:] = 1
        edge_r_mask[:, :num_supports_r, :num_supports_r] = 1
        #set turth label matrix
        #(1)init edge_pos_t
        for num_batch in range(batch_size):
            for i in range(num_supports_r):
                edge_pos_t[num_batch, i, num_supports_r + num_supports_h + i] = 1
                edge_pos_t[num_batch, num_supports_r + num_supports_h + i, i] = 1
        edge_pos_t[:,num_supports_r+num_supports_h+num_supports_t:,:num_supports_r] = edge_ij[:,num_supports_r+num_supports_h+num_supports_t:,num_supports_r+num_supports_h:num_supports_r+num_supports_h+num_supports_t]
        edge_pos_t[:,:num_supports_r,num_supports_r+num_supports_h+num_supports_t:] = edge_ij[:,num_supports_r+num_supports_h:num_supports_r+num_supports_h+num_supports_t,num_supports_r+num_supports_h+num_supports_t:]
        edge_pos_t = edge_pos_t*edge_pos_t_mask
        #(2)init edge_neg_h
        for num_batch in range(batch_size):
            for i in range(num_supports_r):
                edge_neg_h[num_batch, i, num_supports_r + i] = 1
                edge_neg_h[num_batch, num_supports_r + i, i] = 1
        edge_neg_h[:,num_supports_r+num_supports_h+num_supports_t:,:num_supports_r] = edge_ij[:,num_supports_r+num_supports_h+num_supports_t:,num_supports_r:num_supports_r+num_supports_h]
        edge_neg_h[:,:num_supports_r,num_supports_r+num_supports_h+num_supports_t:] = edge_ij[:,num_supports_r:num_supports_r+num_supports_h,num_supports_r+num_supports_h+num_supports_t:]
        edge_neg_h = edge_neg_h * edge_neg_h_mask
        #(3)init edge_r_and_e
        for num_batch in range(batch_size):
            edge_r[num_batch,:num_supports_r,:num_supports_r] = torch.eq(label_i[num_batch,:num_supports_r,:num_supports_r],label_j[num_batch,:num_supports_r,:num_supports_r]).float()
        edge_r = edge_r*edge_r_mask
        # move to cuda
        edge_pos_t = edge_pos_t.to(tt.arg.device)
        edge_neg_h = edge_neg_h.to(tt.arg.device)
        edge_r = edge_r.to(tt.arg.device)
        edge_pos_t_mask = edge_pos_t_mask.to(tt.arg.device)
        edge_neg_h_mask = edge_neg_h_mask.to(tt.arg.device)
        edge_r_mask = edge_r_mask.to(tt.arg.device)
        # expand
        edge_pos_t = edge_pos_t.unsqueeze(1)
        edge_neg_h = edge_neg_h .unsqueeze(1)
        edge_r = edge_r.unsqueeze(1)
        edge_pos_t_mask = edge_pos_t_mask.unsqueeze(1)
        edge_neg_h_mask = edge_neg_h_mask.unsqueeze(1)
        edge_r_mask  = edge_r_mask.unsqueeze(1)
        #cat
        edge_temp = torch.cat([edge_pos_t, edge_neg_h], 1)
        edge = torch.cat([edge_temp,edge_r],1)
        edge_mask_temp = torch.cat([edge_pos_t_mask, edge_neg_h_mask],1)
        edge_mask = torch.cat([edge_mask_temp,edge_r_mask],1)
        return edge, edge_mask

    def Compute_P_dis(self,support_data_h_result, support_data_t_result, query_data_entity_result_reshaped,query_data_entity_mask,num_ways,num_shots,num_unkonw_entity):
        query_data_entity_reshaped = query_data_entity_result_reshaped.reshape(tt.arg.meta_batch_size * num_ways, num_unkonw_entity, -1)
        node_feat_head_entity_reshaped = support_data_h_result.unsqueeze(1).repeat(1,num_ways,1,1).reshape(tt.arg.meta_batch_size*num_ways, num_ways, num_shots,768)
        node_feat_tail_entity_reshaped = support_data_t_result.unsqueeze(1).repeat(1,num_ways,1,1).reshape(tt.arg.meta_batch_size*num_ways, num_ways, num_shots,768)
        node_feat_head_proto = torch.mean(node_feat_head_entity_reshaped, 2)
        node_feat_tail_proto = torch.mean(node_feat_tail_entity_reshaped, 2)
        head_proto_tailed = node_feat_head_proto.unsqueeze(2).repeat(1, 1, num_unkonw_entity, 1).view(tt.arg.meta_batch_size*num_ways,-1, 768)
        tail_proto_tailed = node_feat_tail_proto.unsqueeze(2).repeat(1, 1, num_unkonw_entity, 1).view(tt.arg.meta_batch_size*num_ways,-1, 768)
        entity_mask_reshape = query_data_entity_mask.reshape(tt.arg.meta_batch_size*num_ways, num_unkonw_entity, 768)
        entity_mask_tailed = entity_mask_reshape.unsqueeze(1).repeat(1, num_ways, 1, 1).view(tt.arg.meta_batch_size*num_ways, -1, 768)
        node_unknown_entity_tailed = query_data_entity_reshaped.unsqueeze(1).repeat(1, num_ways, 1, 1).view(tt.arg.meta_batch_size*num_ways, -1, 768)
        dis_unknown_entity_with_h = self.__dist__(head_proto_tailed,node_unknown_entity_tailed,2)
        dis_unknown_entity_with_t = self.__dist__(tail_proto_tailed,node_unknown_entity_tailed,2)
        P_dis_unknown_entity_with_h_1 = 1/ ( 1 + dis_unknown_entity_with_h)
        P_dis_unknown_entity_with_t_1 = 1/( 1 + dis_unknown_entity_with_t)
        P_dis_unknown_entity_with_h = P_dis_unknown_entity_with_h_1 * entity_mask_tailed[:,:,0]
        P_dis_unknown_entity_with_t = P_dis_unknown_entity_with_t_1 * entity_mask_tailed[:,:,0]

        P_dis_unknown_entity_with_head_normalized = F.normalize(P_dis_unknown_entity_with_h.reshape(tt.arg.meta_batch_size*num_ways, -1), p=1, dim=-1)
        P_dis_unknown_entity_with_tail_normalized = F.normalize(P_dis_unknown_entity_with_t.reshape(tt.arg.meta_batch_size*num_ways, -1), p=1, dim=-1)

        query_entity_with_t_1 = P_dis_unknown_entity_with_tail_normalized.reshape(-1,num_unkonw_entity,num_ways).transpose(1,2)
        query_entity_with_h_1 = P_dis_unknown_entity_with_tail_normalized.reshape(-1,num_unkonw_entity,num_ways).transpose(1,2)
        query_entity_with_t = query_entity_with_t_1.reshape(tt.arg.meta_batch_size,num_ways*num_unkonw_entity,num_ways).unsqueeze(3).repeat(1, 1, 1, num_shots).reshape(tt.arg.meta_batch_size,num_unkonw_entity*num_ways,num_ways*num_shots)
        query_entity_with_h = query_entity_with_h_1.reshape(tt.arg.meta_batch_size,num_ways*num_unkonw_entity,num_ways).unsqueeze(3).repeat(1, 1, 1, num_shots).reshape(tt.arg.meta_batch_size,num_unkonw_entity*num_ways,num_ways*num_shots)
        #query_entity_with_t = torch.cosine_similarity(query_data_entity_result_reshaped.unsqueeze(2), support_data_t_result.unsqueeze(1),dim = 3)
        #query_entity_with_h = torch.cosine_similarity(query_data_entity_result_reshaped.unsqueeze(2), support_data_h_result.unsqueeze(1),dim = 3)
        return query_entity_with_t, query_entity_with_h


    def pred_triple(self,logit,num_supports,num_supports_r,edge_C_result):
        N = logit.size(2)
        edge_logit = logit[:,0,:,:]
        #num_entities = N - num_supports
        index_num = list(range(N))
        entity_index = index_num[num_supports:]
        entity_combine = list(itertools .permutations(entity_index,2))
        score_arry = torch.zeros(logit.size(0),num_supports_r,len(entity_combine))
        #query_label = full_label[:,num_supports:]
        for r in range(num_supports_r):
            for j , entity_pair in enumerate(entity_combine):
                score = edge_logit[:,entity_pair[0],r]*edge_C_result[:,entity_pair[0],r]+edge_logit[:,r,entity_pair[1]]*edge_C_result[:,r,entity_pair[1]]+edge_logit[:,entity_pair[0],entity_pair[1]]*edge_C_result[:,entity_pair[0],entity_pair[1]]
                score_arry[:,r,j] = score

        h_pos = torch.zeros(logit.size(0))
        t_pos = torch.zeros(logit.size(0))
        h_pos_label = torch.zeros(logit.size(0))
        t_pos_label = torch.zeros(logit.size(0))
        batch_max_score = torch.zeros(logit.size(0))

        for B in range(logit.size(0)):
            batch_max = torch.max(score_arry[B,:,:])
            batch_max_pos = (score_arry[B,:,:] == batch_max).nonzero(as_tuple=False)
            rel_pos = batch_max_pos[0][0]
            h_p = entity_combine[batch_max_pos[0][1]][0]
            t_p = entity_combine[batch_max_pos[0][1]][1]
            h_p_label =int(str(rel_pos.item())+'1')
            t_p_label = int(str(rel_pos.item())+'2')
            h_pos[B] = h_p
            t_pos[B] = t_p
            h_pos_label[B] = h_p_label
            t_pos_label[B] = t_p_label
            batch_max_score[B] =  batch_max

        return batch_max_score,h_pos,t_pos,h_pos_label,t_pos_label

    def pred_triple_test(self, logit_t, logit_h, num_supports, num_supports_r, full_label,num_queries_e):
        N = logit_t.size(1)
        num_entities = N - num_supports
        num_class = int(num_entities/num_queries_e)
        edge_logit_t_with_r = logit_t # 4*2*5*4
        edge_logit_h_with_r = logit_h
        index_num = list(range(N))
        entity_index_tensor = torch.tensor(index_num[num_supports:]).reshape(num_class,num_queries_e)

        entity_pair_accr_all = 0
        triple_accr_without_order_all = 0
        triple_accr_only_relation_all = 0
        triple_accr_all = 0

        for c_i in range(num_class):   # c_i--> entity class
            entity_index_tensor_class = entity_index_tensor[c_i,:] # 5
            entity_combine = list(itertools.permutations(entity_index_tensor_class.numpy().tolist(), 2))
            score_arry = torch.zeros(logit_t.size(0), num_class, len(entity_combine))

            for r_i in range(num_class): #r_i-->relations
                edge_logit_h_with_r_class = edge_logit_h_with_r[:, :, r_i]
                edge_logit_t_with_r_class = edge_logit_t_with_r[:, :, r_i]
                for j, entity_pair in enumerate(entity_combine):
                    score = edge_logit_t_with_r_class[:,entity_pair[1]] * edge_logit_h_with_r_class[:, entity_pair[0]]
                    score_arry[:, r_i, j] = score
            #h_accr = 0
            #t_accr =0
            triple_accr = 0
            entity_pair_accr = 0
            triple_accr_without_order = 0
            triple_accr_only_relation = 0
            for B in range(score_arry.size(0)):
                score_arry_without_r = torch.sum(score_arry,dim = 1)
                entity_pair_for_batch_max = torch.max(score_arry_without_r[B,:])
                entity_pair_for_batch_max_pos = (score_arry_without_r[B,:] == entity_pair_for_batch_max).nonzero(as_tuple=False)
                h_p = entity_combine[entity_pair_for_batch_max_pos[0][0]][0]
                t_p = entity_combine[entity_pair_for_batch_max_pos[0][0]][1]
                h_p_label_without_r = 1
                t_p_label_without_r = 2
                h_p_label_truth_for_entity_pair = full_label[B,h_p]
                t_p_label_truth_for_entity_pair = full_label[B,t_p]
                h_p_label_truth_without_r = h_p_label_truth_for_entity_pair % 10
                t_p_label_truth_without_r = t_p_label_truth_for_entity_pair % 10
                #if (h_p_label_truth_without_r == 1 and  t_p_label_truth_without_r ==2) or (h_p_label_truth_without_r == 2 and  t_p_label_truth_without_r ==1):
                    #entity_pair_accr = entity_pair_accr + 1
                if (h_p_label_truth_without_r == 1 and t_p_label_truth_without_r == 2) :
                    entity_pair_accr = entity_pair_accr + 1
                entity_pair_for_batch_max_r = torch.max(score_arry[B, :, :])
                entity_pair_for_batch_max_r_pos = (score_arry[B, :, :] == entity_pair_for_batch_max_r).nonzero(as_tuple=False)
                h_p_for_r = entity_pair_for_batch_max_r_pos[0][0]
                t_p_for_r = entity_pair_for_batch_max_r_pos[0][0]
                h_p_r = entity_combine[entity_pair_for_batch_max_r_pos[0][1]][0]
                t_p_r = entity_combine[entity_pair_for_batch_max_r_pos[0][1]][1]
                h_p_label_truth = full_label[B, h_p_r]
                t_p_label_truth = full_label[B, t_p_r]
                h_p_r_label = int(str(h_p_for_r.item())+'1')
                t_p_r_label = int(str(t_p_for_r.item())+'2')
                if (h_p_label_truth == h_p_r_label and t_p_label_truth == t_p_r_label) or(h_p_label_truth ==  t_p_r_label and t_p_label_truth == h_p_r_label):
                    triple_accr_without_order = triple_accr_without_order + 1
                if (int(h_p_for_r.item())==int(h_p_label_truth %100/10)) and(int(t_p_for_r.item())==int(t_p_label_truth %100/10)):
                    triple_accr_only_relation = triple_accr_only_relation + 1
                if h_p_label_truth == h_p_r_label and t_p_label_truth == t_p_r_label:
                    triple_accr = triple_accr + 1
            entity_pair_accr_all = entity_pair_accr_all + entity_pair_accr
            triple_accr_without_order_all = triple_accr_without_order_all + triple_accr_without_order
            triple_accr_only_relation_all = triple_accr_only_relation_all + triple_accr_only_relation
            triple_accr_all = triple_accr_all + triple_accr

        return entity_pair_accr_all,  triple_accr_only_relation_all, triple_accr_without_order_all, triple_accr_all

    def hit(self, logit, label):
        pred = logit.max(1)[1]
        hit = torch.eq(pred, label).float()
        return hit

    def one_hot_encode(self, num_classes, class_idx):
        return torch.eye(num_classes)[class_idx].to(tt.arg.device)

    def save_checkpoint(self, state, is_best):
        torch.save(state, 'asset/checkpoints/{}/'.format(tt.arg.experiment) + 'checkpoint.pth.tar')
        if is_best:
            shutil.copyfile('asset/checkpoints/{}/'.format(tt.arg.experiment) + 'checkpoint.pth.tar',
                            'asset/checkpoints/{}/'.format(tt.arg.experiment) + 'model_best.pth.tar')

def set_exp_name():
    exp_name = 'D-{}'.format(tt.arg.dataset)
    exp_name += '_N-{}_K-{}'.format(tt.arg.num_ways, tt.arg.num_shots)
    exp_name += '_L-{}_B-{}'.format(tt.arg.num_layers, tt.arg.meta_batch_size)
    #exp_name += '_T-{}'.format(tt.arg.transductive)
    exp_name+='_Lr-{}'.format(tt.arg.lr)
    exp_name+='_Train_iters-{}'.format(tt.arg.train_iteration)
    exp_name+='_Test_iters-{}'.format(tt.arg.test_iteration / tt.arg.meta_batch_size)
    exp_name+='_Test_span-{}'.format(tt.arg.test_interval)
    exp_name += '_SEED-{}'.format(tt.arg.seed)
    return exp_name

if __name__ == '__main__':

    tt.arg.device = 'cuda:2' if tt.arg.device is None else tt.arg.device
    tt.arg.device = torch.device('cuda:2')
    # replace dataset_root with your own
    tt.arg.dataset_root = 'data'
    tt.arg.dataset = 'fewrel' if tt.arg.dataset is None else tt.arg.dataset
    tt.arg.num_ways = 5 if tt.arg.num_ways is None else tt.arg.num_ways
    tt.arg.num_shots = 1 if tt.arg.num_shots is None else tt.arg.num_shots
    tt.arg.num_unlabeled = 0 if tt.arg.num_unlabeled is None else tt.arg.num_unlabeled
    tt.arg.num_layers = 2 if tt.arg.num_layers is None else tt.arg.num_layers
    tt.arg.meta_batch_size = 4 if tt.arg.meta_batch_size is None else tt.arg.meta_batch_size
    tt.arg.seed = 222 if tt.arg.seed is None else tt.arg.seed
    tt.arg.num_gpus = 1 if tt.arg.num_gpus is None else tt.arg.num_gpus

    tt.arg.num_ways_train = tt.arg.num_ways
    tt.arg.num_ways_test = tt.arg.num_ways

    tt.arg.num_shots_train = tt.arg.num_shots
    tt.arg.num_shots_test = tt.arg.num_shots

    # model parameter related
    tt.arg.num_edge_features = 768
    tt.arg.num_node_features = 768
    tt.arg.hidden_size = 768
    tt.arg.num_relations = tt.arg.num_ways * tt.arg.num_shots
    tt.arg.num_head_entity = tt.arg.num_ways * tt.arg.num_shots
    tt.arg.num_tail_entity = tt.arg.num_ways * tt.arg.num_shots
    tt.arg.num_unkonw_entity = 5

    # train, test parameters
    tt.arg.train_iteration = 30000 if tt.arg.dataset == 'fewrel' else 200000 #100000
    tt.arg.test_iteration = 4000#10000
    tt.arg.test_interval = 1000 if tt.arg.test_interval is None else tt.arg.test_interval #5000
    tt.arg.train_loss_everycheck = 50
    tt.arg.test_batch_size = 4
    tt.arg.log_step = 1000 if tt.arg.log_step is None else tt.arg.log_step

    tt.arg.lr = 1e-3
    tt.arg.grad_clip = 5
    tt.arg.weight_decay = 1e-5
    tt.arg.dec_lr = 15000 if tt.arg.dataset == 'fewrel' else 30000
    tt.arg.dropout = 0.1 if tt.arg.dataset == 'fewrel' else 0.0

    tt.arg.experiment = set_exp_name() if tt.arg.experiment is None else tt.arg.experiment
    print(set_exp_name())
    #set random seed
    np.random.seed(tt.arg.seed)
    torch.manual_seed(tt.arg.seed)
    torch.cuda.manual_seed_all(tt.arg.seed)
    random.seed(tt.arg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    tt.arg.log_dir_user = tt.arg.log_dir if tt.arg.log_dir_user is None else tt.arg.log_dir_user
    tt.arg.log_dir = tt.arg.log_dir_user
    max_length = 128

    if not os.path.exists('asset/checkpoints'):
        os.makedirs('asset/checkpoints')
    if not os.path.exists('asset/checkpoints/' + tt.arg.experiment):
        os.makedirs('asset/checkpoints/' + tt.arg.experiment)
    if not os.path.exists(tt.arg.log_dir):
        os.makedirs(tt.arg.log_dir)
    logger = get_logger(tt.arg.log_dir)
    logger.info(set_exp_name())

    enc_module = EmbeddingSentences(hidden_size= tt.arg.hidden_size)

    gnn_module = GraphNetwork(in_features=tt.arg.hidden_size,
                              node_features=tt.arg.num_edge_features,
                              edge_features=tt.arg.num_node_features,
                              num_layers=tt.arg.num_layers,
                              dropout=tt.arg.dropout)

    cnn_module = CnnForEntity(hidden_size= tt.arg.hidden_size )


    if tt.arg.dataset == 'fewrel':
        train_loader = fewrelDataLoader(root=tt.arg.dataset_root, file_name='./data/fewrel_cut_train.json', max_length=max_length, partition='train')
        valid_loader = fewrelDataLoader(root=tt.arg.dataset_root, file_name='./data/fewrel_cut_test.json', max_length=max_length,partition='test')
    else:
        print('Unknown dataset!')

    data_loader = {'train': train_loader,
                   'val': valid_loader
                   }

    # create trainer
    trainer = ModelTrainer(enc_module=enc_module,
                           gnn_module=gnn_module,
                           cnn_module=cnn_module,
                           data_loader=data_loader)


    trainer.train()
