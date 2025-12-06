import torch
import numpy as np
from torch.nn.modules.module import Module
from torch.autograd import Variable
from utils import *
from sklearn.linear_model import LogisticRegression
from torch.nn.functional import softmax
import scipy.sparse as sp
from sklearn.metrics import f1_score, accuracy_score, precision_score
import random
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch.nn as nn
import math


class IBETGLF(Module):
    def __init__(self, save_path, save_step, epoch_num, embedding_dim, time_dim, hidden_dim, L, class_num, node_num, manifold, gpu, alpha, r, t, train_node_list, test_node_list, gamma1, gamma2):
        super(IBETGLF, self).__init__()
        self.save_path = save_path
        self.save_step = save_step
        self.epoch_num = epoch_num
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.L = L
        self.class_num = class_num
        self.node_num = node_num
        self.time_dim = time_dim
        self.gpu = gpu
        self.manifold = manifold
        self.alpha = alpha
        self.r = r
        self.t = t
        self.train_node_list = train_node_list
        self.test_node_list = test_node_list
        self.gamma1 = gamma1
        self.gamma2 = gamma2

        # Attribution Decoder
        self.W_1 = torch.nn.Parameter(torch.zeros(size=(self.hidden_dim, self.hidden_dim), dtype=torch.float64))
        torch.nn.init.xavier_uniform_(self.W_1.data, gain=1.414)
        self.b_1 = torch.nn.Parameter(torch.zeros(size=(self.hidden_dim, 1), dtype=torch.float64))
        torch.nn.init.xavier_uniform_(self.b_1.data, gain=1.414)
        self.W_2 = torch.nn.Parameter(torch.zeros(size=(self.hidden_dim, self.embedding_dim), dtype=torch.float64))
        torch.nn.init.xavier_uniform_(self.W_2.data, gain=1.414)
        self.b_2 = torch.nn.Parameter(torch.zeros(size=(self.embedding_dim, 1), dtype=torch.float64))
        torch.nn.init.xavier_uniform_(self.b_2.data, gain=1.414)

        # Attribution Transformation
        self.W_att = torch.nn.Parameter(torch.zeros(size=(self.embedding_dim, self.hidden_dim), dtype=torch.float64))
        torch.nn.init.xavier_uniform_(self.W_att.data, gain=1.414)

        # Space Transformation
        self.W_tr = torch.nn.Parameter(torch.zeros(size=(self.hidden_dim, self.hidden_dim), dtype=torch.float64))
        torch.nn.init.xavier_uniform_(self.W_tr.data, gain=1.414)

        # Transformer
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=8)
        self.transformer= torch.nn.TransformerEncoder(encoder_layer, num_layers=1)

        # time encoder
        self.time_encoder = TimeEncode(self.time_dim)

        # PH
        self.PH = torch.nn.Parameter(torch.zeros(size=(self.node_num, self.hidden_dim), dtype=torch.float64))
        torch.nn.init.xavier_uniform_(self.PH.data, gain=1.414)

        # Hyperbolic Graph Attention Network
        self.HGAT = HyperGAT(self.L, self.hidden_dim, self.manifold, self.node_num)

        # classification
        self.W_nc = torch.nn.Parameter(torch.zeros(size=(self.hidden_dim, self.class_num), dtype=torch.float64))
        torch.nn.init.xavier_uniform_(self.W_nc.data, gain=1.414)



    def forward(self, graphs, T):
        loss = 0
        H_euc = []
        H_hyp = []
        H_euc.append(torch.matmul(graph.ndata["feature"], self.W_att))
        H_hyp.append(self.manifold.expmap0(H_euc))
        H_hyp_temp_list = []
        loss = 0
        his_graph = []
        for t in range(T):
            graph = graphs[t]
            H_hyp_temp = self.HyperGAT(graph.to_dense(), H_hyp[-1])
            H_hyp_temp_list.append(H_hyp_temp)
            time_embedding = self.time_encoder(t)
            H_euc_temp = self.manifold.logmap0(torch.matmul(H_hyp_temp, self.W_tr)) + time_embedding.repeat(self.node_num, dim=0)
            his_graph.append(H_euc_temp)
            stack_his = torch.cat(his_graph, dim=1)
            stack_his = torch.cat([stack_his, self.PH+time_embedding.repeat(self.node_num, dim=0)]).view(self.node_num, t+2, -1)
            H_euc_t = self.transformer(stack_his)[:,-1,:].view(self.node_num, self.hidden_dim)
            H_euc.append(H_euc_t)
            H_hyp_t = manifold.expmap0(H_euc_t+torch.matmul(graph.ndata["feature"], self.W_att))
            H_hyp.append(H_hyp_t)
            if t+1 < T:
                loss_rec = graph_reconstruction(H_euc_t, H_hyp_t, graphs[t+1])
            else:
                loss_rec = 0

            if t >= 1:
                loss_con = graph_contrastive(H_euc[-1], H_euc[-2], H_hyp[-1], H_hyp[-2])
            else:
                loss_con = 0
            loss_sup = graph_supervised(H_hyp_temp, graph)
            loss = loss + loss_sup + gamma1 * loss_rec + gamma2 * loss_con

        PRE, ACC, F1 = evaluation(H_hyp_temp_list, T, graphs)
        return loss

    def graph_reconstruction(self, H_euc, H_hyp, graph):
        X_pre = F.sigmoid(torch.matmul(H_euc, self.W_1) + self.b_1)
        X_pre = torch.matmul(X_pre, self.W_2) + self.b_2
        mse_loss = nn.MSELoss()
        loss_nf = mse_loss(X_pre, graph.ndata["feature"])

        A_pre = torch.zeros(size=(self.node_num, self.node_num), dtype=torch.float64)
        for i in range(node_num):
            for j in range(node_num):
                A_pre[i][j] = torch.pow(torch.exp((torch.pow(self.manifold.sqdist(H_hyp[i], H_hyp[i]), 2) - self.r)/self.t + 1), -1)
        ind = torch.where(graph.to_dense() > 0, 1, 0)
        loss_ts = mse_loss(torch.mul(ind, A_pre-graph.to_dense()))

        return loss_nf + self.alpha * loss_ts

    def graph_contrastive(self, euc_emb2, euc_emb1, hyp_emb2, hyp_emb1):
        pos_loss = self.manifold.sqdist(hyp_emb2, self.manifold.expmap0(euc_emb1)) + mse_loss(euc_emb2, self.manifold.logmap0(hyp_emb1))
        indices = torch.randperm(euc_emb2.size(0))
        neg_loss = 0
        for _ in range(5):
            neg_loss = neg_loss + self.manifold.sqdist(hyp_emb2, self.manifold.expmap0(euc_emb1)[indices]) + mse_loss(euc_emb2,self.manifold.logmap0(hyp_emb1)[indices])
        return pos_loss + neg_loss

    def graph_supervised(self, H, graph):
        Y_pre = F.softmax(torch.matmul(self.manifold.logmap0(H), self.W_nc))
        loss = nn.CrossEntropyLoss(Y_pre[self.train_node_list], graph.ndata["class"][self.train_node_list])
        return loss

    def evaluation(self, H_list, T, graphs):
        with torch.no_grad():
            PRE = 0
            ACC = 0
            F1 = 0
            for t in range(T):
                Y_pre = F.softmax(torch.matmul(self.manifold.logmap0(H_list[t]), self.W_nc))[self.test_node_list]
                _,Y_pre = torch.max(Y_pre, dim=1)
                Y_true = graphs[t].ndata["class"][self.test_node_list]
                ACC = ACC + accuracy_score(Y_true, Y_pre)
                F1 = F1 + f1_score(Y_true, Y_pre)
                PRE = PRE + precision_score
        return PRE/T, ACC/T, F1/T

class TimeEncode(torch.nn.Module):
    def __init__(self, expand_dim):
        super(TimeEncode, self).__init__()
        self.time_dim = expand_dim
        self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, 2*self.time_dim))).float())

    def forward(self, ts):
        time_embedding =  []
        for _ in self.time_dim:
            time_embedding.append(ts)
            time_embedding.append(ts + math.pi)
        time_embedding = torch.from_numpy(np.array(time_embedding))
        cos_ts = ts * self.basis_freq
        harmonic = torch.cos(cos_ts)
        return harmonic

class HyperGAT(torch.nn.Module):
    def __init__(self, L, hidden_dim, manifold, node_num):
        super(HyperGAT, self).__init__()
        self.L = L
        self.hidden_dim = hidden_dim
        self.att_W = []
        self.node_num = node_num
        self.manifold = manifold
        for i in range(L):
            self.W = torch.nn.Parameter(torch.zeros(size=(self.hidden_dim, self.hidden_dim), dtype=torch.float64))
            torch.nn.init.xavier_uniform_(self.W_a.data, gain=1.414)
            self.att_W.append(W)

    def forward(self, H, A):
        for l in range(L):
            att_score = Variable(torch.zeros(node_num,node_num))
            for i in range(node_num):
                for j in range(node_num):
                    Hi = self.manifold.mobius_mat(self.att_W[l],H[i])
                    Hj = self.manifold.mobius_mat(self.att_W[l],H[j])
                    att_score[i][j] = self.manifold.sqdist(Hi, Hj)
            att_score = torch.where(A > 0, att_score, -10000)
            att_score = F.softmax(att_score, dim=1)
            att_score = torch.where(A > 0, att_score, 0)
            H = torch.matmul(att_score, H)
        return H

















