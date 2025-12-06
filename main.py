import time
import torch
import numpy as np
import argparse
import os
from utils import *
from model import IBETGLF
from torch.optim import SGD, Adam
import random
from poincare import *
from euclidean import *



# Initiation
parser = argparse.ArgumentParser()
parser.add_argument('--path', default="../dataset/", help='Directory of the data file')
parser.add_argument('--save-path', default="./result/", help='Directory of the trained model')
parser.add_argument('--dataset', default="Wiki", help='Name of the network/dataset')
parser.add_argument('--gpu', default=-1, type=int, help='ID of the GPU to run on. If set to -1, CPU will be chosen')
parser.add_argument('--epochs', default=20, type=int, help='Number of epochs to train the model')
parser.add_argument('--embedding-dim', default=128, type=int, help='Number of dimensions of the node embedding')
parser.add_argument('--hidden-dim', default=128, type=int, help='Number of dimensions of the hidden embedding')
parser.add_argument('--time-dim', default=128, type=int, help='Number of dimensions of the time embedding')
parser.add_argument('--lr', default=0.001, type=float, help='Number of learning rate')
parser.add_argument('--leakyrelu-rate', default=0.2, type=float, help='rate of leakyrelu function')
parser.add_argument('--save-step', default=100, type=int, help='Saving model after step')
parser.add_argument('--optimization', default="Adam", help='Model optimizer')
parser.add_argument('--gpu', default=1, help='GPU Availability')
parser.add_argument('--weight-decay', default="0.001", type=float, help='L2 regularization strength')
parser.add_argument('--c', default=1, type=float, help='The radius of the poincareball')
parser.add_argument('--L', default=2, type=float, help='The layer number of the HGAT')
parser.add_argument('--alpha', default=0.5, type=float, help='The loss weight')
parser.add_argument('--r', default=1, type=float, help='Adjacency Reconstruction')
parser.add_argument('--t', default=1, type=float, help='Adjacency Reconstruction')
parser.add_argument('--gamma1', default=0.5, type=float, help='The loss weight')
parser.add_argument('--gamma2', default=0.5, type=float, help='The loss weight')
args = parser.parse_args()

# Set GPU
if args.gpu >= 0:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5"
    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda:"+str(args.gpu))

# Load Data
graphs, T = load_data(args.path, args.dataset)
print("Data loading finish!")

# Manifold
manifold = PoincareBall(args.c)


# Split Train_set, Test_set
nodes_num = graphs[0].num_nodes()
train_nodes = graphs[0].nodes()[:int(edges_num*0.9)]
test_nodes = graphs[0].nodes()[int(edges_num*0.9):]
class_num = graphs[0].ndata["class"].size()[1]


# Model Initiation
if args.gpu >= 0:
    model = IBETGLF(save_path=args.save_path,
                 save_step=args.save_step,
                 epoch_num=args.epochs,
                 embedding_dim=args.embedding_dim,
                 time_dim=args.time_dim,
                 hidden_dim=args.hidden_dim,
                 L=args.L,
                 class_num=class_num,
                 node_num=len(node_set),
                 manifold=manifold,
                 gpu=args.gpu,
                 alpha=args.alpha,
                 r=args.r,
                 t=args.t,
                 train_node_list=train_nodes,
                 test_node_list=test_nodes,
                 gamma1=args.gamma1,
                 gamma2=args.gamma2).to(device)
else:
    model = IBETGLF(save_path=args.save_path,
                    save_step=args.save_step,
                    epoch_num=args.epochs,
                    embedding_dim=args.embedding_dim,
                    time_dim=args.time_dim,
                    hidden_dim=args.hidden_dim,
                    L=args.L,
                    class_num=class_num,
                    node_num=len(node_set),
                    manifold=manifold,
                    gpu=args.gpu,
                    alpha=args.alpha,
                    r=args.r,
                    t=args.t,
                    train_node_list=train_nodes,
                    test_node_list=test_nodes,
                    gamma1=args.gamma1,
                    gamma2=args.gamma2)

# optimizer
optimization = Adam(lr=args.lr, params=model.parameters(), weight_decay=args.weight_decay)
print("Initiation finish!")

# Train Starting
start_time = time.time()
print("Start Training Time:{}   Total Epoch Number:{}".format(time.asctime(time.localtime(start_time)), args.epochs))
best_epoch = -1
best_performance_PRE = -1
best_performance_AUC = -1
best_performance_F1 = -1

for epoch in range(args.epochs):
    # Train
    random.shuffle(train_nodes)
    model.train()
    optimization.zero_grad()
    if args.gpu >= 0:
        loss, PRE, ACC, F1 = model(graphs, T).to(device)
    loss.backward()
    optimization.step()

    # Test
    if epoch % 5 == 0 and epoch != 0:
        model.eval()
        print("Test_Epoch: {}  PRE: {:.4f}  AUC: {:.4f}  F1: {:.4f}".format(epoch, PRE, AUC, F1))
        print("---------Train_epoch:{}  Loss:{:.4f}---------".format(epoch, loss.item()))
        if best_performance_PRE < PRE:
            best_performance_PRE = PRE
            best_performance_AUC = AUC
            best_performance_F1 = F1
            best_epoch = epoch
            torch.save(model.state_dict(), args.save_path+args.dataset+"_epoch_"+str(epoch)+"_model")

    print("Best_Epoch: {}  PRE: {:.4f}  AUC: {:.4f}  F1: {:.4f}".format(best_epoch, best_performance_PRE, best_performance_AUC, best_performance_F1))
end_time = time.time()
print("End Training Time:" + time.asctime(time.localtime(end_time)))
print("Total Training Time Cost:"+str(end_time-start_time)+"s")
