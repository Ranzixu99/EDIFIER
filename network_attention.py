import torch
import torch.nn.functional as F
from torch_geometric.data import Data,DataLoader
from torch_geometric.nn import GATConv,GCNConv
from torch_geometric.nn import global_mean_pool as gap
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import  plotLoss , ConvertInput
import pickle


class AttentionClassifier(torch.nn.Module):
    def __init__(self,num_classes = 2 , in_channels = 32, attention_num = 37):
        super(AttentionClassifier, self).__init__()
        self.linear1 = torch.nn.Linear(in_channels,attention_num)# 64 : x , 32 : y
        self.class_head = torch.nn.Linear(attention_num,num_classes) # 32 : y , 2 : z 

    def get_feature(self,x):
        x = self.linear1(x)
        att = torch.matmul(x.T,x)
        soft_att = torch.softmax(att , dim=1)
        return soft_att

    def forward(self,x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.class_head(x)
        return x


class FirstPath(torch.nn.Module):
    def __init__(self,select_feature = None  , out_channels = 32):
        super(FirstPath, self).__init__()
        self.select_feature = select_feature
        self.node_feature_num = len(select_feature)
        self.conv1 = GCNConv(self.node_feature_num, 32)  
        self.conv2 = GCNConv(32, 32)  
        self.conv3 = GCNConv(32, 48)
        self.conv4 = GCNConv(48, 64)
        self.conv5 = GCNConv(64, 96)
        self.conv6 = GCNConv(96, 128)
        self.linear1 = torch.nn.Linear(128,64) 
        self.linear2 = torch.nn.Linear(64,out_channels) 

    def forward(self, data):
        batch = data.batch
        x, edge_index = data.x, data.edge_index
        pdb_feature =  x[:,self.select_feature]
        x = pdb_feature
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = self.conv5(x, edge_index)
        x = F.relu(x)
        x = self.conv6(x, edge_index)
        x = F.relu(x)
        x = gap(x, batch = batch)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x
    
class SecPath(torch.nn.Module):
    def __init__(self,feature_num, output_channels = 32):
        super(SecPath, self).__init__()
        self.linear1 = torch.nn.Linear(feature_num,32)
        self.linear2 = torch.nn.Linear(32,64)
        self.linear3 = torch.nn.Linear(64,128)
        self.linear4 = torch.nn.Linear(128,64)
        self.linear5 = torch.nn.Linear(64,output_channels)

    def forward(self,data):
        batch = data.batch
        x = data.x
        bert_feature = x[:,37:]
        x = gap(bert_feature,batch = batch)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.linear4(x)
        x = F.relu(x)
        x = self.linear5(x)
        x = F.relu(x)
        return x 


class Net(torch.nn.Module):
    # all net
    def __init__(self, bert_feature_num = 1024, select_feature= [ 0,1],fusion_mode = "attention"):
        super(Net, self).__init__()
        self.pdb_path = FirstPath(select_feature =select_feature) # 37 feature
        self.bert_path = SecPath(bert_feature_num) # 1024
        self.fusion_mode = fusion_mode
        self.attention_num = len(select_feature)
        if self.fusion_mode =="attention":
            # attention fusion  ->pdb feature and bert feature
            self.fusion_layer = torch.nn.MultiheadAttention(32,4, )
            self.classifier = AttentionClassifier(num_classes=2 , in_channels=32 , attention_num=self.attention_num)
        elif self.fusion_mode =="linear":
            # MLP fusion -> pdb feature and bert feature
            self.fusion_layer = torch.nn.Linear(32+32 , 32)
            self.classifier = AttentionClassifier(num_classes=2 , in_channels= 32, attention_num=self.attention_num)
        else:
            # concat fusion -> pdb feature and bert feature
            self.classifier = AttentionClassifier(num_classes=2 , in_channels= 32+32, attention_num=self.attention_num)

    def get_relative(self, t):
        x  = self.pdb_path(t)
        x1 = self.bert_path(t)
        if self.fusion_mode =="attention":
            x = x.unsqueeze(0)
            x1 = x1.unsqueeze(0)
            fusion_feature,_ = self.fusion_layer(x,x,x1,need_weights=False)
            fusion_feature = fusion_feature.squeeze(0)
            att = self.classifier.get_feature(fusion_feature)
            return att
        elif self.fusion_mode =="linear":
            input = torch.cat([x,x1],dim=1)
            fusion_feature = self.fusion_layer(input)
            att = self.classifier.get_feature(fusion_feature)
            return att
        else:
            input = torch.cat([x,x1],dim=1)
            fusion_feature = input
            att = self.classifier.get_feature(fusion_feature)
            return att

    def forward(self, t):
        x  = self.pdb_path(t)
        x1 = self.bert_path(t)
        if self.fusion_mode =="attention":
            x = x.unsqueeze(0)
            x1 = x1.unsqueeze(0)
            fusion_feature,_ = self.fusion_layer(x,x,x1,need_weights=False)
            fusion_feature = fusion_feature.squeeze(0)
            output = self.classifier(fusion_feature)
            return output
        elif self.fusion_mode == "linear":
            input = torch.cat([x,x1],dim=1)
            fusion_feature = self.fusion_layer(input)
            output = self.classifier(fusion_feature)
            return output
        else:
            input = torch.cat([x,x1],dim=1)
            fusion_feature = input
            output = self.classifier(fusion_feature)
            return output
def get_dataset(mode = "train"):
    data_list = []
    N_P = []
    with open(f"./ann/{mode}_ann.txt","r")as f:
        N_P = f.readlines()
    for i in N_P:
        i = i.strip()
        dirname = os.path.basename(os.path.dirname(i))
        if dirname == "N":
            d = ConvertInput(i , 0)
        else:
            d = ConvertInput(i , 1)
        data_list.append(d)
    return data_list
