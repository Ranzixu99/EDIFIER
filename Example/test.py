import argparse
import torch
from graphein.protein.graphs import construct_graph
from graphein.protein.edges.distance import add_k_nn_edges
from functools import partial
import torch
from torch_geometric.data import Data
from graphein.protein.config import ProteinGraphConfig
import numpy as np
import matplotlib.pyplot as plt
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
from Bio.PDB import DSSP
from Bio.PDB import HSExposureCB
from Bio.PDB import NACCESS
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from bio_embeddings.embed.prottrans_bert_bfd_embedder import ProtTransBertBFDEmbedder
from Bio.PDB.PDBParser import PDBParser
import torch
import torch.nn.functional as F
from torch_geometric.data import Data,DataLoader
from torch_geometric.nn import GATConv,GCNConv
from torch_geometric.nn import global_mean_pool as gap
import numpy as np
import matplotlib.pyplot as plt
import os




aa_codes = {
    'ALA': 'A',
    'CYS': 'C',
    'ASP': 'D',
    'GLU': 'E',
    'PHE': 'F',
    'GLY': 'G',
    'HIS': 'H',
    'LYS': 'K',
    'ILE': 'I',
    'LEU': 'L',
    'MET': 'M',
    'ASN': 'N',
    'PRO': 'P',
    'GLN': 'Q',
    'ARG': 'R',
    'SER': 'S',
    'THR': 'T',
    'VAL': 'V',
    'TYR': 'Y',
    'TRP': 'W',
}


def get_seq(pdb_path):
    '''
    get the seq using for protein_bert
    >>> return the seq from pdb file
    '''
    p = PDBParser()
    s = p.get_structure('1d3z', pdb_path)
    seq = ""
    for i in s.get_residues():
        res_name = i.get_resname()
        seq += aa_codes[res_name]
    return seq



def _get_four_feature(pdb_path):
    sasa_list = []
    hse_list = []
    dssp_list = []
    naccess_list = []
    p = PDBParser(QUIET=1)
    '''
    >>> strcture
    '''
    struct = p.get_structure("1LCD", pdb_path)

    sr = ShrakeRupley()
    sr.compute(struct, level="R")
    for chain in struct[0]:
        for res in chain:
            sasa_list.append(res.sasa)
            # print(res.get_resname(), res.sasa)

    hse = HSExposureCB(struct[0])
    for i in hse.property_keys:
        """comment
        >>> (21, 10, 0.0)
        >>> hse_u, hse_d, angle
        """
        hse_list.append(hse[i])
    return sasa_list, hse_list, dssp_list, naccess_list

def getProteinNodeAndEdge(pdb_path):
    '''
    based on Residuce layer.

    create the graph of protein and generate the input tensor of protein
    >>> return :
        nodes : Protein of node information
        edge_info : Protein of edge information
    '''
    # key : value , id muse be same
    sasa_list, hse_list, dssp_list  ,naccess_list= _get_four_feature(pdb_path=pdb_path)
    embedder = ProtTransBertBFDEmbedder(model_directory="./prottrans_bert_bfd")

    pdb_seq = get_seq(pdb_path=pdb_path)
    embedding = embedder.embed(pdb_seq)
    #print(embedding.shape) # 1024
    # ------#
    ## create the graph part
    # ------#
    new_edge_funcs = {
        "edge_construction_functions": [
            partial(add_k_nn_edges, k=20, long_interaction_threshold=0)
        ]
    }
    config = ProteinGraphConfig(**new_edge_funcs)
    # obtain the protein graph
    g = construct_graph(config=config, pdb_path=pdb_path)
    node_info = []
    edge_s = []
    edge_e = []
    # node part
    # node -> residucal j
    for e, d in enumerate(g.nodes(data=True)):
        node_num = d[0].split(':')[-1]
        # coords 3:0 1 2
        node_coords = np.array(d[1]['coords'])
        # b_factor 1:3
        node_factor = [float(d[1]['b_factor'])]
        # meiler 7 :4 5 6 7 8 9 10
        node_meiler = np.array([float(d[1]['meiler'][i]) for i in range(7)])
        # sasa 1:11
        node_four_feature = [float(sasa_list[e])]
        # hse 3:12 13 14
        node_five_feature = [*hse_list[e]]
        # dssp feature 11:15 16 17 18 19 20 21 22 23 24 25 
        node_six_feature = [0 for i in range(11)]
        # naccess 11:26 27 28 29 30 31 32 33 34 35 36 
        node_naccess_feature = [0 for i in range(11)]

        # feature concatenate
        node_s_info = np.concatenate(
            (
                node_coords,
                node_factor,
                node_meiler,
                node_four_feature,
                node_five_feature,
                node_six_feature,
                # naccess feature
                node_naccess_feature,
            )
        )

        # node_s_info = node_coords
        node_info.append(node_s_info)

    # edge part
    for u, v, d in g.edges(data=True):
        # start
        edge_start = u.split(":")[-1]
        # end
        edge_end = v.split(":")[-1]
        edge_s.append(int(edge_start))
        edge_e.append(int(edge_end))

    edge_info = np.array([edge_s, edge_e])
    nodes = np.array(node_info)
    # bert  feature
    nodes = np.concatenate((nodes , embedding) , axis = 1)
    return nodes, edge_info


def plotLoss(output:str , title:str , loss: list ,y_label :str):
    '''
    >>> Plot the line Loss in train parse.
    '''
    plt.clf()
    if(len(loss) == 0):
        return
    x = range(len(loss))

    plt.plot(x,loss,label = y_label)
    plt.xlabel("iteration")
    plt.ylabel(f"value")
    plt.title(title)
    plt.legend()
    plt.savefig(output)

def ConvertInput(pdb_path : str , label : int) :
    '''
    >>> label : 0 is N , 1 is P
    >>> return:
        d : Tensor of Protein information including nodes and edges.
    '''
    # get the protein info
    nodes , edges =  getProteinNodeAndEdge(pdb_path=pdb_path)
    # convert to tensor , node
    x = torch.tensor(nodes).float()
    # convert to tensor , edge
    edges = torch.tensor(edges).long() -1
    d = Data(x=x, edge_index=edges.contiguous(),t=int(label))
    return d


#------------------------#
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
    def __init__(self, bert_feature_num = 1024, select_feature= [0,1],fusion_mode = "attention"):
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
#------------------------#

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


if __name__ =="__main__":
    '''
    >>> Run script example : 
            python ./test.py -p example.pdb -c best.pth
    '''
    args = argparse.ArgumentParser()
    args.add_argument("-r", "--rank" , default="cpu", help="use cpu or cuda")
    args.add_argument("-p", "--pdb_path" , default="t.pdb", help="the path of pdb file")
    args.add_argument("-c", "--checkpoint" , default="c.pth", help="Checkpoint path")
    config = args.parse_args()
    classes = ["N", "P"]
    pdb = config.pdb_path
    # Read input of protein 
    # convert tensor
    input = ConvertInput(pdb , 1)
    # Loading the checkpoint for model
    checkpoint_path = config.checkpoint
    checkpoint =  torch.load(checkpoint_path , map_location="cpu")
    model = Net(select_feature=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],bert_feature_num=1024,fusion_mode="linear")
    # Network is converted to eval model
    model.eval()
    model.load_state_dict(checkpoint , strict= True)
    with torch.no_grad():

        if not config.rank == "cpu":
            model = model.cuda()
            input = input.cuda()
        # Forward Part
        output = model(input)
        # Inference Part
        result = torch.argmax(output , dim =1)
        prediction = classes[result.item()]
        # Print the Output of network
        print(pdb , "category is:", prediction)
        score =torch.softmax(output,dim =1)
        score = score.cpu().numpy()[0]
        print("N score:",score[0])
        print("P score:",score[1])
