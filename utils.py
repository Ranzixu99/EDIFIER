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
from bio_embeddings.embed import ProtTransBertBFDEmbedder
from get_seq import get_seq
import pickle
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
    d1 , d2 = NACCESS.run_naccess(struct[0],pdb_path, naccess="/root/autodl-nas/naccess/naccess",temp_path="/root/autodl-nas/naccess")

    for i in d1:
        data = i.strip().split()
        if data[0] == "RES":
            feature = np.array([ float(item) for item in data[3:]])
            naccess_list.append(feature)
    sr = ShrakeRupley()
    sr.compute(struct, level="R")
    for chain in struct[0]:
        for res in chain:
            sasa_list.append(res.sasa)
            # print(res.get_resname(), res.sasa)

    dssp = DSSP(struct[0], pdb_path)
    # (dssp index, amino acid, secondary structure, relative ASA, phi, psi,
    # NH_O_1_relidx, NH_O_1_energy, O_NH_1_relidx, O_NH_1_energy,
    # NH_O_2_relidx, NH_O_2_energy, O_NH_2_relidx, O_NH_2_energy)
    for i in dssp.keys():
        dssp_list.append(dssp[i][3:])

    hse = HSExposureCB(struct[0])
    for i in hse.property_keys:
        """comment
        >>> (21, 10, 0.0)
        >>> hse_u, hse_d, angle
        """
        hse_list.append(hse[i])
        # print(hse[i])
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
    # embedder = ProtTransBertBFDEmbedder()

    #pdb_seq = get_seq(pdb_path=pdb_path)
    #embedding = embedder.embed(pdb_seq)
    embedding = pickle.load(open(f"./bert_feature/{os.path.basename(pdb_path)[:-4]}.pkl","rb"))
    #print(embedding.shape) # 1024  , [  seq_length , 1024]
    # ------#B
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
        node_six_feature = [*dssp_list[e]]
        # naccess 11:26 27 28 29 30 31 32 33 34 35 36 
        node_naccess_feature = [*naccess_list[e]]

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
    nodes , edges = getProteinNodeAndEdge(pdb_path)
    # convert to tensor , node
    x = torch.tensor(nodes).float()
    # convert to tensor , edge
    edges = torch.tensor(edges).long() -1
    d = Data(x=x, edge_index=edges.contiguous(),t=int(label))
    return d
