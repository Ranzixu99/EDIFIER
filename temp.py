import glob
from Bio.PDB import PDBParser
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
    '''This function is used for obtaining sequence from pdb file.'''
    p = PDBParser()
    s = p.get_structure('1d3z', pdb_path)
    seq = ""
    for i in s.get_residues():
        res_name = i.get_resname()
        seq += aa_codes[res_name]
    return seq

if __name__ =="__main__":
    N_dataset = glob.glob("./dataset/train_data/N/*")
    P_dataset = glob.glob("./dataset/train_data/P/*")
    index = 0
    with open("train_P.fasta","w") as f:
        for  n in  P_dataset:
            seq = get_seq(pdb_path=n)
            f.write(f">id{index}\n{seq}\n")
            index = index +1
