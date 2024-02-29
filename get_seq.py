from Bio.PDB.PDBParser import PDBParser

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


