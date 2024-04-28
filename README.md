# EDIFIER: structure-aware secretion system effector protein prediction using geometric deep learning and pretrained language model
The Type III secretion system (T3SS) plays a pivotal role in host-pathogen interactions by mediating the secretion of Type III secretion system effector proteins (T3SEs) into host cells. These T3SEs mimic host cell protein functions, influencing interactions between gram-negative bacterial pathogens and their hosts. Identifying T3SEs is essential in biomedical research for comprehending bacterial pathogenesis and its implications on human cells. This study presents EDIFIER, a novel multi-channel model designed for accurate T3SE prediction. It incorporates a graph structure channel, utilising Graph Convolutional Neural Networks (GCN) to capture tertiary structure information, and a sequence channel based on the ProteinBERT pretrained model to extract the sequence profiles of T3SEs. Rigorous benchmarking, including ablation studies and comparative analysis, validates that EDIFIER outperforms current state-of-the-art tools in T3SEs prediction. To enhance EDIFIER’s accessibility to the broader scientific community, we have developed a user-friendly webserver, accessible at http://edifier.unimelb-biotools.cloud.edu.au/.
We anticipate that EDIFIER will contribute to the field by providing reliable T3SEs predictions, thereby advancing our understanding of host-pathogen dynamics.
## Data
The original dataset is a file in Fasta format, and PDB format files can be obtained based on the information contained within it. Some can be downloaded from the UniProt database based on the sequence ID, or the corresponding PDB file can be predicted in ESMFold.
## Environment
- Ubuntu
- Anaconda
- python 3.8
## Dependency
- numpy 1.24.3
- scikit-learn 0.24.2
- graphein 1.1.0
- matplotlib 3.7.1
- biopython 1.81
- bio-embeddings 0.2.2
- transformers 4.30.1
- torch 1.11.0
- torch-geometric 2.4.0
## Create Environment with Conda
First, create the environment.
```python
conda create -n EDIFIER python=3.7
```
Then, activate the "EDIFIER" environment and enter into the workspace.  
```python
conda activate EDIFIER
pip install -r requirements.txt
```
## Usage
For example:
- using the example file
```python
python train.py
```
Output:
```python
t.pdb 的类别是: P
N score: 0.097184576
P score: 0.90281546
```

