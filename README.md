# EDIFIER: characterizing secretion system effector proteins with structure-aware graph neural networks and pre-trained language models
The type III secretion systems (T3SSs) play a pivotal role in host-pathogen interactions by mediating the secretion of type III secretion system effectors (T3SEs) into host cells. These T3SEs mimic host cell protein functions, influencing interactions between Gram-negative bacterial pathogens and their hosts. Identifying T3SEs is essential in biomedical research for comprehending bacterial pathogenesis and its implications on human cells. This study presents EDIFIER, a novel multi-channel model designed for accurate T3SEs prediction. It incorporates a graph structure channel, utilizing graph convolutional networks (GCN) to capture protein 3D structural features and a sequence channel based on the ProteinBERT pre-trained model to extract the sequence context features of T3SEs. Rigorous benchmarking tests, including ablation studies and comparative analysis, validate that EDIFIER outperforms current state-of-the-art tools in T3SEs prediction. To enhance EDIFIER’s accessibility to the broader scientific community, we developed a webserver which is publicly accessible at http://edifier.unimelb-biotools.cloud.edu.au/. We anticipate EDIFIER will contribute to the field by providing reliable T3SEs predictions, thereby advancing our understanding of host-pathogen dynamics.
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
python test.py
```
Output:
```python
t.pdb 的类别是: P
N score: 0.097184576
P score: 0.90281546
```

