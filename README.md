# EDIFIER: structure-aware secretion system effector protein prediction using geometric deep learning and pretrained language model
## Data
The original dataset is a file in Fasta format, and PDB format files can be obtained based on the information contained within it. Some can be downloaded from the UniProt database based on the sequence ID, or the corresponding PDB file can be predicted in ESMFold.
## Installation
```python
conda create --name myenv python=3.8
conda activate myenv                          
pip install numpy==1.24.3
pip install scikit-learn==0.24.2
pip install graphein==1.1.0
pip install matplotlib==3.7.1
pip install biopython==1.81
pip install bio-embeddings==0.2.2
pip install bio-embeddings-plus==0.1.1
pip install transformers==4.30.1
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install torch-geometric==2.4.0
  torch-cluster==1.6.0
  torch-scatter==2.0.9
  torch-spline-conv==1.2.1
  torch-sparse==0.6.13



