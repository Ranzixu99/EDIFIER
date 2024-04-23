from network_attention import Net 
from utils import   ConvertInput
import argparse
import torch

if __name__ =="__main__":
    '''
    >>> Run script example : 
            python ./test.py -p ./dataset/N/test_0a9df_ptm0.909_r3_default.pdb -c ./log/weight/best.pth
    '''
    args = argparse.ArgumentParser()
    args.add_argument("-r", "--rank" , default="cpu", help="use cpu or cuda")
    args.add_argument("-p", "--pdb_path" , default="./dataset/N/test_0a9df_ptm0.909_r3_default.pdb", help="the path of pdb file")
    args.add_argument("-c", "--checkpoint" , default="./log/weight/best.pth", help="Checkpoint path")
    config = args.parse_args()
    classes = ["N", "P"]
    pdb = config.pdb_path
    # Read input of protein 
    # convert tensor
    input = ConvertInput(pdb , 1)
    # Loading the checkpoint for model
    checkpoint_path = config.checkpoint
    checkpoint =  torch.load(checkpoint_path , map_location="cpu")
    model = Net()
    # Network is converted to eval model
    model.eval()
    model.load_state_dict(checkpoint , strict= True)

    if not config.rank == "cpu":
        model = model.cuda()
        input = input.cuda()
    # Forward Part
    output = model(input)
    # Inference Part
    result = torch.argmax(output , dim =1)
    prediction = classes[result.item()]
    # Print the Output of network
    print(pdb , "的类别是:", prediction)