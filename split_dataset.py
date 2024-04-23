import glob
import random
    
if __name__ =="__main__":
   
    data_N = glob.glob(f"./test_data/N/*.pdb")
    data_P = glob.glob(f"./test_data/P/*.pdb")
    test_set = data_N+ data_P
    with open("./ann/test_ann.txt","w") as f:
        for i in test_set :
            f.write(f"{i}\n")
    
   