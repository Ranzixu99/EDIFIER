'''
>>> this script is used for training
'''

from utils import  plotLoss , ConvertInput
from torch_geometric.data import DataLoader
import torch
from network_attention import Net 
import argparse
import os
from sklearn import metrics
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
    import glob
    import random
    import pickle
    import os
    import tqdm

    # parameter
    args = argparse.ArgumentParser()
    #--------------#
    # use cpu or gpu to train model
    #--------------#
    args.add_argument("-r", "--rank" , default="gpu", help="use cpu or cuda: gpu")
    #--------------#
    # HyperParameters
    # batchsize : the number of protein
    # learning_rate  
    # epoch : the num of train iteration  
    # >>> total dataset number : 1600  -> one epoch
    #--------------#
    args.add_argument("-b", "--batch_size" , default=16 ,type=int, help="Batch size")
    args.add_argument("-l", "--learning_rate" , default=1e-3,type=float, help="")
    args.add_argument("-e", "--epoch" , default=50,type=int, help="")
    #--------------#
    # Log 
    # temp_datset : cache
    # output_dir : the path saving model weight (checkpoint)
    # dataset_path : the path of dataset
    #--------------#
    args.add_argument("-t", "--temp_dataset" , default="./train_dataset.pkl" ,type=str, help="")
    args.add_argument("-o", "--output_dir" , default = f"exp_name" ,type=str, help="")
    args.add_argument("-d", "--dataset_path" , default = "./dataset" ,type=str, help="")
    args.add_argument("-f", "--select_feature" ,nargs='+', help="select the structure feature,(0-36)",type=int)
    args.add_argument("-m", "--fusion_modes" , default = 1 ,type=int, help="fusion mode , 0 : attention ,1:linear ,2:concat")
    # parse the args and save into config
    config = args.parse_args()
    best_acc = 0
    select_feature =  config.select_feature
    print(select_feature)
    fusion_modes = ["attention", "linear" , "concat"]
    fusion_mode = fusion_modes[config.fusion_modes]

    # create the directory named log.

    # obtain hyperparameters
    temp_dataset = config.temp_dataset
    import time
    output_dir= os.path.join(config.output_dir,str(int(time.time())))
    os.makedirs(output_dir , exist_ok=True)
    batch_size = config.batch_size
    lr = config.learning_rate
    epoch = config.epoch
    #---------------#
    # Data Load
    if os.path.exists(temp_dataset):
        # if it exists  
        # load the cache file
        data_list = pickle.load(open(temp_dataset , "rb"))
    else:
        # first runing this part
        data_list = get_dataset(mode = "train")
    
    # first runing this part
    # save cache 
    if not os.path.exists(temp_dataset):
        pickle.dump(data_list , open(temp_dataset , "wb"))

    # DataLoader
    # put the input stack into batch
    trainloader = DataLoader(data_list, batch_size=batch_size, shuffle=True)
    test_data_list = get_dataset(mode = "test")
    
    testloader = DataLoader(test_data_list, batch_size=1, shuffle=True)
    
    # Create Model 
    model = Net(select_feature=select_feature,bert_feature_num=1024,fusion_mode=fusion_mode)
   
    if not config.rank == "cpu":
        model = model.cuda()

    loss_fn = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters() , lr = lr) # lr : learn_rate
    # Record loss curve
    loss_record = []
    # Start to Train
    for e in range(epoch):
        model.train()
        # Train Part
        for index , i in tqdm.tqdm(enumerate(trainloader),leave=True,total=len(trainloader)):
            # i is data
            if not config.rank == "cpu":
                i = i.cuda()

            label = i.t
            # forward
            output = model(i)
            optim.zero_grad()
            loss = loss_fn(output , label)
            # used to plt the loss cureve
            loss_record.append(loss.item())
            # backward
            loss.backward()
            # update the model parameter
            optim.step()
            # plot the loss curve
            if (index +1 ) %10 ==0:
                plotLoss(f"{output_dir}/loss.jpg","loss" , loss_record ,"loss")
        # Val Part
        model.eval()
        if False:
            total = len(data_list)
            acc = 0
            with torch.no_grad():
                for index , i in enumerate(trainloader):
                    if not config.rank == "cpu":
                        i = i.cuda()
                    label = i.t
                    # forward
                    output = model(i)
                    # prediction
                    result = torch.argmax(output , dim =1)
                    # compute the acc
                    pre = result.eq(label)
                    acc = acc + pre.sum().item()
            acc_per = acc /total
            print("ACC:" , acc_per)
        total = len(test_data_list)
        acc = 0
        labels = []
        predictions = []
        with torch.no_grad():
            for index , i in enumerate(testloader):
                if not config.rank == "cpu":
                    i = i.cuda()
                label = i.t
                # forward
                output = model(i)
                # prediction
                result = torch.argmax(output , dim =1)
                labels.append(label.item())
                predictions.append(result.item())
                
                # compute the acc
                pre = result.eq(label)
                acc = acc + pre.sum().item()
        f1_value = metrics.f1_score(labels, predictions)
        recall = metrics.recall_score(labels,predictions)
        
        auc = metrics.roc_auc_score(labels,predictions)
                
        acc_per = acc /total
        if best_acc < acc_per:
            best_acc = acc_per
        print("ACC:" , acc_per)
        print("F1:" , f1_value)
        print("Recall:" , recall)
        print("AUC:" , auc)
        print("************************************")
        print("Best_ACC:" , best_acc)
        print("************************************")
        # Save Model
        os.makedirs(os.path.join(output_dir,"weight"),exist_ok=True)
        if (e+1) % 1 ==0:
            print("Save model in ", e , "epoch" , "path :" , output_dir)
            torch.save(model.state_dict() , f"./{output_dir}/weight/{e}_weight_acc{acc_per:0.4f}_f_{f1_value:04f}_r_{recall:04f}_a_{auc:04f}.pth")
    with open(os.path.join(output_dir,"record_ablation_acc.txt"),"a") as f:
        f.write(f"{fusion_mode},{config.select_feature},best acc:{best_acc}\n")
        
                
