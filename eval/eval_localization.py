#script to test the heatmap regression
import os
import argparse 
import random
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
import torch.nn.functional as F
import sys
sys.path.append(".")
from models.ResUnet import UNetWithResnet50Encoder
#from models.drr_net import drr_net
#from models.deeplabv3 import *
from dataset.drr_dataset import drr_dataset
# from data.multichannel_heatmap import drr_dataset
import SimpleITK as sitk


class log_writer():
    def __init__(self, txt_name = ""):
        super().__init__()
        self.name = txt_name
    def write(self,info):
        writer = open(self.name,"a+")
        data = info + "\n"
        writer.writelines(data)
        print(info)
        writer.close()

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser(description='Localization training')
parser.add_argument('-lr',                          default=0.0001,     type=float, help='learning rate')
parser.add_argument('-batch_size',                  default=1,          type=int,   help='batch size')
parser.add_argument('-epochs',                      default=120,        type=int,   help='training epochs')
parser.add_argument('-eval_epoch',                  default=4,          type=int,   help='evaluation epoch')
parser.add_argument('-log_path',                    default="logs",     type=str,   help='the path of the log') 
parser.add_argument('-log_inter',                   default=50,         type=int,   help='log interval')
parser.add_argument('-read_params',                 default=True,      type=bool,  help='if read pretrained params')
parser.add_argument('-params_path',                 default="checkpoints_localization/logs119.pth",         type=str,   help='the path of the pretrained model')
parser.add_argument('-basepath',                    default="/projects/MAD3D/Zhuoli/MICCAI/VerSe2019",         type=str,   help='base dataset path')
parser.add_argument('-augmentation',                default=False,      type=bool,  help='if augmentation')
parser.add_argument('-num_view',                    default=10,         type=int,   help='the number of views')

if __name__=="__main__":
    setup_seed(33)
    args = parser.parse_args()
    n_views = args.num_view
    log_path = args.log_path
    log_name = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    writer = SummaryWriter(log_path+log_name)
    txt_name = log_path + log_name + ".txt"
    txt_writer = log_writer(txt_name)
    base_train_path = f"{args.basepath}/dataset-verse19training/enhance_drr" 
    base_test_path = f"{args.basepath}/dataset-verse19test/enhance_drr" 
    text_txt = "/projects/MAD3D/Zhuoli/MICCAI/clean_code/Multi-View-Vertebra-Localization-and-Identification-from-CT-Images/DRR_localization_test.txt"
    
    # Initialize a list to store the lines
    lines = []

    # Read the file line by line
    with open(text_txt, 'r') as file:
        for line in file:
            first_item = line.split('*')[0]
            lines.append(first_item)
    
    # The variable 'lines' now contains all lines from the file as a list
    print(lines)
    
    save_path = log_path 
    epochs = args.epochs
    base_lr = args.lr
    eval_epoch = args.eval_epoch
    log_inter = args.epochs
    compare = []
    model = UNetWithResnet50Encoder(1, pretrained=args.read_params)
    if args.read_params:
        state_dict = torch.load(args.params_path)
        model.load_state_dict(state_dict['net'])
    model = model.cuda()
    #model.train()
    loss_func = nn.MSELoss()
    loss_func = loss_func.cuda()

    #optimizer = optim.Adam(model.parameters(), lr=base_lr)
    #schedule = MultiStepLR(optimizer, milestones=[epochs//4, epochs//4*2, epochs//4*3], gamma=0.1)
    #txt_writer.write("-"*8 + "reading data" + "-"*8)
    train = drr_dataset(drr_path=base_train_path, mode="train", if_identification=False, n_views=n_views)
    test = drr_dataset(drr_path=base_test_path, mode="test", if_identification=False, n_views=n_views)
    
    trainset = DataLoader(dataset=train, batch_size=1, shuffle = False)
    testset = DataLoader(dataset=test, batch_size=1, shuffle = False)
    
    #txt_writer.write("-"*8 + "start localization training" + "-"*8)
    start_time = time.time()
    #if epoch%eval_epoch == eval_epoch-1:
    model.train()
    test_loss = 0.0
    line_i = 0
    with torch.no_grad():
        for (data, target) in testset:
            data, target = data.float(), target.float()
            data = data.cuda()
            target = target.cuda()
            output = model(data)  
            loss = loss_func(output, target)
            test_loss += loss.item()
            output = output.squeeze().cpu().numpy()  # Remove batch and channel dimensions and convert to NumPy

            # Convert to SimpleITK image
            sitk_image = sitk.GetImageFromArray(output)
            output_file = lines[line_i].replace("enhance_drr", "heatmap_predict")
            
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            # Save as NIfTI file
            sitk.WriteImage(sitk_image, output_file)
            print(f"Saved output tensor as {output_file}")
            #print(loss)
            line_i += 1
            #import pdb; pdb.set_trace()
    print("total loss", test_loss / len(testset))
    #writer.add_scalar('Test Loss', test_loss / len(testset), epoch)

    now = time.time()
    period = str(datetime.timedelta(seconds=int(now-start_time)))
    #txt_writer.write(f'[*]Test finish, test epoch {epoch+1} loss : {test_loss / len(testset):.8f} , training time is {period}')
    
