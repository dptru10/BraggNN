from model import model_init, BraggNN
import torch, argparse, os, time, sys, shutil, logging
from util import str2bool, str2tuple, s2ituple
from plot import plot_loss 
from torch.utils.data import DataLoader
from dataset import MidasDataset, PatchWiseDataset, BraggNNDataset 
from collections import defaultdict
import numpy as np
import torch.nn as nn
import pandas as pd 
from sklearn.metrics import mean_squared_error 
from torch.optim import lr_scheduler
from ray import tune 
from ray.tune.schedulers import AsyncHyperBandScheduler
import copy

parser = argparse.ArgumentParser(description='Bragg peak finding for HEDM.')
#parser.add_argument('-p_file', type=str, help='frame/patch metadata h5 file')
#parser.add_argument('-f_file', type=str, help='frame/patch h5 file')
parser.add_argument('-m_file', type=str, help='midas output h5 file')
parser.add_argument('-n_cpus', type=int, default=4, help='list of visiable GPUs')
parser.add_argument('-expName',type=str, default="debug", help='Experiment name')
parser.add_argument('-lr',     type=float,default=3e-4, help='learning rate')
parser.add_argument('-mbsz',   type=int, default=512, help='mini batch size')
parser.add_argument('-maxep',  type=int, default=500, help='max training epoches')
parser.add_argument('-fcsz',   type=s2ituple, default='16_8_4_2', help='size of dense layers')
parser.add_argument('-psz',    type=int, default=15, help='working patch size')
parser.add_argument('-aug',    type=int, default=1, help='augmentation size')

args, unparsed = parser.parse_known_args()

if len(unparsed) > 0:
    print('Unrecognized argument(s): \n%s \nProgram exiting ... ... ' % '\n'.join(unparsed))
    exit(0)

itr_out_dir = args.expName + '-itrOut'
if os.path.isdir(itr_out_dir): 
    shutil.rmtree(itr_out_dir)
os.mkdir(itr_out_dir) # to save temp output
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, optimizer, criterion, dataloader, device): 
    best_model_wts=copy.deepcopy(model.state_dict())
    dataset_size = len(dataloader) 
    for epoch in range(args.maxep):                                                                    
        # Each epoch has a training and validation phase
        for phase in ['train']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                                                                                                      
            running_loss = 0.0
            
            # Iterate over data.
            losses = [] 
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                                                                                                      
                # zero the parameter gradients
                optimizer.zero_grad()
                                                                                                      
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                                                                                                      
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
            '''                                                                                          
            if phase == 'train':
                scheduler.step()
            '''
            with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir,"checkpoint")
                torch.save((model.state_dict(),optimizer.state_dict()),path)
                                                                                                      

def test(model, dl_valid, device):
    model.eval()
    mse    = 0.0
    ds_tot = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dl_valid):
            inputs = inputs.to(device)
            y_pred = model(inputs)
            y_pred = y_pred.cpu().numpy()
            labels = labels.cpu().numpy()
            mse    += np.power(y_pred[:,0].squeeze() - labels[:,0].squeeze(),2) + np.power(y_pred[:,1].squeeze() - labels[:,1].squeeze(),2)
            ds_tot += len(inputs)
    return np.sum(mse)/ds_tot

def get_data_loaders(bs):
    logging.info("[%.3f] loading data into CPU memory, it will take a while ... ..." % (time.time(), ))
    ds_train = MidasDataset(psz=args.psz, rnd_shift=args.aug, use='train', mfile=args.m_file)
    dl_train = DataLoader(dataset=ds_train, batch_size=bs, shuffle=True,\
                          num_workers=4, drop_last=True, pin_memory=True)
    #ds_train = PatchWiseDataset(psz=args.psz, rnd_shift=args.aug, use='train', pfile=args.p_file, ffile=args.f_file)
    #ds_train = BraggNNDataset(psz=args.psz, rnd_shift=args.aug, use='train', pfile=args.p_file, ffile=args.f_file)

    #ds_valid = BraggNNDataset(psz=args.psz, rnd_shift=0, use='validation', pfile=args.p_file, ffile=args.f_file)
    #ds_valid = PatchWiseDataset(psz=args.psz, rnd_shift=0, use='validation', pfile=args.p_file, ffile=args.f_file)
    ds_valid = MidasDataset(psz=args.psz, rnd_shift=0, use='validation', mfile=args.m_file)
    dl_valid = DataLoader(dataset=ds_valid, batch_size=bs, shuffle=True, \
                          num_workers=4, drop_last=True, pin_memory=True)
 
    logging.info("[%.3f] loaded training set with %d samples, and validation set with %d samples " % (\
                 time.time(), len(ds_train), len(ds_valid)))
    return dl_train, dl_valid


def train_bnn(config):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    train_loader, test_loader = get_data_loaders(config['batch_size'])
    model = BraggNN(imgsz=args.psz, fcsz=args.fcsz)
    _ = model.apply(model_init) # init model weights and bias
    if torch.cuda.device_count() > 0:
        print("Using ", torch.cuda.device_count(),"gpus!")
        model = nn.DataParallel(model)
    else: 
        print("training on cpu...")
    model = model.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"]) 
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    #while True:
    train(model, optimizer, criterion, train_loader, device)
    mse = test(model, test_loader, device)
    # Set this to run Tune.
    tune.report(mse=mse)

def main(args):

    sched = AsyncHyperBandScheduler()
    
    analysis = tune.run(
        train_bnn,
        metric="mse",
        mode="min",
        name="exp",
        scheduler=sched,
        stop={
            "mse": 1e-6,
            "training_iteration": 100
        },
        resources_per_trial={
            "cpu": args.n_cpus,
            "gpu": torch.cuda.device_count()  # set this for GPUs
        },
        num_samples=50, 
        config={
            "lr": tune.loguniform(1e-4, 1e-2),
            "batch_size": tune.choice([128,256,512])
        })
    
    print("Best config is:", analysis.best_config)


    #time_on_training = 0
    #metrics = defaultdict(list)

if __name__ == "__main__":
    main(args)
