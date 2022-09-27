from model import model_init, BraggNN
import torch, argparse, os, time, sys, shutil, logging
from util import str2bool, str2tuple, s2ituple
from plot import plot_loss, plot_error 
from torch.utils.data import DataLoader
from dataset import BraggNNDataset, PatchWiseDataset, MidasDataset, FrameReaderDataset
from collections import defaultdict
import numpy as np
import torch.nn as nn
import pandas as pd 
from sklearn.metrics import mean_squared_error 
from torch.optim import lr_scheduler
import copy

parser = argparse.ArgumentParser(description='Bragg peak finding for HEDM.')
parser.add_argument('-p_file', type=str, default="debug", help='frame/patch metadata h5 file')
parser.add_argument('-f_file', type=str, default="debug", help='frame h5 file')
parser.add_argument('-ge_ffile', type=str, default="debug", help='frame ge3 file')
parser.add_argument('-ge_dfile', type=str, default="debug", help='frame ge3 file')
parser.add_argument('-midas_file', type=str, default="debug", help='midas h5 file')
parser.add_argument('-gpus',   type=str, default="0", help='list of visiable GPUs')
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

logging.basicConfig(filename=os.path.join(itr_out_dir, 'BraggNN.log'), level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict(data, model, device, class_names):
    model.eval()
    inputs   = data.to(device)
    output   = model(inputs)
    return output

def main(args):
    logging.info("[%.3f] loading data into CPU memory, it will take a while ... ..." % (time.time(), ))
    #ds_train = BraggNNDataset(psz=args.psz, ge_dataset=True, ge_ffile=args.ge_ffile, ge_dfile=args.ge_dfile, rnd_shift=args.aug, use='train', pfile=args.p_file)
    #ds_train = BraggNNDataset(psz=args.psz, rnd_shift=args.aug, use='train', pfile=args.p_file, ffile=args.f_file)
    ds_train = PatchWiseDataset(psz=None, ge_dataset=True, ge_ffile=args.ge_ffile, ge_dfile=args.ge_dfile, rnd_shift=args.aug, use='train', pfile=args.p_file, ffile=args.f_file)
    #ds_train = MidasDataset(psz=args.psz, rnd_shift=args.aug, use='train', mfile=args.midas_file)
    dl_train = DataLoader(dataset=ds_train, batch_size=args.mbsz, shuffle=True,\
                          num_workers=4, prefetch_factor=args.mbsz, drop_last=True, pin_memory=True)

    #ds_valid = BraggNNDataset(psz=args.psz, ge_dataset=True, ge_ffile=args.ge_ffile, ge_dfile=args.ge_dfile, rnd_shift=0, use='validation', pfile=args.p_file, ffile=args.f_file)
    ds_valid = PatchWiseDataset(psz=None, ge_dataset=True, ge_ffile=args.ge_ffile, ge_dfile=args.ge_dfile, rnd_shift=0, use='validation', pfile=args.p_file, ffile=args.f_file)
    #ds_valid = MidasDataset(psz=args.psz, rnd_shift=0, use='validation', mfile=args.midas_file)
    dl_valid = DataLoader(dataset=ds_valid, batch_size=len(ds_valid), shuffle=True, \
                          num_workers=4, prefetch_factor=args.mbsz, drop_last=False, pin_memory=True)
 
    logging.info("[%.3f] loaded training set with %d samples, and validation set with %d samples " % (\
                 time.time(), len(ds_train), len(ds_valid)))

    model = BraggNN(imgsz=args.psz, fcsz=args.fcsz)
    _ = model.apply(model_init) # init model weights and bias
    #if torch.cuda.device_count() > 0:
    #    print("Using ", torch.cuda.device_count(),"gpus!")
    #    model = nn.DataParallel(model)
    #else: 
    print("training on cpu...")
    model = model.to(device)
    
    image_datasets = {'train':ds_train, 'val': ds_valid}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],batch_size=args.mbsz,
                                             shuffle=True, num_workers=4)
                                             for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) 
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_model_wts=copy.deepcopy(model.state_dict())
    
    l2_norm = defaultdict(list)
    metrics = defaultdict(list)
    
    clock_init = time.time()
    for epoch in range(args.maxep):                                                                    
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                                                                                                      
            running_loss = 0.0
                                                                                                      
            # Iterate over data.
            losses = [] 
            for inputs, labels in dataloaders[phase]:
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
                                                                                                      
                running_loss += loss.item() * inputs.size(0) 

            if phase == 'train':
                scheduler.step()
 
            epoch_loss = running_loss / dataset_sizes[phase]
            #l2norm[phase] = np.sqrt((labels[:,0] - outputs[:,0])**2   + (labels[:,1] - outputs[:,1])**2) * args.psz
	    
            metrics[phase].append(epoch_loss) 
            print('%s Epoch [%d]/[%d] loss: %.6f' %
                      (phase, epoch + 1, args.maxep, epoch_loss))
                                                                                                      
            # deep copy the model
            if epoch == 0 :
                best_loss = epoch_loss
            else:  
                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                                                                                                      
        print()
                                                                                                      
    time_elapsed = time.time() - clock_init
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
                                                                                                      
    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model, "%s/mdl_%s.pth" % (itr_out_dir,args.expName))

    #outputs = defaultdict(list)
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dl_valid):
            inputs = inputs.to(device)
            y_pred = model(inputs)
            y_pred = y_pred.cpu().numpy()
            labels = labels.cpu().numpy()
            l2norm_val   = np.sqrt((labels[:,0] - y_pred[:,0])**2     + (labels[:,1] - y_pred[:,1])**2)   * args.psz
            #outputs['pred'] = y_pred
            #outputs['true'] = labels
            plot_error(y_pred,labels,args.expName)
    df_m=pd.DataFrame(metrics)
    df_m.to_csv('metrics_%s.csv' %args.expName)
    plot_loss(metrics,args.expName)
    
    #df_o=pd.DataFrame(outputs)
    #df_o.to_csv('outputs_%s.csv' %args.expName)

if __name__ == "__main__":
    main(args)
