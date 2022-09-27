from model import model_init, BraggNN
from torch.utils.data import Dataset, DataLoader
from dataset import clean_patch, MidasDataset,PatchWiseDataset
import torch, argparse, os
from util import str2bool, str2tuple, s2ituple
from plot import plot_loss, plot_error 
import numpy as np
import pandas as pd 
from torchvision import transforms 
import h5py 

parser = argparse.ArgumentParser(description='Bragg peak finding for HEDM.')
#parser.add_argument('-p_file', type=str, default="debug", help='frame/patch metadata h5 file')
#parser.add_argument('-f_file', type=str, default="debug", help='frame/patch h5 file')
#parser.add_argument('-ge_ffile', type=str, default="debug", help='frame ge3 file')
#parser.add_argument('-ge_dfile', type=str, default="debug", help='frame ge3 file')
#parser.add_argument('-m_file', type=str, default="debug", help='pt model file')
#parser.add_argument('-midas_file', type=str, default="debug", help='midas h5 file')
parser.add_argument('-psz',    type=int, default=15, help='working patch size')
parser.add_argument('-fcsz',   type=s2ituple, default='16_8_4_2', help='size of dense layers')
parser.add_argument('-expName',type=str, default="debug", help='Experiment name')
args, unparsed = parser.parse_known_args()

def main(args):
    device = torch.device("cpu")#"cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        model=torch.load('model.pt',map_location=torch.device('cpu'))
    else: 
        model=torch.load('model.pt')
    model.eval()

    positions = torch.from_numpy(np.load('positions.npy'))

    patch_h5   = h5py.File('patches.h5','r')
    n_frames   = len(patch_h5)
    
    transform_norm = transforms.ToTensor()
    patches  = torch.zeros(size=(len(positions),15,15))
    frame_nr = np.zeros(shape=(len(positions),))
    patch_nr = np.zeros(shape=(len(positions),))
    
    j=0
    for i in range(1,n_frames+1):
        k=0
        for patch in patch_h5['frame_nr%s' %i]:
            patches[j]  = transform_norm(np.array(patch,dtype=np.uint8))
            frame_nr[j] = i 
            patch_nr[j] = k
            j+=1
            k+=1 

    transformed_patches = patches

    transformed_patches = torch.reshape(transformed_patches,(len(patches),1,15,15))
    
    with torch.no_grad():
        y_pred = model(transformed_patches)
        y_pred = y_pred.cpu().numpy()*15 
    
        dataframe=pd.DataFrame()
        dataframe['peakNr']      = list(patch_nr) 
        dataframe['frameNr']     = list(frame_nr)
        dataframe['BNN_horPos']  = y_pred[:,0]
        dataframe['BNN_vertPos'] = y_pred[:,1]
        dataframe['label_centroid_horPos']  = positions[:,0]
        dataframe['label_centroid_vertPos'] = positions[:,1]
        dataframe.to_csv('midas_reconstruction_data_%s.csv' %args.expName)
if __name__ == "__main__":
    main(args)

