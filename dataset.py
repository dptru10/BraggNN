from torch.utils.data import Dataset
import numpy as np
import h5py, torch, random, logging
from skimage.feature import peak_local_max
from skimage import measure
from skimage.measure import label, regionprops
import os 
#from cc_torch import connected_components_labeling
from torchvision import transforms
from time import time
import gc
#from torch.utils.data import TensorDataset,Dataloader 

def connected_components_torch(images,crop_size=15,NrPixels = 2048):
    window=int(crop_size/2)
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    
    ccs     =[]

    start.record()
    
    for image in images:
        cc_out = connected_components_labeling(image).cpu().numpy()
        ccs.append(cc_out)

    end.record() 
    torch.cuda.synchronize()
    print('cc_time: ', start.elapsed_time(end)/1000)
    
    return ccs

def region_props(ccs,images,crop_size=15,NrPixels = 2048):
    window=7#int(crop_size/2)
 
    masks   =[]
    centers =[]
    
    i=0
    start = time()
    for cc in ccs:
        cc=np.array(cc,dtype=np.uint8) 
        for region_nr,region in enumerate(regionprops(cc)):
            if region.area > 4 or region.area < 150: 
                x,y = region.centroid
                start_x = int(x)-window
                end_x   = int(x)+window+1
                start_y = int(y)-window
                end_y   = int(y)+window+1
                if start_x < 0 or end_x > NrPixels - 1 or start_y < 0 or end_y > NrPixels - 1:
                    continue
                sub_img = cc 
                sub_img[cc != region_nr+1] = 0
                sub_img = sub_img[start_y:end_y,start_x:end_x]
                masks.append(sub_img)
                centers.append((start_x,start_y))
        i+=1 
    end=time()
    print('get_regionprops_time: ',end-start)

    #return masks,centers


def connected_components_skimage(images,crop_size=15,NrPixels = 2048):
    window=int(crop_size/2)
    masks   =[]
    centers =[]
    ccs     =[]
    start = time() 
    for image in images:
        cc_out  = measure.label(image)
        ccs.append(cc_out)
    end = time()
    print('cc_time', end - start)
    return ccs

def clean_patch(p, center):
    w, h = p.shape
    cc = measure.label(p > 0)
    if cc.max() == 1:
        return p

    # logging.warn(f"{cc.max()} peaks located in a patch")
    lmin = np.inf
    cc_lmin = None
    for _c in range(1, cc.max()+1):
        lmax = peak_local_max(p * (cc==_c), min_distance=1)
        if lmax.shape[0] == 0:continue # single pixel component
        lc = lmax.mean(axis=0)
        dist = ((lc - center)**2).sum()
        if dist < lmin:
            cc_lmin = _c
            lmin = dist
    return p * (cc == cc_lmin)

class FrameReaderDataset(Dataset): 
    def __init__(self, ffile, dfile,NrPixels=2048, nFrames=1440, nrFiles=1, thresh = 100, fHead = 8192):
        print("dark file:",dfile)
        print("frames file:",ffile)
        self.ffile = ffile 
        self.dark = np.zeros(NrPixels*NrPixels)
        if os.path.exists(dfile):
            darkf       = open(dfile,'rb')
            nFramesDark = int((os.path.getsize(dfile) - 8192) / (2*NrPixels*NrPixels))
            darkf.seek(8192,os.SEEK_SET)
            for nr in range(nFramesDark):
                self.dark += np.fromfile(darkf,dtype=np.uint16,count=(NrPixels*NrPixels))
            self.dark = self.dark.astype(float)
            self.dark /= nFramesDark
            self.dark = np.reshape(self.dark,(NrPixels,NrPixels))
        else:
            self.dark = np.zeros((NrPixels,NrPixels)).astype(float)

        self.frames = []
        self.len = nFrames 
        for fnr in range(nrFiles):
            startFrameNr = (nFrames//nrFiles)*fnr
            endFrameNr = (nFrames//nrFiles)*(fnr+1)
            f = open(ffile,'rb')
            f.seek(fHead,os.SEEK_SET)
            for frameNr in range(startFrameNr,endFrameNr):
                self.thisFrame = np.fromfile(f,dtype=np.uint16,count=(NrPixels*NrPixels))
                self.thisFrame = np.reshape(self.thisFrame,(NrPixels,NrPixels))
                self.thisFrame = self.thisFrame.astype(float)
                self.thisFrame = self.thisFrame - self.dark
                self.thisFrame[self.thisFrame < thresh] = 0
                self.frames.append(self.thisFrame)

    def get_frames(self):
        return np.array(self.frames)

    def write_frames_torch(self):
        f_name = self.ffile.split('/')[-1]
        torch.save(self.frames,'frames_%s.pt' %f_name.split('.ge3')[0])

    def write_frames_numpy(self):
        f_name = self.ffile.split('/')[-1]
        np.save('frames_%s.npy' %f_name.split('.ge3')[0],self.frames)

    def get_peaks_torch(self, psz=15):
        peaks  = connected_components_torch(np.array(self.frames))
        return peaks 
    
    def get_peaks_skimage(self, psz=15):
        peaks  = connected_components_skimage(self.frames)
        return peaks 
    
    def write_peaks_torch(self):
        f_name = self.ffile.split('/')[-1]
        peaks = self.get_peaks_skimage() 
        torch.save(peaks,'peaks_%s.pt' %f_name.split('.ge3')[0])

    def write_peaks_numpy(self):
        f_name = self.ffile.split('/')[-1]
        peaks = self.get_peaks_skimage() 
        np.save('peaks_%s.npy' %f_name.split('.ge3')[0],peaks)

class BraggNNDataset(Dataset):
    def __init__(self, pfile=None, ffile=None, ge_dataset=False, ge_ffile=None, ge_dfile=None, psz=15, rnd_shift=0, use='train', train_frac=0.8):
        self.psz = psz 
        self.rnd_shift = rnd_shift


        with h5py.File(pfile, "r") as h5fd: 
            if use == 'train':
                sti, edi = 0, int(train_frac * h5fd['peak_fidx'].shape[0])
            elif use == 'validation':
                sti, edi = int(train_frac * h5fd['peak_fidx'].shape[0]), None
            else:
                logging.error(f"unsupported use: {use}. This class is written for building either training or validation set")

            mask = h5fd['npeaks'][sti:edi] == 1 # use only single-peak patches
            mask = mask & ((h5fd['deviations'][sti:edi] >= 0) & (h5fd['deviations'][sti:edi] < 1))

            self.peak_fidx= h5fd['peak_fidx'][sti:edi][mask]
            self.peak_row = h5fd['peak_row'][sti:edi][mask]
            self.peak_col = h5fd['peak_col'][sti:edi][mask]

        self.fidx_base = self.peak_fidx.min()
        # only loaded frames that will be used
        if ge_dataset is True: 
            self.frames = FrameReaderDataset(ge_ffile,ge_dfile).get_frames()#[self.peak_fidx.min():self.peak_fidx.max()+1] 
            self.len = self.peak_fidx.shape[0]
            print(self.len)
        else: 
            with h5py.File(ffile, "r") as h5fd: 
                self.frames = h5fd['frames'][self.peak_fidx.min():self.peak_fidx.max()+1]
            self.len = self.peak_fidx.shape[0]


    def __getitem__(self, idx):
        _frame = self.frames[self.peak_fidx[idx] - self.fidx_base]
        if self.rnd_shift > 0:
            row_shift = np.random.randint(-self.rnd_shift, self.rnd_shift+1)
            col_shift = np.random.randint(-self.rnd_shift, self.rnd_shift+1)
        else:
            row_shift, col_shift = 0, 0
        prow_rnd  = int(self.peak_row[idx]) + row_shift
        pcol_rnd  = int(self.peak_col[idx]) + col_shift

        row_base = max(0, prow_rnd-self.psz//2)
        col_base = max(0, pcol_rnd-self.psz//2 )

        crop_img = _frame[row_base:(prow_rnd + self.psz//2 + self.psz%2), \
                            col_base:(pcol_rnd + self.psz//2  + self.psz%2)]
        # if((crop_img > 0).sum() == 1): continue # ignore single non-zero peak
        if crop_img.size != self.psz ** 2:
            c_pad_l = (self.psz - crop_img.shape[1]) // 2
            c_pad_r = self.psz - c_pad_l - crop_img.shape[1]

            r_pad_t = (self.psz - crop_img.shape[0]) // 2
            r_pad_b = self.psz - r_pad_t - crop_img.shape[0]

            logging.warn(f"sample {idx} touched edge when crop the patch: {crop_img.shape}")
            crop_img = np.pad(crop_img, ((r_pad_t, r_pad_b), (c_pad_l, c_pad_r)), mode='constant')
        else:
            c_pad_l, r_pad_t = 0 ,0

        _center = np.array([self.peak_row[idx] - row_base + r_pad_t, self.peak_col[idx] - col_base + c_pad_l])
        crop_img = clean_patch(crop_img, _center)
        if crop_img.max() != crop_img.min():
            _min, _max = crop_img.min().astype(np.float32), crop_img.max().astype(np.float32)
            feature = (crop_img - _min) / (_max - _min)
        else:
            logging.warn("sample %d has unique intensity sum of %d" % (idx, crop_img.sum()))
            feature = crop_img

        px = (self.peak_col[idx] - col_base + c_pad_l) / self.psz
        py = (self.peak_row[idx] - row_base + r_pad_t) / self.psz

        return feature[np.newaxis], np.array([px, py]).astype(np.float32)

    def __len__(self):
        return self.len


class MidasDataset(Dataset):
    def __init__(self, mfile, psz=15, rnd_shift=0, use='train', train_frac=0.8):
        self.psz = psz 
        self.rnd_shift = rnd_shift
        with h5py.File(mfile, "r") as h5fd:
            if use == 'train':
                sti, edi = 0, int(train_frac * len(h5fd['peakLoc']))
            elif use == 'validation':
                sti, edi = int(train_frac * len(h5fd['peakLoc'])), None
            else:
                logging.error(f"unsupported use: {use}. This class is written for building either training or validation set")
            
            npeaks = []
            mask   = []
            for loc in h5fd['peakLoc'][sti:edi]:
                npeaks.append(len(loc))
                mask.append(len(loc)==2)
            
            #mask = npeaks[sti:edi] == 2 # use only single-peak patches
            #mask = mask & ((h5fd['deviations'][sti:edi] >= 0) & (h5fd['deviations'][sti:edi] < 1))
            
            self.npeaks     = npeaks
            self.peak_locs  = h5fd["peakLoc"][sti:edi][mask]
            self.peak_row   = [loc[0] for loc in self.peak_locs]
            self.peak_col   = [loc[1] for loc in self.peak_locs]
            self.deviations = np.zeros(shape=(len(self.peak_locs),))
            self.diffY      = h5fd["diffY"][sti:edi][mask]
            self.diffZ      = h5fd["diffZ"][sti:edi][mask]
            self.peak_fidx  = np.zeros(shape=(len(self.peak_locs),))

            self.crop_img = h5fd['patch'][sti:edi][mask]
            self.len = len(self.peak_locs)#.shape[0]
    
    def __getitem__(self, idx):
        crop_img = self.crop_img[idx] 
        
        row_shift, col_shift = 0, 0
        c_pad_l, r_pad_t = 0 ,0
        prow_rnd  = int(self.peak_row[idx]) + row_shift
        pcol_rnd  = int(self.peak_col[idx]) + col_shift

        row_base = max(0, prow_rnd-self.psz//2)
        col_base = max(0, pcol_rnd-self.psz//2)
        
        if crop_img.max() != crop_img.min():
            _min, _max = crop_img.min().astype(np.float32), crop_img.max().astype(np.float32)
            feature = (crop_img - _min) / (_max - _min)
        else:
            #logging.warn("sample %d has unique intensity sum of %d" % (idx, crop_img.sum()))
            feature = crop_img

        px = (self.peak_col[idx] - col_base + c_pad_l) / self.psz
        py = (self.peak_row[idx] - row_base + r_pad_t) / self.psz

        return feature[np.newaxis], np.array([px, py]).astype(np.float32)

    def __len__(self):
        return self.len

class PatchWiseDataset(Dataset):
    def __init__(self, pfile=None, ffile=None, ge_dataset=False, ge_ffile=None, ge_dfile=None, psz=15, rnd_shift=0, use='train', train_frac=0.8):
        self.ge_dataset = ge_dataset
        self.psz = psz 
        self.rnd_shift = rnd_shift
        if ge_dataset is True: 
            self.peaks = FrameReaderDataset(ge_ffile,ge_dfile).get_peaks_skimage()
            self.len = len(self.peaks)
            print(self.len)
            if use == 'train':
                sti, edi = 0, int(train_frac * self.len)
            elif use == 'validation':
                sti, edi = int(train_frac * self.len), None
            else:
                logging.error(f"unsupported use: {use}. This class is written for building either training or validation set")
            self.crop_img = self.peaks[sti:edi]
        else: 
            with h5py.File(pfile, "r") as h5fd: 
                if use == 'train':
                    sti, edi = 0, int(train_frac * h5fd['peak_fidx'].shape[0])
                elif use == 'validation':
                    sti, edi = int(train_frac * h5fd['peak_fidx'].shape[0]), None
                else:
                    logging.error(f"unsupported use: {use}. This class is written for building either training or validation set")

                mask = h5fd['npeaks'][sti:edi] == 1 # use only single-peak patches
                mask = mask & ((h5fd['deviations'][sti:edi] >= 0) & (h5fd['deviations'][sti:edi] < 1))

                self.peak_fidx= h5fd['peak_fidx'][sti:edi][mask]
                self.peak_row = h5fd['peak_row'][sti:edi][mask]
                self.peak_col = h5fd['peak_col'][sti:edi][mask]
            self.fidx_base = self.peak_fidx.min()
            with h5py.File(ffile, 'r') as h5fd: 
                if use == 'train':
                    sti, edi = 0, int(train_frac * h5fd['frames'].shape[0])
                elif use == 'validation':
                    sti, edi = int(train_frac * h5fd['frames'].shape[0]), None
                else:
                    logging.error(f"unsupported use: {use}. This class is written for building either training or validation set")
                self.crop_img = h5fd['frames'][sti:edi]
                self.len = self.peak_fidx.shape[0]
    
    def __getitem__(self, idx):
        ge_dataset = self.ge_dataset
        print(idx)
        crop_img = self.crop_img[idx] 
        
        if ge_dataset is True: 
            return crop_img
        else: 
            row_shift, col_shift = 0, 0
            c_pad_l, r_pad_t = 0 ,0
            prow_rnd  = int(self.peak_row[idx]) + row_shift
            pcol_rnd  = int(self.peak_col[idx]) + col_shift

            row_base = max(0, prow_rnd-self.psz//2)
            col_base = max(0, pcol_rnd-self.psz//2)
            
            if crop_img.max() != crop_img.min():
                _min, _max = crop_img.min().astype(np.float32), crop_img.max().astype(np.float32)
                feature = (crop_img - _min) / (_max - _min)
            else:
                #logging.warn("sample %d has unique intensity sum of %d" % (idx, crop_img.sum()))
                feature = crop_img

            px = (self.peak_col[idx] - col_base + c_pad_l) / self.psz
            py = (self.peak_row[idx] - row_base + r_pad_t) / self.psz

            return feature[np.newaxis], np.array([px, py]).astype(np.float32)

    def __len__(self):
        return self.len

