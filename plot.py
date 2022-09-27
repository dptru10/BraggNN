import numpy as np
import matplotlib 
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors
from tqdm import tqdm 

def plot_loss(metrics,expName):
    plt.figure() 
    plt.plot(metrics['train'],color='blue',label='train')
    plt.plot(metrics['val'],color='red',label='val')
    plt.xlabel('iterations')
    plt.ylabel('L2 Loss')
    #plt.yscale('log')
    plt.legend(loc="best")
    plt.tight_layout() 
    plt.savefig('loss_%s.pdf' %expName)

def plot_error(y_pred,y_from_ds,expName):
    plt.figure() 
    plt.hist([y_pred[:,0],y_from_ds[:,0]],label=['estimate','true'],bins=int(np.sqrt(len(y_pred[:,0]))),color=['red','blue'],alpha=0.5)
    #plt.hist(y_from_ds[:,0], label='true',  bins=int(np.sqrt(len(y_from_ds[:,0]))),color='blue',alpha=0.5)
    plt.xlabel('x coord')
    plt.ylabel('frequency')
    plt.xlim([0.4,0.55])
    plt.legend() 
    plt.tight_layout()
    plt.savefig('x_coord_kde_%s.pdf' %expName)

    plt.figure() 
    plt.hist([y_pred[:,1],y_from_ds[:,1]],label=['estimate','true'],bins=int(np.sqrt(len(y_pred[:,1]))),color=['red','blue'],alpha=0.5)
    #plt.hist(y_from_ds[:,1], label='true',  bins=int(np.sqrt(len(y_from_ds[:,1]))),color='blue',alpha=0.5)
    plt.xlabel('y coord')
    plt.ylabel('frequency')
    plt.xlim([0.4,0.55])
    plt.legend()
    plt.tight_layout()
    plt.savefig('y_coord_kde_%s.pdf' %expName)
    
    plt.figure() 
    plt.hist(y_from_ds[:,0].squeeze()-y_pred[:,0].squeeze(),bins=int(np.sqrt(len(y_from_ds[:,0]))),label='error')
    plt.xlabel('X Coordinate Absolute Error')
    plt.ylabel('frequency')
    plt.xlim([-0.005,0.005])
    #plt.yscale('log')
    plt.tight_layout()
    plt.savefig('x_coord_mse_%s.pdf' %expName)

    plt.figure() 
    plt.hist(y_from_ds[:,1].squeeze()-y_pred[:,1].squeeze(),bins=int(np.sqrt(len(y_from_ds[:,1]))),label='error')
    plt.xlabel('Y Coordinate Absolute Error')
    plt.ylabel('frequency')
    plt.xlim([-0.005,0.005])
    #plt.yscale('log')
    plt.tight_layout()
    plt.savefig('y_coord_mse_%s.pdf' %expName)

    plt.figure() 
    plt.hist(np.sqrt(np.power(y_from_ds[:,0].squeeze()-y_pred[:,0].squeeze(),2)),bins=int(np.sqrt(len(y_from_ds[:,0]))),label='error')
    plt.xlabel('X Coordinate RMSE')
    plt.ylabel('frequency')
    #plt.yscale('log')
    plt.tight_layout()
    plt.savefig('x_coord_rmse_%s.pdf' %expName)

    plt.figure() 
    plt.hist(np.sqrt(np.power(y_from_ds[:,1].squeeze()-y_pred[:,1].squeeze(),2)),bins=int(np.sqrt(len(y_from_ds[:,1]))),label='error')
    plt.xlabel('Y Coordinate RMSE')
    plt.ylabel('frequency')
    #plt.yscale('log')
    plt.tight_layout()
    plt.savefig('y_coord_rmse_%s.pdf' %expName)

    plt.figure() 
    plt.scatter(y_from_ds[:,0],np.power(y_from_ds[:,0].squeeze()-y_pred[:,0].squeeze(),2))
    plt.xlabel('X Coordinate')
    plt.ylabel('Absolute Error')
    #plt.yscale('log')
    plt.tight_layout()
    plt.savefig('x_coord_positional_mse_%s.pdf' %expName)

    plt.figure() 
    plt.scatter(y_from_ds[:,1],np.power(y_from_ds[:,1].squeeze()-y_pred[:,1].squeeze(),2))
    plt.xlabel('Y Coordinate')
    plt.ylabel('Absolute Error')
    #plt.yscale('log')
    plt.tight_layout()
    plt.savefig('y_coord_positional_mse_%s.pdf' %expName)

    plt.figure()
    x_err = y_from_ds[:,0].squeeze()-y_pred[:,0].squeeze()  
    y_err = y_from_ds[:,1].squeeze()-y_pred[:,1].squeeze() 
    plt.hist2d(x_err,y_err,bins=int(np.sqrt(len(y_from_ds[:,0]))),norm=colors.LogNorm())
    plt.xlabel('X Coordinate Absolute Error')
    plt.ylabel('Y Coordinate Absolute Error')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('error_map_mse_%s.pdf' %expName)
   
    '''
    n_pts = 1000
    data_range_x = np.linspace(min(y_from_ds[:,0]),max(y_from_ds[:,0]),100)
    data_range_y = np.linspace(min(y_from_ds[:,1]),max(y_from_ds[:,1]),100)
    x,y = np.meshgrid(data_range_x,data_range_y)
    i,j = 0,0
    error = np.zeros(shape=(100,100))
    for m in range(0,len(y_from_ds[:,0]),n_pts):
        for n in range(0,len(y_from_ds[:,1]),n_pts):
            error[int(m/n_pts)][int(n/n_pts)] = np.sqrt(np.power(y_from_ds[m,0] - y_pred[m,0],2) + \
            np.power(y_from_ds[n,1] - y_pred[n,1],2))
    plt.figure() 
    plt.imshow(error,extent=[min(y_from_ds[:,0]),max(y_from_ds[:,0]),min(y_from_ds[:,1]),max(y_from_ds[:,1])])
    plt.xlabel('X Coordinate MSE')
    plt.ylabel('Y Coordinate MSE')
    plt.colorbar() 
    plt.tight_layout()
    plt.savefig('test_%s.pdf' %expName)
    '''

    np.save('x_coord_true.npy',y_from_ds[:,0])
    np.save('y_coord_true.npy',y_from_ds[:,1])
    np.save('x_coord_pred.npy',y_pred[:,0])
    np.save('y_coord_pred.npy',y_pred[:,1])

    bins_x = np.linspace(min(y_from_ds[:,0]),max(y_from_ds[:,0]),20)
    bins_y = np.linspace(min(y_from_ds[:,1]),max(y_from_ds[:,1]),20)
    num_pix = 15
    #bins_x = np.linspace(0.42,0.51,20)
    #bins_y = np.linspace(0.42,0.51,20)
    width_x = bins_x[1]-bins_x[0]
    width_y = bins_y[1]-bins_y[0]
    

    x_real = y_from_ds[:,0]
    y_real = y_from_ds[:,1]
    x_sim = y_pred[:,0]
    y_sim = y_pred[:,1]
    
    distrib = np.zeros((bins_x.shape[0],bins_y.shape[0]))
    for idx_x,bin_x in enumerate(bins_x):
        for idx_y,bin_y in enumerate(bins_y):
            idxs_x_within_bin = (x_real-bin_x < width_x) & (x_real >= bin_x)
            idxs_y_within_bin = (y_real-bin_y < width_y) & (y_real >= bin_y)
            # We need to find common idxs
            common_idxs = idxs_x_within_bin & idxs_y_within_bin
            x_err = x_real[common_idxs] - x_sim[common_idxs]
            y_err = y_real[common_idxs] - y_sim[common_idxs]
            len_err = np.mean(np.sqrt(np.power(x_err,2)+np.power(y_err,2)))
            distrib[idx_x,idx_y] = len_err

    axis_values = [num_pix*min(y_from_ds[:,0]),num_pix*max(y_from_ds[:,0]),num_pix*min(y_from_ds[:,1]),num_pix*max(y_from_ds[:,1])] 
    plt.figure() 
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.imshow(distrib,extent=axis_values); plt.colorbar();#plt.show()
    plt.savefig('test_%s_pix.pdf' %expName)

    plt.figure() 
    plt.hist2d(y_from_ds[:,0],y_from_ds[:,0].squeeze()-y_pred[:,0].squeeze()+y_from_ds[:,1].squeeze()-y_pred[:,1].squeeze(),bins=int(np.sqrt(len(y_from_ds[:,0]))),norm=colors.LogNorm())
    plt.xlabel('X Coordinate Absolute Error')
    plt.ylabel('Y Coordinate Absolute Error')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('error_map_mse_no_norm_%s.pdf' %expName)

