import glob
import gzip
import pickle
import torch

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


sns.set_palette(sns.hls_palette(8, l=.3, s=.8))
sns.set(style='ticks')

def load(filename):
    """
    SOURCE: https://github.com/ucl-exoplanets/ExoGAN_public/blob/master/util.py
    Loads a compressed object from disk
    """
    file = gzip.GzipFile(filename, 'rb')
    data = file.read()
    dict_ = pickle.loads(data, encoding='latin1') # Python 2 and 3 incompatibility... https://stackoverflow.com/questions/11305790/pickle-incompatibility-of-numpy-arrays-between-python-2-and-3
    file.close()
    # Free memory
    del data, file
    return dict_
    
def load_all_to_array(path, filenames):
    """
    path, path to directory containing chunks
    """ 
    paths = glob.glob(path+filenames)
 
    X = []
    for i in tqdm(range(len(paths))):
        s = load(paths[i])
        for j in s.keys():
            X.append(s[j])
    X = np.array(X)
    return X

def load_wavelengths():
    """
    Loads ExoGAN wavelengths file
    """
    path = '/datb/16011015/ExoGAN_data//'
    wavelengths = pd.read_csv(path+'wnw_grid.txt', header=None)
    wavelengths.columns = ['wavelength']
    return wavelengths

def combine_spectrum_with_wavelengths(spectrum, wavelengths):
    """
    Combines R/R and spectrum into one df
    """
    df = pd.DataFrame(spectrum, columns=['y'])
    df['x'] = wavelengths
    return df

def plot_spectrum(x, y, log_scale=True, figsize=(10,5)):
    """
    Plots spectrum
    x = wavelengths
    y = R/R
    """
    plt.figure(figsize=figsize)

    plt.plot(x, y, '.-', color='r')

    plt.xlabel(r'Wavelength [µm]')
    plt.ylabel(r'$(R_P / R_S)^2$')

    plt.grid(True, which="both", ls='-')
    
    if log_scale:
        plt.xscale('log')
    
    return plt


def plot_trans(x,y, multi=False, savefig=False, label=None, x_max=None):
    """
    multi: plot multiple arrays
    if true, x must be a list of arrays [x1, x2, x3], same for y.
    note that all trans spectra must have the same range & spectral res
    """
    
    plt.figure(figsize=(13,4))
    

    
    
    if multi:
        for i in range(len(x)):
            plt.plot(x[i], y[i], '.-', linewidth=1, ms=3, label=str(i))
    else:
        plt.plot(x, y, '.-', color='black', linewidth=1, ms=3, label=label)
    


    plt.xlabel(u'$\lambda$ [µm]')
    plt.ylabel(u'$(R_P / R_*)^2$')


    """Figure formatting"""
    max_value = np.array(x).flatten().max()
    
    plt.gca().set_xscale('log')
    #plt.gca().set_xticks([1, 10]) # Michiel uses this range for ARIEL retrieval challenge
    plt.gca().set_xticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, max_value]) # Taken from doi:10.1038/nature16068 (up till 5)
    plt.gca().get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    plt.gca().yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.4f}')) # specify y-axis to have 3 decimals
    plt.gca().xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.1f}')) # specify y-axis to have 3 decimals
    
    plt.grid()
    plt.tight_layout()
    
    if multi:
        legendloc = (1, 1-len(x)*0.07)
    else:
        legendloc = (1, 0.85)
    
    plt.legend(loc='lower left', bbox_to_anchor=legendloc, borderaxespad=0, frameon=False)
    
    if savefig:
        plt.savefig('/home/brian/notebooks/plots/plot.png', dpi=500)
    return








class numpy_dataset(Dataset):
    def __init__(self, data, transform=None, to_vram=False):
        if to_vram:
            self.data = torch.from_numpy(data).float().to('cuda') # fixes data to vram
        else:
            self.data = torch.from_numpy(data).float()
        #self.target = torch.from_numpy(target).long()
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        #y = self.target[index]
        
        if self.transform:
            x = self.transform(x)
        
        return x
    
    def __len__(self):
        return len(self.data)
