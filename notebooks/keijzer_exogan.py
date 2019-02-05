import glob
import gzip
import pickle
import torch

import numpy as np

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

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