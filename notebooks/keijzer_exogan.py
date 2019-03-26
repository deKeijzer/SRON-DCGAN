import glob
import gzip
import pickle
import torch

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import copy # copy.deepcopy(dict_variable) to actually copy a dict without problems
import math

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler


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

    plt.figure(figsize=(15,4))
    
    # Set the max value
    if x_max == None:
        max_value = np.array(x).max()
    else:
        max_value = x_max
        
    
    # Convert ndarrays to df to slice up till max value
    dfs = []
    if type(x) == list: # accounts for x being a list of arrays
        for i in range(len(x)):
            df = pd.DataFrame([x[i],y[i]]).T
            df.columns = ['x', 'y']
            df = df[(df.x <= max_value)]
            dfs.append(df)
    else:
        df = pd.DataFrame([x,y]).T
        df.columns = ['x', 'y']
        df = df[(df.x <= max_value)]
        dfs.append(df)
    
    
    if multi: # TODO: remove if multi... and add a color list so first df is black
        for df in dfs:
            plt.plot(df.x, df.y, '.-', linewidth=1, ms=3, label=str(i))
    else:
        plt.plot(dfs[0].x, dfs[0].y, '.-', color='black', linewidth=1, ms=3, label=label)
    


    plt.xlabel(u'$\lambda$ [µm]')
    plt.ylabel(u'$(R_P / R_*)^2$')


    """Figure formatting"""
    
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


def scale_param(X, X_min, X_max):
    """
    Formule source: 
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
    
    In this case 1 is max, 0 is min
    """
    std = (X-X_min)/ (X_max - X_min)
    return std*(1 - -1)+-1


def inverse_scale(X_, X_min, X_max):
    """
    X_ is scaled X
    X is unscaled X_
    """
    X = X_ * (X_max - X_min) + X_min
    return X


def ASPA_v3(x, wavelengths, max_wavelength=16):
    """
    x: dict 
    y: ?
    max_wavelength: max wavelength in micron to decode in aspa
    """
    
    #print('='*100)
    x = copy.deepcopy(x)
    wavelengths = wavelengths.copy()
    
    spectrum = x['data']['spectrum']
    #print('Original spectrum length: ', len(spectrum))
    spectrum = np.expand_dims(spectrum, axis=1) # change shape from (515,) to (515,1)
    params = x['param']

    for param in params:
        #print('Param: ', param)
        if 'mixratio' in param: 
            params[param] = np.log(np.abs(params[param])) # transform mixratio's because they are generated on logarithmic scale
    
    """
    Normalize params
    """
    # Min max values from training set, in the same order as params above: planet mass, temp, .... co mixratio.
    min_values = [1.518400e+27, 
                  1.000000e+03, 
                  -1.842068e+01, 
                  5.592880e+07, 
                  -1.842068e+01, 
                  -1.842068e+01, 
                  -1.842068e+01]
    
    max_values = [3.796000e+27, 
                  2.000000e+03, 
                  -2.302585e+00, 
                  1.048665e+08, 
                  -2.302585e+00, 
                  -2.302585e+00,
                  -2.302585e+00]

    for i,param in enumerate(params):
        params[param] = scale_param(params[param], min_values[i], max_values[i])
        #print('%s: %s' % (param, params[param]))
        
    #print('-'*5)
    """
    Select bins
    """
    data = np.concatenate([wavelengths,spectrum], axis=1)
    #print('Original data length: ', len(data))
    data = pd.DataFrame(data)
    data.columns = ['x', 'y'] # x is wavelength, y is (R_p / R_s)^2
    data = data.loc[data['x'] <= 16] # select only wavelengths <= 16
    
    # Could loop this, but right now this is more visual
    bin1 = data[data.x <= 0.8]
    bin2 = data[(data.x > 0.8) & (data.x <= 1.3)] # select data between 2 and 4 micron
    bin3 = data[(data.x > 1.3) & (data.x <= 2)]
    bin4 = data[(data.x > 2) & (data.x <= 4)]
    bin5 = data[(data.x > 4) & (data.x <= 6)]
    bin6 = data[(data.x > 6) & (data.x <= 10)]
    bin7 = data[(data.x > 10) & (data.x <= 14)]
    bin8 = data[data.x > 14]

    bins = [bin8, bin7, bin6, bin5, bin4, bin3, bin2, bin1]
    #print('Total bins length: ', len(np.concatenate(bins)))
    """
    Normalize bins
    """
    scalers = [MinMaxScaler(feature_range=(-1,1)).fit(b) for b in bins] # list of 8 scalers for the 8 bins
    mins = [ b.iloc[:,1].min() for b in bins] # .iloc[:,1] selects the R/R (y) only
    maxs = [ b.iloc[:,1].max() for b in bins]
    stds = [ b.iloc[:,1].std() for b in bins]
    #print(min(mins), max(maxs))
    bins_scaled = []
    for i,b in enumerate(bins):
        bins_scaled.append(scalers[i].transform(b))
        
    spectrum_scaled = np.concatenate(bins_scaled, axis=0)
    spectrum_scaled = spectrum_scaled[:,1]
    #print('spectrum scaled shape: ', spectrum_scaled.shape)
    
    """
    Create the ASPA
    """
    
    """Spectrum"""
    aspa = np.zeros((32,32))

    row_length = 25 # amount of pixels used per row
    n_rows = math.ceil(len(spectrum_scaled) / row_length) # amount of rows the spectrum needs in the aspa, so for 415 data points, 415/32=12.96 -> 13 rows
    #print('Using %s rows' % n_rows)

    for i in range(n_rows): # for i in 

        start = i*row_length
        stop = start+row_length
        spec = spectrum_scaled[start:stop]

        if len(spec) != row_length:
            n_missing_points = row_length-len(spec)
            spec = np.append(spec, [0 for _ in range(n_missing_points)]) # for last row, if length != 32, fill remaining with 0's
            #print('Filled row with %s points' % n_missing_points)

        aspa[i, :row_length] = spec
        
    """ExoGAN params"""
    for i,param in enumerate(params):
        aspa[:16, 25+i:26+i] = params[param]
        
    """min max std values for spectrum bins"""
    for i in range(len(mins)):
        min_ = scale_param(mins[i], 0.005, 0.03)
        max_ = scale_param(maxs[i], 0.005, 0.03)
        std_ = scale_param(stds[i], 9e-6, 2e-4)

        aspa[16:17, i*4:i*4+4] = min_
        aspa[17:18, i*4:i*4+4] = std_
        aspa[18:19, i*4:i*4+4] = max_
        
    """Fill unused space with noice"""
    for i in range(13):
        noise = np.random.rand(32) # random noise betweem 0 and 1 for each row
        aspa[19+i:20+i*1, :] = noise
        
    return aspa


def decode_params_from_aspa(aspa):
    """The values of the image parts from the aspa (still encoded)"""
    # Grab the values 
    spectrum = aspa[:16, :25].flatten()
    params = [aspa[:16, 25+i:26+i].mean() for i in range(7)]
    
    
    """Decode to arrays"""
    # min max values for params used to decode
    min_values = [1.518400e+27, 
                      1.000000e+03, 
                      -1.842068e+01, 
                      5.592880e+07, 
                      -1.842068e+01, 
                      -1.842068e+01, 
                      -1.842068e+01]

    max_values = [3.796000e+27, 
                      2.000000e+03, 
                      -2.302585e+00, 
                      1.048665e+08, 
                      -2.302585e+00, 
                      -2.302585e+00,
                      -2.302585e+00]

    # Initialize dict to be used for the param values
    params_dict = {
        'planet_mass': 0,
        'temp_profile': 0,
        'ch4_mixratio': 0,
        'planet_radius': 0,
        'h2o_mixratio': 0,
        'co2_mixratio': 0,
        'co_mixratio': 0
    }
    
    
    for i,param in enumerate(params_dict):
        # put aspa values in dict
        params_dict[param] = params[i]

        # inverse scale these values
        params_dict[param] = inverse_scale(params[i], min_values[i], max_values[i])

        # scale mixratios from log back to linear
        #if 'mixratio' in param: 
            #params_dict[param] = np.exp(params_dict[param])
            #print(param, params_dict[param])
        
    return params_dict


def decode_spectrum_from_aspa(aspa, max_wavelength=16):
    """
    Returns x: wavelength in micron, y: R/R 
    It's currently hard coded to work with 
    
    """
    mins_ = [aspa[16:17, i*4:i*4+4].mean() for i in range(8)]
    maxs_ = [aspa[18:19, i*4:i*4+4].mean() for i in range(8)]


    """min max std values for spectrum bins"""
    mins = [] # globally decoded values
    maxs = []
    for i in range(8):
        mins.append(inverse_scale(mins_[i], 0.005, 0.03))
        maxs.append(inverse_scale(maxs_[i], 0.005, 0.03))


    """Select bins"""
    df = ke.load_wavelengths()

    df.columns = ['x']
    df = df.loc[df['x'] <= max_wavelength] # select only wavelengths <= 16 (max wavelength ASPA has been encoded with)
    df['y'] = spectrum

    # Could loop this, but right now this is more visual
    bin1 = df[df.x <= 0.8]
    bin2 = df[(df.x > 0.8) & (df.x <= 1.3)] # select data between 2 and 4 micron
    bin3 = df[(df.x > 1.3) & (df.x <= 2)]
    bin4 = df[(df.x > 2) & (df.x <= 4)]
    bin5 = df[(df.x > 4) & (df.x <= 6)]
    bin6 = df[(df.x > 6) & (df.x <= 10)]
    bin7 = df[(df.x > 10) & (df.x <= 14)]
    bin8 = df[df.x > 14]

    bins = [bin8, bin7, bin6, bin5, bin4, bin3, bin2, bin1]

    """Inverse scale bins"""
    for i,b in enumerate(bins):
        b.y = inverse_scale(b.y, mins[i], maxs[i])

    x = np.concatenate([b.x for b in bins])
    y = np.concatenate([b.y for b in bins])

    return x, y


def ASPA_v4(x, wavelengths, max_wavelength=16):
    """
    x: dict 
    max_wavelength: max wavelength in micron to decode to aspa
    
    returns: 1x32x32 ndarray
    """
    
    #print('='*100)
    x = copy.deepcopy(x)
    wavelengths = wavelengths.copy()
    
    spectrum = x['data']['spectrum']
    #print('Original spectrum length: ', len(spectrum))
    spectrum = np.expand_dims(spectrum, axis=1) # change shape from (515,) to (515,1)
    params = x['param']

    for param in params:
        #print('Param: ', param)
        if 'mixratio' in param: 
            params[param] = np.log10(params[param]) # transform mixratio's to dex because they are generated on logarithmic scale
    
    """
    Normalize params
    """
    # Min max values from training set, in the same order as params above: planet mass, temp, .... co mixratio.
    min_values = [1.518400e+27, 
                  1.000000e+03, 
                  -8, 
                  5.592880e+07, 
                  -8, 
                  -8, 
                  -8]
    
    max_values = [3.796000e+27, 
                  2.000000e+03, 
                  -1, 
                  1.048665e+08, 
                  -1, 
                  -1,
                  -1]

    for i,param in enumerate(params):
        params[param] = scale_param(params[param], min_values[i], max_values[i])
        #print('%s: %s' % (param, params[param]))
        
    #print('-'*5)
    """
    Select bins
    """
    data = np.concatenate([wavelengths,spectrum], axis=1)
    #print('Original data length: ', len(data))
    data = pd.DataFrame(data)
    data.columns = ['x', 'y'] # x is wavelength, y is (R_p / R_s)^2
    data = data.loc[data['x'] <= 16] # select only wavelengths <= 16
    
    # Could loop this, but right now this is more visual
    # H2O bins
    bin1 = data[data.x <= 0.44]
    bin2 = data[(data.x > 0.44) & (data.x <= 0.495)]
    bin3 = data[(data.x > 0.495) & (data.x <= 0.535)]
    bin4 = data[(data.x > 0.535) & (data.x <= 0.58)]
    bin5 = data[(data.x > 0.58) & (data.x <= 0.635)]
    bin6 = data[(data.x > 0.635) & (data.x <= 0.71)]
    bin7 = data[(data.x > 0.71) & (data.x <= 0.79)]
    bin8 = data[(data.x > 0.79) & (data.x <= 0.9)]
    bin9 = data[(data.x > 0.9) & (data.x <= 1.08)]
    bin10 = data[(data.x > 1.08) & (data.x <= 1.3)]
    bin11 = data[(data.x > 1.3) & (data.x <= 1.7)]
    bin12 = data[(data.x > 1.7) & (data.x <= 2.35)]
    
    # Manually chosen bins
    bin13 = data[(data.x > 2.35) & (data.x <= 4)]
    bin14 = data[(data.x > 4) & (data.x <= 6)]
    bin15 = data[(data.x > 6) & (data.x <= 10)]
    bin16 = data[(data.x > 10) & (data.x <= 14)]
    bin17 = data[data.x > 14]

    bins = [bin17, bin16, bin15, bin14, bin13, bin12, bin11, bin10, bin9, bin8, bin7, bin6, bin5, bin4, bin3, bin2, bin1]
    #print('Total bins length: ', len(np.concatenate(bins)))
    #for i,b in enumerate(bins):
    #    print('Bin , shape: ',(len(bins)-i), b.values.shape)
    
    """
    Normalize bins
    """
    scalers = [MinMaxScaler(feature_range=(-1,1)).fit(b) for b in bins] # list of 8 scalers for the 8 bins
    mins = [ b.iloc[:,1].min() for b in bins] # .iloc[:,1] selects the R/R (y) only
    maxs = [ b.iloc[:,1].max() for b in bins]
    stds = [ b.iloc[:,1].std() for b in bins]
    #print(min(mins), max(maxs))
    bins_scaled = []
    for i,b in enumerate(bins):
        bins_scaled.append(scalers[i].transform(b))
        
    spectrum_scaled = np.concatenate(bins_scaled, axis=0)
    spectrum_scaled = spectrum_scaled[:,1]
    #print('spectrum scaled shape: ', spectrum_scaled.shape)
    
    """
    Create the ASPA
    """
    
    """Spectrum"""
    aspa = np.zeros((32,32))

    row_length = 25 # amount of pixels used per row
    n_rows = math.ceil(len(spectrum_scaled) / row_length) # amount of rows the spectrum needs in the aspa, so for 415 data points, 415/32=12.96 -> 13 rows
    #print('Using %s rows' % n_rows)

    for i in range(n_rows): # for i in 

        start = i*row_length
        stop = start+row_length
        spec = spectrum_scaled[start:stop]

        if len(spec) != row_length:
            n_missing_points = row_length-len(spec)
            spec = np.append(spec, [0 for _ in range(n_missing_points)]) # for last row, if length != 32, fill remaining with 0's
            #print('Filled row with %s points' % n_missing_points)

        aspa[i, :row_length] = spec
        
    """ExoGAN params"""
    for i,param in enumerate(params):
        aspa[:16, 25+i:26+i] = params[param]
        
    """min max std values for spectrum bins"""
    for i in range(len(mins)):
        min_ = scale_param(mins[i], 0.00648 , 0.02877)
        max_ = scale_param(maxs[i], 0.00648 , 0.02877)
        #std_ = scale_param(stds[i], 9e-6, 2e-4)
        
        #min_ = mins[i]
        #max_ = maxs[i]
        
        aspa[16:17, i*2:i*2+2] = min_
        aspa[17:18, i*2:i*2+2] = max_
            
    """Fill unused space with noice"""
    for i in range(14):
        noise = np.random.rand(32) # random noise betweem 0 and 1 for each row
        aspa[18+i:19+i*1, :] = noise
        
    return aspa




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
