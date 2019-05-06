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
    sigma = (X-X_min) / (X_max - X_min)
    return sigma*(1 - -1)+-1


def inverse_scale(X_, X_min, X_max):
    """
    X_ is scaled
    X is unscaled
    """
    # with min=0, max=1
    #sigma = X_ 
    X = X_ * (X_max - X_min) + X_min
    
    #with min=-1, max=1
    X = (1/2)*(X_+1)*(X_max-X_min)+X_min 
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
    
    params_dict = decode_params_from_list(params)
    
    return params_dict

def decode_params_from_list(params):
    
    """Decode to arrays"""
    # min max values for params used to decode
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
    spectrum = aspa[:16, :25].flatten()
    
    n_bins = 17
    mins_ = [aspa[16:17, i*2:i*2+2].mean() for i in range(n_bins)]
    maxs_ = [aspa[17:18, i*2:i*2+2].mean() for i in range(n_bins)]


    """min max std values for spectrum bins"""
    mins = [] # globally decoded values
    maxs = []
    for i in range(n_bins):
        mins.append(inverse_scale(mins_[i], 0.00648 , 0.02877))
        maxs.append(inverse_scale(maxs_[i], 0.00648 , 0.02877))


    """Select bins"""
    df = load_wavelengths()

    df.columns = ['x']
    df = df.loc[df['x'] <= max_wavelength] # select only wavelengths <= 16 (max wavelength ASPA has been encoded with)
    df['y'] = spectrum

    # Could loop this, but right now this is more visual
    # H2O bins
    bin1 = df[df.x <= 0.44]
    bin2 = df[(df.x > 0.44) & (df.x <= 0.495)]
    bin3 = df[(df.x > 0.495) & (df.x <= 0.535)]
    bin4 = df[(df.x > 0.535) & (df.x <= 0.58)]
    bin5 = df[(df.x > 0.58) & (df.x <= 0.635)]
    bin6 = df[(df.x > 0.635) & (df.x <= 0.71)]
    bin7 = df[(df.x > 0.71) & (df.x <= 0.79)]
    bin8 = df[(df.x > 0.79) & (df.x <= 0.9)]
    bin9 = df[(df.x > 0.9) & (df.x <= 1.08)]
    bin10 = df[(df.x > 1.08) & (df.x <= 1.3)]
    bin11 =df[(df.x > 1.3) & (df.x <= 1.7)]
    bin12 = df[(df.x > 1.7) & (df.x <= 2.35)]
    
    # Manually chosen bins
    bin13 = df[(df.x > 2.35) & (df.x <= 4)]
    bin14 = df[(df.x > 4) & (df.x <= 6)]
    bin15 = df[(df.x > 6) & (df.x <= 10)]
    bin16 = df[(df.x > 10) & (df.x <= 14)]
    bin17 = df[df.x > 14]

    bins = [bin17, bin16, bin15, bin14, bin13, bin12, bin11, bin10, bin9, bin8, bin7, bin6, bin5, bin4, bin3, bin2, bin1]

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



def match_arcis_to_exogan_lambda_values(trans):
    """
    trans: np.loadtxt() loaded trans file from ARCiS MakeAI
    output: trans converted to dataframe that match the ExoGAN wavelengths up to 16 micron.
    
    Example:
    exogan wavelengths: 0.30, 0.35. 0.40
    trans wavelengths: 0.31, 0.32, 0.35, 0.39, 0.45
    Then the output df would contain: 0.31, 0.35, 0.39
    """
    
    """prepare exogan wavelengths"""
    # load exogan wavelengths to match trans to
    df = pd.read_csv('wavelengths_and_indices.csv', header=None, skiprows=[0], usecols=[1]) # load wavelengths
    df.columns = ['x']
    df = df.loc[df['x'] <= 16] # select only wavelengths <= 16
    
    # dfe for 'df exogan'
    dfe = df.iloc[::-1] # flip rows
    dfe = dfe.reset_index(drop=True) # reset index
    
    """prepare trans"""
    x = trans[:, 0] # expected MakeAI ARCiS format
    y = trans[:, 1]
    
    # dfa for 'df ARCiS'
    dfa = pd.DataFrame([x,y]).T
    dfa.columns = ['x', 'y']
    
    """Get the lambda values that are the closest match to exogan"""
    closest_matches = []
    for target in dfe['x']:
        delta_previous = 99999
        previous_point = 99999

        for current_point in dfa['x']: # loop over all points in dfa
            delta_current = np.abs(target-current_point) # the absolute difference between the target and selected point

            if delta_current < delta_previous:
                delta_previous = delta_current
                previous_point = current_point

            else:
                closest_matches.append(previous_point)
                break
    
    dfa_selection = dfa[dfa['x'].isin(closest_matches)] # grab the values from dfa that match closest_matches
    
    return dfa_selection



def get_arcis_makeai_params(path_to_file):
    """
    Returns dict {'param':value}
    """
    params = np.genfromtxt(path_to_file, usecols=np.arange(0,3), skip_header=1, dtype='str')
    params[:,2] = params[:,2]
    params = params[:, [0,2]]
    
    params = dict(params)
    
    for key in params:
        params[key] = float(params[key])
    
    return params


def get_makeai_abundances(path_to_file):
    """
    Opens path_to_file and returns dict with abundances
    path_to_file: 'contr_trans_ARIEL' file path
    
    """
    
    
    # Open file, read raw data
    with open(path_to_file) as file:
        words = file.readlines()
        abundances = words[:17]
        
    arr = np.char.split(abundances)
    
    # Put names & values in lists
    ab_names = []
    ab_values = []
    for array in arr:
        ab_names.append(array[1])
        ab_values.append(array[-1])
    
    # Convert these lists to dict like exogan params
    abundances = np.array([ab_names, ab_values])
    
    abundances = dict.fromkeys(ab_names, 0)
    
    for i,key in enumerate(abundances.keys()):
        abundances[key] = float(ab_values[i])
        
    return abundances


def ASPA_complex_v1(trans_file, params_file, ariels_file):
    """
    trans_file, params_file, ariels_file: file paths to respective arcis makeai files
    
    output: ndarray shape(32,32)
    """
    
    """Get spectrum"""
    trans = np.loadtxt(trans_file)
    data = match_arcis_to_exogan_lambda_values(trans) # contains df.x and df.y (lambda and R/R)

    """Get params"""
    params = get_arcis_makeai_params(params_file)
    abundances = get_makeai_abundances(ariels_file)

    params = {**params, **abundances} # create one dict from params and abundances dicts, calling it params 

    del params['Tform'] # delete this because it's only one value...

    """Transform certain parameters from linear to log10 scale"""
    # TO DO: Find out form Michiel what's been sampled in which way

    to_log10 = ['H2O', 'CO2', 'CO', 'CH4', 'SO2', 'NH3', 'HCN', 'C2H2', 'C2H4', 'H2', 'He', 'Na', 'K', 'TiO', 'VO']
    to_log = ['Dplanet', 'TeffP', 'fdry', 'fwet', 'cloud1:Sigmadot', 'cloud1:Kzz', 'f_dry', 'f_wet', 'P', ]

    # convert scales
    for key in to_log10:
        params[key] = np.log10(params[key])

    for key in to_log:
        params[key] = np.log(params[key])

    # make sure everything is a ndarray
    for key in params.keys():
        params[key] = np.array(params[key])

    """seperate dict into exo dict and arcis dict"""
    exo_param_names = ['Mp', 'T', 'CH4', 'Rp', 'H2O', 'CO2', 'CO']

    exo = {key: params[key] for key in exo_param_names} # create dict with exo params

    # create arcis params dict
    arcis = copy.deepcopy(params)

    for key in exo_param_names: # remove exo params so only arcis params are left
        del arcis[key]


    """normalization"""

    """
    exo params

    in order: ''Mp', 'T', 'CH4', 'Rp', 'H2O', 'CO2', 'CO'
    """

    mins = [0.042567043,
     162.714,
     -9.601192269796735,
     0.14990851,
     -8.113734940970243,
     -13.041149548320321,
     -13.29362364416031]

    maxs = [9.9972309,
     2481.179,
     -2.6158258611929663,
     2.9987519,
     -2.3853136577179876,
     -4.269621531412357,
     -2.599807511407424]

    for i,key in enumerate(exo):
        exo[key] = scale_param(exo[key], mins[i], maxs[i])

    """
    arcis params

    in order: 'Dplanet', 'betaT', 'TeffP', 'fdry', 'fwet', 'cloud1:Sigmadot', 'cloud1:Kzz', 'Tform', 'f_dry', 'f_wet', 'COratio', 'metallicity', 'P', 'SO2', 'NH3', 'HCN', 'C2H2', 'C2H4', 'H2', 'He', 'Na', 'K', 'TiO', 'VO'
    """

    mins = np.array([-1.99982975,   0.1000388 ,   1.00126179,  -0.99978456,
            -0.99937355, -18.99565657,   6.00145523,  -0.99978456,
            -0.99937355,   0.05947539,  -0.37582021,  -2.14218522,
           -33.71421773,  -9.68739956, -21.28433086, -31.69229008,
           -21.57430279,  -0.39308147,  -0.95821268, -14.73306309,
           -16.52432881, -18.10646016, -18.50182734])

    maxs = np.array([-0.69903435,   0.2499561 ,   2.39778685,   0.99962839,
             0.99885968, -11.00131038,   9.99344688,   0.99962839,
             0.99885968,   1.1501613 ,   0.81627806,   1.76797172,
            -7.95078198,  -3.9465374 ,  -4.73565449,  -4.37582107,
            -7.20558167,  -0.06844043,  -0.83624248,  -4.53106219,
            -5.74617756,  -6.01139646,  -7.22628667])

    for i,key in enumerate(arcis):
        arcis[key] = scale_param(arcis[key], mins[i], maxs[i])

    """ Select spectrum bins """
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

    """
    Normalize bins
    """
    scalers = [MinMaxScaler(feature_range=(-1,1)).fit(b) for b in bins] # list of ... scalers for the ... bins
    mins = [ b.iloc[:,1].min() for b in bins] # .iloc[:,1] selects all rows from column 1 (which is R/R)
    maxs = [ b.iloc[:,1].max() for b in bins]

    bins_scaled = []
    for i,b in enumerate(bins):
        bins_scaled.append(scalers[i].transform(b))

    spectrum_scaled = np.concatenate(bins_scaled, axis=0)
    spectrum_scaled = spectrum_scaled[:,1]

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

    """exo params"""
    for i,key in enumerate(exo_param_names): # !!!!!! Need to grab the keys list which is in order, python <3.7 reorders the created dict
        aspa[:16, 25+i:26+i] = exo[key]

    """min max values for spectrum bins"""
    for i in range(len(mins)):
        min_ = scale_param(mins[i], 0.00024601534 , 0.1710588)
        max_ = scale_param(maxs[i], 0.00024601534 , 0.1710588)

        aspa[16:17, i*2:i*2+2] = min_
        aspa[17:18, i*2:i*2+2] = max_






    #"""Fill unused space with noice"""
    #for i in range(14):
    #    noise = np.random.rand(32) # random noise betweem 0 and 1 for each row
    #    aspa[18+i:19+i*1, :] = noise

    for i,key in enumerate(arcis.keys()):
        value = arcis[key]
        aspa[18:, i:i+1] = value

    """Fill unused space with some params, just so there is structure to the unused space"""
    for i,key in enumerate(arcis.keys()):
        value = arcis[key]
        aspa[18+i:19+i, 23:] = value
    
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
