import os
import time
import math
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import torch.utils.data as Data
import torch
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import logging
import logging.config
np.set_printoptions(suppress=True)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

MAX_LENGTH=6
SEED=9

def load_model_from_checkpoint(checkpoint_file, model, optimizer, scheduler):
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict((checkpoint['opimizer_state_dict']))
    scheduler.load_state_dict((checkpoint['scheduler_state_dict']))
    device = checkpoint['device']
    return checkpoint['epoch'], model, optimizer, scheduler, device

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))	

def showPlot(points_lists, save_dir='', fig_name='losses.png'):
    num_point_arrays = len(points_lists)
    fig, axs = plt.subplots(num_point_arrays, 1)
    for i, ax in enumerate(axs.flatten()):
        ax.plot(points_lists[i])

    if os.path.isdir(save_dir): 
        plt.savefig(os.path.join(save_dir, f"{fig_name}")) 

    plt.show()
    #plt.close()

def data_normalize1(input_array, min_val, max_val):
    x = input_array.reshape(-1, 1)
    y = (x - min_val)/(max_val - min_val)
    return y

def data_normalize(input_array, min_val, max_val, multipler=1, inverse_transform=False):
    x = input_array.reshape(-1, 1)
    if inverse_transform:
        y = x/multipler * (max_val - min_val) + min_val
    else:
        y = multipler*(x - min_val)/(max_val - min_val)
    return y

##f = [0,1], Da = [1.5 3], Depar = [0.5 2], Deperp = [0 1.5], kappa = [2 128]
def min_max_scale(input_data, ranges=None, multipler=1, inverse_transform=False):
    #input_data: 2D numpy array, nSamples x nVariables
    if ranges==None: ranges=[[0,1], [0, 3], [0, 3], [0, 3], [0, 1]]
    assert input_data.shape[1] == len(ranges)
    input_scaled = np.zeros(input_data.shape)

    for i in range(len(ranges)):
        input_scaled[:, i] = data_normalize(input_data[:, i], *ranges[i], multipler=multipler, inverse_transform=inverse_transform)[:,0]

    return input_scaled.astype(np.float32)

def sk_data_scaler(input_data, scale_type='MinMax', mm_range=(0, 1)):
    #input_data: nSamples x nFeatures
    scaler = MinMaxScaler(feature_range=mm_range) if scale_type is 'MinMax' else StandardScaler()
        
    scaler.fit(input_data)
    normalized = scaler.transform(input_data)

    return normalized.astype(np.float32), scaler

def custom_scale(input_data, scale=[100, 100, 100, 100, 1], b=[0,0,0,0,0], inverse_transform=False):
    #output: y=x*scale + b
    if b==None: b=[0,0,0,0,0]
    b = np.array(b).reshape(1,-1)
    if inverse_transform:
        scaled = (input_data - b) / np.array(scale).reshape(1,-1)
    else:
        scaled = input_data * np.array(scale).reshape(1,-1) + b
    return scaled.astype(np.float32)

def log_scale(input_data, scale, inverse_transform=False):
    if inverse_transform:
        scaled = np.power(10, input_data / np.array(scale).reshape(1,-1))
    else:
        scaled = np.log10(input_data)*np.array(scale).reshape(1,-1)

    return scaled.astype(np.float32)

def prep_datasets(datapath, mat_filename, batch_size=128, train_perc=0.6, val_perc=0.2, data_norm=True, x_scaler=None,
                        device = device, scale_type=0, scale=None, intercept=None, dki_scale_type='MinMax',
                        wmti_ranges=[[0,1], [0, 3], [0, 3], [0, 3], [0, 1]], seed=SEED): #wmti_ranges=[[0,1], [1.5, 3], [0.5, 2], [0, 1.5], [2, 128]]
    '''
    output: train, val, test datasets
    each: input and target tensor pairs
    exmaple: 
    train[0]: (tensor([ 0.7696,  2.1042,  0.1024,  0.2498,  0.0322, 25.6437]), tensor([  0.9111,   2.1900,   1.4116,   1.0584, 122.4471]))
    '''
    datafile = os.path.join(datapath, mat_filename)
    mats = sio.loadmat(datafile)
    dki = mats['dki'].astype(np.float32)
    wmti = mats['wmti_paras'].astype(np.float32)
    if data_norm:
        #input scaling
        if (x_scaler == None) and (dki_scale_type!=None):
            if dki_scale_type==3:
                dki = min_max_scale(dki, multipler=100, ranges=[[0,3], [0, 3], [0, 3],[0, 10], [0, 10], [0, 10]])
            else:
                dki, x_scaler = sk_data_scaler(dki, scale_type=dki_scale_type)
            
        elif x_scaler != None:
            normalized = x_scaler.transform(dki)
            dki = normalized.astype(np.float32)

        if scale_type==1:
            wmti = custom_scale(wmti, scale, b=intercept) #[100, 100, 100, 100, 1]#[10, 10, 10, 10, 0.1]
        elif scale_type==2:
            wmti = min_max_scale(wmti, multipler=scale, ranges=wmti_ranges)
        elif scale_type==3:
            wmti = log_scale(wmti, scale)
        else:
            assert scale_type == 0

    x, y = Variable(torch.tensor(dki, device=device)), Variable(torch.tensor(wmti, device=device))

    assert x.shape[0] == y.shape[0]

    torch_dataset = Data.TensorDataset(x, y)
    train_size = int(train_perc * x.shape[0])
    val_size = int(val_perc * x.shape[0])
    test_size = x.shape[0] - train_size - val_size

    print(f"--train_size: {train_size}, val_size: {val_size}, test_size: {test_size}")
    train_data, val_data, test_data = Data.random_split(torch_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(seed))  

    train_loader = Data.DataLoader(
        dataset=train_data, 
        batch_size=batch_size, 
        shuffle=True)#   , num_workers=4

    val_loader = Data.DataLoader(
        dataset=val_data, 
        batch_size=batch_size, 
        shuffle=True)  

    test_loader = Data.DataLoader(
        dataset=test_data, 
        batch_size=batch_size, 
        shuffle=True)  

    return train_loader, val_loader, test_loader    

def setup_log(path, logbsname):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    # create logger
    logger = logging.getLogger(__name__)
    # logger.handlers = []
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    # logger.handlers = []
    logger.addHandler(ch)
    # file handler
    fh = logging.FileHandler(os.path.join(path, f"{logbsname}-{timestr}.log"))
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger    



def prep_datasets_sel(datapath, mat_filename, batch_size=128, train_perc=0.6, val_perc=0.2, idx_wmti=[0], data_norm=True, 
                        device = device, scale_type=0, scale=None, intercept=None, dki_scale_type='MinMax', mm_range=(0, 1),
                        wmti_ranges=None, seed=SEED):
    '''
    output: train, val, test datasets
        each: input and target tensor pairs
    exmaple: 
        train[0]: (tensor([ 0.7696,  2.1042,  0.1024,  0.2498,  0.0322, 25.6437]), tensor([ 0.9111, ...]))
    select a part of wmti sequence in target data
        dki seq (input): md, ad, rd, mk, ak, rk
        wmti seq (output):f, Da, Depar, Deperp, kappa (c2)
    '''
    datafile = os.path.join(datapath, mat_filename)
    mats = sio.loadmat(datafile)
    dki = mats['dki'].astype(np.float32)
    wmti_all = mats['wmti_paras'].astype(np.float32)
    if intercept==None: intercept=[0]*len(idx_wmti)

    wmti = wmti_all[:, idx_wmti[0]]
    if len(idx_wmti)>1:
        for i, idx in enumerate(idx_wmti):
            if i==0: continue
            wmti = np.stack((wmti, wmti_all[:, idx_wmti[idx]]), axis=1)   

    wmti = wmti.reshape(-1, len(idx_wmti))

    if wmti_ranges==None:
        wmti_ranges=[[0,1], [1.5, 3], [0.5, 2], [0, 1.5], [2, 128]]
        wmti_range = []
        for i in idx_wmti: wmti_range.append(wmti_ranges[i])
        wmti_ranges = wmti_range

    if data_norm:
        dki, _ = sk_data_scaler(dki, scale_type=dki_scale_type, mm_range=mm_range)
        if scale_type==1:
            wmti = custom_scale(wmti, scale, b=intercept) #[100]#
        elif scale_type==2:
            wmti = min_max_scale(wmti, multipler=scale, ranges=wmti_ranges)
        elif scale_type==3:
            wmti = log_scale(wmti, scale)

    x, y = Variable(torch.tensor(dki, device=device)), Variable(torch.tensor(wmti, device=device))

    assert x.shape[0] == y.shape[0]

    torch_dataset = Data.TensorDataset(x, y)
    train_size = int(train_perc * x.shape[0])
    val_size = int(val_perc * x.shape[0])
    test_size = x.shape[0] - train_size - val_size

    print(f"--train_size: {train_size}, val_size: {val_size}, test_size: {test_size}")
    train_data, val_data, test_data = Data.random_split(torch_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(seed))  

    train_loader = Data.DataLoader(
        dataset=train_data, 
        batch_size=batch_size, 
        shuffle=True)#   , num_workers=4

    val_loader = Data.DataLoader(
        dataset=val_data, 
        batch_size=batch_size, 
        shuffle=True)  

    test_loader = Data.DataLoader(
        dataset=test_data, 
        batch_size=batch_size, 
        shuffle=True)  

    return train_loader, val_loader, test_loader    