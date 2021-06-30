import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np
import random
import os

import matplotlib.pyplot as plt
from my_utils import *

filepath = os.path.dirname(os.path.realpath(__file__))
logger = setup_log(filepath, 'mlp')
logger.info(f"device: {device}")

######### model #########
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_output, n_hiddens, dropout=0):
        super(Net, self).__init__()
        layers = []
        for i, n in enumerate(n_hiddens):
            if i==0:
                self.hidden = torch.nn.Linear(n_feature, n)
            else:
                self.hidden = torch.nn.Linear(n_hiddens[i-1], n)
            self.bn = nn.BatchNorm1d(n)        
            self.relu = nn.LeakyReLU() #nn.ReLU() 
            self.dropout = nn.Dropout(dropout)
            layers += [self.hidden, self.bn, self.relu, self.dropout]

        self.network = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.out=nn.Linear(n_hiddens[-1], n_output)

    def forward(self, x):
        x = self.network(x)
        return self.dropout(self.out(x))

    def train(self, data_loader, epoch, model_path, log_freq=10, val_loader=None, model_name='', learning_rate = 0.01, load_ckp_file=''):
        checkpoint_path = os.path.join(model_path, 'checkpoints')
        if not os.path.exists(checkpoint_path): os.makedirs(checkpoint_path)  
        if model_name: model_file = os.path.join(model_path, model_name)

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        loss_func = torch.nn.MSELoss()
        st_epoch=0
        save_freq = max(1, int(epoch/20))

        num_batches = len(data_loader)
        batch_size = data_loader.batch_size    
        n_iters = epoch * num_batches #total batches
        logger.info(f"batch_size: {batch_size}, nBatches: {num_batches}, nEpochs: {epoch}, totalIters: {n_iters}")

        factor = 0.95
        patience = 4
        min_lr = 1e-6
        logger.info(f"Epoch: lr_scheduler.ReduceLROnPlateau: factor={factor}, patience={patience}, min_lr={min_lr}")

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, patience=patience, min_lr=min_lr)        

        if os.path.exists(load_ckp_file):
            logger.info(f"-- loaded: {load_ckp_file}")
            last_epoch, optimizer, train_loss, _ = self.load_model_from_checkpoint(load_ckp_file, optimizer)
            st_epoch = last_epoch + 1 
            logger.info(f"starting epoch: {st_epoch}")

        for ep in range(epoch):
            train_loss = 0.0
            for step, datas in enumerate(data_loader): # for each training step
                batch_x, batch_y = datas[0].to(device), datas[1].to(device)
                optimizer.zero_grad()   

                bz = batch_x.size(0)
                b_x = Variable(batch_x)
                b_y = Variable(batch_y)
                prediction = self(b_x)

                loss = loss_func(prediction, b_y)     

                loss.backward()         
                optimizer.step()       

                train_loss += loss.item()

            # #update after each epoch
            epoch_loss = round(train_loss/num_batches, 4)
            scheduler.step(epoch_loss)   
            logger.info(f"---> epoch: {st_epoch+ep}/{st_epoch+epoch}, nBatches: {step+1}, avg loss: {epoch_loss}")         
            
            if ((ep+1) % log_freq)==0:
                logger.info(f"learning_rate: {optimizer.param_groups[0]['lr']}")
                index_in_batch = random.randint(0, bz-1)
                logger.info('-------------------------train-------------------------------')
                logger.info(f"GT: {b_y.cpu().detach().data.numpy()[index_in_batch,:]}")
                logger.info(f"PD: {prediction.cpu().detach().data.numpy()[index_in_batch,:]}")
                logger.info('-----------------------------------------------------------------')

            if ((ep+1) % (log_freq*3))==0:
                if val_loader is not None: self.evaluate(loss_func, val_loader, device)

            if ((ep+1) % save_freq)==0:                
                checkpoint_file = os.path.join(checkpoint_path, f"checkpoint_epoch_{st_epoch + ep}.pt")
                torch.save({'epoch': ep + st_epoch,
                    'model_state_dict': self.state_dict(),
                    'opimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': train_loss,
                    'device': device
                }, checkpoint_file)

        if model_name: torch.save(self.state_dict(), model_file)   

    def _evaluate(self, criterion, input_tensor, target_tensor):
        with torch.no_grad():
            batch_x, batch_y = input_tensor, target_tensor
            bz = batch_x.size(0)
            b_x = Variable(batch_x)
            b_y = Variable(batch_y)
            prediction = self(b_x)
            loss = criterion(prediction, b_y) 
            index_in_batch = random.randint(0, bz-1)
            pred_sample = prediction.cpu().detach().numpy()[index_in_batch]

        return loss.item(), prediction, index_in_batch, pred_sample


    def evaluate(self, criterion, data_loader, device = device):
        val_loss = 0.0
        predictions = []
        targets = []
        for step, datas in enumerate(data_loader): 
            batch_x, batch_y = datas[0].to(device), datas[1].to(device)  
            loss, prediction, _,_ = self._evaluate(criterion, batch_x, batch_y)                  
            if step==0:
                predictions = prediction
                targets = batch_y
            else:
                predictions = torch.cat((predictions, prediction), dim=0) 
                targets = torch.cat((targets, batch_y), dim=0)

            val_loss += loss

        val_loss /= (step+1)
        index = random.randint(0, predictions.size(0)-1)
        pred_sample = predictions.cpu().detach().numpy()[index]
        target_sample = targets.cpu().detach().numpy()[index]

        logger.info('-------------------------evaluation-------------------')
        logger.info(f"sampleIndex: {index}, eval_loss: [{val_loss}]")
        logger.info(f"GT: {target_sample}")
        logger.info(f"PD: {pred_sample}")
        logger.info('------------------------------------------------------')

        return val_loss, pred_sample, target_sample, predictions


    def load_model_from_checkpoint(self, checkpoint_file, optimizer):
        checkpoint = torch.load(checkpoint_file)
        self.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict((checkpoint['opimizer_state_dict']))
        train_loss = checkpoint['loss']
        device = checkpoint['device']
        return checkpoint['epoch'], optimizer, train_loss, device

def test_model(test_logger, model, test_loader, criterion=nn.MSELoss(), scale_type=0, out_scale=None, wmti_ranges=None,
                            intercept=None, model_file='', model_ckpt='', save_dir=''):

    if os.path.isfile(model_file):
        model = torch.load(model_file)

    elif os.path.isfile(model_ckpt):
        checkpoint1 = torch.load(model_ckpt)
        model.load_state_dict(checkpoint1['model_state_dict'])
    else:
        test_logger.info("No model files found!!!")
        return False   

    batch_errors = []
    for i, datas in enumerate(test_loader):
        input_tensor, target_tensor = datas[0].to(device), datas[1].to(device)    

        loss, prediction, _index_in_batch, pred_sample = model._evaluate(criterion, input_tensor, target_tensor)  
        outputs = prediction.cpu().detach().numpy()
        targets = target_tensor.cpu().detach().numpy() 

        if scale_type == 1:
            outputs = custom_scale(outputs, out_scale, b=intercept, inverse_transform=True)
            targets = custom_scale(targets, out_scale, b=intercept, inverse_transform=True)
        elif scale_type == 2:
            outputs=min_max_scale(outputs, ranges=wmti_ranges, multipler=out_scale, inverse_transform=True)
            targets=min_max_scale(targets, ranges=wmti_ranges, multipler=out_scale, inverse_transform=True)
        elif scale_type == 3:
            outputs=log_scale(outputs, scale=out_scale, inverse_transform=True)
            targets=log_scale(targets, scale=out_scale, inverse_transform=True)

        if i==0: 
            predictions = np.copy(outputs)  
            targets_all = np.copy(targets)
        else:
            predictions = np.concatenate((predictions, outputs), axis=0)
            targets_all = np.concatenate((targets_all, targets), axis=0)

        rmse = mean_squared_error(targets, outputs, squared=False)
        errs = mean_squared_error(targets, outputs, multioutput='raw_values', squared=False)
        batch_errors.append(rmse)
        test_logger.info(f"--batch#{i}, mean errors: {errs}")
        test_logger.info(f"--batch#{i}, GT: {np.round(targets[_index_in_batch, :], 5)}")
        test_logger.info(f"--batch#{i}, PD: {np.round(outputs[_index_in_batch, :], 5)} \n")
        #test_logger.info(rmse)

    plt.plot(batch_errors)    
    
    if os.path.isdir(save_dir): 
        plt.savefig(os.path.join(save_dir, "batch_rms_errors.png")) 
    plt.show()
    pred_targets = np.stack((predictions, targets_all), axis=2)   

    return pred_targets, batch_errors

def test(logger, model, test_data, scale_type, out_scale, intercept=None, wmti_ranges=None,
                            model_file='', model_ckpt='', save_dir=''):

    pred_and_targets, batch_rms_errors = test_model(logger, model, test_data, scale_type=scale_type, out_scale=out_scale, wmti_ranges=wmti_ranges,
                            intercept=intercept, model_file=model_file, model_ckpt=model_ckpt, save_dir=save_dir)

    if os.path.isdir(save_dir):
        np.save(f"{save_dir}/test_pred_tgt.npy", pred_and_targets)
        np.save(f"{save_dir}/batch_rms_errors.npy", batch_rms_errors)  


def main():
    datapath = '/home/yujian/Desktop/cibmaitsrv1/DeepLearnings/whiteMatterModel'
    model_path = os.path.join(datapath, 'MLP_norm3_scale2_1e-3_2')
    if not os.path.exists(model_path): os.mkdir(model_path) 
    logger.info(f"model path: {model_path}")
    mode = 'train'#'test'#

    data_norm=True
    dki_scale_type = 3
    
    #dki_scaler_filename = 'dki_norm_scaler.gz'
    #dki_scaler = joblib.load(os.path.join(model_path, dki_scaler_filename))
    dki_scaler = None
    logger.info(f"data norm: {data_norm}, norm type: {dki_scale_type}, scaler: {dki_scaler}")

    scale_type = 2
    out_scale = 100
    wmti_ranges=[[0,1], [0, 3], [0, 3], [0, 3], [0, 1]]
    out_intercept = None
    logger.info(f"scale_type: {scale_type}, target scale: {out_scale}, target intercept: {out_intercept}")
    dataset_fn = 'datasets/real_dki_wmti_brain_constraint2_coh1.mat' # 
    batch_size=256
    logger.info(f"dataset: {dataset_fn}, batch size: {batch_size}")
    train_data, val_data, test_data = prep_datasets(datapath, mat_filename=dataset_fn, batch_size=batch_size,  x_scaler=dki_scaler, dki_scale_type=dki_scale_type, train_perc=0.04, val_perc=0.01,
                                        wmti_ranges=wmti_ranges, data_norm=data_norm, scale_type=scale_type, scale=out_scale, intercept=out_intercept) 
    #3 hiddens is the best
    dropout = 0.0
    logger.info(f"dropout: {dropout}")
    mlp = Net(n_feature=6, n_output=5, n_hiddens=[2048, 1024, 256], dropout=dropout).to(device) 
    logger.info(mlp)

    if mode is 'train':
        mlp.train(train_data, 600, model_path, log_freq=1, load_ckp_file='', val_loader=val_data,
                                        learning_rate = 0.001, model_name='fcn_regression.pt')
    elif mode is 'test':
    ##################### Test ################
        logger.info("*********************** Testing model *********************")
        ckpt_epoch = 599
        model_ckt = f'{model_path}/checkpoints/checkpoint_epoch_{ckpt_epoch}.pt'  
        test(logger, mlp, test_data, scale_type, out_scale, intercept=out_intercept, model_ckpt = model_ckt, save_dir=model_path, wmti_ranges=wmti_ranges)


if __name__ == '__main__':
    main()
