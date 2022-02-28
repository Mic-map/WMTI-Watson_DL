import os
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from util_dev import device
from utils import min_max_scale, custom_scale
from train import evaluate
import scipy.io as sio


def test_model(logger, encoder, decoder, test_loader, criterion=nn.MSELoss(), scale_type=0, out_scale=None, output_length=None,
                            intercept=None, encoder_file='', decoder_file='', encoder_ckpt='', decoder_ckpt='', save_dir='',
                            wmti_ranges=None):

    logger.info(f"device: {device}")

    #if wmti_ranges is None: wmti_ranges=[[0,1], [0, 4], [0, 3], [0, 3], [0, 1]]

    if os.path.isfile(encoder_file) and os.path.isfile(decoder_file):
        encoder.load_state_dict(torch.load(encoder_file))
        decoder.load_state_dict(torch.load(decoder_file))
    elif os.path.isfile(encoder_ckpt) and os.path.isfile(decoder_ckpt):
        checkpoint1 = torch.load(encoder_ckpt)
        encoder.load_state_dict(checkpoint1['model_state_dict'])
        checkpoint2 = torch.load(decoder_ckpt)
        decoder.load_state_dict(checkpoint2['model_state_dict'])     
    else:
        logger.info("No model files found!!!")
        return False   

    batch_errors = []
    n_batches = len(test_loader)
    checks = [0, int(n_batches*0.3), (n_batches*0.5), int(n_batches*0.8), n_batches-1]
    for i, datas in enumerate(test_loader):
        input_tensor, target_tensor = datas[0].to(device), datas[1].to(device)    

        _loss, _decoder_outputs, _index_in_batch, _outputs, _attentions = evaluate(encoder, decoder, criterion, input_tensor, target_tensor, output_length)  
        outputs = _outputs.cpu().detach().numpy()
        targets = target_tensor.cpu().detach().numpy() 
        attentions = _attentions.cpu().detach().numpy()
        if scale_type == 1:
            outputs = custom_scale(outputs, out_scale, b=intercept, inverse_transform=True)
            targets = custom_scale(targets, out_scale, b=intercept, inverse_transform=True)
        elif scale_type == 2:
            outputs=min_max_scale(outputs, ranges=wmti_ranges, multipler=out_scale, inverse_transform=True)
            targets=min_max_scale(targets, ranges=wmti_ranges, multipler=out_scale, inverse_transform=True)

        if i==0: 
            predictions = np.copy(outputs)  
            targets_all = np.copy(targets)
            attentions_all = np.copy(attentions)
        else:
            predictions = np.concatenate((predictions, outputs), axis=0)
            targets_all = np.concatenate((targets_all, targets), axis=0)
            attentions_all = np.concatenate((attentions_all, attentions), axis=0)

        rmse = mean_squared_error(targets, outputs, squared=False)
        errs = mean_squared_error(targets, outputs, multioutput='raw_values', squared=False)
        batch_errors.append(rmse)

        if i in checks:
            logger.info(f"--batch#{i}, mean errors: {errs}")
            logger.info(f"--batch#{i}, GT: {np.round(targets[_index_in_batch, :], 5)}")
            logger.info(f"--batch#{i}, PD: {np.round(outputs[_index_in_batch, :], 5)} \n")

    plt.plot(batch_errors)    
    
    if os.path.isdir(save_dir): 
        plt.savefig(os.path.join(save_dir, "batch_rms_errors.png")) 
    #plt.show()
    pred_targets = np.stack((predictions, targets_all), axis=2)   
    return pred_targets, batch_errors, attentions_all

def test(logger, encoder, decoder, test_data, scale_type, out_scale, intercept=None, output_length=None, wmti_ranges=None,
                            encoder_file='', decoder_file='', encoder_ckpt='', decoder_ckpt='', save_dir=''):

    pred_and_targets, batch_rms_errors, attentions = test_model(logger, encoder, decoder, test_data, scale_type=scale_type, out_scale=out_scale, intercept=intercept, output_length=output_length,
                            encoder_file=encoder_file, decoder_file=decoder_file, encoder_ckpt=encoder_ckpt, decoder_ckpt=decoder_ckpt, save_dir=save_dir, wmti_ranges=wmti_ranges)
    if os.path.isdir(save_dir):
        pred = pred_and_targets[:, :, 0]
        wmti = pred_and_targets[:, :, 1]
        matdata = {'pred_wmti_': pred, 'target_wmti_': wmti}
        sio.savemat(os.path.join(save_dir, 'test_pred_tgt.mat'), matdata)
        np.save(f"{save_dir}/test_pred_tgt.npy", pred_and_targets)
        np.save(f"{save_dir}/batch_rms_errors.npy", batch_rms_errors)   
        #np.save(f"{save_dir}/attention_weights.npy", attentions)

        # attention = np.mean(attentions, axis=0)
        # fig, ax = plt.subplots()
        # cax = ax.imshow(attention)   
        # plt.colorbar(cax,ticks=[0, 0.25, 0.5, 0.75, 1])
        # ax.set_xticks((0,1,2,3,4,5))
        # ax.set_xticklabels(('MD', 'AD', 'RD', 'MK', 'AK', 'RK'))
        # ax.set_yticks((0,1,2,3,4))
        # ax.set_yticklabels([r'$f$', r'$D_a$', r'$D_{e,\parallel}$', r'$D_{e,\perp}$', r'$c_2$'])     
        # plt.savefig(os.path.join(save_dir, "attention.png"))   