import os
import random
import argparse
import numpy as np
import json
import torch
import joblib
# import torch.nn as nn

from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from util_dev import device
from utils import min_max_scale, custom_scale, preprocess_datasets, preprocess_input_dataset, setup_log
from RNNEncoderDecoder import EncoderRNN, AttnDecoderRNN

def predict(encoder, decoder, input_tensor, output_length):
    with torch.no_grad():
        bz = input_tensor.size(0)

        outputs = torch.zeros(bz, output_length, device=device)

        encoder_outputs, encoder_hidden, encoder_cell = encoder(input_tensor)

        decoder_hidden = encoder_hidden[-1]
        decoder_cell = encoder_cell[-1]
        #decoder_attentions = torch.zeros(input_length, input_length)
        decoder_input = input_tensor[:, -1].view(-1,1)

        for di in range(output_length):
            decoder_output, decoder_hidden, decoder_cell, decoder_attention = decoder(
                decoder_input, decoder_hidden, decoder_cell, encoder_outputs)
            #decoder_attentions[di] = decoder_attention.data
            decoder_input = decoder_output.detach().view(-1,1)
            outputs[:, di] = decoder_input[:,0]

    return outputs 

def prediction(encoder, decoder, data_loader, scale_type=0, out_scale=None, output_length=5, 
                has_target=False, wmti_ranges=[[0, 1], [0, 3], [0, 3], [0, 3], [0, 1]], intercept=None, encoder_file='', 
                decoder_file='', encoder_ckpt='', decoder_ckpt='', logger=None, save_dir=''):

    if os.path.isfile(encoder_file) and os.path.isfile(decoder_file):
        encoder.load_state_dict(torch.load(encoder_file))
        decoder.load_state_dict(torch.load(decoder_file))
    elif os.path.isfile(encoder_ckpt) and os.path.isfile(decoder_ckpt):
        checkpoint1 = torch.load(encoder_ckpt)
        encoder.load_state_dict(checkpoint1['model_state_dict'])
        checkpoint2 = torch.load(decoder_ckpt)
        decoder.load_state_dict(checkpoint2['model_state_dict'])     
    else:
        print("No model files found!!!")
        return False   

    batch_errors = []
    for i, datas in enumerate(data_loader):
        input_tensor = datas[0].to(device)     
        if has_target: 
            bz = input_tensor.size(0)
            _index_in_batch = random.randint(0, bz-1)
            target_tensor = datas[1].to(device)  
            targets = target_tensor.cpu().detach().numpy() 

        outputs_ = predict(encoder, decoder, input_tensor, output_length)  
        outputs = outputs_.cpu().detach().numpy()
 
        if scale_type == 1:
            outputs = custom_scale(outputs, out_scale, b=intercept, inverse_transform=True)
            if has_target: targets = custom_scale(targets, out_scale, b=intercept, inverse_transform=True)
        elif scale_type == 2:
            outputs=min_max_scale(outputs, multipler=out_scale, ranges=wmti_ranges, inverse_transform=True)
            if has_target: targets=min_max_scale(targets, multipler=out_scale, ranges=wmti_ranges, inverse_transform=True)

        if i==0: 
            predictions = np.copy(outputs)  
            if has_target: targets_all = np.copy(targets)
        else:
            predictions = np.concatenate((predictions, outputs), axis=0)  
            if has_target: targets_all = np.concatenate((targets_all, targets), axis=0)

        if has_target:
            rmse = mean_squared_error(targets, outputs, squared=False)
            errs = mean_squared_error(targets, outputs, multioutput='raw_values', squared=False)
            batch_errors.append(rmse)
            if logger != None:
                logger.info(f"--batch#{i}, mean errors: {errs}")
                logger.info(f"--batch#{i}, GT: {np.round(targets[_index_in_batch, :], 5)}")
                logger.info(f"--batch#{i}, PD: {np.round(outputs[_index_in_batch, :], 5)} \n")

    if has_target: 
        plt.plot(batch_errors)  
        plt.show() 

    if os.path.isdir(save_dir) and has_target:
        np.save(f"{save_dir}/wmti_target.npy", targets_all)

    return predictions

def main(args):
    mode = "estimation"
    datapath = args.datapath
    model_path = os.path.join(datapath, args.model_folder)
    if not os.path.exists(model_path): os.mkdir(model_path) 
    logpath = model_path if args.logpath==None else args.logpath
    logger = setup_log(logpath, f"{args.logname}-{mode}")

    logger.info(f"device: {device}")
    logger.info(f"model path: {model_path}")
  
    with open(f"{model_path}/estimation_args.json", 'w') as jf:
        json.dump(args.__dict__, jf, indent=2)

    ## input normalization
    data_norm=args.data_normalizatioin   
    dki_norm_scaler_file = None
    dki_norm_scaler = None
    if args.dki_scaler_filename: 
        dki_norm_scaler_file = f"{model_path}/{args.dki_scaler_filename}"
        dki_norm_scaler = joblib.load(dki_norm_scaler_file)
    logger.info(f"data norm: {data_norm}, dki normalization scaler: {dki_norm_scaler_file}")

    ## output scaling
    scale_type = args.output_scale_type 
    out_scale = args.out_scale 
    intercept = args.out_intercept 
    logger.info(f"scale_type: {scale_type}, target scale: {out_scale}, intercept: {intercept}")

    dataset_fn = args.dataset 
    batch_size= args.batch_size
    has_target=args.has_target

    logger.info(f"dataset: {dataset_fn}, batch size: {batch_size}")

    if has_target:
         data_loader, _, _, _ = preprocess_datasets(datapath, mat_filename=dataset_fn, batch_size=batch_size, train_perc=0.95, val_perc=0.01,
                                                    data_norm=data_norm, scale_type=scale_type, scale=out_scale, intercept=intercept, 
                                                    dki_norm_type=args.dki_norm_type, x_scaler=dki_norm_scaler) 
    else:
        data_loader, dki, _ = preprocess_input_dataset(datapath, mat_filename=dataset_fn, batch_size=batch_size, x_scaler=dki_norm_scaler, dki_norm_type=args.dki_norm_type) 
    ##
    hidden_size = args.hidden_size
    norm = args.model_normalization
    input_seq_length = args.input_seq_length
    output_seq_length = args.output_seq_length    
    
    encoder = EncoderRNN(1, hidden_size, seq_length=input_seq_length, normalization=norm, num_layers=1).to(device)
    attn_decoder = AttnDecoderRNN(hidden_size, 1, x_seq_length=input_seq_length, dropout=args.dropout, normalization=norm).to(device)

    enc_file = f'{model_path}/lstm_encoder.pt'
    dec_file = f'{model_path}/lstm_decoder.pt'    
    wmti = prediction(encoder, attn_decoder, data_loader, scale_type, out_scale, logger=logger, has_target=has_target, intercept=intercept, output_length=output_seq_length,
                            encoder_file=enc_file, decoder_file=dec_file, encoder_ckpt='', decoder_ckpt='', save_dir=model_path) 

    np.save(f"{model_path}/wmti_estimation.npy", wmti) 
    if not has_target:
        np.save(f"{model_path}/input_dki.npy", dki)   

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--has_target", default=False, type=bool)
    parser.add_argument("--dataset", default='datasets/synthetic_wmti_gt_nl3.mat', type=str) #  #wmti_dki_validate_estimation.mat
    parser.add_argument("--dropout", default=0., type=float)
    parser.add_argument("--logpath", default=None)
    parser.add_argument("--logname", default='lstm', type=str)

    parser.add_argument("--datapath", default='/home/yujian/Desktop/cibmaitsrv1/DeepLearnings/whiteMatterModel', type=str)
    parser.add_argument("--model_folder", default='LSTM_96_norm3_scale2_1e-3_sim3', type=str)
    parser.add_argument("--dki_norm_type", default=3, type=bool)
    parser.add_argument("--dki_scaler_filename", default=None, type=str)
    parser.add_argument("--output_scale_type", default=2, type=int)
    parser.add_argument("--out_scale", nargs="+", default=[100], type=int)
    parser.add_argument("--out_intercept", default=None)
    parser.add_argument("--data_normalizatioin", default=True)

    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--hidden_size", default=96, type=int)
    parser.add_argument("--model_normalization", default=True, type=bool)
    parser.add_argument("--input_seq_length", default=6, type=int)
    parser.add_argument("--output_seq_length", default=5, type=int)

    args = parser.parse_args()
    main(args)   