import os
import argparse
import numpy as np
import json
import torch
import torch.nn as nn

from util_dev import device
from utils import min_max_scale, custom_scale, preprocess_input_dataset, setup_log
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
                            intercept=None, encoder_file='', decoder_file='', encoder_ckpt='', decoder_ckpt=''):

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

    for i, datas in enumerate(data_loader):
        input_tensor = datas[0].to(device)     

        outputs_ = predict(encoder, decoder, input_tensor, output_length)  
        outputs = outputs_.cpu().detach().numpy()
 
        if scale_type == 1:
            outputs = custom_scale(outputs, out_scale, b=intercept, inverse_transform=True)
        elif scale_type == 2:
            outputs=min_max_scale(outputs, multipler=out_scale, inverse_transform=True)

        if i==0: 
            predictions = np.copy(outputs)  
        else:
            predictions = np.concatenate((predictions, outputs), axis=0)  

    return predictions

def estimate(encoder, decoder, input_data_loader, scale_type, out_scale, intercept=None, output_length=5,
                            encoder_file='', decoder_file='', encoder_ckpt='', decoder_ckpt='', save_dir=''):

    predictions = prediction(encoder, decoder, input_data_loader, scale_type=scale_type, out_scale=out_scale, intercept=intercept, output_length=output_length,
                            encoder_file=encoder_file, decoder_file=decoder_file, encoder_ckpt=encoder_ckpt, decoder_ckpt=decoder_ckpt)
    if os.path.isdir(save_dir):
        np.save(f"{save_dir}/wmti_estimation.npy", predictions)

    return predictions

def main(args):
    logpath = os.path.dirname(os.path.realpath(__file__)) if args.logpath==None else args.logpath
    logger = setup_log(logpath, args.logname)
    logger.info(f"device: {device}")
    datapath = args.datapath
    model_path = os.path.join(datapath, args.model_folder)
    if not os.path.exists(model_path): os.mkdir(model_path) 
    logger.info(f"model path: {model_path}")
  
    with open(f"{model_path}/estimation_args.json", 'w') as jf:
        json.dump(args.__dict__, jf, indent=2)

    ##
    data_norm=args.data_norm
    norm_type=args.dki_norm_type#'MinMax'#'Standard'#
    logger.info(f"data norm: {data_norm}, dki normalization type: {norm_type}")
    scale_type = args.output_scale_type #1 #2 #wmti scale type
    out_scale = args.out_scale #[100, 100, 100, 100, 100]#[100, 100, 100, 100, 1]#[60, 100, 100, 60, 1]#100
    intercept = args.out_intercept #[0, 0, 0, 0, 0 ]#[40, 0, 0, 40, 0 ]
    logger.info(f"scale_type: {scale_type}, target scale: {out_scale}, intercept: {intercept}")
    dataset_fn = args.dataset #'wmti_dki_constraint_c2.mat'#'wmti_dki_constraint.mat'
    batch_size= args.batch_size
    logger.info(f"dataset: {dataset_fn}, batch size: {batch_size}")


    data_loader, dki = preprocess_input_dataset(datapath, mat_filename=dataset_fn, batch_size=batch_size, 
                                                    data_norm=data_norm, scale_type=norm_type) 
    ##
    hidden_size = args.hidden_size
    norm = args.model_norm
    input_seq_length = args.input_seq_length
    output_seq_length = args.output_seq_length    
    
    encoder = EncoderRNN(1, hidden_size, seq_length=input_seq_length, normalization=norm, num_layers=1).to(device)
    attn_decoder = AttnDecoderRNN(hidden_size, 1, x_seq_length=input_seq_length, dropout=args.dropout, normalization=norm).to(device)

    # ckpt_epoch = args.load_checkpoint
    # enc_ckt = f'{model_path}/checkpoints/ckpt_ep-{ckpt_epoch}_encoder.pt'
    # dec_ckt = f'{model_path}/checkpoints/ckpt_ep-{ckpt_epoch}_decoder.pt'
    enc_file = f'{model_path}/lstm_encoder.pt'
    dec_file = f'{model_path}/lstm_decoder.pt'    
    wmti = estimate(encoder, attn_decoder, data_loader, scale_type, out_scale, intercept=intercept, output_length=output_seq_length,
                            encoder_file=enc_file, decoder_file=dec_file, encoder_ckpt='', decoder_ckpt='', save_dir=model_path) 

    np.save(f"{model_path}/input_dki.npy", dki)   
    np.save(f"{model_path}/output_wmti.npy", wmti)                            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default='wmti_dki_constraint.mat', type=str)
    parser.add_argument("--dropout", default=0., type=float)
    parser.add_argument("--logpath", default=None)
    parser.add_argument("--logname", default='lstm-estimation', type=str)
    parser.add_argument("--datapath", default='/home/yujian/Desktop/cibmaitsrv1/DeepLearnings/whiteMatterModel', type=str)
    parser.add_argument("--model_folder", default='saveLSTM_96_scale3_100_1e-3', type=str)
    parser.add_argument("--data_norm", default=True, type=bool)
    parser.add_argument("--dki_norm_type", default="MinMax", type=str)
    parser.add_argument("--output_scale_type", default=2, type=int)
    parser.add_argument("--out_scale", default=100)
    parser.add_argument("--out_intercept", default=None)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--hidden_size", default=96, type=int)
    parser.add_argument("--model_norm", default=True, type=bool)
    parser.add_argument("--input_seq_length", default=6, type=int)
    parser.add_argument("--output_seq_length", default=5, type=int)
    # parser.add_argument("--load_checkpoint", default=1599, type=int)

    args = parser.parse_args()
    main(args)   