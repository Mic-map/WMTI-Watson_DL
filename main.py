
import os
import argparse
import json
import joblib

from util_dev import device
from utils import preprocess_datasets, setup_log
from RNNEncoderDecoder import EncoderRNN, AttnDecoderRNN
from train import trainIters
from test import test


def main(args):
    mode = args.mode
    datapath = args.datapath
    model_path = os.path.join(datapath, args.model_folder)
    if not os.path.exists(model_path): os.mkdir(model_path) 

    logpath = model_path if args.logpath==None else args.logpath
    logger = setup_log(logpath, f"{args.logname}-{mode}")
    logger.info(f"device: {device}")
    logger.info(f"model path: {model_path}")
     
    with open(f"{model_path}/{mode}_args.json", 'w') as jf:
        json.dump(args.__dict__, jf, indent=2)

    ## input normalization
    data_norm=args.data_normalization
    dki_norm_type=args.input_scale_type
    if args.input_scaler_filename != None:
        logger.info(f"Input normalization: {data_norm}, using norm scaler: {args.input_scaler_filename}")
        dki_norm_scaler = joblib.load(os.path.join(model_path, args.input_scaler_filename))
    else:
        logger.info(f"Input normalization: {data_norm}, norm type: {dki_norm_type}")
        dki_norm_scaler = None

    ## output scaling
    scale_type = args.output_scale_type 
    out_scale = args.output_scale if scale_type==1 else args.output_scale[0]
    intercept = args.output_intercept 
    if scale_type ==2:
        wmti_ranges=[[0,1], [0, 3], [0, 3], [0, 3], [0, 1]]
    else:
        wmti_ranges = None

    logger.info(f"scale_type: {scale_type}, target scale: {out_scale}, intercept: {intercept}, wmti_ranges: {wmti_ranges}")

    dataset_fn = args.dataset 
    batch_size= args.batch_size
    logger.info(f"dataset: {dataset_fn}, batch size: {batch_size}")
    train_data, val_data, test_data, dki_norm_scaler = preprocess_datasets(datapath, mat_filename=dataset_fn, batch_size=batch_size, train_perc=0., val_perc=0., 
                            data_norm=data_norm, scale_type=scale_type, scale=out_scale, intercept=intercept, dki_name = 'dki', wmti_name='wmti_paras',
                            dki_norm_type=dki_norm_type, x_scaler=dki_norm_scaler, wmti_ranges=wmti_ranges) 
    ## save scaler
    if (args.input_scaler_filename == None) and (dki_norm_scaler != None):
        scaler_saveto = os.path.join(model_path, "dki_norm_scaler.gz")
        joblib.dump(dki_norm_scaler, scaler_saveto) 
        logger.info(f"dki normalization scaler saved to : {scaler_saveto}")

    ##
    hidden_size = args.hidden_size
    norm = args.model_norm
    input_seq_length = args.input_seq_length
    output_seq_length = args.output_seq_length    
    
    encoder = EncoderRNN(1, hidden_size, seq_length=input_seq_length, normalization=norm, num_layers=1).to(device)
    attn_decoder = AttnDecoderRNN(hidden_size, 1, x_seq_length=input_seq_length, dropout=args.dropout, normalization=norm).to(device)

    ##################### Train ################
    if mode == "train":
        teacher_forcing_ratio = args.teacher_forcing_ratio 
        logger.info(f"teacher_forcing_ratio: {teacher_forcing_ratio}")
        trainIters(encoder, attn_decoder, train_data, n_epochs=args.num_epochs, logger=logger, st_epoch=args.start_epoch, save_dir=model_path, 
                    val_data_loader=val_data, print_every=args.print_every, plot_every=args.plot_every, learning_rate=args.learning_rate, 
                    teacher_forcing_ratio=teacher_forcing_ratio, output_length=output_seq_length)
    elif mode=="test":
    ##################### Test ################
        logger.info("*********************** Testing model *********************")
        ckpt_epoch = args.load_checkpoint
        enc_ckt = f'{model_path}/checkpoints/ckpt_ep-{ckpt_epoch}_encoder.pt'
        dec_ckt = f'{model_path}/checkpoints/ckpt_ep-{ckpt_epoch}_decoder.pt'
        test(logger, encoder, attn_decoder, test_data, scale_type, out_scale, intercept=intercept, output_length=output_seq_length, wmti_ranges=wmti_ranges,
                                encoder_file='', decoder_file='', encoder_ckpt=enc_ckt, decoder_ckpt=dec_ckt, save_dir=model_path)    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="train", type=str) 
    parser.add_argument("--logpath", default=None)
    parser.add_argument("--logname", default='lstm', type=str)    
    parser.add_argument("--print_every", default=2000, type=int)
    parser.add_argument("--plot_every", default=20000, type=int)

    parser.add_argument("--dataset", default='datasets/training.mat', type=str)  
    parser.add_argument("--datapath", default='my/root/path', type=str)
    parser.add_argument("--model_folder", default='model', type=str)
    parser.add_argument("--data_normalization", default=True, type=bool)
    parser.add_argument("--input_scale_type", default=3, type=str) 
    parser.add_argument("--input_scaler_filename", default=None , type=str) 
    parser.add_argument("--output_scale_type", default=2, type=int) #1 #2
    parser.add_argument("--output_scale",  nargs="+", default=[100], type=int) 
    parser.add_argument("--output_intercept", default=None)  

    parser.add_argument("--num_epochs", default=1200, type=int)
    parser.add_argument("--learning_rate", default=0.001, type=float)
    parser.add_argument("--start_epoch", default=-1)
    parser.add_argument("--dropout", default=0)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--hidden_size", default=96, type=int)
    parser.add_argument("--model_norm", default=True, type=bool)
    parser.add_argument("--input_seq_length", default=6, type=int)
    parser.add_argument("--output_seq_length", default=5, type=int)
    parser.add_argument("--teacher_forcing_ratio", default=0.3, type=float)
    parser.add_argument("--load_checkpoint", default=1199, type=int)

    args = parser.parse_args()
    main(args)
