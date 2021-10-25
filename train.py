import random
import os
import time
import numpy as np
# import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch import optim

from util_dev import device
from utils import showPlot, timeSince, load_model_from_checkpoint

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, teacher_forcing_ratio, output_length=None):
    '''
    input_tensor/target_tensor: batch_size, seq_len
    '''
    bz = input_tensor.size(0)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    target_length = target_tensor.size(1) if output_length==None else output_length

    loss = 0

    encoder_outputs, encoder_hidden, encoder_cell = encoder(input_tensor)
    index_in_batch = random.randint(0, bz-1)

    decoder_hidden = encoder_hidden[-1]
    decoder_cell = encoder_cell[-1]

    decoder_input = input_tensor[:,-1].view(-1,1) # the last input element, batch x 1

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    decoder_outputs = []

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_cell, decoder_attention = decoder(
                decoder_input, decoder_hidden, decoder_cell, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[:,di].view(-1,1))
            decoder_input = target_tensor[:,di].view(-1,1) # Teacher forcing    
            decoder_outputs.append(decoder_output.cpu().detach().numpy().flatten()[index_in_batch])        
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_cell, decoder_attention = decoder(
                decoder_input, decoder_hidden, decoder_cell, encoder_outputs)

            decoder_input = decoder_output.view(-1,1)
            loss += criterion(decoder_output, target_tensor[:,di].view(-1,1))   
            decoder_outputs.append(decoder_output.cpu().detach().numpy().flatten()[index_in_batch])  #store 1st item in the batch

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item(), decoder_outputs, index_in_batch

def evaluate(encoder, decoder, criterion, input_tensor, target_tensor, output_length=None):
    with torch.no_grad():
        bz = input_tensor.size(0)
        #input_length = input_tensor.size(1)
        target_length = target_tensor.size(1) if output_length==None else output_length

        # encoder_outputs = torch.zeros(bz, input_length, encoder.hidden_size, device=device)
        outputs = torch.zeros(bz, target_length, device=device)

        loss = 0

        encoder_outputs, encoder_hidden, encoder_cell = encoder(input_tensor)

        decoder_hidden = encoder_hidden[-1]
        decoder_cell = encoder_cell[-1]
        #decoder_attentions = torch.zeros(input_length, input_length)
        decoder_input = input_tensor[:, -1].view(-1,1)

        decoder_outputs=[]
        index_in_batch = random.randint(0, bz-1)

        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_cell, decoder_attention = decoder(
                decoder_input, decoder_hidden, decoder_cell, encoder_outputs)
            #decoder_attentions[di] = decoder_attention.data
            decoder_input = decoder_output.detach().view(-1,1)
            loss += criterion(decoder_output, target_tensor[:,di].view(-1,1))  
            decoder_outputs.append(decoder_output.cpu().detach().numpy().flatten()[index_in_batch])
            outputs[:, di] = decoder_input[:,0]

    return loss.item(), decoder_outputs, index_in_batch, outputs 


def trainIters(encoder, decoder, train_data_loader, n_epochs, logger, save_dir='', st_epoch=-1, val_data_loader=None,
                                device=device, print_every=1000, plot_every=1000, learning_rate=0.001, teacher_forcing_ratio=0.3, output_length=None):
    start = time.time()
    plot_losses = []
    plot_val_losses=[]
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    itr = 0   

    logger.info(encoder)
    logger.info(decoder)
    logger.info(f"lr: {learning_rate}")
    
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        checkpoint_dir = os.path.join(save_dir, "checkpoints")
        if not os.path.exists(checkpoint_dir): os.mkdir(checkpoint_dir)

    num_batches = len(train_data_loader)
    batch_size = train_data_loader.batch_size    
    n_iters = n_epochs * num_batches #total batches
    logger.info(f"batch_size: {batch_size}, nBatches: {num_batches}, nEpochs: {n_epochs}, totalIters: {n_iters}")

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=1e-2)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate, weight_decay=1e-2)

    factor = 0.95
    patience = 6
    min_lr = 1e-6
    logger.info(f"Epoch: lr_scheduler.ReduceLROnPlateau: factor={factor}, patience={patience}, min_lr={min_lr}")

    encoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, factor=factor, patience=patience, min_lr=min_lr)
    decoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(decoder_optimizer, factor=factor, patience=patience, min_lr=min_lr)

    criterion = nn.MSELoss()

    if st_epoch > 0:
        enc_ckp_file = os.path.join(checkpoint_dir, f"ckpt_ep-{st_epoch}_encoder.pt")
        dec_ckp_file = os.path.join(checkpoint_dir, f"ckpt_ep-{st_epoch}_decoder.pt")

        st_epoch, encoder, encoder_optimizer, encoder_scheduler, _ = load_model_from_checkpoint(enc_ckp_file, encoder, 
                                            encoder_optimizer, encoder_scheduler)
        st_epoch, decoder, decoder_optimizer, decoder_scheduler, _ = load_model_from_checkpoint(dec_ckp_file, decoder, 
                                            decoder_optimizer, decoder_scheduler)
                                 
                
    st_epoch += 1
   
    check_freq = max(1, int(n_epochs/20))
    logger.info(f"epoch starts from : {st_epoch}")

    for epoch in range(n_epochs):
        epoch_loss = 0

        for i, datas in enumerate(train_data_loader):

            input_tensor, target_tensor = datas[0].to(device), datas[1].to(device)

            loss, predictions, index_in_batch = train(input_tensor, target_tensor, encoder,
                        decoder, encoder_optimizer, decoder_optimizer, criterion, teacher_forcing_ratio, output_length)

            print_loss_total += loss
            plot_loss_total += loss
            epoch_loss += loss

            itr += 1
            if (itr % print_every == 0) or itr ==n_iters:
                if itr ==n_iters and (itr % print_every > 0):
                    print_loss_avg = print_loss_total / (itr % print_every)
                else:
                    print_loss_avg = print_loss_total / print_every

                print_loss_total = 0
                logger.info(f"learning_rate: {encoder_optimizer.param_groups[0]['lr']}")
                logger.info('Epoch-#%d %s (iter-%d %.2f%%) [%.4f]' % (epoch, timeSince(start, itr / n_iters),
                                            itr , itr / n_iters * 100, print_loss_avg))
                logger.info(f"groundTruth: {target_tensor.cpu().detach().numpy()[index_in_batch,:]}")
                logger.info(f"predicted: {predictions}")

                if val_data_loader is not None:
                    logger.info("++++++++++++++++++++ Evaluation ++++++++++++++++++++++")
                    x, y = iter(val_data_loader).next()
                    val_input, val_target = x.to(device), y.to(device)
                    val_loss, val_pred, val_index, _ = evaluate(encoder, decoder, criterion, val_input, val_target, output_length)
                    logger.info(f"evaluation loss: [{val_loss}]")
                    logger.info(f"val GT: {val_target.cpu().detach().numpy()[val_index,:]}")
                    logger.info(f"val pred: {val_pred}")  

                logger.info("\n")                 

            if (itr % plot_every == 0) or itr ==n_iters:
                if itr ==n_iters and (itr % plot_every > 0):
                    plot_loss_avg = plot_loss_total / (itr % plot_every)
                else:
                    plot_loss_avg = plot_loss_total / plot_every

                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

                if val_data_loader is not None:
                    vloss_total = 0
                    for ival, (xval, yval) in enumerate(val_data_loader):
                        vloss, _, _ , _ = evaluate(encoder, decoder, criterion, xval.to(device), yval.to(device))
                        vloss_total += vloss    
                    plot_val_losses.append(vloss_total/(ival + 1))

        if save_dir and ((epoch+1) % check_freq == 0 or epoch==n_epochs):
            checkpoint_encoder = os.path.join(checkpoint_dir, f"ckpt_ep-{st_epoch+epoch}_encoder.pt")
            torch.save({'epoch': epoch + st_epoch,
                'model_state_dict': encoder.state_dict(),
                'opimizer_state_dict': encoder_optimizer.state_dict(),
                'scheduler_state_dict': encoder_scheduler.state_dict(),
                'device': device
            }, checkpoint_encoder) 
            #           
            checkpoint_decoder = os.path.join(checkpoint_dir, f"ckpt_ep-{st_epoch+epoch}_decoder.pt")
            torch.save({'epoch': epoch + st_epoch,
                'model_state_dict': decoder.state_dict(),
                'opimizer_state_dict': decoder_optimizer.state_dict(),
                'scheduler_state_dict': decoder_scheduler.state_dict(),
                'device': device
            }, checkpoint_decoder) 

        # #update after each epoch
        epoch_loss = round(epoch_loss/num_batches, 4)
        encoder_scheduler.step(epoch_loss)
        decoder_scheduler.step(epoch_loss)              

    #save model
    torch.save(encoder.state_dict(), os.path.join(save_dir, 'lstm_encoder.pt'))
    torch.save(decoder.state_dict(), os.path.join(save_dir, 'lstm_decoder.pt'))

    if len(plot_losses)>1:
        plot_losses_all = [plot_losses, plot_val_losses] if len(plot_val_losses)>1 else [plot_losses]
        showPlot(plot_losses_all, checkpoint_dir)
        if os.path.isdir(checkpoint_dir):
            np.save(os.path.join(checkpoint_dir, 'training_avg_losses.npy'), np.array(plot_losses))
            if len(plot_val_losses)>1: 
                np.save(os.path.join(checkpoint_dir, 'val_avg_losses.npy'), np.array(plot_val_losses))
