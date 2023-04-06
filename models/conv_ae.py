import os
import torch.nn as nn
import torch
from tqdm import tqdm
import sys
sys.path.append('..')
from utils.opt import EarlyStopping
from utils.layers import conv_block, deconv_block
import matplotlib.pyplot as plt

class Encoder(nn.Module):
    def __init__(self, in_channel=1, kernel_size=3, filter_num_list=None, latent_dim=10,
                 img_heigth=16, img_width=16, activation=nn.ReLU(), flattened=True, padding=1):
        super(Encoder, self).__init__()

        self.nn_enc = nn.Sequential()

        if filter_num_list is None:
            self.filter_num_list = [1, 32, 64]

        self.in_channel = in_channel
        self.kernel_size = kernel_size
        self.filter_num_list = filter_num_list
        self.latent_dim = latent_dim
        self.h = img_heigth
        self.w = img_width
        self.act = activation
        self.flattened = flattened
        self.padding = padding

        for i, num in enumerate(self.filter_num_list):
            if i + 2 == len(self.filter_num_list):
                self.nn_enc.add_module('enc_lay_{}'.format(i), conv_block(num, self.filter_num_list[i + 1],
                                                                          self.kernel_size,  activation=self.act,
                                                                          padding=self.padding))
                break
            self.nn_enc.add_module('enc_lay_{}'.format(i), conv_block(num, self.filter_num_list[i+1],
                                                                  self.kernel_size, activation=self.act,
                                                                      padding=self.padding))

        self.flattened_size, self.h_enc, self.w_enc = self._get_final_flattened_size()

        if self.flattened:
            self.encoder_layer = nn.Linear(self.flattened_size, self.latent_dim)

        self.init_kaiming_normal()

    def init_kaiming_normal(self, mode='fan_in'):
        print('Initializing conv2d weights with Kaiming He normal')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode=mode)
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(
                (1, self.in_channel, self.h, self.w)
            )
            x = self.nn_enc(x)
            _, c, w, h = x.size()
        return c * w * h, w, h

    def forward(self, x):
        enc = self.nn_enc(x)
        if self.flattened:
            enc = enc.view(-1, self.flattened_size)
            enc = self.encoder_layer(enc)
        return enc


class Decoder(nn.Module):
    def __init__(self, in_channel=1, kernel_size=3, filter_num_list=None, latent_dim=10, flattened_size=None,
                 img_heigth=16, img_width=16, h_enc=2, w_enc=2, activation=nn.ReLU(), flattened=True):
        super(Decoder, self).__init__()

        self.nn_dec = nn.Sequential()

        if filter_num_list is None:
            self.filter_num_list = [32, 64]
        self.in_channel = in_channel
        self.kernel_size = kernel_size
        self.filter_num_list = filter_num_list
        self.latent_dim = latent_dim
        self.flattened_size = flattened_size
        self.h = img_heigth
        self.w = img_width
        self.h_enc = h_enc
        self.w_enc = w_enc
        self.filter_num_list = self.filter_num_list[::-1]
        self.act = activation
        self.flattened = flattened

        if self.flattened:
            self.reshape = nn.Linear(self.latent_dim, self.flattened_size)

        for i, num in enumerate(self.filter_num_list):
            if i + 2 == len(self.filter_num_list):
                self.nn_dec.add_module('dec_lay_{}'.format(i), deconv_block(num, self.filter_num_list[i+1], activation=self.act))
                break
            self.nn_dec.add_module('dec_lay_{}'.format(i), deconv_block(num, self.filter_num_list[i + 1],  activation=self.act))

        self.decoder_layer = nn.Conv2d(self.filter_num_list[i+1], self.in_channel, kernel_size=1)
        self.init_kaiming_normal()

    def init_kaiming_normal(self, mode='fan_in'):
        print('Initializing conv2d weights with Kaiming He normal')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode=mode)
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        if self.flattened:
            x = self.reshape(x)
            x = x.view((-1, self.filter_num_list[0], self.h_enc, self.w_enc))
        dec = self.nn_dec(x)
        out = self.decoder_layer(dec)
        return out

# define the NN architecture
class CONV_AE(nn.Module):
    def __init__(self, in_channel=1,  heigth=16, width=16, kernel_size=3, dilation=1, filter_num=2,
                 latent_dim=50, n_layers=1, activation = nn.ReLU(), increasing=False, flattened=True):
        super(CONV_AE, self).__init__()

        self.in_channel = in_channel
        self.filter_num = filter_num
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.latent_dim = latent_dim
        self.act = activation
        self.h = heigth  # number of features (to transpose respect to the df format)
        self.w = width  # sequence length (to transpose respect to the df format)
        self.increasing = increasing
        self.flattened = flattened
        self.dilation = dilation

        self.padding = int((self.dilation * (self.kernel_size - 1) / 2))

        if self.increasing:
            self.filter_num_list = [int(self.filter_num * ((ix + 1) * 2)) for ix in range(self.n_layers)]
        else:
            self.filter_num_list = [int(self.filter_num / ((ix + 1)*2)) for ix in range(self.n_layers)]

        self.filter_num_list = [self.in_channel] + [self.filter_num] + self.filter_num_list

        self.encoder = Encoder(self.in_channel, kernel_size=self.kernel_size, filter_num_list=self.filter_num_list,
                               latent_dim=self.latent_dim,
                               img_heigth=self.h, img_width=self.w, activation=self.act, padding=self.padding, flattened=self.flattened)
        self.flattened_size = self.encoder.flattened_size
        self.decoder = Decoder(self.in_channel, kernel_size=self.kernel_size, filter_num_list=self.filter_num_list,
                               latent_dim=self.latent_dim, flattened_size=self.flattened_size,
                               img_heigth=self.h, img_width=self.w, h_enc=self.encoder.h_enc, w_enc=self.encoder.w_enc,
                               activation=self.act, flattened=self.flattened)

    def forward(self, x):
        enc = self.encoder(x)
        out = self.decoder(enc)
        return out


def train_conv_ae(param_conf, train_iter, test_iter, model, criterion, optimizer,scheduler, device,
           out_dir, model_name, epochs=100, es_patience=10):
    """
    Training function.

    Args:
        train_iter: (DataLoader): train data iterator
        test_iter: (DataLoader): test data iterator
        model: model
        criterion: loss to use
        optimizer: optimizer to use
        config:
    """

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    early_stopping = EarlyStopping(patience=es_patience)

    val_loss = 10 ** 16
    val_losses = []
    train_losses = []
    for epoch in tqdm(range(epochs), unit='epoch'):
        train_loss = 0.0
        train_steps = 0
        for i, batch in tqdm(enumerate(train_iter), total=len(train_iter), unit="batch"):
            model.train()
            optimizer.zero_grad()

            # y.requires_grad_(True)
            y_o = model(batch[0].to(device))
            loss = criterion(y_o.to(device), batch[1].to(device))
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            train_loss += loss.item()
            train_steps += 1

            # if (i + 1) % config['gradient_accumulation_steps'] == 0:
            optimizer.step()

            if i % 10 == 0:
                print("Loss:")
                print(loss.item())

        print('train loss at the end of epoch is ', train_loss/train_steps)
        train_losses.append(train_loss/train_steps)

        model.eval()
        val_steps = 0
        temp_val_loss = 0
        with torch.no_grad():
            for i, batch in tqdm(enumerate(test_iter), total=len(test_iter), desc="Evaluating"):

                y_o = model(batch[0].to(device))
                loss = criterion(y_o.to(device), batch[1].to(device)).item()
                temp_val_loss += loss
                val_steps += 1

            temp_val_loss= temp_val_loss / val_steps
            print('eval loss {}'.format(temp_val_loss))
            scheduler.step(temp_val_loss)
            
            val_losses.append(temp_val_loss)     
            epochs = [x for x in range(len(train_losses))]
            
      
            fig = plt.figure(figsize=(4,3))

            plt.plot(epochs, train_losses, marker='.',label = "train mse loss")
            plt.plot(epochs, val_losses,marker='.', label = "val mse loss")
            plt.xlabel('epochs', fontsize=18)
            plt.ylabel('mse value', fontsize=18)
            plt.yticks(fontsize=16)
            plt.xticks(fontsize=16)
            plt.legend(fontsize=14)
            plt.show()

            early_stopping(temp_val_loss)
            if early_stopping.early_stop:
                break

            if temp_val_loss < val_loss:
                print('val_loss improved from {} to {}, saving model  {} to {}' \
                      .format(val_loss, temp_val_loss, model_name, out_dir))
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'param_conf': param_conf,
                }, out_dir + '/{}.pth'.format(model_name))
                val_loss = temp_val_loss