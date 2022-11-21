import os
import torch
import torch.nn as nn
# from .early_stopping import *
from json import dump
import torch
from tqdm import tqdm
torch.manual_seed(0)


####################
# LSTM Autoencoder #
####################
# code inspired by  https://github.com/shobrook/sequitur/blob/master/sequitur/autoencoders/rae.py
# annotation sourced by  https://pytorch.org/docs/stable/nn.html#torch.nn.LSTM
class Encoder(nn.Module):
    def __init__(self, seq_in, n_features, embedding_size, latent_dim, n_layers=1):
        super().__init__()

        # self.seq_len = seq_in
        self.n_features = n_features  # The number of expected features(= dimension size) in the input x
        self.embedding_size = embedding_size  # the number of features in the embedded points of the inputs' number of features
        self.n_layers = n_layers
        self.hidden_size = (2 * embedding_size)  # The number of features in the hidden state h
        self.latent_dim = latent_dim

        self.LSTMenc = nn.LSTM(
            input_size=n_features,
            hidden_size=self.hidden_size,
            num_layers=n_layers,
            batch_first=True
        )
        self.LSTM1 = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=embedding_size,
            num_layers=n_layers,
            batch_first=True
        )
        self.enc = nn.Linear(embedding_size, self.latent_dim)

        self.apply(self.weight_init)
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        # Inputs: input, (h_0, c_0). -> If (h_0, c_0) is not provided, both h_0 and c_0 default to zero.
        # x is the output and so the size is > (batch, seq_len, hidden_size)
        # x, (_,_) = self.LSTM1(x)
        x, (hidden_state, cell_state) = self.LSTMenc(x)
        x, (hidden_state, cell_state) = self.LSTM1(x) ### to switch to x because it needs repeated sequence
        #  ht[:,0,:] == x[0, -1, :] (or also x[0,-1,:]==hidden_state[-1,0,:]) with x ([32, 10, 64]) and h_state ([1, 32, 64])
        last_lstm_layer_hidden_state = hidden_state[-1, :, :] #take only the last layer
        #we need hidden state only here because is like our encoding of the time series
        enc = self.enc(last_lstm_layer_hidden_state)
        return enc

# (2) Decoder
class Decoder(nn.Module):
    def __init__(self, seq_out, embedding_size, output_size, latent_dim, n_layers=1):
        super().__init__()

        self.seq_len = seq_out
        self.embedding_size = embedding_size
        self.hidden_size = (2 * embedding_size)
        self.n_layers = n_layers
        self.output_size = output_size
        self.latent_dim = latent_dim

        self.LSTMdec = nn.LSTM(
            input_size=latent_dim,
            hidden_size=embedding_size,
            num_layers=n_layers,
            batch_first=True
        )

        self.LSTM1 = nn.LSTM(
            input_size=embedding_size,
            hidden_size=self.hidden_size,
            num_layers=n_layers,
            batch_first=True
        )
        self.fc = nn.Linear(self.hidden_size, output_size)

        self.apply(self.weight_init)
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1) # x[0,0,:]==x[0,1,:] ## we nedd to repeat to have an output of secquences (how is our target)
        x, (hidden_state, cell_state) = self.LSTMdec(x)
        x, (_, _) = self.LSTM1(x)
        x = x.reshape((-1, self.seq_len, self.hidden_size))    #fc layer input (input_size, output_size) but if you have (batch, seq_len, input_size) it takes the same operation batch by bath and for
        # each sequence
        # we use the output to target a regression o a label
        out = self.fc(x) #it needs ([32, n, 64]) because in the next operation needs to output a sequence of n
                        #if you don't reshape with sequence lenght in dimension 1 we don'n have out.size = [batch,n , n_features)
                        #also for this we need: x = x.unsqueeze(1).repeat(1, self.seq_len, 1) >>> lstm output >>> reshape
        return out

class LSTM_AE(nn.Module):
    def __init__(self, seq_in = 7, seq_out = 7, n_features = 1, output_size=1, embedding_dim=64, latent_dim=50, n_layers=1):
        super().__init__()

        self.seq_in = seq_in
        self.seq_out = seq_out
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.output_size = output_size
        self.latent_dim = latent_dim

        self.encoder = Encoder(self.seq_in, self.n_features,
                               self.embedding_dim, self.latent_dim, self.n_layers)
        self.decoder = Decoder(self.seq_out, self.embedding_dim,
                               self.output_size, self.latent_dim, self.n_layers)

    def forward(self, x):
        torch.manual_seed(0)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return x, encoded, decoded

    def encode(self, x):
        self.eval()
        encoded = self.encoder(x)
        return encoded

    def decode(self, x):
        self.eval()
        decoded = self.decoder(x)
        squeezed_decoded = decoded.squeeze()
        return squeezed_decoded


def train_lstm_ae(param_conf, train_iter, test_iter, model, criterion, optimizer, scheduler,
                  device,
           out_dir, model_name, epochs=100):
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

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=0.001, gamma=0.5)

    val_loss = 10 ** 16
    for epoch in tqdm(range(epochs), unit='epoch'):
        train_loss = 0.0
        train_steps = 0
        for i, batch in tqdm(enumerate(train_iter), total=len(train_iter), unit="batch"):
            model.train()
            optimizer.zero_grad()

            # y.requires_grad_(True)
            x, enc, y_o = model(batch[0].to(device))
            loss = criterion(y_o.to(device), batch[1].to(device))
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            train_loss += loss.item()
            train_steps += 1

            # if (i + 1) % config['gradient_accumulation_steps'] == 0:
            optimizer.step()
            scheduler.step()

            if i % 10 == 0:
                print("Loss:")
                print(loss.item())

        print('train loss at the end of epoch is ', train_loss/train_steps)

        model.eval()
        val_steps = 0
        temp_val_loss = 0
        with torch.no_grad():
            for i, batch in tqdm(enumerate(test_iter), total=len(test_iter), desc="Evaluating"):

                x_o, enc, y_o = model(batch[0].to(device))
                loss = criterion(y_o.to(device), batch[1].to(device)).item()
                temp_val_loss += loss
                val_steps += 1

            temp_val_loss= temp_val_loss / val_steps
            print('eval loss {}'.format(temp_val_loss))
            if temp_val_loss < val_loss:
                print('val_loss improved from {} to {}, saving model  {} to {}' \
                      .format(val_loss, temp_val_loss, model_name, out_dir))
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'param_conf': param_conf,
                }, out_dir + '/{}.pth'.format(model_name))
                #torch.save(model, out_dir + '{}.pth'.format(model_name))
                val_loss = temp_val_loss