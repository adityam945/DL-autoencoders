import torch
import torch.nn as nn
import torch.optim as optim


from AE_modules import *

from DataProcessing import *

from model_parameters import *

import os
import datetime

class VAE_MNIST(nn.Module):
    def __init__(self):

        super().__init__()
        # model
        self.model_name = 'mnist'
        # features defined
        self.n_latent_features = 64
        pooling_kernel = [2, 2]
        encoder_output_size = 7
        color_channels = 1
        n_neurons_middle_layer = 256 * encoder_output_size * encoder_output_size
        # AE model
        # encoder
        self.encoder = EncoderNetwork(color_channels, pooling_kernel, n_neurons_middle_layer)
        # middel layer
        self.middle_layer1 = nn.Linear(n_neurons_middle_layer, self.n_latent_features)
        self.middle_layer2 = nn.Linear(n_neurons_middle_layer, self.n_latent_features)
        self.middle_layer3 = nn.Linear(self.n_latent_features, n_neurons_middle_layer)
        # decoder
        self.decoder = DecoderNetwork(color_channels, pooling_kernel, encoder_output_size)
        # load data
        self.train_loader, self.test_loader = load_data_mnist()
        # init a dictionary to store data
        self.history = {"loss":[], "val_loss":[]}
        # model name
        check_create_path(self.model_name)
        if not os.path.exists(self.model_name):
            os.mkdir(self.model_name)

    def forward(self, x):
        # Encoder
        # image
        
        h = self.encoder(x)
        # Bottle-neck
        z, mu, logvar = bottleneck_params(self.middle_layer1, self.middle_layer2, h)
        # decoder
        z = self.middle_layer3(z)
        
        d = self.decoder(z)
        return d, mu, logvar


    def init_model(self):
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self
    # Train 
    def fit_train(self, epoch):
        self.train()
        print(f"\nEpoch: {epoch+1:d} {datetime.datetime.now()}")
        train_loss = 0
        samples_cnt = 0
        for batch_idx, (inputs, _) in enumerate(self.train_loader):
            inputs = inputs
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self(inputs)
            loss = loss_function(recon_batch, inputs, mu, logvar)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            samples_cnt += inputs.size(0)
            if batch_idx%50 == 0:
                print(batch_idx, len(self.train_loader), f"Loss: {train_loss/samples_cnt:f}")


        self.history["loss"].append(train_loss/samples_cnt)

    def test(self, epoch):
        self.eval()
        val_loss = 0
        samples_cnt = 0
        with torch.no_grad():
            for batch_idx, (inputs, _) in enumerate(self.test_loader):

                recon_batch, mu, logvar = self(inputs)


                val_loss += loss_function(recon_batch, inputs, mu, logvar).item()
                samples_cnt += inputs.size(0)
                if batch_idx == 0:
                    save_image(inputs, f"{self.model_name}/encdoed_input{str(epoch)}.png", nrow=8)
                    save_image_for_epoch(recon_batch, self.model_name, epoch)

        # print(batch_idx, len(self.test_loader), f"Total Loss: {val_loss/samples_cnt:f}")
        print(f"Total Loss: {val_loss/samples_cnt:f}", 'Test accuracy: {:.4f}'.format(batch_idx/len(self.test_loader)))
        self.history["val_loss"].append(val_loss/samples_cnt)

