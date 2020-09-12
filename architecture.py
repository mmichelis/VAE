# ------------------------------------------------------------------------------
# PyTorch implementation of the network architecture. 
# ------------------------------------------------------------------------------

import torch 
import numpy as np


class VAE(torch.nn.Module):
    """
    Variational Autoencoder architecture with Conv2D in the encoder, and ConvTranspose2d in the decoder. The encoder outputs mean and log(variance) through two linear layers (which follow the Conv2D layers before that).

    Encoder expects an input of shape [N, 1, 28, 28]
    Decoder expects an input of shape [N, hidden_dim]
    """
    def __init__(self):
        super(VAE, self).__init__()

        self.hidden_dimension = 64
        
        self.affine = True

        # Encoder
        self.conv1 = torch.nn.Conv2d(1, 8, kernel_size=7)
        self.relu1 = torch.nn.ReLU()
        self.batchn1 = torch.nn.BatchNorm2d(8, affine=self.affine)
        # Should now be 22
        self.conv2 = torch.nn.Conv2d(8, 32, kernel_size=5)
        self.relu2 = torch.nn.ReLU()
        self.batchn2 = torch.nn.BatchNorm2d(32, affine=self.affine)
        # Should now be 18
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=3)
        self.relu3 = torch.nn.ReLU()
        self.batchn3 = torch.nn.BatchNorm2d(64, affine=self.affine)
        # Should now be 16

        self.linear1 = torch.nn.Linear(16*16*64, self.hidden_dimension)
        self.linear2 = torch.nn.Linear(16*16*64, self.hidden_dimension)

        # Decoder
        self.linear3 = torch.nn.Linear(self.hidden_dimension, 256)

        self.convT1 = torch.nn.ConvTranspose2d(1, 8, kernel_size=3)
        self.relud1 = torch.nn.ReLU()
        self.batchnd1 = torch.nn.BatchNorm2d(8, affine=self.affine)
        self.convT2 = torch.nn.ConvTranspose2d(8, 16, kernel_size=5)
        self.relud2 = torch.nn.ReLU()
        self.batchnd2 = torch.nn.BatchNorm2d(16, affine=self.affine)
        self.convT3 = torch.nn.ConvTranspose2d(16, 32, kernel_size=7)
        self.relud3 = torch.nn.ReLU()
        self.batchnd3 = torch.nn.BatchNorm2d(32, affine=self.affine)

        self.convFin = torch.nn.Conv2d(32, 1, kernel_size=1)
        self.sigmoid = torch.nn.Sigmoid()
    

    def encoder(self, X):
        # Input of shape [N, 1, 28, 28]
        #out = torch.reshape(X, [-1, 1, 28, 28])
        out = self.batchn3(self.relu3(self.conv3(self.batchn2(self.relu2(self.conv2(self.batchn1(self.relu1(self.conv1(X)))))))))

        out = torch.reshape(out, [-1, 16*16*64])
        mean = self.linear1(out)
        log_var = self.linear2(out)

        xi = torch.normal(torch.zeros_like(mean))
        # std is actually log(std**2)
        Z = mean + xi * torch.exp(0.5 * log_var)
        #Z = mean + xi * log_var

        return Z, mean, log_var


    def decoder(self, Z):
        # Input of shape [N, hidden_dim]
        out = self.linear3(Z)
        out = torch.reshape(out, [-1, 1, 16, 16])

        out = self.batchnd3(self.relud3(self.convT3(self.batchnd2(self.relud2(self.convT2(self.batchnd1(self.relud1(self.convT1(out)))))))))
        # Ensure output between 0 and 1
        X_hat = self.sigmoid(self.convFin(out))
        #X_hat = torch.reshape(X_hat, [-1, 28, 28])

        return X_hat


    # def forward(self, X):
    #     X_hat = self.decoder(self.encoder(X))
    #     return X_hat


    @staticmethod
    def initialize_weight(module):
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.normal_(module.weight, 0.0, 0.02)
        elif isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, 0.0, 0.02)
        elif isinstance(module, torch.nn.BatchNorm2d):
            torch.nn.init.normal_(module.weight, 0.0, 0.02)
            torch.nn.init.constant_(module.bias, 0)


