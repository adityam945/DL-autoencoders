import torch
import torch.nn.functional as F


def reparametrized_params(mu, logvar):
    std = logvar.mul(0.5).exp_()
    esp = torch.randn(*mu.size())
    z = mu + std * esp
    return z

def bottleneck_params(middle_layer1, middle_layer2, h):
    mu, logvar = middle_layer1(h), middle_layer2(h)
    z = reparametrized_params(mu, logvar)
    return z, mu, logvar
    

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, size_average=False)        
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD