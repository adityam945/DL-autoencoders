import torch.nn as nn


class EncoderLayer(nn.Module):
    def __init__(self, input_channels, output_channels, stride, kernel, pad):
        super().__init__()
        self.conv_encoder = nn.Conv2d(input_channels, output_channels, kernel_size=kernel, padding=pad, stride=stride)
        self.batchNorm_encoder = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.batchNorm_encoder(self.conv_encoder(x)))

class EncoderNetwork(nn.Module):
    def __init__(self, color_channels, pooling_kernels, n_neurons_in_middle_layer):
        self.n_neurons_in_middle_layer = n_neurons_in_middle_layer
        super().__init__()
        self.layer_initial = EncoderLayer(color_channels, 32, stride=1, kernel=1, pad=0)
        self.layer1 = EncoderLayer(32, 64, stride=1, kernel=3, pad=1)
        self.layer2 = EncoderLayer(64, 128, stride=pooling_kernels[0], kernel=3, pad=1)
        self.layer3 = EncoderLayer(128, 256, stride=pooling_kernels[1], kernel=3, pad=1)

    def forward(self, x):
        out = self.layer3(self.layer2(self.layer1(self.layer_initial(x))))
        return out.view(-1, self.n_neurons_in_middle_layer)
        
class DecoderLayer(nn.Module):
    def __init__(self, input_channels, output_channels, stride, activation="relu"):
        super().__init__()
        self.conv_decoder = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=stride, stride=stride)
        self.batchNorm_decoder = nn.BatchNorm2d(output_channels)
        if activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.batchNorm_decoder(self.conv_decoder(x)))

class DecoderNetwork(nn.Module):
    def __init__(self, color_channels, pooling_kernels, decoder_input_size):
        self.decoder_input_size = decoder_input_size
        super().__init__()
        self.layer1 = DecoderLayer(256, 128, stride=1)
        self.layer2 = DecoderLayer(128, 64, stride=pooling_kernels[1])
        self.layer3 = DecoderLayer(64, 32, stride=pooling_kernels[0])
        self.last_decoder = DecoderLayer(32, color_channels, stride=1, activation="sigmoid")

    def forward(self, x):
        out = x.view(-1, 256, self.decoder_input_size, self.decoder_input_size)
        out = self.layer3(self.layer2(self.layer1(out)))
        out = self.last_decoder(out)
        return out
