import torch
import torch.nn as nn

class PixelRNN(nn.Module):
    def __init__(self, input_channels=3, hidden_channels=128, kernel_size=3):
        """
        Simple PixelRNN model
        - Two convolutional layers for encoding
        - One LSTM layer for sequence modeling
        - Two deconvolutional layers for decoding
        - Sigmoid output for [0,1] pixel range
        """
        super(PixelRNN, self).__init__()
        padding = kernel_size // 2

        # Encoder
        self.conv1 = nn.Conv2d(input_channels, hidden_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding)

        # LSTM
        self.rnn = nn.LSTM(input_size=hidden_channels, hidden_size=hidden_channels, batch_first=True)

        # Decoder
        self.deconv1 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding)
        self.deconv2 = nn.Conv2d(hidden_channels, input_channels, kernel_size, padding=padding)

        # Activations
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))

        B, C, H, W = out.size()
        # Flatten spatial dimensions for LSTM
        out = out.view(B, C, H * W).permute(0, 2, 1)  # [B, H*W, C]
        out, _ = self.rnn(out)
        out = out.permute(0, 2, 1).view(B, C, H, W)

        # Decoder
        out = self.relu(self.deconv1(out))
        out = self.deconv2(out)
        out = self.sigmoid(out)  # Clamp output to [0,1]
        return out
