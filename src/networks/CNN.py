import torch.nn as nn

class CNN_encoder(nn.Module):
    '''
     CNN_encoder is a Convolutional Neural Network (CNN) designed for encoding time series data.
     It processes input time series data through convolutional layers to extract meaningful features,
     which are then flattened and passed through a fully connected layer to generate a fixed-size representation.
    '''
    def __init__(self, window_size, input_size = 1, hidden_size = 8):
        super().__init__()
        self.rep_dim = window_size // (2*2) # two pooling layers with stride = 2

        # Encoder part
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels = input_size, out_channels = hidden_size, kernel_size = 5, stride = 1, padding = 2, bias=False),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(p = 0.2),
            nn.Conv1d(in_channels = hidden_size, out_channels = hidden_size // 2, kernel_size = 5, stride = 1, padding = 2, bias=False),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(p = 0.2)
        )
        # Forward layer
        self.fc = nn.LazyLinear(out_features = self.rep_dim, bias = False)        

    def forward(self, x):
        x = x.permute(0, 2, 1) # The default input shape is (batch_size, window_size, feature_size)
        x = self.encoder(x)
        x = x.view(x.size(0), -1) # reshape the tensor to feed into the linear layer
        x = self.fc(x)

        return x
    

class CNN_Autoencoder(nn.Module): 
    '''
    CNN_Autoencoder is a Convolutional Autoencoder designed for time series data.
    It includes both an encoder (similar to CNN_encoder) to compress the input data into a lower-dimensional embedding
    and a decoder that reconstructs the original time series from this embedding. 
    '''
    def __init__(self, window_size, input_size = 1, hidden_size = 8):
        super().__init__()
        self.rep_dim = window_size // (2*2) # two pooling layers with stride = 2

        # Encoder (must match the Deep SVDD network above)
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels = input_size, out_channels = hidden_size, kernel_size = 5, stride = 1, padding = 2, bias=False),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(p = 0.2),
            nn.Conv1d(in_channels = hidden_size, out_channels = hidden_size // 2, kernel_size = 5, stride = 1, padding = 2, bias=False),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(p = 0.2)
        )
        self.encoder_fc = nn.LazyLinear(out_features = self.rep_dim, bias = False)  ### Continue here      

        self.decoder_cv = nn.LazyConvTranspose1d(out_channels = hidden_size // 2, kernel_size = 5, stride = 1, padding = 2, bias=False)
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode = 'linear'),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(p = 0.2),
            nn.ConvTranspose1d(in_channels = hidden_size // 2, out_channels = hidden_size, kernel_size = 5, stride = 1, padding = 2, bias=False),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode = 'linear'),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(p = 0.2),
            nn.ConvTranspose1d(in_channels = hidden_size, out_channels = input_size, kernel_size = 5, padding = 2, bias=False)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1) # The default input shape is (batch_size, window_size, feature_size), convert to (batch_size, channel_size, window_size)
        encoded = self.encoder(x) # Encoder part
        batch_size = encoded.size(0) # Batch size
        encoded = encoded.view(batch_size, -1) # flatten the tensor
        encoded = self.encoder_fc(encoded) # create embedding

        decoded = encoded.view(batch_size, 1, -1) # reshape embedding
        decoded = self.decoder_cv(decoded) # process the reshaped embedding
        decoded = self.decoder(decoded) # Decoder part
        x = decoded.permute(0, 2, 1) # The default output shape is (batch_size, window_size, feature_size), convert from (batch_size, channel_size, window_size)
        return x