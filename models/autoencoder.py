import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AutoEncoder(nn.Module):
    """
    A simple autoencoder for images. 
    self.linear1 generates the intermediate embeddings that we use for the normalizing flow.
    """
    def __init__(self, data_dim, z_dim, hidden_dim=128):
        super(AutoEncoder, self).__init__()

        self.data_shape_out = tuple(int(l / 4) for l in data_dim)
        self.data_dim_out = int(np.prod(self.data_shape_out)) # Downsampling size 
        self.hidden_dim = hidden_dim
        
        conv_kwargs = dict(padding=1, kernel_size=3, stride=2)

        # Encoding layers
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=int(hidden_dim / 2), bias=False, **conv_kwargs
        )
        self.conv2 = nn.Conv2d(
            in_channels=int(hidden_dim / 2), out_channels=hidden_dim, bias=False, **conv_kwargs
        )
        self.linear1 = nn.Linear(
            in_features=hidden_dim * self.data_dim_out, out_features=z_dim
        )
        
        # Decoding layers
        self.linear2 = nn.Linear(
            in_features=z_dim, out_features=hidden_dim * self.data_dim_out
        )
        self.convt1 = nn.ConvTranspose2d(
            in_channels=hidden_dim, out_channels=int(hidden_dim / 2), output_padding=1, **conv_kwargs
        )
        self.convt2 = nn.ConvTranspose2d(
            in_channels=int(hidden_dim / 2), out_channels=1, output_padding=1, **conv_kwargs
        )

    def forward(self, x):
        emb = self.encoder(x)
        _x = self.decoder(emb)
        return _x, emb
    
    def decoder(self, emb):
        _x = self.linear2(emb)
        _x = F.dropout(_x, p=0.2, training=self.training)
        _x = torch.tanh(_x)
        _x = _x.view(-1, self.hidden_dim, *self.data_shape_out)
        _x = self.convt1(_x)
        _x = F.dropout(_x, p=0.2, training=self.training)
        _x = torch.tanh(_x)
        _x = self.convt2(_x)
        # _x = F.dropout(_x, p=0.2, training=self.training)
        _x = torch.tanh(_x)
        # print(_x.shape)
        return _x
    
    def encoder(self, x):
        _x = self.conv1(x)
        _x = F.dropout(_x, p=0.2, training=self.training)
        _x = torch.tanh(_x)
        _x = self.conv2(_x)
        _x = F.dropout(_x, p=0.2, training=self.training)
        _x = torch.tanh(_x)
        _x = torch.flatten(_x, 1)
        # print(_x.shape)
        emb = self.linear1(_x)
        return emb


class AutoEncoderC(nn.Module):
    """
    A simple autoencoder for images. 
    self.linear1 generates the intermediate embeddings that we use for the normalizing flow.
    """
    def __init__(self, data_dim, z_dim, hidden_dim=128):
        super(AutoEncoderC, self).__init__()

        self.data_shape_out = tuple(int(l / 4) for l in data_dim)
        self.data_dim_out = int(np.prod(self.data_shape_out)) # Downsampling size 
        self.hidden_dim = hidden_dim
        
        conv_kwargs = dict(padding=1, kernel_size=3, stride=2)

        # Encoding layers
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=int(hidden_dim / 2), bias=False, **conv_kwargs
        )
        self.conv2 = nn.Conv2d(
            in_channels=int(hidden_dim / 2), out_channels=hidden_dim, bias=False, **conv_kwargs
        )
        self.linear1 = nn.Linear(
            in_features=hidden_dim * self.data_dim_out, out_features=z_dim
        )
        
        # Decoding layers
        self.linear2 = nn.Linear(
            in_features=z_dim, out_features=hidden_dim * self.data_dim_out
        )
        self.convt1 = nn.ConvTranspose2d(
            in_channels=hidden_dim, out_channels=int(hidden_dim / 2), output_padding=1, **conv_kwargs
        )
        self.convt2 = nn.ConvTranspose2d(
            in_channels=int(hidden_dim / 2), out_channels=1, output_padding=1, **conv_kwargs
        )

    def forward(self, x):
        emb = self.encoder(x)
        _x = self.decoder(emb)
        return _x, emb
    
    def decoder(self, emb):
        _x = self.linear2(emb)
        _x = F.dropout(_x, p=0.2, training=self.training)
        _x = torch.tanh(_x)
        _x = _x.view(-1, self.hidden_dim, *self.data_shape_out)
        _x = self.convt1(_x)
        _x = F.dropout(_x, p=0.2, training=self.training)
        _x = torch.tanh(_x)
        _x = self.convt2(_x)
        # _x = F.dropout(_x, p=0.2, training=self.training)
        _x = torch.tanh(_x)
        # print(_x.shape)
        return _x
    
    def encoder(self, x):
        _x = self.conv1(x)
        _x = F.dropout(_x, p=0.2, training=self.training)
        _x = torch.tanh(_x)
        _x = self.conv2(_x)
        _x = F.dropout(_x, p=0.2, training=self.training)
        _x = torch.tanh(_x)
        _x = torch.flatten(_x, 1)
        # print(_x.shape)
        emb = self.linear1(_x)
        return emb


class AutoEncoderA(nn.Module):
    """
    A simple autoencoder for images. 
    self.linear1 generates the intermediate embeddings that we use for the normalizing flow.
    """
    def __init__(self, data_dim, z_dim, parameter_dim, hidden_dim=128):
        super(AutoEncoderA, self).__init__()

        self.data_shape_out = tuple(int(l / (2 ** 4)) for l in data_dim) # 2 ** (n_layers)
        self.data_dim_out = int(np.prod(self.data_shape_out)) # Downsampling size 
        self.hidden_dim = hidden_dim
        conv_kwargs = dict(padding="same", kernel_size=3, stride=1)
        convt_kwargs = dict(padding="same", kernel_size=3, stride=1)

        """
            Max pooling
        """
        # # Encoding layers
        # self.conv1 = nn.Conv2d(
        #     in_channels=1, out_channels=int(hidden_dim / 8), bias=False, **conv_kwargs
        # )
        # self.conv2 = nn.Conv2d(
        #     in_channels=int(hidden_dim / 8), out_channels=int(hidden_dim / 4), bias=False, **conv_kwargs
        # )
        # self.conv3 = nn.Conv2d(
        #     in_channels=int(hidden_dim / 4), out_channels=int(hidden_dim / 2), bias=False, **conv_kwargs
        # )
        # self.conv4 = nn.Conv2d(
        #     in_channels=int(hidden_dim / 2), out_channels=hidden_dim, bias=False, **conv_kwargs
        # )
        # self.linear1 = nn.Linear(
        #     in_features=hidden_dim * self.data_dim_out + parameter_dim, out_features=z_dim
        # )
        
        # # Decoding layers
        # self.linear2 = nn.Linear(
        #     in_features=z_dim + parameter_dim, out_features=hidden_dim * self.data_dim_out
        # )
        # self.convt1 = nn.Conv2d(
        #     in_channels=hidden_dim, out_channels=int(hidden_dim / 8), **convt_kwargs
        # )
        # self.convt2 = nn.Conv2d(
        #     in_channels=int(hidden_dim / 8), out_channels=int(hidden_dim / 4), **convt_kwargs #output_padding=1,  
        # )
        # self.convt3 = nn.Conv2d(
        #     in_channels=int(hidden_dim / 4), out_channels=int(hidden_dim / 2), **convt_kwargs
        # )
        # self.convt4 = nn.Conv2d(
        #     in_channels=int(hidden_dim / 2), out_channels=1, **convt_kwargs
        # )
        # Encoding layers
        bias = True
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=hidden_dim, bias=bias, **conv_kwargs
        )
        self.conv2 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim, bias=bias, **conv_kwargs
        )
        self.conv3 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim, bias=bias, **conv_kwargs
        )
        self.conv4 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim, bias=bias, **conv_kwargs
        )
        self.linear1 = nn.Linear(
            in_features=hidden_dim * self.data_dim_out + parameter_dim, out_features=z_dim
        )
        
        # Decoding layers
        self.linear2 = nn.Linear(
            in_features=z_dim + parameter_dim, out_features=hidden_dim * self.data_dim_out
        )
        self.convt1 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim, **convt_kwargs
        )
        self.convt2 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim, **convt_kwargs #output_padding=1,  
        )
        self.convt3 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim, **convt_kwargs
        )
        self.convt4 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=1, **convt_kwargs
        )


    def forward(self, x, y):
        emb = self.encoder(x, y)
        _x = self.decoder(emb, y)
        return _x, emb
    
    def decoder(self, e, y):
        # Map an embedding to an initial 'image' array
        _x = self.linear2(torch.cat([e, y], dim=1))
        # _x = F.dropout(_x, p=0.1, training=self.training)
        _x = torch.tanh(_x)
        _x = _x.view(-1, self.hidden_dim, *self.data_shape_out)
        # Upsample this image to data space
        _x = self.convt1(_x)
        _x = F.dropout(_x, p=0.1, training=self.training)
        _x = torch.tanh(_x)
        _x = F.interpolate(_x, size=(8, 8), mode='bilinear', align_corners=False)
        _x = self.convt2(_x)
        _x = F.dropout(_x, p=0.1, training=self.training)
        _x = torch.tanh(_x)
        _x = F.interpolate(_x, size=(16, 16), mode='bilinear', align_corners=False)
        _x = self.convt3(_x)
        _x = F.dropout(_x, p=0.1, training=self.training)
        _x = torch.tanh(_x)
        _x = F.interpolate(_x, size=(32, 32), mode='bilinear', align_corners=False)
        _x = self.convt4(_x)
        _x = F.dropout(_x, p=0.1, training=self.training)
        _x = torch.tanh(_x) # NOTE: only if data scaled to [-1, 1]
        # print(_x.shape)
        return _x
    
    def encoder(self, x, y):
        # Encode an image in data space to an embedding
        _x = self.conv1(x)
        _x = F.max_pool2d(_x, kernel_size=1, stride=2)
        _x = F.dropout(_x, p=0.1, training=self.training)
        _x = torch.tanh(_x)
        _x = self.conv2(_x)
        _x = F.max_pool2d(_x, kernel_size=1, stride=2)
        _x = F.dropout(_x, p=0.1, training=self.training)
        _x = torch.tanh(_x)
        _x = self.conv3(_x)
        _x = F.max_pool2d(_x, kernel_size=1, stride=2)
        _x = F.dropout(_x, p=0.1, training=self.training)
        _x = torch.tanh(_x)
        _x = self.conv4(_x)
        _x = F.max_pool2d(_x, kernel_size=1, stride=2)
        _x = F.dropout(_x, p=0.1, training=self.training)
        _x = torch.tanh(_x)
        _x = torch.flatten(_x, start_dim=1)
        # print(_x.shape)
        emb = self.linear1(torch.cat([_x, y], dim=1))
        emb = torch.tanh(emb)
        return emb


class AutoEncoderA2(nn.Module):
    """
    A simple autoencoder for images. 
    self.linear1 generates the intermediate embeddings that we use for the normalizing flow.
    """
    def __init__(self, data_dim, z_dim, parameter_dim, hidden_dim=128):
        super(AutoEncoderA2, self).__init__()

        self.data_shape_out = tuple(int(l / (2 ** 4)) for l in data_dim) # 2 ** (n_layers)
        self.data_dim_out = int(np.prod(self.data_shape_out)) # Downsampling size 
        self.hidden_dim = hidden_dim
        conv_kwargs = dict(padding="same", kernel_size=3, stride=1)
        convt_kwargs = dict(padding="same", kernel_size=3, stride=1)

        """
            Max pooling
        """
        # # Encoding layers
        # self.conv1 = nn.Conv2d(
        #     in_channels=1, out_channels=int(hidden_dim / 8), bias=False, **conv_kwargs
        # )
        # self.conv2 = nn.Conv2d(
        #     in_channels=int(hidden_dim / 8), out_channels=int(hidden_dim / 4), bias=False, **conv_kwargs
        # )
        # self.conv3 = nn.Conv2d(
        #     in_channels=int(hidden_dim / 4), out_channels=int(hidden_dim / 2), bias=False, **conv_kwargs
        # )
        # self.conv4 = nn.Conv2d(
        #     in_channels=int(hidden_dim / 2), out_channels=hidden_dim, bias=False, **conv_kwargs
        # )
        # self.linear1 = nn.Linear(
        #     in_features=hidden_dim * self.data_dim_out + parameter_dim, out_features=z_dim
        # )
        
        # # Decoding layers
        # self.linear2 = nn.Linear(
        #     in_features=z_dim + parameter_dim, out_features=hidden_dim * self.data_dim_out
        # )
        # self.convt1 = nn.Conv2d(
        #     in_channels=hidden_dim, out_channels=int(hidden_dim / 8), **convt_kwargs
        # )
        # self.convt2 = nn.Conv2d(
        #     in_channels=int(hidden_dim / 8), out_channels=int(hidden_dim / 4), **convt_kwargs #output_padding=1,  
        # )
        # self.convt3 = nn.Conv2d(
        #     in_channels=int(hidden_dim / 4), out_channels=int(hidden_dim / 2), **convt_kwargs
        # )
        # self.convt4 = nn.Conv2d(
        #     in_channels=int(hidden_dim / 2), out_channels=1, **convt_kwargs
        # )
        # Encoding layers
        bias = True
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=hidden_dim, bias=bias, **conv_kwargs
        )
        self.conv2 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim, bias=bias, **conv_kwargs
        )
        self.conv3 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim, bias=bias, **conv_kwargs
        )
        self.conv4 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim, bias=bias, **conv_kwargs
        )
        self.linear1 = nn.Linear(
            in_features=hidden_dim * self.data_dim_out, out_features=z_dim
        )
        
        # Decoding layers
        self.linear2 = nn.Linear(
            in_features=z_dim, out_features=hidden_dim * self.data_dim_out
        )
        self.convt1 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim, **convt_kwargs
        )
        self.convt2 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim, **convt_kwargs #output_padding=1,  
        )
        self.convt3 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim, **convt_kwargs
        )
        self.convt4 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=1, **convt_kwargs
        )

    def forward(self, x):
        emb = self.encoder(x)
        _x = self.decoder(emb)
        return _x, emb
    
    def decoder(self, e):
        # Map an embedding to an initial 'image' array
        _x = self.linear2(e)
        # _x = F.dropout(_x, p=0.1, training=self.training)
        _x = torch.tanh(_x)
        _x = _x.view(-1, self.hidden_dim, *self.data_shape_out)
        # Upsample this image to data space
        _x = self.convt1(_x)
        _x = F.dropout(_x, p=0.1, training=self.training)
        _x = torch.tanh(_x)
        _x = F.interpolate(_x, size=(8, 8), mode='bilinear', align_corners=False)
        _x = self.convt2(_x)
        _x = F.dropout(_x, p=0.1, training=self.training)
        _x = torch.tanh(_x)
        _x = F.interpolate(_x, size=(16, 16), mode='bilinear', align_corners=False)
        _x = self.convt3(_x)
        _x = F.dropout(_x, p=0.1, training=self.training)
        _x = torch.tanh(_x)
        _x = F.interpolate(_x, size=(32, 32), mode='bilinear', align_corners=False)
        _x = self.convt4(_x)
        _x = F.dropout(_x, p=0.1, training=self.training)
        _x = torch.tanh(_x) # NOTE: only if data scaled to [-1, 1]
        # print(_x.shape)
        return _x
    
    def encoder(self, x):
        # Encode an image in data space to an embedding
        _x = self.conv1(x)
        _x = F.max_pool2d(_x, kernel_size=1, stride=2)
        _x = F.dropout(_x, p=0.1, training=self.training)
        _x = torch.tanh(_x)
        _x = self.conv2(_x)
        _x = F.max_pool2d(_x, kernel_size=1, stride=2)
        _x = F.dropout(_x, p=0.1, training=self.training)
        _x = torch.tanh(_x)
        _x = self.conv3(_x)
        _x = F.max_pool2d(_x, kernel_size=1, stride=2)
        _x = F.dropout(_x, p=0.1, training=self.training)
        _x = torch.tanh(_x)
        _x = self.conv4(_x)
        _x = F.max_pool2d(_x, kernel_size=1, stride=2)
        _x = F.dropout(_x, p=0.1, training=self.training)
        _x = torch.tanh(_x)
        _x = torch.flatten(_x, start_dim=1)
        # print(_x.shape)
        emb = self.linear1(_x)
        emb = torch.tanh(emb)
        return emb


class AutoencoderA3(nn.Module):
    def __init__(self, input_channels, num_layers):
        super(AutoencoderA3, self).__init__()
        
        # Ensure the number of layers is at least 2 (one encoder and one decoder layer)
        assert num_layers >= 2, "Number of layers should be at least 2"
        
        # Encoder layers
        encoder_layers = []
        in_channels = input_channels
        out_channels = 2  # Starting number of channels for the encoder
        for _ in range(num_layers):
            encoder_layers.append(
                nn.Conv2d(
                    in_channels, 
                    out_channels, 
                    kernel_size=3, 
                    stride=1, 
                    padding="same"
                )
            )
            encoder_layers.append(nn.MaxPool2d(kernel_size=1, stride=2))
            encoder_layers.append(nn.LeakyReLU())
            in_channels = out_channels
            out_channels *= 2
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder layers
        decoder_layers = []
        out_channels = in_channels // 2
        for _ in range(num_layers):
            decoder_layers.append(
                nn.ConvTranspose2d(
                    in_channels, 
                    out_channels, 
                    kernel_size=3, 
                    stride=2, 
                    padding=1, 
                    output_padding=1
                )
            )
            decoder_layers.append(nn.LeakyReLU())
            in_channels = out_channels
            out_channels //= 2
        
        # Final layer to reconstruct the image
        decoder_layers.append(nn.Conv2d(in_channels, input_channels, kernel_size=3, stride=1, padding="same"))
        decoder_layers.append(nn.Tanh())  # Using Sigmoid to get output in range [-1, 1]
        
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x, z

    def encode(self, x):
      return self.encoder(x)

    def decode(self, z):
      return self.decoder(z)


class AutoEncoderB(nn.Module):
    """
    A simple autoencoder for images. 
    self.linear1 generates the intermediate embeddings that we use for the normalizing flow.
    """
    def __init__(self, data_dim, z_dim, parameter_dim, hidden_dim=128):
        super(AutoEncoderB, self).__init__()

        self.data_shape_out = tuple(int(l / (2 ** 4)) for l in data_dim) # 2 ** (n_layers)
        self.data_dim_out = int(np.prod(self.data_shape_out)) # Downsampling size 
        self.hidden_dim = hidden_dim
        conv_kwargs = dict(padding=1, kernel_size=3, stride=2)

        # Encoding layers
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=int(hidden_dim / 8), bias=False, **conv_kwargs
        )
        self.conv2 = nn.Conv2d(
            in_channels=int(hidden_dim / 8), out_channels=int(hidden_dim / 4), bias=False, **conv_kwargs
        )
        self.conv3 = nn.Conv2d(
            in_channels=int(hidden_dim / 4), out_channels=int(hidden_dim / 2), bias=False, **conv_kwargs
        )
        self.conv4 = nn.Conv2d(
            in_channels=int(hidden_dim / 2), out_channels=hidden_dim, bias=False, **conv_kwargs
        )
        self.linear1 = nn.Linear(
            in_features=hidden_dim * self.data_dim_out + parameter_dim, out_features=z_dim
        )
        
        # Decoding layers
        self.linear2 = nn.Linear(
            in_features=z_dim + parameter_dim, out_features=hidden_dim * self.data_dim_out
        )
        self.convt1 = nn.ConvTranspose2d(
            in_channels=hidden_dim, out_channels=int(hidden_dim / 8), output_padding=1, **conv_kwargs
        )
        self.convt2 = nn.ConvTranspose2d(
            in_channels=int(hidden_dim / 8), out_channels=int(hidden_dim / 4), output_padding=1, **conv_kwargs
        )
        self.convt3 = nn.ConvTranspose2d(
            in_channels=int(hidden_dim / 4), out_channels=int(hidden_dim / 2), output_padding=1, **conv_kwargs
        )
        self.convt4 = nn.ConvTranspose2d(
            in_channels=int(hidden_dim / 2), out_channels=1, output_padding=1, **conv_kwargs
        )

    def forward(self, x, y):
        emb = self.encoder(x, y)
        _x = self.decoder(emb, y)
        return _x, emb
    
    def decoder(self, emb, y):
        # Map an embedding to an initial 'image' array
        _x = self.linear2(torch.cat([emb, y], dim=1))
        # _x = F.dropout(_x, p=0.1, training=self.training)
        _x = torch.tanh(_x)
        _x = _x.view(-1, self.hidden_dim, *self.data_shape_out)
        # Upsample this image to data space
        _x = self.convt1(_x)
        # _x = F.dropout(_x, p=0.1, training=self.training)
        _x = torch.tanh(_x)
        _x = self.convt2(_x)
        # _x = F.dropout(_x, p=0.1, training=self.training)
        _x = torch.tanh(_x)
        _x = self.convt3(_x)
        # _x = F.dropout(_x, p=0.1, training=self.training)
        _x = torch.tanh(_x)
        _x = self.convt4(_x)
        # _x = F.dropout(_x, p=0.1, training=self.training)
        _x = torch.tanh(_x)
        # print(_x.shape)
        return _x
    
    def encoder(self, x, y):
        # Encode an image in data space to an embedding
        _x = self.conv1(x)
        # _x = F.dropout(_x, p=0.1, training=self.training)
        _x = torch.tanh(_x)
        _x = self.conv2(_x)
        # _x = F.dropout(_x, p=0.1, training=self.training)
        _x = torch.tanh(_x)
        _x = self.conv3(_x)
        # _x = F.dropout(_x, p=0.1, training=self.training)
        _x = torch.tanh(_x)
        _x = self.conv4(_x)
        # _x = F.dropout(_x, p=0.1, training=self.training)
        _x = torch.tanh(_x)
        _x = torch.flatten(_x, start_dim=1)
        # print(_x.shape)
        emb = self.linear1(torch.cat([_x, y], dim=1))
        emb = torch.tanh(emb)
        return emb