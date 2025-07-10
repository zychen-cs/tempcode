import torch
import torch.nn as nn
import torch.nn.functional as F
from .Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from .SelfAttention_Family import FullAttention, AttentionLayer
from .embed import DataEmbedding
import numpy as np


class Transformer(nn.Module):
    """
    Vanilla Transformer
    with O(L^2) complexity
    Paper link: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """

    def __init__(self,enc_in, c_out, d_model=32, n_heads=8, e_layers=3, d_ff=512, factor=1,
                 dropout=0.0, activation='gelu',output_attention=True):
        super(Transformer, self).__init__()
        self.output_attention = output_attention
        # Embedding
        self.enc_embedding = DataEmbedding(enc_in,d_model,dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=output_attention), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.projection = nn.Linear(d_model,c_out, bias=True)


    def anomaly_detection(self, x_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out)
        return dec_out


    def forward(self, x_enc):
        x_enc=x_enc.unsqueeze(-1).permute(0,2,1)    # B,1,84
        dec_out = self.anomaly_detection(x_enc)
        return dec_out.permute(0,2,1).squeeze(-1)
        # x_enc=x_enc.unsqueeze(-1)    # B,84,1
        # return dec_out.squeeze(-1)  # B,84


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    
class OutConv(nn.Module):
    def __init__(self, d_model, out_ch):
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, out_ch, kernel_size=1)

    def forward(self, x):  # x: [B, d_model, T]
        out = self.conv1(x)
        return out
        
class ConvTransformer(nn.Module):

    def __init__(self,enc_in, c_out, d_channel=64, d_model=32, n_heads=8, e_layers=3, d_ff=512, factor=1,
                 dropout=0.0, activation='gelu',output_attention=True):
        super(ConvTransformer, self).__init__()
        self.output_attention = output_attention
        
        # Embedding
        self.channel_embedding = InConv(1, d_channel)
        self.enc_embedding = DataEmbedding(enc_in,d_model,dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=output_attention), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.channel_projection = OutConv(d_channel, 1)
        self.projection = nn.Linear(d_model,c_out, bias=True)


    def anomaly_detection(self, x_enc):
        # Embedding
        x_enc=self.channel_embedding(x_enc) # B,d_channel,L
        enc_out = self.enc_embedding(x_enc)   # B,d_channel,d_model
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        enc_out=F.relu(self.channel_projection(enc_out))
        dec_out = self.projection(enc_out)
        return dec_out


    def forward(self, x_enc):
        x_enc=x_enc.unsqueeze(-1).permute(0,2,1)    # B,1,84
        dec_out = self.anomaly_detection(x_enc)
        return dec_out.permute(0,2,1).squeeze(-1)
        # x_enc=x_enc.unsqueeze(-1)    # B,84,1
        # return dec_out.squeeze(-1)  # B,84
