# _*_ coding=utf-8 _*_

import torch
import torch.nn as nn
import numpy as np

from module import EncoderLayer
from module import DecoderLayer
from module import get_sinusoid_encoding_table
from module import get_attn_pad_mask
from module import get_attn_subsequence_mask
import torch.optim as optim
import torch.nn.functional as F

''' torch version 1.7.1'''


# Encoder
class Encoder(nn.Module):
    def __init__(self, model_config):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(model_config) \
                                     for _ in range(model_config.enc_num_block)])

    def forward(self, enc_input, enc_self_attn_mask):
        '''
            enc_input = [batch_size, src_len, d_model]
            enc_self_attn_mask = [batch_size, n_heads, src_len, src_len]
        '''
        enc_output = enc_input
        enc_self_attns = []
        for layer in self.layers:
            # enc_output: [batch_size, src_len, d_model]
            enc_output, enc_self_attn = layer(enc_output, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_output, enc_self_attns


# Decoder
class Decoder(nn.Module):
    def __init__(self, model_config):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(model_config) \
                                     for _ in range(model_config.dec_num_block)])

    def forward(self, enc_output, dec_input,
                dec_self_attn_mask, dec_enc_attn_mask):
        '''
            enc_output = [batch_size, src_len, d_model]
            dec_input = [batch_size, tgt_len, d_model]
            dec_self_attn_pad_mask = [batch_size, tgt_len, tgt_len]
            dec_self_attn_subsequence_mask = [batch_size, tgt_len, tgt_len]
            dec_enc_attn_mask = [batc_size, tgt_len, src_len]
        '''
        dec_output = dec_input
        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model],
            # dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len],
            # dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_output, dec_self_attn, dec_enc_attn = layer(dec_output, enc_output,
                                                            dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_output, dec_self_attns, dec_enc_attns


# Transformer
class Transformer(nn.Module):
    def __init__(self, model_config, device='cpu'):
        super(Transformer, self).__init__()
        if model_config.embed_path is None:
            self.embedding_matrix = nn.Embedding(model_config.vocab_size, model_config.emb_dim)
            print('Initializing embedding matrix using random initialization.')
        else:
            from data_utils import read_word2vec
            _, word2vec = read_word2vec(model_config.embed_path)
            word2vec = torch.FloatTensor(word2vec)
            self.embedding_matrix = nn.Embedding.from_pretrained(word2vec)
            print('Initializing embedding matrix form pretrained file.')
        self.pos_emb = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(model_config.vocab_size, model_config.d_model),
            freeze=True) # freeze : default is True
        self.encoder = Encoder(model_config)
        self.decoder = Decoder(model_config)
        self.projection = nn.Linear(model_config.d_model, model_config.vocab_size, bias=False)
        self.device = device
        self.config = model_config

    def forward(self, src, tgt, use_teacher_forcing=True):
        # src = [batch_size, src_len]
        # tgt = [batch_size, tgt_len]
        # teacher_foring_radio is probability to use teacher forcing

        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.config.vocab_size

        # tensor to store decoder outputs
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        enc_input_web = self.embedding_matrix(src)
        enc_input_peb = self.pos_emb(src)
        enc_input = enc_input_web+enc_input_peb
        enc_self_attn_mask = get_attn_pad_mask(src, src)
        # enc_output is all hidden states of the input sequence, back and forwards
        # s is the final forward and backward hidden states, passed through a linear layer
        enc_output, att = self.encoder(enc_input, enc_self_attn_mask)

        dec_self_attn_pad_mask = get_attn_pad_mask(tgt, tgt).to(self.device)  # [batch_size, tgt_len, tgt_len]
        dec_self_attn_subsequent_mask = get_attn_subsequence_mask(tgt).to(self.device)  # [batch_size, 1]
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask),
                                      0).to(self.device)  # [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask = get_attn_pad_mask(tgt, src).to(self.device)

        if use_teacher_forcing:
            dec_input_web = self.embedding_matrix(tgt)
            dec_input_peb = self.pos_emb(tgt)
            dec_input = dec_input_web + dec_input_peb
            # insert dec_input token embeddings, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state

            dec_output, dec_self_attns, dec_enc_attns = self.decoder(dec_input, enc_output,
                                                                     dec_self_attn_mask, dec_enc_attn_mask)
            # place predictions in a tensor holding predictions for each token
            logits = self.projection(dec_output)
            outputs = logits
        else:
            preds = torch.LongTensor(np.zeros((batch_size, tgt_len))).to(self.device)
            preds[:, 0] = tgt[:, 0]
            for t in range(1, tgt_len):
                # if teacher foring, use actual next token as next input
                # if not, use predicted token
                dec_input_web = self.embedding_matrix(preds)
                dec_input_peb = self.pos_emb(preds)
                dec_input = dec_input_web + dec_input_peb
                # insert dec_input token embeddings, previous hidden state and all encoder hidden states
                # receive output tensor (predictions) and new hidden state

                dec_output, dec_self_attns, dec_enc_attns = self.decoder(dec_input, enc_output,
                                                                         dec_self_attn_mask, dec_enc_attn_mask)
                # place predictions in a tensor holding predictions for each token
                logits = self.projection(dec_output)
                outputs[:, t, :] = logits[:,t-1,:]
                # get the highest predicted token from our predictions
                preds[:,t] = torch.argmax(logits[:,t-1,:], dim=-1)

        return outputs
