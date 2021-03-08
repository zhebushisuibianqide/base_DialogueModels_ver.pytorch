#_*_ coding=utf-8 _*_

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
''' torch version 1.7.1'''
# Encoder
class Encoder(nn.Module):
    def __init__(self, embedding_matrix, rnn_type, is_bidirectional, enc_layer, \
        emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        self.embedding = embedding_matrix
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size = emb_dim, # The number of expected features in the input x
                hidden_size = enc_hid_dim, # The number of features in the hidden state h
                num_layers = enc_layer, # Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two LSTMs together to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results. Default: 1
                bias = True, # If False, then the layer does not use bias weights b_ih and b_hh. Default: True
                batch_first = True, # If True, then the input and output tensors are provided as (batch, seq, feature). Default: False
                dropout = dropout, # If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to dropout. Default: 0
                bidirectional = is_bidirectional, # If True, becomes a bidirectional LSTM. Default: False
                proj_size = 0 # If > 0, will use LSTM with projections of corresponding size. Default: 0
                )
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(
                input_size = emb_dim, # The number of expected features in the input x
                hidden_size = enc_hid_dim, # The number of features in the hidden state h
                num_layers = enc_layer, # Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two GRUs together to form a stacked GRU, with the second GRU taking in outputs of the first GRU and computing the final results. Default: 1
                bias = True, # If False, then the layer does not use bias weights b_ih and b_hh. Default: True
                batch_first = True, # If True, then the input and output tensors are provided as (batch, seq, feature). Default: False
                dropout = dropout, # If non-zero, introduces a Dropout layer on the outputs of each GRU layer except the last layer, with dropout probability equal to dropout. Default: 0
                bidirectional = is_bidirectional # If True, becomes a bidirectional GRU. Default: False
                )
        else:
            raise ValueError('No rnn_type is {}, check the config.'.format(rnn_type))
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        '''
        src = [batch_size, src_len]
        '''
        # we set the batch_first is TRUE, so the batch_size is first than src_len.
        embedded = self.dropout(self.embedding(src))  # embedded = [batch_size, src_len, emb_dim]
        # enc_output = [batch_size, src_len, hid_dim * num_directions]
        # enc_hidden = [batch_size, n_layers * num_directions, hid_dim]
        enc_output, enc_hidden = self.rnn(embedded)  # if h_0 is not give, it will be set 0 acquiescently

        # enc_hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # enc_output are always from the last layer

        # enc_hidden [ : ,-2, : ] is the last of the forwards RNN
        # enc_hidden [ : ,-1, : ] is the last of the backwards RNN

        # initial decoder hidden is final hidden state of the forwards and backwards
        # encoder RNNs fed through a linear layer
        # s = [batch_size, dec_hid_dim]
        dec_init_state = torch.tanh(self.fc(torch.cat((enc_hidden[:, -2, :], enc_hidden[:, -1, :]), dim=1)))

        return enc_output, dec_init_state


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear((enc_hid_dim) * 2 + dec_hid_dim, dec_hid_dim, bias=False)
        self.V = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, s, enc_output):
        # s = [batch_size, dec_hid_dim]
        # enc_output = [batch_size, seq_len, enc_hid_dim * 2]

        batch_size = enc_output.shape[0]
        src_len = enc_output.shape[1]

        # repeat decoder hidden state src_len times
        # s = [batch_size, seq_len, dec_hid_dim]
        # enc_out_put = [batch_size, seq_len, enc_hid_dim * 2]
        s = s.unsqueeze(1).repeat(1, src_len, 1)

        # energy = [batch_size, src_len, dec_hid_dim]
        energy = torch.tanh(self.attn(torch.cat((s, enc_output), dim=2)))

        # attention = [batch_size, src_len]
        attention = self.V(energy).squeeze(2)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, embedding_matrix, emb_dim, vocab_size, rnn_type,\
        dec_layer, enc_hid_dim, dec_hid_dim, dropout, is_bidirectional, attention):

        super().__init__()
        self.attention = attention
        self.embedding = embedding_matrix
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size = enc_hid_dim * 2 + emb_dim, # The number of expected features in the input x
                hidden_size = dec_hid_dim, # The number of features in the hidden state h
                num_layers = dec_layer, # Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two LSTMs together to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results. Default: 1
                bias = True, # If False, then the layer does not use bias weights b_ih and b_hh. Default: True
                batch_first = True, # If True, then the input and output tensors are provided as (batch, seq, feature). Default: False
                dropout = dropout, # If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to dropout. Default: 0
                bidirectional = is_bidirectional, # If True, becomes a bidirectional LSTM. Default: False
                proj_size = 0 # If > 0, will use LSTM with projections of corresponding size. Default: 0
                )
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(
                input_size = enc_hid_dim * 2 + emb_dim, # The number of expected features in the input x
                hidden_size = dec_hid_dim, # The number of features in the hidden state h
                num_layers = dec_layer, # Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two GRUs together to form a stacked GRU, with the second GRU taking in outputs of the first GRU and computing the final results. Default: 1
                bias = True, # If False, then the layer does not use bias weights b_ih and b_hh. Default: True
                batch_first = True, # If True, then the input and output tensors are provided as (batch, seq, feature). Default: False
                dropout = dropout, # If non-zero, introduces a Dropout layer on the outputs of each GRU layer except the last layer, with dropout probability equal to dropout. Default: 0
                bidirectional = is_bidirectional # If True, becomes a bidirectional GRU. Default: False
                )
        else:
            raise ValueError('No rnn_type is {}, check the config.'.format(rnn_type))
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, dec_input, s, enc_output):
        # dec_input = [batch_size]
        # s = [batch_size, dec_hid_dim]
        # enc_output = [src_len, batch_size, enc_hid_dim * 2]

        dec_input = dec_input.unsqueeze(1)  # dec_input = [batch_size, 1]

        embedded = self.dropout(self.embedding(dec_input))  # embedded = [batch_size, 1, emb_dim]

        # a = [batch_size, 1, src_len]
        a = self.attention(s, enc_output).unsqueeze(1)

        # enc_output = [batch_size, src_len, enc_hid_dim * 2]
        # c = [batch_size, 1, enc_hid_dim * 2]
        c = torch.bmm(a, enc_output)

        # rnn_input = [1, batch_size, (enc_hid_dim * 2) + emb_dim]
        rnn_input = torch.cat((embedded, c), dim=2)

        # dec_output = [src_len(=1), batch_size, dec_hid_dim]
        # dec_hidden = [n_layers * num_directions, batch_size, dec_hid_dim]
        dec_output, dec_hidden = self.rnn(rnn_input, s.unsqueeze(0))

        # embedded = [batch_size, emb_dim]
        # dec_output = [batch_size, dec_hid_dim]
        # c = [batch_size, enc_hid_dim * 2]
        embedded = embedded.squeeze(1)
        dec_output = dec_output.squeeze(1)
        c = c.squeeze(1)

        # pred = [batch_size, output_dim]
        pred = self.fc_out(torch.cat((dec_output, c, embedded), dim=1))

        return pred, dec_hidden.squeeze(1)


# seq2seq
class Seq2Seq(nn.Module):
    def __init__(self, model_config):
        super(Seq2Seq, self).__init__()
        if model_config.word2vec is None:
            self.embedding_matrix = nn.Embedding(model_config.vocab_size, model_config.emb_dim)
        else:
            pass
        self.attention = Attention(config.enc_hid_dim, config.dec_hid_dim)
        self.encoder = Encoder(
            self.embedding_matrix, model_config.enc_rnn_type, model_config.enc_is_bidirectional, \
            model_config.enc_layer, model_config.emb_dim, model_config.enc_hid_dim, \
            model_config.dec_hid_dim, model_config.dropout
            )
        self.decoder = Decoder(
            self.embedding_matrix, model_config.emb_dim, model_config.vocab_size, \
            model_config.dec_rnn_type, model_config.dec_layer, config.enc_hid_dim, \
            model_config.dec_hid_dim, model_config.dropout, \
            model_config.dec_is_bidirectional, self.attention
            )
        self.device = model_config.device
        self.config = model_config

    def forward(self, src, trg, use_teacher_forcing=True):
        # src = [src_len, batch_size]
        # trg = [trg_len, batch_size]
        # teacher_foring_radio is probability to use teacher forcing

        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # enc_output is all hidden states of the input sequence, back and forwards
        # s is the final forward and backward hidden states, passed through a linear layer
        enc_output, s = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        dec_input = trg[0, :]

        for t in range(1, trg_len):
            # insert dec_input token embeddings, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            # dec_output = [batch_size, output_dim]
            # s = [batch_size, dec_hid_dim]
            dec_output, s = self.decoder(dec_input, s, enc_output)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = dec_output

            # get the highest predicted token from our predictions
            top1 = dec_output.argmax(1)

            # if teacher foring, use actual next token as next input
            # if not, use predicted token
            dec_input = trg[t] if use_teacher_forcing else top1

        return outputs