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
        self.rnn_type = rnn_type
        self.is_bidirectional = is_bidirectional
        self.num_layer = enc_layer
        self.embedding = embedding_matrix
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size = emb_dim, # The number of expected features in the input x
                hidden_size = enc_hid_dim, # The number of features in the hidden state h
                num_layers = enc_layer, # Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two LSTMs together to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results. Default: 1
                bias = True, # If False, then the layer does not use bias weights b_ih and b_hh. Default: True
                batch_first = True, # If True, then the input and output tensors are provided as (batch, seq, feature). Default: False
                dropout = dropout, # If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to dropout. Default: 0
                bidirectional = is_bidirectional # If True, becomes a bidirectional LSTM. Default: False
                #proj_size = 0 # If > 0, will use LSTM with projections of corresponding size. Default: 0
                )
            #TypeError: __init__() got an unexpected keyword argument 'proj_size'
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
        # if rnn is LSTM -> output are enc_output, (h_n, c_n) = LSTM(input, (h_0,c_0)) h_0,c_0 can be ingored.
        # enc_output = [batch_size, src_len, hid_dim * num_directions] : all hidden state of each t
        # enc_hidden = (h_n, c_n) ->
        # h_n -> [num_layers * num_directions, batch_size, hid_dim]
        # c_n -> [num_layers * num_directions, batch_size, hid_dim]
        # elif rnn is GRU -> output are enc_output, h_n = GRU(input, h_0) h_0 can be ingored
        # enc_output = [batch_size, src_len, hid_dim * num_directions] : all hidden state of each t
        # h_n -> [num_layers * num_directions, batch_size, hid_dim]
        if self.rnn_type == 'LSTM':
            enc_output, (h_n, c_n) = self.rnn(embedded)  # if h_0 is not give, it will be set 0 acquiescently
            if self.is_bidirectional:
                enc_h_n = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
                enc_c_n = torch.cat((c_n[-2,:,:], c_n[-1,:,:]), dim=1)
            else:
                enc_h_n = h_n[-1,:,:]
                enc_c_n = c_n[-1,:,:]
        elif self.rnn_type == 'GRU':
            enc_output, h_n = self.rnn(embedded)
            if self.is_bidirectional:
                enc_h_n = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
            else:
                enc_h_n = h_n[-1,:,:]
        else:
            raise ValueError('No rnn type is {}, chooes \'LSTM\' or \'GRU\''.format(self.rnn_type))

        # if rnn is bidirectional:
        # h_n is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # enc_output are always from the last layer
        # h_n[-2,:,:] is the last of the forwards RNN -> shape(batch_size, hid_dim)
        # h_n[-1,:,:] is the last of the backwards RNN -> shape(batch_size, hid_dim)
        # same as c_n
        # else:
        # h_n is stacked [h_n_1, h_n_2, ...] 1,2,... mean the layer of the rnn
        # c_n is the same

        # initial decoder hidden is final hidden state of the forwards and backwards
        # encoder RNNs fed through a linear layer
        # s = [batch_size, dec_hid_dim]
        # if rnn is LSTM : input are input, (h_0, c_0)
        # if rnn is GRU : input are input, h_0
        if self.rnn_type == 'LSTM':
            dec_init_state = (torch.tanh(self.fc(enc_h_n)), torch.tanh(self.fc(enc_c_n)))
        elif self.rnn_type == 'GRU':
            dec_init_state = torch.tanh(self.fc(enc_h_n))
        else:
            raise ValueError('No rnn type is {}, chooes \'LSTM\' or \'GRU\''.format(self.rnn_type))

        return enc_output, dec_init_state


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super(Attention, self).__init__()
        #self.attn = nn.Linear((enc_hid_dim) * 2 + dec_hid_dim, dec_hid_dim, bias=False)
        #self.V = nn.Linear(dec_hid_dim, 1, bias=False)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim, bias=False)

    def forward(self, s, enc_output):
        # s = [batch_size, dec_hid_dim]
        # enc_output = [batch_size, seq_len, enc_hid_dim * 2]

        batch_size = enc_output.shape[0]
        src_len = enc_output.shape[1]

        # repeat decoder hidden state src_len times
        # s = [batch_size, 1, dec_hid_dim]
        # enc_out_put = [batch_size, seq_len, enc_hid_dim * 2]
        #s = s.unsqueeze(1).repeat(1, src_len, 1)
        s = s.unsqueeze(1)

        context = torch.tanh(self.fc(enc_output))

        attention = torch.bmm(s, context.transpos(1,2)).squeeze(1)

        # energy = [batch_size, src_len, dec_hid_dim]
        # energy = torch.tanh(self.attn(torch.cat((s, enc_output), dim=2)))

        # attention = [batch_size, src_len]
        # attention = self.V(energy).squeeze(2)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, embedding_matrix, emb_dim, vocab_size, rnn_type,\
        dec_layer, enc_hid_dim, dec_hid_dim, dropout, is_bidirectional, attention):

        super().__init__()
        self.attention = attention
        self.embedding = embedding_matrix
        self.rnn_type = rnn_type
        self.is_bidirectional = is_bidirectional
        self.num_layer = dec_layer
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size = enc_hid_dim * 2 + emb_dim, # The number of expected features in the input x
                hidden_size = dec_hid_dim, # The number of features in the hidden state h
                num_layers = dec_layer, # Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two LSTMs together to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results. Default: 1
                bias = True, # If False, then the layer does not use bias weights b_ih and b_hh. Default: True
                batch_first = True, # If True, then the input and output tensors are provided as (batch, seq, feature). Default: False
                dropout = dropout, # If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to dropout. Default: 0
                bidirectional = is_bidirectional # If True, becomes a bidirectional LSTM. Default: False
                #proj_size = 0 # If > 0, will use LSTM with projections of corresponding size. Default: 0
                )
            #TypeError: __init__() got an unexpected keyword argument 'proj_size'
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
        # s equals the dec_init_state
        # if rnn of encoder is LSTM:
        # s = (h_n, c_n) -> h_n = c_n = [batch_size, dec_hid_dim]
        # if rnn of encoder is GRE:
        # s = h_n -> [batch_size, dec_hid_dim]
        # if encoder is bidirectional rnn:
        # enc_output = [batch_size, src_len, enc_hid_dim * 2]
        # else 
        # enc_output = [batch_size, src_len, enc_hid_dim * 1]

        dec_input = dec_input.unsqueeze(1)  # dec_input [batch_size] -> [batch_size, 1]

        embedded = self.dropout(self.embedding(dec_input))  # embedded = [batch_size, 1, emb_dim]

        # a = [batch_size, 1, src_len]
        if isinstance(s, tuple):
            h_t, c_t = s
        else:
            h_t = s
        a = self.attention(h_t, enc_output).unsqueeze(1)

        if isinstance(s, tuple):
            if self.num_layer > 1:
                h_t = h_t.unsqueeze(0).repeat(self.num_layer,1,1)
                c_t = c_t.unsqueeze(0).repeat(self.num_layer,1,1)
            else:
                h_t = h_t.unsqueeze(0)
                c_t = c_t.unsqueeze(0)
        else:
            if self.num_layer > 1:
                h_t = h_t.unsqueeze(0).repeat(self.num_layer,1,1)
            else:
                h_t = h_t.unsqueeze(0)

        # enc_output = [batch_size, src_len, enc_hid_dim * 2]
        # c = [batch_size, 1, enc_hid_dim * 2]
        c = torch.bmm(a, enc_output)

        # rnn_input = [batch_size, 1, (enc_hid_dim * 2) + emb_dim]
        rnn_input = torch.cat((embedded, c), dim=2)

        # dec_output = [batch_size, src_len(=1), dec_hid_dim]
        # dec_hidden = [n_layers * num_directions, batch_size, dec_hid_dim]
        if self.rnn_type == 'LSTM':
            if isinstance(s, tuple):
                dec_output, (dec_h_n, dec_c_n) = self.rnn(rnn_input, (h_t, c_t))
            else:
                dec_output, (dec_h_n, dec_c_n) = self.rnn(rnn_input, (h_t, h_t))
        elif self.rnn_type == 'GRU':
            dec_output, dec_h_n = self.rnn(rnn_input, h_t)

        # embedded = [batch_size, 1, emb_dim] -> [batch_size, emb_dim]
        # dec_output = [batch_size, dec_hid_dim]
        # c = [batch_size, enc_hid_dim * 2]
        embedded = embedded.squeeze(1)
        dec_output = dec_output.squeeze(1)
        c = c.squeeze(1)

        # pred = [batch_size, output_dim]
        pred = self.fc_out(torch.cat((dec_output, c, embedded), dim=1))

        if self.rnn_type == 'LSTM':
            if self.is_bidirectional:
                out_h_n = torch.cat((dec_h_n[-2,:,:], dec_h_n[-1,:,:]), dim=1)
                out_c_n = torch.cat((dec_c_n[-2,:,:], dec_c_n[-1,:,:]), dim=1)
            else:
                out_h_n = dec_h_n[-1,:,:]
                out_c_n = dec_c_n[-1,:,:]
            return pred, (out_h_n, out_c_n)
        elif self.rnn_type == 'GRU':
            if self.is_bidirectional:
                out_h_n = torch.cat((dec_h_n[-2,:,:], dec_h_n[-1,:,:]), dim=1)
            else:
                out_h_n = dec_h_n[-1,:,:]
            return pred, out_h_n
        else:
            raise ValueError('No rnn type is {}, chooes \'LSTM\' or \'GRU\''.format(self.rnn_type))


# seq2seq
class Seq2Seq(nn.Module):
    def __init__(self, model_config, device='cpu'):
        super(Seq2Seq, self).__init__()
        if model_config.embed_path is None:
            self.embedding_matrix = nn.Embedding(model_config.vocab_size, model_config.emb_dim)
        else:
            pass
        self.attention = Attention(model_config.enc_hid_dim, model_config.dec_hid_dim)
        self.encoder = Encoder(
            self.embedding_matrix, model_config.enc_rnn_type, model_config.enc_is_bidirectional, \
            model_config.enc_num_layer, model_config.emb_dim, model_config.enc_hid_dim, \
            model_config.dec_hid_dim, model_config.dropout
            )
        self.decoder = Decoder(
            self.embedding_matrix, model_config.emb_dim, model_config.vocab_size, \
            model_config.dec_rnn_type, model_config.dec_num_layer, model_config.enc_hid_dim, \
            model_config.dec_hid_dim, model_config.dropout, \
            model_config.dec_is_bidirectional, self.attention
            )
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

        # enc_output is all hidden states of the input sequence, back and forwards
        # s is the final forward and backward hidden states, passed through a linear layer
        enc_output, s = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        dec_input = tgt[:, 0]

        for t in range(1, tgt_len):
            # insert dec_input token embeddings, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            # dec_output = [batch_size, output_dim]
            # s = [batch_size, dec_hid_dim]
            dec_output, s = self.decoder(dec_input, s, enc_output)

            # place predictions in a tensor holding predictions for each token
            outputs[:,t,:] = dec_output

            # get the highest predicted token from our predictions
            top1 = dec_output.argmax(1)

            # if teacher foring, use actual next token as next input
            # if not, use predicted token
            dec_input = tgt[:,t] if use_teacher_forcing else top1

        return outputs
