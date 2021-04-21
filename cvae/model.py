# _*_ coding=utf-8 _*_

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

''' torch version 1.7.1'''


# Encoder
class Encoder(nn.Module):
    def __init__(self, embedding_matrix, rnn_type, is_bidirectional, enc_layer, \
                 emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super(Encoder, self).__init__()
        self.rnn_type = rnn_type
        self.is_bidirectional = is_bidirectional
        self.num_layer = enc_layer
        self.embedding = embedding_matrix
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=emb_dim,  # The number of expected features in the input x
                hidden_size=enc_hid_dim,  # The number of features in the hidden state h
                num_layers=enc_layer,
                # Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two LSTMs together to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results. Default: 1
                bias=True,  # If False, then the layer does not use bias weights b_ih and b_hh. Default: True
                batch_first=True,
                # If True, then the input and output tensors are provided as (batch, seq, feature). Default: False
                dropout=dropout,
                # If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to dropout. Default: 0
                bidirectional=is_bidirectional  # If True, becomes a bidirectional LSTM. Default: False
                # proj_size = 0 # If > 0, will use LSTM with projections of corresponding size. Default: 0
            )
            # TypeError: __init__() got an unexpected keyword argument 'proj_size'
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=emb_dim,  # The number of expected features in the input x
                hidden_size=enc_hid_dim,  # The number of features in the hidden state h
                num_layers=enc_layer,
                # Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two GRUs together to form a stacked GRU, with the second GRU taking in outputs of the first GRU and computing the final results. Default: 1
                bias=True,  # If False, then the layer does not use bias weights b_ih and b_hh. Default: True
                batch_first=True,
                # If True, then the input and output tensors are provided as (batch, seq, feature). Default: False
                dropout=dropout,
                # If non-zero, introduces a Dropout layer on the outputs of each GRU layer except the last layer, with dropout probability equal to dropout. Default: 0
                bidirectional=is_bidirectional  # If True, becomes a bidirectional GRU. Default: False
            )
        else:
            raise ValueError('No rnn_type is {}, check the config.'.format(rnn_type))
        self.fc = nn.Linear(enc_hid_dim * (int(is_bidirectional) + 1), dec_hid_dim)
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
                enc_h_n = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
                enc_c_n = torch.cat((c_n[-2, :, :], c_n[-1, :, :]), dim=1)
            else:
                enc_h_n = h_n[-1, :, :]
                enc_c_n = c_n[-1, :, :]
        elif self.rnn_type == 'GRU':
            enc_output, h_n = self.rnn(embedded)
            if self.is_bidirectional:
                enc_h_n = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
            else:
                enc_h_n = h_n[-1, :, :]
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


class Priori_net(nn.Module):
    def __init__(self, enc_hid_dim, latent_dim):
        super(Priori_net, self).__init__()
        self.priori_dense = nn.Linear(enc_hid_dim, latent_dim * 2)
        self.activate_layer = nn.Tanh()
        self.priori_mulgovar = nn.Linear(latent_dim * 2, latent_dim * 2)
        self.latent_dim = latent_dim

    def forward(self, enc_output):
        mu_logvar = self.activate_layer(self.priori_dense(enc_output))
        mu_logvar = self.priori_mulgovar(mu_logvar)
        mu, logvar = torch.split(mu_logvar, [self.latent_dim, self.latent_dim], -1)
        return mu, logvar


class Recognize_net(nn.Module):
    def __init__(self, enc_hid_dim, latent_dim):
        super(Recognize_net, self).__init__()
        self.posterior_dense = nn.Linear(enc_hid_dim + enc_hid_dim, latent_dim * 2)
        #self.activate_layer = nn.Tanh()
        self.latent_dim = latent_dim

    def forward(self, enc_output_c, enc_output_r):
        rec_input = torch.cat([enc_output_c, enc_output_r], -1)
        #mu_logvar = self.activate_layer(self.posterior_dense(rec_input))
        mu_logvar = self.posterior_dense(rec_input)
        mu, logvar = torch.split(mu_logvar, [self.latent_dim, self.latent_dim], -1)
        return mu, logvar


class Generation_net(nn.Module):
    def __init__(self, enc_hid_dim, latent_dim, bow_dim, vocab_size, dropout):
        super(Generation_net, self).__init__()
        self.bow_dense1 = nn.Linear(enc_hid_dim+latent_dim, bow_dim)
        self.bow_dense_out = nn.Linear(bow_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, bow_input):
        bow_logits = self.bow_dense1(bow_input)
        bow_logits = self.dropout(bow_logits)
        bow_logits = self.bow_dense_out(bow_logits).unsqueeze(1)
        return bow_logits


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super(Attention, self).__init__()
        # self.attn = nn.Linear((enc_hid_dim) * 2 + dec_hid_dim, dec_hid_dim, bias=False)
        # self.V = nn.Linear(dec_hid_dim, 1, bias=False)
        self.fc = nn.Linear(enc_hid_dim, dec_hid_dim, bias=False)
        self.fout = nn.Softmax(dim=1)

    def forward(self, s, enc_output):
        # s = [batch_size, dec_hid_dim]
        # enc_output = [batch_size, seq_len, enc_hid_dim * 2]

        batch_size = enc_output.shape[0]
        src_len = enc_output.shape[1]

        # repeat decoder hidden state src_len times
        # s = [batch_size, 1, dec_hid_dim]
        # enc_out_put = [batch_size, seq_len, enc_hid_dim]
        # s = s.unsqueeze(1).repeat(1, src_len, 1)
        s = s.unsqueeze(1)

        context = torch.tanh(self.fc(enc_output))

        attention = torch.tanh(torch.bmm(s, context.transpose(1, 2)).squeeze(1))

        # energy = [batch_size, src_len, dec_hid_dim]
        # energy = torch.tanh(self.attn(torch.cat((s, enc_output), dim=2)))

        # attention = [batch_size, src_len]
        # attention = self.V(energy).squeeze(2)

        return self.fout(attention)


class Decoder(nn.Module):
    def __init__(self, embedding_matrix, emb_dim, latent_dim, vocab_size, rnn_type, \
                 dec_layer, enc_hid_dim, dec_hid_dim, dropout, is_bidirectional, attention):

        super(Decoder, self).__init__()
        self.attention = attention
        self.embedding = embedding_matrix
        self.rnn_type = rnn_type
        self.is_bidirectional = is_bidirectional
        self.num_layer = dec_layer
        rnn_input_size = enc_hid_dim + emb_dim + latent_dim
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=rnn_input_size,  # The number of expected features in the input x
                hidden_size=dec_hid_dim,  # The number of features in the hidden state h
                num_layers=dec_layer,
                # Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two LSTMs together to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results. Default: 1
                bias=True,  # If False, then the layer does not use bias weights b_ih and b_hh. Default: True
                batch_first=True,
                # If True, then the input and output tensors are provided as (batch, seq, feature). Default: False
                dropout=dropout,
                # If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to dropout. Default: 0
                bidirectional=is_bidirectional  # If True, becomes a bidirectional LSTM. Default: False
                # proj_size = 0 # If > 0, will use LSTM with projections of corresponding size. Default: 0
            )
            # TypeError: __init__() got an unexpected keyword argument 'proj_size'
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=rnn_input_size,  # The number of expected features in the input x
                hidden_size=dec_hid_dim,  # The number of features in the hidden state h
                num_layers=dec_layer,
                # Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two GRUs together to form a stacked GRU, with the second GRU taking in outputs of the first GRU and computing the final results. Default: 1
                bias=True,  # If False, then the layer does not use bias weights b_ih and b_hh. Default: True
                batch_first=True,
                # If True, then the input and output tensors are provided as (batch, seq, feature). Default: False
                dropout=dropout,
                # If non-zero, introduces a Dropout layer on the outputs of each GRU layer except the last layer, with dropout probability equal to dropout. Default: 0
                bidirectional=is_bidirectional  # If True, becomes a bidirectional GRU. Default: False
            )
        else:
            raise ValueError('No rnn_type is {}, check the config.'.format(rnn_type))
        #self.fc_out = nn.Linear(enc_hid_dim + dec_hid_dim + emb_dim + latent_dim, vocab_size)
        self.fc_out = nn.Linear(dec_hid_dim + latent_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, dec_input, s, enc_output, latent_variables):
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

        if self.attention is not None:
            # use attention
            # enc_output = [batch_size, src_len, enc_hid_dim]
            # c = [batch_size, 1, enc_hid_dim * (is_bi+1)]
            a = self.attention(h_t, enc_output).unsqueeze(1)
            c = torch.bmm(a, enc_output)
            # rnn_input = [batch_size, 1, enc_hid_dim * (is_bi+1) + emb_dim]
            rnn_input = torch.cat((embedded, c), dim=2)
        else:
            c = enc_output[:, -1, :].unsqueeze(1)
            rnn_input = torch.cat((embedded, c), dim=2)

        if isinstance(s, tuple):
            if self.num_layer > 1:
                h_t = h_t.unsqueeze(0).repeat(self.num_layer, 1, 1)
                c_t = c_t.unsqueeze(0).repeat(self.num_layer, 1, 1)
            else:
                h_t = h_t.unsqueeze(0)
                c_t = c_t.unsqueeze(0)
        else:
            if self.num_layer > 1:
                h_t = h_t.unsqueeze(0).repeat(self.num_layer, 1, 1)
            else:
                h_t = h_t.unsqueeze(0)

        rnn_input = torch.cat((rnn_input, latent_variables.unsqueeze(1)), -1)

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
        # c = [batch_size, enc_hid_dim]
        embedded = embedded.squeeze(1)
        dec_output = dec_output.squeeze(1)
        # c = c.squeeze(1)

        # pred = [batch_size, output_dim]
        # pred = self.fc_out(torch.cat((dec_output, c, embedded), dim=1))
        c = c.squeeze(1)
        # pred = F.softmax(self.fc_out(torch.cat((dec_output, c, embedded), dim=1)), dim=1)
        #pred = self.fc_out(torch.cat((dec_output, c, embedded, latent_variables), dim=1))
        pred = self.fc_out(torch.cat((dec_output, latent_variables), dim=1))

        if self.rnn_type == 'LSTM':
            if self.is_bidirectional:
                out_h_n = torch.cat((dec_h_n[-2, :, :], dec_h_n[-1, :, :]), dim=1)
                out_c_n = torch.cat((dec_c_n[-2, :, :], dec_c_n[-1, :, :]), dim=1)
            else:
                out_h_n = dec_h_n[-1, :, :]
                out_c_n = dec_c_n[-1, :, :]
            return pred, (out_h_n, out_c_n)
        elif self.rnn_type == 'GRU':
            if self.is_bidirectional:
                out_h_n = torch.cat((dec_h_n[-2, :, :], dec_h_n[-1, :, :]), dim=1)
            else:
                out_h_n = dec_h_n[-1, :, :]
            return pred, out_h_n
        else:
            raise ValueError('No rnn type is {}, chooes \'LSTM\' or \'GRU\''.format(self.rnn_type))


# cvae
class CVAE(nn.Module):
    def __init__(self, model_config, device='cpu'):
        super(CVAE, self).__init__()
        if model_config.embed_path is None:
            self.embedding_matrix = nn.Embedding(model_config.vocab_size, model_config.emb_dim)
            print('Initializing embedding matrix using random initialization.')
        else:
            from data_utils import read_word2vec
            _, word2vec = read_word2vec(model_config.embed_path)
            word2vec = torch.FloatTensor(word2vec)
            self.embedding_matrix = nn.Embedding.from_pretrained(word2vec)
            print('Initializing embedding matrix form pretrained file.')
        if model_config.use_attention:
            self.attention = Attention(
                model_config.enc_hid_dim * (int(model_config.enc_is_bidirectional) + 1),
                model_config.dec_hid_dim)
        else:
            self.attention = None
        self.encoder = Encoder(
            self.embedding_matrix, model_config.enc_rnn_type, model_config.enc_is_bidirectional, \
            model_config.enc_num_layer, model_config.emb_dim, model_config.enc_hid_dim, \
            model_config.dec_hid_dim, model_config.dropout
        )
        self.sample_net = torch.distributions.multivariate_normal.MultivariateNormal(
            torch.zeros(model_config.latent_dim), torch.eye(model_config.latent_dim)
        )
        self.priori_net = Priori_net(model_config.enc_hid_dim * (int(model_config.enc_is_bidirectional) + 1), \
                                     model_config.latent_dim)
        self.recognize_net = Recognize_net(model_config.enc_hid_dim * (int(model_config.enc_is_bidirectional) + 1), \
                                           model_config.latent_dim)
        if model_config.use_bow:
            self.bow_logits = Generation_net(model_config.enc_hid_dim*(int(model_config.enc_is_bidirectional) + 1),\
                                             model_config.latent_dim, model_config.bow_dim, model_config.vocab_size,\
                                             model_config.dropout)
        else:
            self.bow_logits = None
        self.decoder = Decoder(
            self.embedding_matrix, model_config.emb_dim, model_config.latent_dim, \
            model_config.vocab_size, \
            model_config.dec_rnn_type, model_config.dec_num_layer, \
            model_config.enc_hid_dim * (int(model_config.enc_is_bidirectional) + 1), \
            model_config.dec_hid_dim, model_config.dropout, \
            model_config.dec_is_bidirectional, self.attention
        )
        self.device = device
        self.config = model_config

    def forward(self, src, tgt, use_teacher_forcing=True, use_priori=True):
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
        enc_output_src, s = self.encoder(src)
        enc_output_tgt, _ = self.encoder(tgt)

        sample_variables = self.sample_net.rsample((batch_size, 1)).squeeze(1).to(self.device)
        #print('sample shape: {}'.format(sample_variables.shape))

        priori_mu, priori_logvar = self.priori_net(enc_output_src[:,-1,:])
        #print('priori_mu shape: {}'.format(priori_mu.shape))
        #print('priori_logvar shape: {}'.format(priori_logvar.shape))
        recogn_mu, recogn_logvar = self.recognize_net(enc_output_src[:,-1,:], enc_output_tgt[:,-1,:])
        #print('recogn_mu shape: {}'.format(recogn_mu.shape))
        #print('recogn_logvar shape: {}'.format(recogn_logvar.shape))

        # kld(p1|p2) = log(sqrt(var2))-log(sqrt(var1)) + var1/(2var2) + (mu1-mu2)^2/(2var2) -0.5
        #            = 0.5 * (2log(sqrt(var2)) - 2log(sqrt(var1)) + var1/var2 + (mu1-mu2)^2/var2 - 1)
        #            = 0.5 * (logvar2-logvar1) + var1/var2 + (mu1-mu2)^2/var2 - 1)
        kld = -0.5 * torch.sum(1 + (priori_logvar - recogn_logvar)
                                 - torch.div(torch.pow(priori_mu - recogn_mu, 2), torch.exp(priori_logvar))
                                 - torch.div(torch.exp(priori_logvar), torch.exp(recogn_logvar)))
        #print('kld :{}'.format(kld))

        if use_priori:
            latent_variables = (priori_mu + torch.exp(0.5 * priori_logvar) * sample_variables).to(self.device)
        else:
            latent_variables = (recogn_mu + torch.exp(0.5 * recogn_logvar) * sample_variables).to(self.device)

        if self.config.use_bow:
            bow_logits = self.bow_logits(torch.cat((enc_output_src[:,-1,:],latent_variables), -1)).to(self.device)
            bow_logits = bow_logits.repeat(1, tgt_len-1, 1)
        else:
            bow_logits = None

        # first input to the decoder is the <sos> tokens
        dec_input = tgt[:, 0]

        for t in range(1, tgt_len):
            # insert dec_input token embeddings, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            # dec_output = [batch_size, output_dim]
            # s = [batch_size, dec_hid_dim]
            dec_output, s = self.decoder(dec_input, s, enc_output_src, latent_variables)

            # place predictions in a tensor holding predictions for each token
            outputs[:, t, :] = dec_output

            # get the highest predicted token from our predictions
            top1 = dec_output.argmax(1)

            # if teacher foring, use actual next token as next input
            # if not, use predicted token
            dec_input = tgt[:, t] if use_teacher_forcing else top1

        return outputs, kld, bow_logits
