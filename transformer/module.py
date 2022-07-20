# _*_ coding=utf-8 _*_

import math
import torch
import torch.nn as nn
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                    self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
#             warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
#                          self.base_lrs]
            warmup_lr = [base_lr * (self.last_epoch / self.total_epoch + 0.0) for base_lr in
                         self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)

class Scaled_Dot_Product_Attention(nn.Module):
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()
        self.softmax = nn.Softmax(dim=3)

    def forward(self, Q, K, V, masked=None):
        d_k = K.size(-1)
        QKT = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(d_k)

        if len(QKT.shape) != 4:
            QKT.unsqueeze(2)
        if masked is not None:
            QKT = QKT.masked_fill(masked, -1e9)
        att = self.softmax(QKT)
        V = torch.matmul(att, V)
        return V, att


class MultiHead_Attention(nn.Module):
    def __init__(self, d_model, d_q, d_k, d_v, head, att, dropout):
        super(MultiHead_Attention, self).__init__()
        self.Wq = nn.Linear(d_model, d_q * head, bias=False)
        self.Wk = nn.Linear(d_model, d_k * head, bias=False)
        self.Wv = nn.Linear(d_model, d_v * head, bias=False)
        self.Wo = nn.Linear(head * d_v, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)
        self.head = head
        self.d_k = d_k
        self.d_q = d_q
        self.d_v = d_v
        self.scaled_dot_product_att = att
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, masked=None):
        '''
            query: [batch_size, len_q, d_model]
            key: [batch_size, len_k, d_model]
            value: [batch_size, len_v(=len_k), d_model]
            mask: [batch_size, seq_len, seq_len]
        '''
        batch_size = query.size(0)
        # [batch_size, len, d_model] -> [batch_size, head, len, d_q]
        Q = self.Wq(query).view(batch_size, -1, self.head, self.d_q).transpose(1, 2)
        K = self.Wk(key).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
        V = self.Wv(value).view(batch_size, -1, self.head, self.d_v).transpose(1, 2)
        # [batch_size, seq_len, seq_len] -> [batch_size, head, seq_len, seq_len]
        if masked is not None:
            masked = masked.unsqueeze(1).repeat(1, self.head, 1, 1)
        # output: [batch_size, head, len_v, d_v]
        output, att = self.scaled_dot_product_att(Q, K, V, masked)
        # output -> [batch_size, len_v, head * d_v]
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_v)
        output = self.Wo(output)
        output = self.dropout(output)
        output = self.layer_norm(output + query)
        return output, att


class Position_wise_Feed_Forward_Network(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(Position_wise_Feed_Forward_Network, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        '''
            inputs: [batch_size, seq_len, d_model]
        '''
        output = self.ffn(input)
        output = self.dropout(output)
        output = output + input
        return self.layer_norm(output)


class EncoderLayer(nn.Module):
    def __init__(self, model_config):
        super(EncoderLayer, self).__init__()
        self.att = Scaled_Dot_Product_Attention()
        self.enc_self_attn = MultiHead_Attention(model_config.d_model,
                                                 model_config.d_q,
                                                 model_config.d_k,
                                                 model_config.d_v,
                                                 model_config.enc_head,
                                                 self.att,
                                                 model_config.dropout)
        self.pos_ffn = Position_wise_Feed_Forward_Network(model_config.d_model,
                                                          model_config.d_ff,
                                                          model_config.dropout)

    def forward(self, enc_input, enc_self_attn_mask):
        '''
            enc_inputs: [batch_size, src_len, d_model]
            enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_input, enc_input, enc_input,
                                               enc_self_attn_mask)  # enc_input to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn


class DecoderLayer(nn.Module):
    def __init__(self, model_config):
        super(DecoderLayer, self).__init__()
        self.self_att = Scaled_Dot_Product_Attention()
        self.dec_self_attn = MultiHead_Attention(model_config.d_model,
                                                 model_config.d_q,
                                                 model_config.d_k,
                                                 model_config.d_v,
                                                 model_config.dec_head,
                                                 self.self_att,
                                                 model_config.dropout)
        self.att = Scaled_Dot_Product_Attention()
        self.dec_enc_attn = MultiHead_Attention(model_config.d_model,
                                                 model_config.d_q,
                                                 model_config.d_k,
                                                 model_config.d_v,
                                                 model_config.dec_head,
                                                 self.att,
                                                 model_config.dropout)
        self.pos_ffn = Position_wise_Feed_Forward_Network(model_config.d_model,
                                                          model_config.d_ff,
                                                          model_config.dropout)

    def forward(self, dec_input, enc_output, dec_self_attn_mask, dec_enc_attn_mask):
        '''
            dec_input: [batch_size, tgt_len, d_model]
            enc_output: [batch_size, src_len, d_model]
            dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
            dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        '''
        # dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_output, dec_self_attn = self.dec_self_attn(dec_input, dec_input, dec_input,
                                                        dec_self_attn_mask)
        # dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_output, dec_enc_attn = self.dec_enc_attn(dec_output, enc_output, enc_output,
                                                      dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)  # [batch_size, tgt_len, d_model]
        return dec_output, dec_self_attn, dec_enc_attn


def get_sinusoid_encoding_table(n_position, d_model):
    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_model)
    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_model)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table)


def get_attn_pad_mask(seq_q, seq_k):
    '''
        seq_q: [batch_size, seq_len]
        seq_k: [batch_size, seq_len]
        seq_len could be src_len or it could be tgt_len
        seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]


def get_attn_subsequence_mask(seq):
    '''
        seq: [batch_size, tgt_len]
    '''
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1) # Upper triangular matrix
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask # [batch_size, tgt_len, tgt_len]


def greedy_decode(model, srcs, tgts, src_pos, tgt_pos, device, config):
    enc_output = model(srcs, tgts, src_pos, tgt_pos, use_teacher_forcing=False)
    preds = torch.zeros((config.batch_size, config.max_tgtlen, config.vocab_size)).to(device)
    tgt_i = tgts[:,0].unsqueeze(1)
    tgt_pos_i = tgt_pos[:,0].unsqueeze(1)
    Stop_Flag = torch.zeros((config.batch_size)).to(device)
    for i in range(config.max_tgtlen):
        # building mask
        dec_self_attn_pad_mask = get_attn_pad_mask(tgt_i, tgt_i).to(device)  # [batch_size, tgt_len, tgt_len]
        dec_self_attn_subsequent_mask = get_attn_subsequence_mask(tgt_i).to(device)  # [batch_size, 1]
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask),
                                      0).to(device)  # [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask = get_attn_pad_mask(tgt_i, srcs).to(device)
        # building decoder inputing
        dec_input_web = model.embedding_matrix(tgt_i)
        dec_input_peb = model.pos_emb(tgt_pos_i)
        dec_input = dec_input_web + dec_input_peb
        # decoder
        dec_output, dec_self_attns, dec_enc_attns = model.decoder(dec_input, enc_output, 
                                                                  dec_self_attn_pad_mask, dec_enc_attn_mask)
        logits = model.projection(dec_output)
        pred_i = logits[:,-1,:] #torch.argmax(logits[:,-1,:], dim=-1)
        preds[:,i,:] = pred_i
        tgt_i = torch.cat([tgt_i, torch.argmax(pred_i, dim=-1).unsqueeze(1)], 1)
        flag_i = torch.argmax(pred_i, dim=-1).data.eq(3)
        Stop_Flag = torch.logical_or(Stop_Flag, flag_i)
        if (Stop_Flag==True).sum() == srcs.size()[0]:
            print('equal break')
            break
        tgt_pos_i = tgt_pos[:, 0:i+2]
    #print(tgt_i)

    return preds