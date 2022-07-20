import os
import time
import math
import random
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau

from config import Config
from model import Transformer
from data_utils import read_vocab, read_word2vec, load_processed_data
from data_utils import prepare_batch_iterator, idslist2sent
from data_utils import save_metrics_msg, save_step_msg
from data_utils import main as process_data
from module import GradualWarmupScheduler
from module import greedy_decode


import argparse

def weights_init(m):
    if isinstance(m,(nn.Conv2d,nn.Linear)):
        nn.init.xavier_normal_(m.weight)

        
def evaluate(model, data_iterator, criterion, device):
    model.eval()
    epoch_loss = 0
    print('Begin Evaluating ...')
    with torch.no_grad():
#         for i, batch in enumerate(tqdm(data_iterator)):
        for i, batch in enumerate(data_iterator):
            srcs = []
            tgts = []  # tgts = [batch_size，tgt_len]
            src_pos = []
            tgt_pos = []
            for src, tgt, s_pos, t_pos in batch:
                srcs.append(src)
                tgts.append(tgt)
                src_pos.append(s_pos)
                tgt_pos.append(t_pos)

            srcs = torch.LongTensor(srcs).to(device)
            tgts = torch.LongTensor(tgts).to(device)
            src_pos = torch.LongTensor(src_pos).to(device)
            tgt_pos = torch.LongTensor(tgt_pos).to(device)
            # output = [batch_size, tgt_len, output_dim]
            output = model(srcs, tgts, src_pos, tgt_pos, True)  # turn on teacher forcing

            output_dim = output.shape[-1]

            # tgts = [(tgt_len - 1 ) * batch_size]
            # output = [(tgt_len - 1) * batch_size, output_dim]
            output = output.view(-1, output_dim)
            tgts_label = tgts[:,1:].contiguous().view(-1)

            loss = criterion(output, tgts_label)
            epoch_loss += loss.item()

    return epoch_loss


def inferring(model, data_iterator, criterion, id2word, device, config):
    model.eval()
#     epoch_loss = 0
    generate_responses = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_iterator)):
            srcs = []
            tgts = []  # tgt = [batch_size，tgt_len]
            src_pos = []
            tgt_pos = []
            for src, tgt, s_pos, _ in batch:
                srcs.append(src)
                tgts.append(tgt)
                src_pos.append(s_pos)
                tgt_pos.append([x for x in range(1, len(tgt)+1)])

            srcs = torch.LongTensor(srcs).to(device)
            tgts = torch.LongTensor(tgts).to(device)
            src_pos = torch.LongTensor(src_pos).to(device)
            tgt_pos = torch.LongTensor(tgt_pos).to(device)
            # output = [batch_size, tgt_len, output_dim]
            output = greedy_decode(model, srcs, tgts, src_pos, tgt_pos, device, config)
            #output = model(srcs, tgts, False)  # turn off teacher forcing

            output_res_batch = torch.argmax(output, dim=2).tolist()

#             output_dim = output.shape[-1]

            # trg = [(tgt_len - 1 ) * batch_size]
            # output = [(tgt_len - 1) * batch_size, output_dim]
#             output = output[:,1:,:].contiguous().view(-1, output_dim)
#             output = output.view(-1, output_dim)
#             tgts = tgts[:,1:].contiguous().view(-1)

#             loss = criterion(output, tgts)
#             epoch_loss += loss.item()

            response_batch = []
            for output_res in output_res_batch:
                response_batch.append(idslist2sent(output_res, id2word))
            generate_responses.append(response_batch)

    return generate_responses


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def main(args):
    print('load Config')
    Conf = Config()
    os.environ["CUDA_VISIBLE_DEVICES"] = Conf.device

    torch.manual_seed(Conf.graph_seed)
    torch.cuda.manual_seed(Conf.graph_seed)
    torch.cuda.manual_seed_all(Conf.graph_seed)  # if you are using multi-GPU.
    np.random.seed(Conf.graph_seed)  # Numpy module.
    random.seed(Conf.graph_seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    exp_time = len(os.listdir(Conf.samples_dir))-1 if os.path.exists(Conf.samples_dir) else 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('set device {}'.format(device))

    if not os.path.exists('data/train.tok.ids'):
        print('processing data')
        process_data(args.data_dir)
    print('load data')
    test_data, test_length_data = load_processed_data('data', 'test')
    test_data_iterator, test_tgt_lnes = prepare_batch_iterator(test_data, \
                                                               test_length_data['tgt'], Conf.batch_size, shuffle=False)


    _, word2ids, ids2word = read_vocab(os.path.join(args.data_dir, Conf.vocab_path))
    Conf.vocab_size = min(Conf.vocab_size, len(word2ids.keys()))

    print('initilize model, loss, optimizer')
    model = Transformer(Conf, device).to(device)
    model.apply(weights_init)
    PAD_IDX = word2ids['PAD']
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, reduction='sum').to(device)
    
    model_path = os.path.join(Conf.checkpt_dir, 'Transformer-base.pt')

    model.load_state_dict(torch.load(model_path))
    
    valid_loss = evaluate(model, test_data_iterator, criterion, device)
    average_valid_loss = valid_loss / (sum(test_tgt_lnes)-len(test_tgt_lnes))
    valid_ppl = math.exp(average_valid_loss)

    test_responses = inferring(model, test_data_iterator, criterion, ids2word, device,
                                          Conf)
#     average_test_loss = test_loss / (sum(test_tgt_lnes)-len(test_tgt_lnes))
#     test_ppl = math.exp(average_test_loss)

    test_samples_path = os.path.join(Conf.testing_dir, 'exp_time_{}'.format(exp_time))
    if not os.path.exists(test_samples_path): os.makedirs(test_samples_path)
    save_metrics_msg(test_data_iterator, test_responses,
                     0, 0, valid_ppl, ids2word, test_samples_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='input the path of data dir.')
#     parser.add_argument('--model_dir', type=str, help='input the path of model dir.')
    parser.add_argument('--gpu', default='0', type=str, help='choose gpu id.')
#     parser.add_argument('--train_type', default='pretrained', type=str, help='the model type used for training - (new, pretrained, freeze, adapter)')
#     parser.add_argument('--model_name', default='HF-BART-base', type=str, help='the model file.')
    
    args = parser.parse_args()
    main(args)