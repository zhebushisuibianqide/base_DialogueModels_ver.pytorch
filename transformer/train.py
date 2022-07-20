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


def train(model, data_iterator, optimizer, criterion, scheduler, device):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(tqdm(data_iterator)):
        srcs = []
        tgts = []  # trg = [batch_size，trg_len]
        for src, tgt in batch:
            srcs.append(src)
            tgts.append(tgt)

        srcs = torch.LongTensor(srcs).to(device)
        tgts = torch.LongTensor(tgts).to(device)

        # pred = [batch_size, tgt_len, pred_dim]
        pred = model(srcs, tgts)

        pred_dim = pred.shape[-1]

        # tgt = [(tgt_len - 1) * batch_size]
        # pred = [(tgt_len - 1) * batch_size, pred_dim]
        tgts = tgts[:,1:].contiguous().view(-1)
        pred = pred[:,1:,:].contiguous().view(-1, pred_dim)

        loss = criterion(pred, tgts)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        epoch_loss += loss.item()

    return epoch_loss


def train_on_batch(model, data_iterator, epoch, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(tqdm(data_iterator)):
        srcs = []
        tgts = []  # tgts = [batch_size，tgt_len]
        for src, tgt in batch:
            srcs.append(src)
            tgts.append(tgt)

        srcs = torch.LongTensor(srcs).to(device)
        tgts = torch.LongTensor(tgts).to(device)

        # pred = [batch_size, tgt_len, pred_dim]
        pred = model(srcs, tgts)

        pred_dim = pred.shape[-1]

        # tgt = [(tgt_len - 1) * batch_size]
        # pred = [(tgt_len - 1) * batch_size, pred_dim]
        tgts = tgts[:,1:].contiguous().view(-1)
        pred = pred[:,1:,:].contiguous().view(-1, pred_dim)

        loss = criterion(pred, tgts)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss


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

    exp_time = len(os.listdir(Conf.samples_dir)) if os.path.exists(Conf.samples_dir) else 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('set device {}'.format(device))

    if not os.path.exists('data/train.tok.ids'):
        print('processing data')
        process_data(args.data_dir)
    print('load data')
    train_data, train_length_data = load_processed_data('data', 'train')
    valid_data, valid_length_data = load_processed_data('data', 'valid')
    valid_data_iterator, valid_tgt_lens = prepare_batch_iterator(valid_data, \
                                                                 valid_length_data['tgt'], Conf.batch_size,
                                                                 shuffle=False)

    _, word2ids, ids2word = read_vocab(os.path.join(args.data_dir, Conf.vocab_path))
    Conf.vocab_size = min(Conf.vocab_size, len(word2ids.keys()))

    print('initilize model, loss, optimizer')
    model = Transformer(Conf, device).to(device)
    model.apply(weights_init)
    PAD_IDX = word2ids['PAD']
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, reduction='sum').to(device)
    optimizer = optim.Adam(model.parameters(), lr=Conf.lr)

#     scheduler_steplr = StepLR(optimizer, step_size=Conf.StepLR_size, gamma=Conf.StepLR_gamma)
    scheduler_steplr = ReduceLROnPlateau(optimizer, mode='min', factor=Conf.StepLR_gamma, patience=2,
                                        verbose=True, threshold=1e-4, threshold_mode='rel',
                                        cooldown=0, min_lr=0.0, eps=1e-8)
    scheduler_warmup = GradualWarmupScheduler(optimizer, \
                                              multiplier=1, total_epoch=Conf.warmming_up, \
                                              after_scheduler=scheduler_steplr)


#     best_valid_loss = float('inf')
    best_valid_loss = 100.0

    print('start Training...')
    if Conf.eval_per_batch is not None:
        '''start train on batch'''
        train_data_iterator, train_tgt_lens = prepare_batch_iterator(train_data, \
                                                                     train_length_data['tgt'], Conf.batch_size,
                                                                     shuffle=True)
        batch_num = len(train_data_iterator)
        epoch_loss = 0
        start_time = time.time()
        lens_per_batch = 0
        for batch_t in tqdm(range(Conf.total_epoch_num * batch_num)):
#             print(batch_t)
            model.train()
#             print(optimizer.state_dict()['param_groups'][0]['lr'])

            srcs = []
            tgts = []  # trg = [batch_size，trg_len]
            src_pos = []
            tgt_pos = []
            for src, tgt, s_pos, t_pos in train_data_iterator[batch_t % batch_num]:
                srcs.append(src)
                tgts.append(tgt)
                src_pos.append(s_pos)
                tgt_pos.append(t_pos)
            srcs = torch.LongTensor(srcs).to(device)
            tgts = torch.LongTensor(tgts).to(device)
            src_pos = torch.LongTensor(src_pos).to(device)
            tgt_pos = torch.LongTensor(tgt_pos).to(device)
            # pred = [batch_size, trg_len, pred_dim]
            pred = model(srcs, tgts, src_pos, tgt_pos)
            pred_dim = pred.shape[-1]

            tgts_labels = tgts[:,1:].contiguous().view(-1)
            pred = pred.view(-1, pred_dim)

            loss = criterion(pred, tgts_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_t+1 < Conf.warmming_up:
                scheduler_warmup.step()
            epoch_loss += loss.item()
            lens_per_batch += sum(train_tgt_lens[(batch_t % batch_num) * Conf.batch_size: \
                                                 (batch_t % batch_num + 1) * Conf.batch_size])-len(srcs)
            if (batch_t+1) % batch_num == 0 and batch_t != 0:
                train_data_iterator, train_tgt_lens = prepare_batch_iterator(train_data, \
                                                                             train_length_data['tgt'], Conf.batch_size,
                                                                             shuffle=True)
            if (batch_t+1) % Conf.eval_per_batch == 0 and batch_t != 0:
                average_train_loss = epoch_loss / lens_per_batch
                train_ppl = math.exp(average_train_loss)
                epoch_loss = 0
                lens_per_batch = 0

                valid_loss = evaluate(model, valid_data_iterator, criterion, device)
                average_valid_loss = valid_loss / (sum(valid_tgt_lens)-len(valid_tgt_lens))
                valid_ppl = math.exp(average_valid_loss)
                
                if batch_t >= Conf.warmming_up:
                    scheduler_warmup.step(metrics=valid_ppl)

                end_time = time.time()
                epoch_mins, epoch_secs = epoch_time(start_time, end_time)

                if valid_ppl < best_valid_loss:
                    best_valid_loss = valid_ppl

                    infer_responses = inferring(model, valid_data_iterator, criterion, ids2word, device,
                                                            Conf)
#                     average_infer_loss = infer_loss / (sum(valid_tgt_lens)-len(valid_tgt_lens))
#                     infer_ppl = math.exp(average_infer_loss)

                    samples_path = os.path.join(Conf.samples_dir, 'exp_time_{}'.format(exp_time))
                    if not os.path.exists(samples_path): os.makedirs(samples_path)

                    save_metrics_msg(valid_data_iterator, infer_responses,
                                     int((batch_t+1)/batch_num)+1, batch_t+1, valid_ppl, ids2word, samples_path)
#                     save_metrics_msg(valid_data_iterator, None,
#                                      int((batch_t+1)/batch_num)+1, batch_t+1, valid_ppl, ids2word, None)

                    if not os.path.exists(args.model_save_dir): os.makedirs(args.model_save_dir)
                    model_path = os.path.join(args.model_save_dir, 'Transformer-base.pt')
                    torch.save(model.state_dict(), model_path)

                if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)
                step_msg_path = os.path.join(args.log_dir, 'log_msg_{}'.format(exp_time))
                save_step_msg(average_train_loss, valid_ppl,
                              int((batch_t+1)/batch_num)+1, batch_t+1, end_time - start_time, step_msg_path)

                print(f'Epoch: {int((batch_t+1)/batch_num)+1:02} |Batch: {batch_t+1:04} |Time: {epoch_mins}m {epoch_secs}s')
                print("Learning Rate : {}".format(optimizer.state_dict()['param_groups'][0]['lr']))
                print(f'\t Train. Loss: {average_train_loss:.3f} |  Train. PPL: {train_ppl:7.3f}')
                print(f'\t Val. Loss: {average_valid_loss:.3f} |  Val. PPL: {valid_ppl:7.3f}')
#                 print(f'\t Val. infer Loss: {average_infer_loss:.3f} |  Val. infer PPL: {infer_ppl:7.3f}')
                start_time = time.time()

    else:
        # train on epoch
        for epoch in range(Conf.total_epoch_num):
            start_time = time.time()
            # output learning rate at each training epoch
            print(optimizer.state_dict()['param_groups'][0]['lr'])

            train_data_iterator, train_tgt_lens = prepare_batch_iterator(train_data, \
                                                                         train_length_data['tgt'], Conf.batch_size,
                                                                         shuffle=True)
            train_loss = train(model, train_data_iterator, optimizer, criterion, scheduler_warmup, device)
            average_train_loss = train_loss / (sum(train_tgt_lens)-len(train_tgt_lens))
            train_ppl = math.exp(average_train_loss)

            valid_loss = evaluate(model, valid_data_iterator, criterion, device)
            average_valid_loss = valid_loss / (sum(valid_tgt_lens)-len(valid_tgt_lens))
            valid_ppl = math.exp(average_valid_loss)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss

                infer_loss, infer_responses = inferring(model, valid_data_iterator, criterion, ids2word, device,
                                                        Conf)
                average_infer_loss = infer_loss / (sum(valid_tgt_lens)-len(valid_tgt_lens))
                infer_ppl = math.exp(average_infer_loss)

                samples_path = os.path.join(Conf.samples_dir, 'exp_time_{}'.format(exp_time))
                if not os.path.exists(samples_path): os.makedirs(samples_path)

                save_metrics_msg(valid_data_iterator, infer_responses,
                                 epoch, 0, valid_ppl, ids2word, samples_path)

                if not os.path.exists(Conf.checkpt_dir): os.makedirs(Conf.checkpt_dir)
                model_path = os.path.join(Conf.checkpt_dir, 'tut3-model.pt')
                torch.save(model.state_dict(), model_path)

            if not os.path.exists(Conf.logging_dir): os.makedirs(Conf.logging_dir)
            step_msg_path = os.path.join(Conf.logging_dir, 'log_msg_{}'.format(exp_time))
            save_step_msg(average_train_loss, valid_ppl,
                          epoch, 0, end_time - start_time, step_msg_path)
            print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\t Train. Loss: {average_train_loss:.3f} |  Train. PPL: {train_ppl:7.3f}')
            print(f'\t Val. Loss: {average_valid_loss:.3f} |  Val. PPL: {valid_ppl:7.3f}')
#             print(f'\t Val. infer Loss: {average_infer_loss:.3f} |  Val. infer PPL: {infer_ppl:7.3f}')

    # end train, do test
    test_data, test_length_data = load_processed_data('data', 'test')
    test_data_iterator, test_tgt_lnes = prepare_batch_iterator(test_data, \
                                                               test_length_data['tgt'], Conf.batch_size, shuffle=False)

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
    parser.add_argument('--model_save_dir', type=str, help='input the path of model dir.')
    parser.add_argument('--log_dir', type=str, help='input the path of model dir.')
    parser.add_argument('--gpu', default='0', type=str, help='choose gpu id.')
#     parser.add_argument('--train_type', default='pretrained', type=str, help='the model type used for training - (new, pretrained, freeze, adapter)')
#     parser.add_argument('--model_name', default='HF-BART-base', type=str, help='the model file.')
    
    args = parser.parse_args()
    main(args)