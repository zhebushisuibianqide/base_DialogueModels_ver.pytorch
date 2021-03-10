import os
import time
from config import Config
from model import Seq2Seq
from data_utils import read_vocab, read_word2vec, load_processed_data
from data_utils import prepare_batch_iterator, idslist2sent
from data_utils import save_metrics_msg, save_step_msg
import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
import math

def train(model, data_iterator, optimizer, criterion, device):
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

        # pred = [batch_size, trg_len, pred_dim]
        pred = model(srcs, tgts)

        pred_dim = pred.shape[-1]

        # tgt = [(tgt_len - 1) * batch_size]
        # pred = [(tgt_len - 1) * batch_size, pred_dim]
        tgts = tgts.view(-1)
        pred = pred.view(-1, pred_dim)

        loss = criterion(pred, tgts)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss

def evaluate(model, data_iterator, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_iterator)):
            srcs = []
            tgts = []  # trg = [batch_size，trg_len]
            for src, tgt in batch:
                srcs.append(src)
                tgts.append(tgt)

            srcs = torch.LongTensor(srcs).to(device)
            tgts = torch.LongTensor(tgts).to(device)
            # out_put = [batch_size, trg_len, output_dim]
            output = model(srcs, tgts, True)  # turn on teacher forcing

            output_dim = output.shape[-1]

            # trg = [(trg_len - 1 ) * batch_size]
            # output = [(trg_len - 1) * batch_size, output_dim]
            output = output.view(-1, output_dim)
            tgts = tgts.view(-1)

            loss = criterion(output, tgts)
            epoch_loss += loss.item()

    return epoch_loss


def inferring(model, data_iterator, criterion, device):
    model.eval()
    epoch_loss = 0
    generate_responses = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_iterator)):
            srcs = []
            tgts = []  # trg = [batch_size，trg_len]
            for src, tgt in batch:
                srcs.append(src)
                tgts.append(tgt)

            srcs = torch.LongTensor(srcs).to(device)
            tgts = torch.LongTensor(tgts).to(device)
            # out_put = [batch_size, trg_len, output_dim]
            output = model(srcs, tgts, False)  # turn off teacher forcing

            output_dim = output.shape[-1]

            # trg = [(trg_len - 1 ) * batch_size]
            # output = [(trg_len - 1) * batch_size, output_dim]
            output = output.view(-1, output_dim)
            tgts = tgts.view(-1)

            loss = criterion(output, tgts)
            epoch_loss += loss.item()

            output_res_batch = torch.argmax(output, dim=2).tolist()

            for output_res in output_res_batch:
                generate_responses.append(idslist2sent(output_res))

    return epoch_loss, generate_responses

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def main():
    print('load Config')
    Conf = Config()
    os.environ["CUDA_VISIBLE_DEVICES"] = Conf.device

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('set device {}'.format(device))

    print('load data')
    train_data, train_length_data = load_processed_data(Conf.data_dir, 'train')
    valid_data, valid_length_data = load_processed_data(Conf.data_dir, 'valid')
    valid_data_iterator = prepare_batch_iterator(valid_data, Conf.batch_size,\
        shuffle = False)

    _, word2ids, _ = read_vocab(Conf.vocab_path)

    print('initilize model, loss, optimizer')
    model = Seq2Seq(Conf, device).to(device)
    PAD_IDX = word2ids['<pad>']
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, reduction='sum').to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_valid_loss = float('inf')

    print('start Training...')
    for epoch in range(Conf.total_epoch_num):
        start_time = time.time()

        train_data_iterator = prepare_batch_iterator(train_data, Conf.batch_size,\
            shuffle = True)
        train_loss = train(model, train_data_iterator, optimizer, criterion, device)
        average_train_loss = train_loss/sum(train_length_data['tgt'])
        train_ppl = math.exp(average_train_loss)

        valid_loss = evaluate(model, valid_data_iterator, criterion, device)
        average_valid_loss = valid_loss/sum(valid_length_data['tgt'])
        valid_ppl = math.exp(average_valid_loss)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss

            infer_loss, infer_responses = inferring(model, valid_data_iterator, criterion, device)
            average_infer_loss = infer_loss/sum(valid_length_data['tgt'])
            infer_ppl = math.exp(average_infer_loss)

            save_metrics_msg(valid_data_iterator, infer_responses,
                epoch, 0, valid_ppl, Conf.samples_dir)

            model_path = os.path.join(Conf.checkpt_dir, 'tut3-model.pt')
            if not os.path.exists(model_path): os.makedirs(model_path)
            torch.save(model.state_dict(), model_path)

        save_step_msg(average_train_loss, valid_ppl,
            epoch, 0, end_time-start_time, Conf)
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\t Train. Loss: {average_valid_loss:.3f} |  Train. PPL: {train_ppl:7.3f}')
        print(f'\t Val. Loss: {average_valid_loss:.3f} |  Val. PPL: {valid_ppl:7.3f}')
        print(f'\t Val. infer Loss: {average_infer_loss:.3f} |  Val. infer PPL: {infer_ppl:7.3f}')

    test_data, test_length_data = load_processed_data(Conf.data_dir, 'test')
    test_data_iterator = prepare_batch_iterator(test_data, Conf.batch_size,\
        shuffle = False)

    model.load_state_dict(torch.load(model_path))
    test_loss, test_responses = inferring(model, test_data_iterator, criterion, device)
    average_test_loss = test_loss/sum(test_length_data['tgt'])
    test_ppl = math.exp(average_test_loss)

    save_metrics_msg(valid_data_iterator, infer_responses,
                0, 0, valid_ppl, Conf.testing_dir)



if __name__ == '__main__':
    main()
