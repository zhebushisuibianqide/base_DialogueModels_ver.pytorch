import os
from config import Config
from model import Seq2Seq
from data_utils import read_vocab, read_word2vec, load_processed_data
from data_utils import prepare_batch_iterator
import torch.optim as optim
import torch.nn as nn
import torch

def train(model, data_iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(data_iterator):
        srcs = []
        tgts = []  # trg = [batch_size，trg_len]
        for src, tgt in batch:
            srcs.append(src)
            tgts.append(tgt)

        srcs = torch.Tensor(srcs)
        tgts = torch.Tensor(tgts)

        # pred = [batch_size, trg_len, pred_dim]
        pred = model(srcs, tgts)

        pred_dim = pred.shape[-1]

        # trg = [(trg_len - 1) * batch_size]
        # pred = [(trg_len - 1) * batch_size, pred_dim]
        trg = trg.view(-1)
        pred = pred.view(-1, pred_dim)

        loss = criterion(pred, trg)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(data_iterator)

def eval(model, data_iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(data_iterator):
            srcs = []
            tgts = []  # trg = [batch_size，trg_len]
            for src, tgt in batch:
                srcs.append(src)
                tgts.append(tgt)

            srcs = torch.Tensor(srcs)
            tgts = torch.Tensor(tgts)
            # out_put = [trg_len, batch_size, output_dim]
            output = model(srcs, tgts, 0)  # turn off teacher forcing

            output_dim = output.shape[-1]

            # trg = [(trg_len - 1 ) * batch_size]
            # output = [(trg_len - 1) * batch_size, output_dim]
            output = output.view(-1, output_dim)
            tgts = tgts.view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(data_iterator)

def main():
    Conf = Config()
    os.environ["CUDA_VISIBLE_DEVICES"] = Conf.device

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data = load_processed_data(Conf.data_dir, 'train')
    valid_data = load_processed_data(Conf.data_dir, 'valid')
    valid_data_iterator = prepare_batch_iterator(valid_data, Conf.batch_size,\
        shuffle = False)

    _, word2ids, _ = read_vocab(Conf.vocab_path)

    model = Seq2Seq(Conf).to(device)
    PAD_IDX = word2ids['<pad>']
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_valid_loss = float('inf')

    for epoch in range(Conf.total_epoch_num):
        start_time = time.time()

        train_data_iterator = prepare_batch_iterator(train_data, Conf.batch_size,\
            shuffle = True)
        train_loss = train(model, train_data_iterator, optimizer, criterion)
        valid_loss = evaluate(model, valid_data_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut3-model.pt')

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')


if __name__ == '__main__':
    main()