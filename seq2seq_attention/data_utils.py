#_*_ coding=utf-8 _*_
import os
import random
from config import Config

def tokenizer(sentence):
    return [tok for tok in sentence.split()]


def load_data(data_dir, data_type):
    dialogues = []
    data_src_path = os.path.join(data_dir, data_type+'.source')
    data_tgt_path = os.path.join(data_dir, data_type+'.target')

    with open(data_src_path, 'r', encoding='utf-8')as f1, \
    open(data_tgt_path, 'r', encoding='utf-8') as f2:
        for src, tgt in zip(f1, f2):
            dialogues.append([src.strip('\n'), tgt.strip('\n')])
    return dialogues


def read_vocab(vocab_path):
    word2ids, ids2word = dict(), dict()
    vocab = list()
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            word, _ = line.strip('\n').split('\t')
            vocab.append(word)

    for idx, word in enumerate(vocab):
        word2ids[word] = idx
        ids2word[idx] = word
    return vocab, word2ids, ids2word


def read_word2vec(word2vec_path):
    word2vec = dict()
    with open(word2vec_path, 'r', encoding='utf-8') as f:
        for line in f:
            word, vec = line.strip('\n').split('\t')
            vec_f = [float(x) for x in vec.split(' ')]
            word2vec[word] = vec_f
    return word2vec


def sent2ids(sentence, word2ids):
    sent_tokens = tokenizer(sentence)
    sent_tokens_proc = ['__'+x for x in sent_tokens]
    sent_ids = [word2ids.get(x, word2ids['<unk>']) for x in sent_tokens_proc]
    return sent_ids


def process_data(dialogues, word2ids, config):
    dialogue_ids = []
    for dialogue in dialogues:
        src, tgt = dialogue
        src_ids = [word2ids['<s>']] + sent2ids(src, word2ids) + [word2ids['</s>']]
        tgt_ids = [word2ids['<s>']] + sent2ids(tgt, word2ids) + [word2ids['</s>']]
        if len(src_ids) > config.max_srclen:
            src_ids = src_ids[0:config.max_srclen-1] + [word2ids['</s>']]
        elif len(src_ids) < config.max_srclen:
            src_ids = src_ids + [word2ids['<pad>']]*(config.max_srclen-len(src_ids))
        else:
            pass
        if len(tgt_ids) > config.max_tgtlen:
            tgt_ids = tgt_ids[0:config.max_tgtlen-1] + [word2ids['</s>']]
        elif len(tgt_ids) < config.max_tgtlen:
            tgt_ids = tgt_ids + [word2ids['<pad>']]*(config.max_tgtlen-len(tgt_ids))
        else:
            pass

        dialogue_ids.append([src_ids, tgt_ids])
    return dialogue_ids

def save_processed_data(data, data_dir, data_type):
    data_src_path = os.path.join(data_dir, data_type+'.source.tok.ids')
    data_tgt_path = os.path.join(data_dir, data_type+'.target.tok.ids')
    with open(data_src_path, 'w', encoding='utf-8') as f1, \
    open(data_tgt_path, 'w', encoding='utf-8') as f2:
        for src, tgt in data:
            write_src_line = '\t'.join([str(x) for x in src])
            write_tgt_line = '\t'.join([str(x) for x in tgt])
            f1.write(write_src_line+'\n')
            f2.write(write_tgt_line+'\n')


def load_processed_data(data_dir, data_type):
    dialogues = []
    data_src_path = os.path.join(data_dir, data_type+'.source.tok.ids')
    data_tgt_path = os.path.join(data_dir, data_type+'.target.tok.ids')

    with open(data_src_path, 'r', encoding='utf-8') as f1, \
    open(data_tgt_path, 'r', encoding='utf-8') as f2:
        for src, tgt in zip(f1, f2):
            src_str = src.strip('\n').split('\t')
            tgt_str = tgt.strip('\n').split('\t')
            dialogues.append([[int(x) for x in src_str],
                [int(x) for x in tgt_str]])
    return dialogues


def prepare_batch_iterator(data, batch_size, shuffle = False):
    if shuffle:
        random.shuffle(data)

    batch_iterator = []
    data_num = len(data)
    batch_num = int(data_num/batch_size)
    left_data_num = data_num - batch_size*batch_num
    print('total data num is {}, create {} batches, left {} samples'.\
        format(data_num, batch_num, left_data_num))
    assert left_data_num >= 0
    batch = []
    for item in data:
        batch.append(item)
        if len(batch) == batch_size:
            batch_iterator.append(batch)
            batch = []
        else:
            continue
    return batch_iterator

def main():
    conf = Config()
    print('load data')
    train_dialogues = load_data(conf.data_dir, 'train')
    valid_dialogues = load_data(conf.data_dir, 'valid')
    test_dialogues = load_data(conf.data_dir, 'test')
    print('train data')
    print(train_dialogues[0:2])

    print('load vocab')
    _, word2ids, _ = read_vocab(conf.vocab_path)
    for idx, (k, v) in enumerate(word2ids.items()):
        print('word is {}, the id is {}'.format(k,v))
        if idx >=10:
            break

    print('process data')
    train_data_ids = process_data(train_dialogues, word2ids, conf)
    valid_data_ids = process_data(valid_dialogues, word2ids, conf)
    test_data_ids = process_data(test_dialogues, word2ids, conf)

    print('processed train data')
    print(train_data_ids[0:2])

    print('save processed data')
    save_processed_data(train_data_ids, conf.data_dir, 'train')
    save_processed_data(valid_data_ids, conf.data_dir, 'valid')
    save_processed_data(test_data_ids, conf.data_dir, 'test')
    print('done')

if __name__ == '__main__':
    main()
