#_*_ coding=utf-8 _*_
import os
import random
import _pickle as pickle
from config import Config

def tokenizer(sentence):
    return [tok for tok in sentence.split()]


def idslist2sent(idslist, idx2word, ignoreUNK=True):
    if ignoreUNK:
        return ' '.join([idx2word[x] for x in idslist if x not in [0,1,2]]).split('EOS')[0]
    else:
        return ' '.join([idx2word[x] for x in idslist if x not in [0,2]]).split('EOS')[0]


def save_metrics_msg(data, responses, epoch, batch, val_ppl, idx2word, dir_path):
    if responses is None:
        return
    sample_path = os.path.join(dir_path,'samples_epoch_{:0>4d}_batch_{:0>6d}_ppl_{:0>.4f}_result'.format(epoch, batch, val_ppl))
    with open(sample_path, 'w', encoding='utf-8') as f:
        for idx, batch_src_tgt in enumerate(data):
            for idxb, [src, tgt, _, _] in enumerate(batch_src_tgt):
                f.write('sample:\n')
                f.write(idslist2sent(src, idx2word)+'\n')
                f.write(idslist2sent(tgt, idx2word)+'\n')
                f.write(responses[idx][idxb]+'\n\n')


def save_step_msg(train_loss, val_ppl, epoch, batch, cost_time, log_path):

    with open(log_path, 'a', encoding='utf-8') as f:
        f.write('{}\t{}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(
            epoch,
            batch,
            train_loss,
            val_ppl,
            cost_time))


def load_data(data_dir, data_type):
    data_path = os.path.join(data_dir, data_type+'_data.txt')
    dialogues = []
    with open(data_path, 'r', encoding='utf-8')as f:
        for iline in f:
            src, tgt = iline.strip('\n').split('__eou__')
            dialogues.append([src.strip(), tgt.strip()])
    return dialogues


def read_vocab(vocab_path):
    word2ids, ids2word = dict(), dict()
    vocab = []
    with open(vocab_path, 'r', encoding='utf-8') as f:
        #vocab = pickle.load(f)
        for iline in f:
            vocab.append(iline.strip('\n').strip())
            
    for idx, word in enumerate(vocab):
        word2ids[word] = idx
        ids2word[idx] = word
    return vocab, word2ids, ids2word


def read_word2vec(word2vec_path):
    word2vec = dict()
    embedding_matrix = []
    with open(word2vec_path, 'r', encoding='utf-8') as f:
        for line in f:
            word, vec = line.strip('\n').split('\t')
            vec_f = [float(x) for x in vec.split(' ')]
            word2vec[word] = vec_f
            embedding_matrix.append(vec_f)
    return word2vec, embedding_matrix


def sent2ids(sentence, word2ids):
    sent_tokens = tokenizer(sentence)
    sent_tokens_proc = [x for x in sent_tokens]
    sent_ids = [word2ids.get(x, word2ids['UNK']) for x in sent_tokens_proc]
    return sent_ids


def process_data(dialogues, word2ids, config):
    dialogue_ids = []
    src_lens = []
    tgt_lens = []
    for dialogue in dialogues:
#         src, tgt = dialogue['query'], dialogue['response']
        src, tgt = dialogue
        src_ids = [word2ids['GO']] + sent2ids(src, word2ids) + [word2ids['EOS']]
        src_pos = [x for x in range(1, len(src_ids)+1)]
        tgt_ids = [word2ids['GO']] + sent2ids(tgt, word2ids) + [word2ids['EOS']]
        tgt_pos = [x for x in range(1, len(tgt_ids)+1)]
        src_lens.append(len(src_ids) if len(src_ids)<config.max_srclen else config.max_srclen)
        tgt_lens.append(len(tgt_ids) if len(tgt_ids)<config.max_tgtlen else config.max_tgtlen)
        if len(src_ids) > config.max_srclen:
            src_ids = src_ids[0:config.max_srclen-1] + [word2ids['EOS']]
            src_pos = src_pos[0:config.max_srclen]
        elif len(src_ids) < config.max_srclen:
            src_ids = src_ids + [word2ids['PAD']]*(config.max_srclen-len(src_ids))
            src_pos = src_pos + [0]*(config.max_srclen-len(src_pos))
        else:
            pass
        if len(tgt_ids) > config.max_tgtlen:
            tgt_ids = tgt_ids[0:config.max_tgtlen-1] + [word2ids['EOS']]
            tgt_pos = tgt_pos[0:config.max_tgtlen]
        elif len(tgt_ids) < config.max_tgtlen:
            tgt_ids = tgt_ids + [word2ids['PAD']]*(config.max_tgtlen-len(tgt_ids))
            tgt_pos = tgt_pos + [0]*(config.max_tgtlen-len(tgt_pos))
        else:
            pass

        dialogue_ids.append([src_ids, tgt_ids, src_pos, tgt_pos])
        dialogue_length_dict = {'src':src_lens, 'tgt':tgt_lens}
    return dialogue_ids, dialogue_length_dict

def save_processed_data(data, data_dir, data_type):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    data_src_path = os.path.join(data_dir, data_type+'.tok.ids')
    with open(data_src_path, 'wb') as f:
        pickle.dump(data, f)


def save_length_dict(length_dict, data_dir, data_type):
    dict_path = os.path.join(data_dir, data_type+'.length.dict')
    with open(dict_path, 'wb') as f:
        pickle.dump(length_dict, f)


def load_processed_data(data_dir, data_type):
    data_src_path = os.path.join(data_dir, data_type+'.tok.ids')
    with open(data_src_path, 'rb') as f:
        dialogues = pickle.load(f)

    length_dict_path = os.path.join(data_dir, data_type+'.length.dict')
    with open(length_dict_path, 'rb') as f:
        dialogues_len_dict = pickle.load(f)

    return dialogues, dialogues_len_dict


def prepare_batch_iterator(data, length, batch_size, shuffle = False):
    data_num = len(data)
    data_ids = [x for x in range(data_num)]

    if shuffle:
        random.shuffle(data_ids)

    batch_iterator = []
    select_lens = []
    batch_num = int(data_num/batch_size)
    left_data_num = data_num - batch_size*batch_num
    print('total data num is {}, create {} batches, left {} samples'.\
        format(data_num, batch_num, left_data_num))
    assert left_data_num >= 0
    batch = []
    for select_id in data_ids[0:batch_size*batch_num]:
        batch.append(data[select_id])
        select_lens.append(length[select_id])
        if len(batch) == batch_size:
            batch_iterator.append(batch)
            batch = []
        else:
            continue
    return batch_iterator, select_lens

def main(data_dir):
    conf = Config()
    print('load data')
    train_dialogues = load_data(data_dir, 'train')
    valid_dialogues = load_data(data_dir, 'valid')
    test_dialogues = load_data(data_dir, 'test')
    print('train data')
    print(train_dialogues[0:2])

    print('load vocab')
    _, word2ids, _ = read_vocab(os.path.join(data_dir, conf.vocab_path))
    for idx, (k, v) in enumerate(word2ids.items()):
        print('word is {}, the id is {}'.format(k,v))
        if idx >=10:
            break

    print('process data')
    train_data_ids, train_len_dict = process_data(train_dialogues, word2ids, conf)
    valid_data_ids, valid_len_dict = process_data(valid_dialogues, word2ids, conf)
    test_data_ids, test_len_dict = process_data(test_dialogues, word2ids, conf)

    print('processed train data')
    print(train_data_ids[0:2])
    print('processed train data lens')
    print(train_len_dict['src'][0:2], train_len_dict['tgt'][0:2])

    print('save processed data')
    save_processed_data(train_data_ids, 'data', 'train')
    save_length_dict(train_len_dict, 'data', 'train')

    save_processed_data(valid_data_ids, 'data', 'valid')
    save_length_dict(valid_len_dict, 'data', 'valid')

    save_processed_data(test_data_ids, 'data', 'test')
    save_length_dict(test_len_dict, 'data', 'test')

    print('done')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='input the path of data dir.')
    args = parser.parse_args()
    
    main(args.data_dir)