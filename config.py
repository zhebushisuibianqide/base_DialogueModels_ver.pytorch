#_*_ coding=utf-8 _*_


class Config(Object):
    def __init__(self):

        device = '0'

        data_name = 'daily'
        data_dir = 'data'
        vocab_path = 'data/vocab.nltk.bpe'
        embed_path = 'data/vocab.nltk.bpe_embeddings'

        logging_dir = 'log'
        samples_dir = 'samples'
        checkpt_dir = 'checkpoints'

        max_srclen = 25
        max_tgtlen = 25

        vocab_size = 20000
        emb_size = 300

        enc_hid_dim = 300
        enc_num_layer = 2
        enc_rnn_type = 'LSTM' # only 'LSTM' and 'GRU'
        enc_is_bidirectional = True

        dec_hid_dim = 300
        dec_num_layer = 4
        dec_rnn_type = 'LSTM' # only 'LSTM' and 'GRU'
        dec_is_bidirectional = False

        dropout = 0.1
        lr = 1e-3
        batch_size = 64
        total_epoch_num = 50