#_*_ coding=utf-8 _*_


class Config():
    def __init__(self):

        self.device = '1'

        self.data_name = 'daily'
        self.data_dir = '../data'
        self.vocab_path = '../data/vocab.nltk.bpe'
        self.embed_path = '../data/vocab.nltk.bpe_embeddings'
        #self.embed_path = None
        self.emb_dim = 300

        self.logging_dir = 'log'
        self.samples_dir = 'samples'
        self.testing_dir = 'test_samples'
        self.checkpt_dir = 'checkpoints'

        self.max_srclen = 25
        self.max_tgtlen = 25

        self.vocab_size = 20000
        self.emb_size = 300

        self.enc_hid_dim = 300
        self.enc_num_layer = 1
        self.enc_rnn_type = 'GRU' # only 'LSTM' and 'GRU'
        self.enc_is_bidirectional = True

        self.dec_hid_dim = 300
        self.dec_num_layer = 2
        self.dec_rnn_type = 'GRU' # only 'LSTM' and 'GRU'
        self.dec_is_bidirectional = False

        self.dropout = 0.0
        self.lr = 1e-3
        self.batch_size = 64
        self.total_epoch_num = 10
