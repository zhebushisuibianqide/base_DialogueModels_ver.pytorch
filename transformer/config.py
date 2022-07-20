# _*_ coding=utf-8 _*_


class Config():
    def __init__(self):
        self.device = '0'

        self.logging_dir = 'log'
        self.samples_dir = 'samples'
        self.testing_dir = 'test_samples'
        self.checkpt_dir = 'workings'

        self.max_srclen = 42
        self.max_tgtlen = 42
        self.max_uttlen = 42

        self.vocab_path = 'vocab.txt'
        #self.embed_path = '../data/vocab_embeddings'
        self.embed_path = None
        self.vocab_size = 50000
        self.emb_dim = 768

        self.graph_seed = 123456
        self.enc_num_block = 6
        self.enc_head = 8

        self.dec_num_block = 6
        self.dec_head = 8

        self.d_model = 768
        self.d_k = 96
        self.d_q = 96
        self.d_v = 96
        self.d_ff = 4096

        self.dropout = 0.3
        self.lr = 1e-4
        self.warmming_up = 2000
        self.StepLR_size = 5
        self.StepLR_gamma = 0.98
        self.batch_size = 128
        self.total_epoch_num = 50
        self.eval_per_batch = 100  # set 'number' of 'None'