# _*_ coding=utf-8 _*_


class Config():
    def __init__(self):
        self.device = '1'

        self.data_dir = '../data'

        self.logging_dir = 'log'
        self.samples_dir = 'samples'
        self.testing_dir = 'test_samples'
        self.checkpt_dir = 'checkpoints'

        self.max_srclen = 25
        self.max_tgtlen = 25

        self.vocab_path = '../data/vocab'
        #self.embed_path = '../data/vocab_embeddings'
        self.embed_path = None
        self.vocab_size = 20000
        self.emb_dim = 512

        self.graph_seed = 123456
        self.enc_num_block = 6
        self.enc_head = 8

        self.dec_num_block = 6
        self.dec_head = 8

        self.d_model = 512
        self.d_k = 64
        self.d_q = 64
        self.d_v = 64
        self.d_ff = 2048

        self.dropout = 0.1
        self.lr = 1e-3
        self.warmming_up = 2000
        self.StepLR_size = 5
        self.StepLR_gamma = 0.95
        self.batch_size = 512
        self.total_epoch_num = 100
        self.eval_per_batch = None  # set 'number' of 'None'
