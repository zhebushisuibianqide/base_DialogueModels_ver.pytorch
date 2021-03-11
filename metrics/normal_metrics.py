# coding=utf-8

import argparse
import os
import _pickle as cpickle
import numpy as np
from nltk import ngrams, sent_tokenize, word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from collections import Counter
from codes.Metrics_base import Metrics
from nltk import bigrams, FreqDist
from tqdm import tqdm
from math import inf

def _response_tokenize(response):
    """
    Function: 将每个response进行tokenize
    Return: [token1, token2, ......]
    """
    response_tokens = []
    #valid_tokens = set(word2vec.keys())
    for token in response.strip().split(' '):
        #if token in valid_tokens:
        response_tokens.append(token)
#        response_tokens = ["__"+token for token in response_tokens]
    return response_tokens


def _response_tokenize_reduce_stopwords(response):
    from nltk.corpus import stopwords
    response_tokens = []
    for token in response.strip().split(' '):
        if token not in set(stopwords.words('english')):
            response_tokens.append(token)

    return response_tokens


class NormalMetrics(Metrics):
    def __init__(self, file_path, vocab, word2vec, model_path):
        """
        Function: 初始化以下变量
        contexts: [context1, context2, ...]
        true_responses: [true_response1, true_response2, ...]
        gen_responses: [gen_response1, gen_response2, ...]
        """
        self.vocab = vocab
        self.word2vec = word2vec
        self.direction_num = direction_num
        self.model_path = model_path

        contexts, true_responses, generate_responses = \
        self._extract_data(file_path)
        self._stasticAndcleanData([contexts, true_responses, generate_responses])

    def _extract_data(self, path):
        true_responses = []
        generate_responses = []
        contexts = []

        with open(path, 'r', encoding='utf-8') as f:
            sentences = f.readlines()

        for i in range(len(sentences)):
            if sentences[i] == 'sample:\n':
                contexts.append(sentences[i+1].rstrip('\n'))
                true_responses.append(sentences[i+2].rstrip('\n'))
                generate_responses.append(sentences[i+3].rstrip('\n'))
            else:
                pass

        return contexts, true_responses, generate_responses


    def _stasticAndcleanData(self, data):
        [contexts, true_responses, generated_responses] = data
        data_count = len(contexts)

        tmp1 = []
        tmp2 = []
        tmp3 = []
        for context, true_response, gen_response in zip(contexts, true_responses,
                                                        generated_responses):
            if (len(self._response_tokenize(true_response)) !=0 and
                len(self._response_tokenize(gen_response)) >1 and
                len(self._response_tokenize(context.replace(' EOT ',' '))) !=0):
                tmp1.append(true_response)
                tmp2.append(gen_response)
                tmp3.append(context)
        self.true_responses = tmp1
        self.gen_responses = tmp2
        self.contexts = tmp3

        valid_data_count = len(self.contexts)
        average_len_in_contexts = sum([len(self._response_tokenize(sentence))
        for sentence in self.contexts])/valid_data_count
        average_len_in_true_response = sum([len(self._response_tokenize(sentence))
        for sentence in self.true_responses])/valid_data_count
        average_len_in_generated_response = sum([len(self._response_tokenize(sentence))
        for sentence in self.gen_responses])/valid_data_count
        self.datamsg = [data_count, valid_data_count,
                   average_len_in_contexts, average_len_in_true_response,
                   average_len_in_generated_response]


    def _consine(self, v1, v2):
        """
        Function：计算两个向量的余弦相似度
        Return：余弦相似度
        """
        return np.dot(v1,v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


    def get_dp_gan_metrics(self, mode='gen_response'):
        """
        Function：计算所有true_responses、gen_responses的
                  token_gram、unigram、bigram、trigram、sent_gram的数量
        Return：token_gram、unigram、bigram、trigram、sent_gram的数量
        """
        if mode == 'true_response':
            responses = self.true_responses
        else:
            responses = self.gen_responses

        token_gram = []
        unigram = []
        bigram = []
        trigram = []
        sent_gram = []

        for response in responses:
            tokens = self._response_tokenize(response)
            token_gram.extend(tokens)
            unigram.extend([element for element in ngrams(tokens, 1)])
            bigram.extend([element for element in ngrams(tokens, 2)])
            trigram.extend([element for element in ngrams(tokens, 3)])
            sent_gram.append(response)

        return len(token_gram), len(set(unigram)), len(set(bigram)), \
               len(set(trigram)), len(set(sent_gram))


    def get_distinct(self, n, mode='gen_responses'):
        """
        Function: 计算所有true_responses、gen_responses的ngrams的type-token ratio
        Return: ngrams-based type-token ratio
        """
        ngrams_list = []
        if mode == 'true_responses':
            responses = self.true_responses
        else:
            responses = self.gen_responses

        for response in responses:
            tokens = self._response_tokenize(response)
            ngrams_list.extend([element for element in ngrams(tokens, n)])

        if len(ngrams_list) == 0:
            return 0
        else:
            return len(set(ngrams_list)) / len(ngrams_list)


    def get_response_length(self):
        """ Reference:
             1. paper : Iulian V. Serban,et al. A Deep Reinforcement Learning Chatbot
        """
        response_lengths = []
        for gen_response in self.gen_responses:
            response_lengths.append(len(self._response_tokenize(gen_response)))

        if len(response_lengths) == 0:
            return 0
        else:
            return sum(response_lengths)/len(response_lengths)


    def get_bleu(self, n_gram):
        """
        Function: 计算所有true_responses、gen_responses的ngrams的bleu

        parameters:
            n_gram : calculate BLEU-n,
                     calculate the cumulative 4-gram BLEU score, also called BLEU-4.
                     The weights for the BLEU-4 are 1/4 (25%) or 0.25 for each of the 1-gram, 2-gram, 3-gram and 4-gram scores.

        Reference:
            1. https://machinelearningmastery.com/calculate-bleu-score-for-text-python/
            2. https://cloud.tencent.com/developer/article/1042161

        Return: bleu score BLEU-n
        """
        weights = {1:(1.0, 0.0, 0.0, 0.0),
                   2:(1/2, 1/2, 0.0, 0.0),
                   3:(1/3, 1/3, 1/3, 0.0),
                   4:(1/4, 1/4, 1/4, 1/4)}
        total_score = []
        for true_response, gen_response in zip(self.true_responses, self.gen_responses):
            score = sentence_bleu(
                    [self._response_tokenize(true_response)],
                    self._response_tokenize(gen_response),
                    weights[n_gram],
                    smoothing_function=SmoothingFunction().method7)
            total_score.append(score)

        if len(total_score) == 0:
            return 0
        else:
            return sum(total_score) / len(total_score)


    def get_greedy_matching(self):
        """
        Function: 计算所有true_responses、gen_responses的greedy_matching
        Return：greedy_matching
        """
        model = self.word2vec
        total_cosine = []
        for true_response, gen_response in zip(self.true_responses, self.gen_responses):
            true_response_token_wv = np.array([model[item] for item in
                self._response_tokenize(true_response)])
            gen_response_token_wv = np.array([model[item] for item in
                self._response_tokenize(gen_response)])

            true_gen_cosine = np.array([[self._consine(gen_token_vec, true_token_vec)
                for gen_token_vec in gen_response_token_wv] for true_token_vec
                in true_response_token_wv])
            gen_true_cosine = np.array([[self._consine(true_token_vec, gen_token_vec)
                for true_token_vec in true_response_token_wv] for gen_token_vec
                in gen_response_token_wv])

            true_gen_cosine = np.max(true_gen_cosine, 1)
            gen_true_cosine = np.max(gen_true_cosine, 1)
            cosine = (np.sum(true_gen_cosine) / len(true_gen_cosine) + np.sum(gen_true_cosine) / len(gen_true_cosine)) / 2
            total_cosine.append(cosine)

        if len(total_cosine) == 0:
            return 0
        else:
            return sum(total_cosine) / len(total_cosine)


    def get_embedding_average(self):
        model = self.word2vec
        total_cosine = []
        for true_response, gen_response in zip(self.true_responses, self.gen_responses):
            true_response_token_wv = np.array([model[item] for item in
                self._response_tokenize(true_response)])
            gen_response_token_wv = np.array([model[item] for item in
                self._response_tokenize(gen_response)])

            true_response_sentence_wv =  np.sum(true_response_token_wv, 0)
            gen_response_sentence_wv = np.sum(gen_response_token_wv, 0)
            true_response_sentence_wv = true_response_sentence_wv / np.linalg.norm(true_response_sentence_wv)
            gen_response_sentence_wv =  gen_response_sentence_wv / np.linalg.norm(gen_response_sentence_wv)
            cosine = self._consine(true_response_sentence_wv,
                    gen_response_sentence_wv)
            total_cosine.append(cosine)

        if len(total_cosine) == 0:
            return 0
        else:
            return sum(total_cosine) / len(total_cosine)


    def get_vector_extrema(self):
        model = self.word2vec
        total_cosine = []
        for true_response, gen_response in zip(self.true_responses, self.gen_responses):
            true_response_token_wv = np.array([model[item] for item in
                self._response_tokenize(true_response)])
            gen_response_token_wv = np.array([model[item] for item in
                self._response_tokenize(gen_response)])

            true_sent_max_vec = np.max(true_response_token_wv, 0)
            true_sent_min_vec = np.min(true_response_token_wv, 0)
            true_sent_vec = []
            for max_dim, min_dim in zip(true_sent_max_vec, true_sent_min_vec):
                if max_dim > np.abs(min_dim):
                    true_sent_vec.append(max_dim)
                else:
                    true_sent_vec.append(min_dim)
            true_sent_vec = np.array(true_sent_vec)

            gen_sent_max_vec = np.max(gen_response_token_wv, 0)
            gen_sent_min_vec = np.min(gen_response_token_wv, 0)
            gen_sent_vec = []
            for max_dim, min_dim in zip(gen_sent_max_vec, gen_sent_min_vec):
                if max_dim > np.abs(min_dim):
                    gen_sent_vec.append(max_dim)
                else:
                    gen_sent_vec.append(min_dim)
            gen_sent_vec = np.array(gen_sent_vec)

            consine = self._consine(true_sent_vec, gen_sent_vec)
            total_cosine.append(consine)

        if len(total_cosine) == 0:
            return 0
        else:
            return sum(total_cosine) / len(total_cosine)


    def weighted_average_sim_rmpc(self, We, x1, x2, w1, w2):
        """
        Compute the scores between pairs of sentences using weighted average + removing the projection on the first principal component
        :param We: We[i,:] is the vector for word i
        :param x1: x1[i, :] are the indices of the words in the first sentence in pair i
        :param x2: x2[i, :] are the indices of the words in the second sentence in pair i
        :param w1: w1[i, :] are the weights for the words in the first sentence in pair i
        :param w2: w2[i, :] are the weights for the words in the first sentence in pair i
        :return: scores, scores[i] is the matching score of the pair i
        """
        def get_weighted_average(We, x, w):
            """
            Compute the weighted average vectors
            :param We: We[i,:] is the vector for word i
            :param x: x[i, :] are the indices of the words in sentence i
            :param w: w[i, :] are the weights for the words in sentence i
            :return: emb[i, :] are the weighted average vector for sentence i
            """
            n_samples = x.shape[0]
            emb = np.zeros((n_samples, We[0].shape[0]))
            for i in range(n_samples):
                emb[i,:] = w[i,:].dot(np.array([We[token] for token in x[i]])) / np.count_nonzero(w[i,:])
            return emb

        emb1 = get_weighted_average(We, x1, w1)
        emb2 = get_weighted_average(We, x2, w2)

        inn = (emb1 * emb2).sum(axis=1)
        emb1norm = np.sqrt((emb1 * emb1).sum(axis=1))
        emb2norm = np.sqrt((emb2 * emb2).sum(axis=1))
        scores = inn / emb1norm / emb2norm
        return scores


    def prepare_parameters(self, data, data_type, smooth_a):

        def calculate_weight(data_tokens, data_lengths, smooth_a, wordProb):
            weights = np.zeros([len(data_tokens), max(data_lengths)])
            for i in range(len(data_tokens)):
                for j in range(len(data_tokens[i])):
                    weights[i,j] = smooth_a / (smooth_a + wordProb[data_tokens[i][j]])
            return weights


        def tokens2idx(data, data_lengths, vocab_word2idx):
            X = np.zeros([len(data), max(data_lengths)])
            for i in range(len(data)):
                x = [vocab_word2idx.get(token, 0) for token in data[i]]
                X[i, :data_lengths[i]] = x
            return X


        def create_input_of_average_embedding(data, data_type):
            tokens = []
            if data_type == 'context':
                for sent in data:
                    tokens.append(self._response_tokenize(sent.replace(' EOT ', ' ')))
            else: # true_response, or gen_responses
                for sent in data:
                    tokens.append(self._response_tokenize(sent))
            tokens_lengths = [len(x) for x in tokens]
            return tokens, tokens_lengths

        # 1 分词
        tokens, tokens_lengths = create_input_of_average_embedding(data, data_type)
        # 2 统计词频
        [unigramsProb, unigramsFreqDist]= get_language_model('unigrams')
        # 3 计算权重
        weights = calculate_weight(tokens, tokens_lengths, smooth_a, unigramsProb)
        # 4 处理数据
        vocab_word2idx = {t:idx for idx,t in enumerate(self.vocab)}
        X = tokens2idx(tokens, tokens_lengths, vocab_word2idx)
        # 5 准备Wrod2Vec
        word2vec = self.word2vec
        Word2Vec = dict()
        for token, idx in vocab_word2idx.items():
            Word2Vec[idx] = np.array(word2vec[token])

        return Word2Vec, X, weights


    def cal_coherence(self, smooth_a = 10e-3):
        """ Reference:
             1. paper : Better Conversations by Modeling, Filtering, and Optimizing for Coherence and Diversity
             2. paper : A SIMPLE BUT TOUGH-TO-BEAT BASELINE FOR SENTENCE EMBEDDINGS
             3. github : https://github.com/PrincetonML/SIF
        """
        Word2Vec, X_context, weights_context = self.prepare_parameters(self.contexts, 'context', smooth_a)
        _, X_respons, weights_respons = self.prepare_parameters(self.gen_responses, 'gen_responses', smooth_a)

        coherences = self.weighted_average_sim_rmpc(Word2Vec,
                                               X_context,
                                               X_respons,
                                               weights_context,
                                               weights_respons)

        if len(coherences) == 0:
            return 0
        else:
            return sum(coherences)/len(coherences)


    def get_embedding_average_with_weight(self, smooth_a = 10e-3):
        """ Reference:
             1. paper : A SIMPLE BUT TOUGH-TO-BEAT BASELINE FOR SENTENCE EMBEDDINGS
             2. github : https://github.com/PrincetonML/SIF
        """
        Word2Vec, X_true_responses, weights_true_responses = self.prepare_parameters(
                self.true_responses,
                'true_responses',
                smooth_a)
        _, X_gene_responses, weights_gene_responses = self.prepare_parameters(
                self.gen_responses,
                'gen_responses',
                smooth_a)

        average_embedding_with_weight = self.weighted_average_sim_rmpc(Word2Vec,
                                               X_true_responses,
                                               X_gene_responses,
                                               weights_true_responses,
                                               weights_gene_responses)

        if len(average_embedding_with_weight) == 0:
            return 0
        else:
            return sum(average_embedding_with_weight)/len(average_embedding_with_weight)


def get_language_model(model_path, model_name, data_path = '../data'):
    """
    :param model_name:
    :return:
    :function: 通过统计获得ngrams language model，使用Additive 1 smoothing
    """

    if os.path.exists(os.path.join(model_path, model_name)):
        # model has been saved
        with open(os.path.join(model_path, model_name), 'rb') as f:
            model_Prob_FreqDist = cpickle.load(f, encoding='bytes')
            return model_Prob_FreqDist

    else:
        # no model file, create a new model
        print('there no exist language model, creating...')
        with open(os.path.join(data_path, 'train.source'), 'r', encoding='utf-8') as f1, \
        open(os.path.join(data_path, 'train.target'), 'r', encoding='utf-8') as f2:
            train_data = []
            for source, target in zip(f1.readlines(), f2.readlines()):
                train_data.append([source.strip('\n'), target.strip('\n')])

        if model_name == 'unigrams':
            unigramsFreqDist = FreqDist()

            for session in train_data:
                for sent in session:
                    sent_unigramsFreqDist = FreqDist(_response_tokenize(sent))
                    for j in sent_unigramsFreqDist:
                        if j in unigramsFreqDist:
                            unigramsFreqDist[j] += sent_unigramsFreqDist[j]
                        else:
                            unigramsFreqDist[j] = sent_unigramsFreqDist[j]

            model_Prob = FreqDist()
            for i in unigramsFreqDist:
                # Additive 1 smoothing
                model_Prob[i] = (unigramsFreqDist[i]+1) / (unigramsFreqDist.N() + unigramsFreqDist.B())
            Model = [model_Prob, unigramsFreqDist]

        elif model_name == 'bigrams':
            [unigramsProb, unigramsFreqDist] = get_language_model('unigrams')
            bigramsFreqDist = FreqDist()

            for session in train_data:
                for sent in session:
                    sent_bigramsFreqDist = FreqDist(bigrams(_response_tokenize(sent)))
                    for j in sent_bigramsFreqDist:
                        if j in bigramsFreqDist:
                            bigramsFreqDist[j] += sent_bigramsFreqDist[j]
                        else:
                            bigramsFreqDist[j] = sent_bigramsFreqDist[j]

            model_Prob = FreqDist()
            for i in bigramsFreqDist:
                # Additive 1 smoothing
                model_Prob[i] = (bigramsFreqDist[i]+1) / (unigramsFreqDist[i[0]] + bigramsFreqDist.B())
            Model = [model_Prob, bigramsFreqDist]

        else:
            raise ValueError('no model be named as {}'.format(model_name))

        # save model
        print('new model is created over')
        with open(os.path.join(self.model_path, model_name), 'wb') as f:
            cpickle.dump(Model, f)

        return Model


def getMetricsMsg(file_path, vocab, word2vec, model_path):

    metrics = NormalMetrics(file_path, vocab, word2vec, model_path)

    frequence_results = metrics.frequence_results

    Token, Dist_1, Dist_2, Dist_3, Dist_S = metrics.get_dp_gan_metrics()
    print('Token : {}\n'.format(Token))
    print('Dist-1,2,3 : {},{},{}\n'.format(Dist_1,Dist_2,Dist_3))
    print('Dist-S : {}\n'.format(Dist_S))

    distinct_1 = metrics.get_distinct(1)
    distinct_2 = metrics.get_distinct(2)
    distinct_3 = metrics.get_distinct(3)
    print('distinct-1,2,3 : {:0>.4f},{:0>.4f},{:0>.4f}\n'.format(distinct_1,distinct_2,distinct_3))
    response_length = metrics.get_response_length()
    print('response_length : {}\n'.format(response_length))

    bleu_1 = metrics.get_bleu(1)
    bleu_2 = metrics.get_bleu(2)
    bleu_3 = metrics.get_bleu(3)
    bleu_4 = metrics.get_bleu(4)
    print('Bleu-1,2,3,4 : {:0>.4f},{:0>.4f},{:0>.4f},{:0>.4f}\n'.format(bleu_1,bleu_2,bleu_3,bleu_4))

    greedy_matching = metrics.get_greedy_matching()
    embedding_average = metrics.get_embedding_average()
    vector_extrema = metrics.get_vector_extrema()
    print('embedding-\{greedy, average, extrema\} : {:0>.4f},{:0>.4f},{:0>.4f}\n'.\
        format(greedy_matching, embedding_average, vector_extrema))
    embedding_distance_average = sum([greedy_matching, embedding_average, vector_extrema])/3
    print('Average emb-based : {:0>.4f}\n'.format(embedding_distance_average))
    coherence = metrics.cal_coherence(smooth_a = 10e-3)
    print('Coherence : {:0>.4f}\n'.format(coherence))

    data_msg = metrics.datamsg
    metrics_results = [Token, Dist_1, Dist_2, Dist_3, Dist_S,
                       distinct_1, distinct_2, distinct_3,
                       response_length,
                       bleu_1, bleu_2, bleu_3, bleu_4,
                       greedy_matching, embedding_average, vector_extrema,
                       embedding_distance_average,
                       coherence]
    return data_msg, metrics_results


def interaction():
    ''' the interface of users
    checking every needable files, if not exists, creating them.
    '''
    print('Welcome to use this metric tool.\n')
    print('  checking external materials..., please wait\n')

    print('  checking \'data dir\'...')
    data_dir_res = '  ... OK' if os.path.exists('../data/train.source') \
    and os.path.exists('../data/train.target') else '  ... Error'
    print(data_dir_res)
    if data_dir_res is '  ... Error': return False

    print('  checking \'word2vec file\'...')
    data_dir_res = '  ... OK' if os.path.exists('../data/vocab.nltk.bpe_embeddings') \
    else '  ... Not Exist'
    print(data_dir_res)
    if data_dir_res is '  ... Not Exist': return False

    print('  checking \'vocab file\'...')
    data_dir_res = '  ... OK' if os.path.exists('../data/vocab.nltk.bpe') \
    else '  ... Not Exist'
    print(data_dir_res)
    if data_dir_res is '  ... Not Exist': return False

    print('  checking \'language model\'...')
    data_dir_res = '  ... OK' if os.path.exists('lm_path/unigrams') \
    else '  ... Not Exist'
    print(data_dir_res)
    if data_dir_res is '  ... Not Exist':
        if not os.path.exists('lm_path'):os.mkdir('lm_path')
        _ = get_language_model('lm_path', 'unigrams')
    else:
        pass

    dialogue_models = ['seq2seq_base',
    'seq2seq_attention','cvae_bow',
    'transformer']
    def check_computable_models(model_name):
        model_exist, computable = False, False
        if not os.path.exists('../{}'.format(model_name)): return [model_exist, computable]
        model_exist = True
        if os.path.exists('../{}/samples'.format(model_name)):
            if len(os.listdir('../{}/samples'.format(model_name)))>0:
                computable = True
                return [model_exist, computable]
        else:
            os.path.exists('../{}/test_samples'.format(model_name)):
            if len(os.listdir('../{}/test_samples'.format(model_name)))>0:
                computable = True
                return [model_exist, computable]
        return [model_exist, computable]

    print('  Searching computable DialogueModels\n')
    possible_models = []
    for idx, model_name in enumerate(tqdm(dialogue_models)):
        model_exist, computable = check_computable(model_name)
        if model_exist and computable:
            possible_models.append(model_name)

    print('There are {} dialogue models can be calculated with metrics.\n')
    for idx, model_name in enumerate(possible_models):
        print('{} : {}'.format(idx, model_name))
    print('q : quit')
    if len(possible_models) == 0:
        print('No dialogue model need to be computed. Automatically quit.')
        return False
    else:
        while True:
            input_str = input('please input the number \'0-{}\' or \'q\':'.format(idx))
            if input_str not in ['q'] + [x for x in range(len(possible_models))]:
                print('Input Error, please check your input!\n')
                continue
            else:
                if input_str is 'q':
                    print('Program over. quit...\n')
                    return False
                else:
                    print('<{}> has been choosed, starting calculate the metrics...\n'.\
                        format(possible_models[int(input_str)]))
                    return possbile_models[int(input_str)]


def getTheTargetFileName(file_list):
    target_name = None
    target_ppl = inf
    for file_name in file_list:
        file_ppl = float(re.findall('ppl_(.*?).results', file_name))
        if file_ppl < target_ppl:
            target_name = file_name
            target_ppl = file_ppl
    return target_name, target_ppl

if __name__ == '__main__':
    model_name = interaction()
    if isinstance(model_name, bool):
        return

    if os.path.exists('{}/samples'.format(model_name)):
        file_list = os.listdir('{}/samples'.format(model_name))
        if len(file_list)>0:
            print(' Valid samples metrics:\n')
            target_name, target_ppl = getTheTargetFileName(file_list)
            print('  file name : {}'.format(target_names))
            file_path = os.path.join('..', model_name, 'samples', target_name)
            data_msg, metrics_results = getMetricsMsg(file_path,
                vocab = '../data/vocab.nltk.bpe',
                word2vec = '../data/vocab.nltk.bpe_embeddings',
                model_path = 'lm_path')

    if os.path.exists('{}/test_samples'.format(model_name)):
        file_list = os.listdir('{}/test_samples'.format(model_name))
        if len(file_list)>0:
            print(' Test samples metrics:\n')
            target_name, target_ppl = getTheTargetFileName(file_list)
            print('  file name : {}'.format(target_names))
            file_path = os.path.join('..', model_name, 'test_samples', target_name)
            data_msg, metrics_results = getMetricsMsg(file_path,
                vocab = '../data/vocab.nltk.bpe',
                word2vec = '../data/vocab.nltk.bpe_embeddings',
                model_path = 'lm_path')




