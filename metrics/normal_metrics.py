# coding=utf-8

import argparse
import os
import re
import _pickle as cpickle
import numpy as np
from nltk import ngrams, sent_tokenize, word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from collections import Counter
from nltk import bigrams, FreqDist
from tqdm import tqdm
from math import inf
import argparse

# from sacrebleu.metrics import BLEU


def _response_tokenize(response):
    """
    Function: 将每个response进行tokenize
    Return: [token1, token2, ......]
    """
    response_tokens = []
    # valid_tokens = set(word2vec.keys())
    for token in response.strip().split(' '):
        # if token in valid_tokens:
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


class NormalMetrics():
    def __init__(self, file_path, vocab, word2vec, model_path):
        """
        Function: 初始化以下变量
        contexts: [context1, context2, ...]
        true_responses: [true_response1, true_response2, ...]
        gen_responses: [gen_response1, gen_response2, ...]
        """
        self.vocab = vocab
        self.word2vec = word2vec
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
                contexts.append(sentences[i + 1].rstrip('\n'))
                true_responses.append(sentences[i + 2].rstrip('\n'))
                generate_responses.append(sentences[i + 3].rstrip('\n'))
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
            if (len(_response_tokenize(true_response)) != 0 and
                    len(_response_tokenize(gen_response)) > 1 and
                    len(_response_tokenize(context.replace(' EOT ', ' '))) != 0):
                tmp1.append(true_response)
                tmp2.append(gen_response)
                tmp3.append(context)
        self.true_responses = tmp1
        self.gen_responses = tmp2
        self.contexts = tmp3

        valid_data_count = len(self.contexts)
        average_len_in_contexts = sum([len(_response_tokenize(sentence))
                                       for sentence in self.contexts]) / valid_data_count
        average_len_in_true_response = sum([len(_response_tokenize(sentence))
                                            for sentence in self.true_responses]) / valid_data_count
        average_len_in_generated_response = sum([len(_response_tokenize(sentence))
                                                 for sentence in self.gen_responses]) / valid_data_count
        self.datamsg = [data_count, valid_data_count,
                        average_len_in_contexts, average_len_in_true_response,
                        average_len_in_generated_response]

    def _consine(self, v1, v2):
        """
        Function：计算两个向量的余弦相似度
        Return：余弦相似度
        """
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

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
            tokens = _response_tokenize(response)
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
            tokens = _response_tokenize(response)
            ngrams_list.extend([element for element in ngrams(tokens, n)])

        if len(ngrams_list) == 0:
            return 0
        else:
            return len(set(ngrams_list)) / len(ngrams_list)


    def get_batch_distinct(self, n, batch_size, mode='gen_responses'):
        """
        Function: 计算每个batch的 true_responses、gen_responses的ngrams的type-token ratio
        Return: ngrams-based type-token ratio
        """
        ngrams_list = []
        if mode == 'true_responses':
            responses = self.true_responses
        else:
            responses = self.gen_responses

        batch_distinct = []
        for idx, response in enumerate(responses):
            if idx and idx%batch_size == 0:
                if len(ngrams_list) == 0:
                    batch_distinct.append(0)
                else:
                    batch_distinct.append(len(set(ngrams_list)) / len(ngrams_list))
                ngrams_list = []

            tokens = _response_tokenize(response)
            ngrams_list.extend([element for element in ngrams(tokens, n)])

        if len(batch_distinct) == 0:
            return 0
        else:
            return sum(batch_distinct) / len(batch_distinct)


    def get_response_length(self):
        """ Reference:
             1. paper : Iulian V. Serban,et al. A Deep Reinforcement Learning Chatbot
        """
        response_lengths = []
        for gen_response in self.gen_responses:
            response_lengths.append(len(_response_tokenize(gen_response)))

        if len(response_lengths) == 0:
            return 0
        else:
            return sum(response_lengths) / len(response_lengths)

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
        weights = {1: (1.0, 0.0, 0.0, 0.0),
                   2: (1 / 2, 1 / 2, 0.0, 0.0),
                   3: (1 / 3, 1 / 3, 1 / 3, 0.0),
                   4: (1 / 4, 1 / 4, 1 / 4, 1 / 4)}
        total_score = []
        for true_response, gen_response in zip(self.true_responses, self.gen_responses):
            score = sentence_bleu(
                [_response_tokenize(true_response)],
                _response_tokenize(gen_response),
                weights[n_gram],
                smoothing_function=SmoothingFunction().method7)
            total_score.append(score)
            # print(true_response)
            # print(gen_response)
            # print(score)

        if len(total_score) == 0:
            return 0
        else:
            return sum(total_score) / len(total_score)


    # def get_sacrebleu(self, max_n_gram):
    #     '''
    #         _TOKENIZERS = {
    #             'none': 'tokenizer_base.BaseTokenizer',
    #             'zh': 'tokenizer_zh.TokenizerZh',
    #             '13a': 'tokenizer_13a.Tokenizer13a',
    #             'intl': 'tokenizer_intl.TokenizerV14International',
    #             'char': 'tokenizer_char.TokenizerChar',
    #             'ja-mecab': 'tokenizer_ja_mecab.TokenizerJaMecab',
    #             'spm': 'tokenizer_spm.TokenizerSPM',
    #         }
    #         SMOOTH_DEFAULTS: Dict[str, Optional[float]] = {
    #             # The defaults for `floor` and `add-k` are obtained from the following paper
    #             # A Systematic Comparison of Smoothing Techniques for Sentence-Level BLEU
    #             # Boxing Chen and Colin Cherry
    #             # http://aclweb.org/anthology/W14-3346
    #             'none': None,   # No value is required
    #             'floor': 0.1,
    #             'add-k': 1,
    #             'exp': None,    # No value is required
    #         }
    #         :param lowercase: If True, lowercased BLEU is computed.
    #         :param force: Ignore data that looks already tokenized.
    #         :param tokenize: The tokenizer to use. If None, defaults to language-specific tokenizers with '13a' as the fallback default.
    #         :param smooth_method: The smoothing method to use ('floor', 'add-k', 'exp' or 'none').
    #         :param smooth_value: The smoothing value for `floor` and `add-k` methods. `None` falls back to default value.
    #         :param max_ngram_order: If given, it overrides the maximum n-gram order (default: 4) when computing precisions.
    #         :param effective_order: If `True`, stop including n-gram orders for which precision is 0. This should be
    #             `True`, if sentence-level BLEU will be computed.
    #         :param trg_lang: An optional language code to raise potential tokenizer warnings.
    #         :param references: A sequence of reference documents with document being
    #         defined as a sequence of reference strings. If given, the reference n-grams
    #         and lengths will be pre-computed and cached for faster BLEU computation
    #         across many systems.
    #     '''
    #     bleu = BLEU(lowercase=True,
    #                 force=False,
    #                 tokenize='13a',
    #                 smooth_method='exp',
    #                 smooth_value=None,
    #                 max_ngram_order=max_n_gram,
    #                 effective_order=False,
    #                 trg_lang='')
    #     # bleu.corpus_score(hys, refs)
    #     # hys = [str1, str2, ..., strn]
    #     # refs = [[ref1, ref2, ..., refn],
    #     #         [ref1', ref2', ..., refn'],...,]
    #     bleu_result = bleu.corpus_score(self.gen_responses, [self.true_responses])
    #     print(bleu_result)
    #     print(bleu.get_signature())
    #     return bleu_result


    def get_greedy_matching(self):
        """
        Function: 计算所有true_responses、gen_responses的greedy_matching
        Return：greedy_matching
        """
        model = self.word2vec
        total_cosine = []
        for true_response, gen_response in zip(self.true_responses, self.gen_responses):
            true_response_token_wv = np.array([model[item] for item in
                                               _response_tokenize(true_response)])
            gen_response_token_wv = np.array([model[item] for item in
                                              _response_tokenize(gen_response)])

            true_gen_cosine = np.array([[self._consine(gen_token_vec, true_token_vec)
                                         for gen_token_vec in gen_response_token_wv] for true_token_vec
                                        in true_response_token_wv])
            gen_true_cosine = np.array([[self._consine(true_token_vec, gen_token_vec)
                                         for true_token_vec in true_response_token_wv] for gen_token_vec
                                        in gen_response_token_wv])

            true_gen_cosine = np.max(true_gen_cosine, 1)
            gen_true_cosine = np.max(gen_true_cosine, 1)
            cosine = (np.sum(true_gen_cosine) / len(true_gen_cosine) + np.sum(gen_true_cosine) / len(
                gen_true_cosine)) / 2
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
                                               _response_tokenize(true_response)])
            gen_response_token_wv = np.array([model[item] for item in
                                              _response_tokenize(gen_response)])

            true_response_sentence_wv = np.sum(true_response_token_wv, 0)
            gen_response_sentence_wv = np.sum(gen_response_token_wv, 0)
            true_response_sentence_wv = true_response_sentence_wv / np.linalg.norm(true_response_sentence_wv)
            gen_response_sentence_wv = gen_response_sentence_wv / np.linalg.norm(gen_response_sentence_wv)
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
                                               _response_tokenize(true_response)])
            gen_response_token_wv = np.array([model[item] for item in
                                              _response_tokenize(gen_response)])

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
                emb[i, :] = w[i, :].dot(np.array([We[token] for token in x[i]])) / np.count_nonzero(w[i, :])
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
                    weights[i, j] = smooth_a / (smooth_a + wordProb[data_tokens[i][j]])
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
                    tokens.append(_response_tokenize(sent.replace(' EOT ', ' ')))
            else:  # true_response, or gen_responses
                for sent in data:
                    tokens.append(_response_tokenize(sent))
            tokens_lengths = [len(x) for x in tokens]
            return tokens, tokens_lengths

        # 1 分词
        tokens, tokens_lengths = create_input_of_average_embedding(data, data_type)
        # 2 统计词频
        [unigramsProb, unigramsFreqDist] = get_language_model(self.model_path, 'unigrams')
        # 3 计算权重
        weights = calculate_weight(tokens, tokens_lengths, smooth_a, unigramsProb)
        # 4 处理数据
        vocab_word2idx = {t: idx for idx, t in enumerate(self.vocab)}
        X = tokens2idx(tokens, tokens_lengths, vocab_word2idx)
        # 5 准备Wrod2Vec
        word2vec = self.word2vec
        Word2Vec = dict()
        for token, idx in vocab_word2idx.items():
            Word2Vec[idx] = np.array(word2vec[token])

        return Word2Vec, X, weights

    def cal_coherence(self, smooth_a=10e-3):
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
            return sum(coherences) / len(coherences)

    def get_embedding_average_with_weight(self, smooth_a=10e-3):
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
            return sum(average_embedding_with_weight) / len(average_embedding_with_weight)


def read_vocab(vocab_path):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    vocab = dict()
    # for l in lines:
    #     w, idx = l.strip('\n').split('\t')
    #     vocab[w.lstrip('__')] = int(idx.strip('-'))
    for idx, l in enumerate(lines):
        w = l.strip('\n')
        vocab[w] = int(idx)
    return vocab


def read_word2vec(word2vec_path):
    with open(word2vec_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    word2vec = dict()
    for l in lines:
        # w, vec = l.strip('\n').split('\t')
        # word2vec[w.lstrip('__')] = [float(x) for x in vec.split(' ')]
        w, vec = l.strip('\n').split(' ')[0], l.strip('\n').split(' ')[1:]
        word2vec[w] = [float(x) for x in vec]
    return word2vec


def get_language_model(model_path, model_name, data_path='../data'):
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
                model_Prob[i] = (unigramsFreqDist[i] + 1) / (unigramsFreqDist.N() + unigramsFreqDist.B())
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
                model_Prob[i] = (bigramsFreqDist[i] + 1) / (unigramsFreqDist[i[0]] + bigramsFreqDist.B())
            Model = [model_Prob, bigramsFreqDist]

        else:
            raise ValueError('no model be named as {}'.format(model_name))

        # save model
        print('new model is created over')
        with open(os.path.join(model_path, model_name), 'wb') as f:
            cpickle.dump(Model, f)

        return Model


def getMetricsMsg(file_path, vocab, word2vec, model_path):
    metrics = NormalMetrics(file_path, vocab, word2vec, model_path)

    Token, Dist_1, Dist_2, Dist_3, Dist_S = metrics.get_dp_gan_metrics()
    print('Token : {}'.format(Token))
    print('Dist-1,2,3 : {},{},{}'.format(Dist_1, Dist_2, Dist_3))
    print('Dist-S : {}'.format(Dist_S))

    # distinct_1 = metrics.get_distinct(1)
    # distinct_2 = metrics.get_distinct(2)
    # distinct_3 = metrics.get_distinct(3)
    distinct_1 = metrics.get_batch_distinct(1, 64)
    distinct_2 = metrics.get_batch_distinct(2, 64)
    distinct_3 = metrics.get_batch_distinct(3, 64)
    print('distinct-1,2,3 : {:0>.4f},{:0>.4f},{:0>.4f}'.format(distinct_1, distinct_2, distinct_3))
    response_length = metrics.get_response_length()
    print('response_length : {}'.format(response_length))

    bleu_1 = metrics.get_bleu(1)
    bleu_2 = metrics.get_bleu(2)
    bleu_3 = metrics.get_bleu(3)
    bleu_4 = metrics.get_bleu(4)
    print('Bleu-1,2,3,4 : {:0>.4f},{:0>.4f},{:0>.4f},{:0>.4f}'.format(bleu_1, bleu_2, bleu_3, bleu_4))
    # sbleu_1 = metrics.get_sacrebleu(1)
    # sbleu_2 = metrics.get_sacrebleu(2)
    # sbleu_3 = metrics.get_sacrebleu(3)
    # sbleu_4 = metrics.get_sacrebleu(4)

    greedy_matching = metrics.get_greedy_matching()
    embedding_average = metrics.get_embedding_average()
    vector_extrema = metrics.get_vector_extrema()
    print('embedding-greedy, average, extrema : {:0>.4f},{:0>.4f},{:0>.4f}'. \
          format(greedy_matching, embedding_average, vector_extrema))
    embedding_distance_average = sum([greedy_matching, embedding_average, vector_extrema]) / 3
    print('Average emb-based : {:0>.4f}'.format(embedding_distance_average))
    coherence = metrics.cal_coherence(smooth_a=10e-3)
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


def interaction(args):
    ''' the interface of users
    checking every needable files, if not exists, creating them.
    '''
    print('Welcome to use this metric tool.')
    print('  checking external materials..., please wait')

    print('  checking \'data dir\'...')
    data_dir_res = '  ... OK' if os.path.exists('../data/train.source') \
                                 and os.path.exists('../data/train.target') else '  ... Error'
    print(data_dir_res)
    if data_dir_res is '  ... Error': return False

    print('  checking \'word2vec file\'...')
    data_dir_res = '  ... OK' if os.path.exists(args.word2vec_path) \
        else '  ... Not Exist'
    print(data_dir_res)
    if data_dir_res is '  ... Not Exist': return False

    print('  checking \'vocab file\'...')
    data_dir_res = '  ... OK' if os.path.exists(args.vocab_path) \
        else '  ... Not Exist'
    print(data_dir_res)
    if data_dir_res is '  ... Not Exist': return False

    print('  checking \'language model\'...')
    data_dir_res = '  ... OK' if os.path.exists('lm_path/unigrams') \
        else '  ... Not Exist'
    print(data_dir_res)
    if data_dir_res is '  ... Not Exist':
        if not os.path.exists('lm_path'): os.mkdir('lm_path')
        _ = get_language_model('lm_path', 'unigrams')
    else:
        pass

    dialogue_models = ['seq2seq_base',
                       'seq2seq_attention', 'cvae',
                       'transformer']

    def check_computable_model(model_name):
        model_exist, computable = False, False
        if not os.path.exists('../{}'.format(model_name)): return [model_exist, computable]
        model_exist = True
        if os.path.exists('../{}/samples'.format(model_name)):
            if len(os.listdir('../{}/samples'.format(model_name))) > 0:
                if len(os.listdir('../{}/samples/exp_time_0'.format(model_name))) > 0:
                    computable = True
                return [model_exist, computable]
        else:
            if os.path.exists('../{}/test_samples'.format(model_name)):
                if len(os.listdir('../{}/test_samples'.format(model_name))) > 0:
                    if len(os.listdir('../{}/test_samples/exp_time_0'.format(model_name))) > 0:
                        computable = True
                    return [model_exist, computable]
        return [model_exist, computable]

    print('  Searching computable DialogueModels')
    possible_models = []
    for idx, model_name in enumerate(tqdm(dialogue_models)):
        model_exist, computable = check_computable_model(model_name)
        if model_exist and computable:
            possible_models.append(model_name)

    print('There are {} dialogue models can be calculated with metrics.'.format(len(possible_models)))
    for idx, model_name in enumerate(possible_models):
        print('{} : {}'.format(idx, model_name))
    print('q : quit')
    if len(possible_models) == 0:
        print('No dialogue model need to be computed. Automatically quit.')
        return False
    else:
        while True:
            input_str = input('please input the number \'0-{}\' or \'q\':'.format(idx))
            # valid_input = ['q'] + [str(x) for x in range(len(possible_models))]:
            if input_str not in ['q'] + [str(x) for x in range(len(possible_models))]:
                print('Input Error, please check your input!\n')
                continue
            else:
                if input_str is 'q':
                    print('Program over. quit...\n')
                    return False
                else:
                    print('<{}> has been choosed, starting calculate the metrics...\n'. \
                          format(possible_models[int(input_str)]))
                    return possible_models[int(input_str)]


def getTheTargetFileName(file_list):
    target_name = None
    target_ppl = inf
    for file_name in file_list:
        file_ppl = float(re.findall('ppl_(.*?)_result', file_name)[0])
        if file_ppl < target_ppl:
            target_name = file_name
            target_ppl = file_ppl
    return target_name, target_ppl


def save_metrics(model_name, write_data_msg, write_metrics_results):
    if not os.path.exists('metrics_results'): os.makedirs('metrics_results')
    with open('metrics_results/results.txt', 'a', encoding='utf-8') as f:
        f.write(model_name + '\n')
        f.write('valid data msg:\n')
        if len(write_metrics_results['valid']) != 0:
            data_msg_title = '\t'.join(['Total_num', 'Valid_num', 'avg_cxt_len', 'avg_gtr_len', 'avg_ger_len'])
            f.write(data_msg_title + '\n')
            f.write('\t'.join([str(x) for x in write_data_msg['valid']]) + '\n')
            f.write('valid metrics results:\n')
            metrics_titles = '\t'.join(['epx_time', \
                                        'Token', 'Dist-1', 'Dist-2', 'Dist-3', 'Dist-S', \
                                        'distinct-1', 'distinct-2', 'distinct-3', 'response-length', \
                                        'Bleu-1', 'Bleu-2', 'Bleu-3', 'Bleu-4', \
                                        'Greedy', 'Average', 'Extrema', 'embedding_based_avg', \
                                        'Coherence'])
            f.write(metrics_titles + '\n')
            for i in range(len(write_metrics_results['valid'])):
                write_line = 'exp_{}\t{}\t{}\t{}\t{}\t{}\t{:0>.4f}\t{:0>.4f}\t{:0>.4f}\t\
                        {:0>.2f}\t{:0>.4f}\t{:0>.4f}\t{:0>.4f}\t{:0>.4f}\t\
                        {:0>.4f}\t{:0>.4f}\t{:0>.4f}\t{:0>.4f}\t{:0>.4f}\n'.format( \
                    i, *write_metrics_results['valid'][i])
                f.write(write_line)
            f.write('average and std.\n')
            average_metrics = list(np.average(write_metrics_results['valid'], 0))
            std_metrics = list(np.std(write_metrics_results['valid'], axis=0))
            write_line = 'exp_avg\t{}\t{}\t{}\t{}\t{}\t{:0>.4f}\t{:0>.4f}\t{:0>.4f}\t\
                        {:0>.2f}\t{:0>.4f}\t{:0>.4f}\t{:0>.4f}\t{:0>.4f}\t\
                        {:0>.4f}\t{:0>.4f}\t{:0>.4f}\t{:0>.4f}\t{:0>.4f}\n'.format( \
                *average_metrics)
            f.write(write_line)
            write_line = 'exp_avg\t{:0>.3f}\t{:0>.3f}\t{:0>.3f}\t{:0>.3f}\t{:0>.3f}\t\
                    {:0>.3f}\t{:0>.3f}\t{:0>.3f}\t{:0>.3f}\t{:0>.3f}\t{:0>.3f}\t{:0>.3f}\t\
                    {:0>.3f}\t{:0>.3f}\t{:0>.3f}\t{:0>.3f}\t{:0>.3f}\t{:0>.3f}\n'.format( \
                *std_metrics)
            f.write(write_line)

        f.write('\n')

        f.write('test data msg:\n')
        if len(write_metrics_results['test']) != 0:
            data_msg_title = '\t'.join(['Total_num', 'Valid_num', 'avg_cxt_len', 'avg_gtr_len', 'avg_ger_len'])
            f.write(data_msg_title + '\n')
            f.write('\t'.join([str(x) for x in write_data_msg['test']]) + '\n')
            f.write('test metrics results:\n')
            metrics_titles = '\t'.join(['epx_time', \
                                        'Token', 'Dist-1', 'Dist-2', 'Dist-3', 'Dist-S', \
                                        'distinct-1', 'distinct-2', 'distinct-3', 'response-length', \
                                        'Bleu-1', 'Bleu-2', 'Bleu-3', 'Bleu-4', \
                                        'Greedy', 'Average', 'Extrema', 'embedding_based_avg', \
                                        'Coherence'])
            f.write(metrics_titles + '\n')
            for i in range(len(write_metrics_results['test'])):
                write_line = 'exp_{}\t{}\t{}\t{}\t{}\t{}\t{:0>.4f}\t{:0>.4f}\t{:0>.4f}\t\
                        {:0>.2f}\t{:0>.4f}\t{:0>.4f}\t{:0>.4f}\t{:0>.4f}\t\
                        {:0>.4f}\t{:0>.4f}\t{:0>.4f}\t{:0>.4f}\t{:0>.4f}\n'.format( \
                    i, *write_metrics_results['test'][i])
                f.write(write_line)
            f.write('average and std.\n')
            average_metrics = list(np.average(write_metrics_results['test'], 0))
            std_metrics = list(np.std(write_metrics_results['test'], axis=0))
            write_line = 'exp_avg\t{}\t{}\t{}\t{}\t{}\t{:0>.4f}\t{:0>.4f}\t{:0>.4f}\t\
                        {:0>.2f}\t{:0>.4f}\t{:0>.4f}\t{:0>.4f}\t{:0>.4f}\t\
                        {:0>.4f}\t{:0>.4f}\t{:0>.4f}\t{:0>.4f}\t{:0>.4f}\n'.format( \
                *average_metrics)
            f.write(write_line)
            write_line = 'exp_avg\t{:0>.3f}\t{:0>.3f}\t{:0>.3f}\t{:0>.3f}\t{:0>.3f}\t\
                    {:0>.3f}\t{:0>.3f}\t{:0>.3f}\t{:0>.3f}\t{:0>.3f}\t{:0>.3f}\t{:0>.3f}\t\
                    {:0>.3f}\t{:0>.3f}\t{:0>.3f}\t{:0>.3f}\t{:0>.3f}\t{:0>.3f}\n'.format( \
                *std_metrics)
            f.write(write_line)
        f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_path', type=str, help='list your vocab path')
    parser.add_argument('--word2vec_path', type=str, help='list your word2vec path')

    args = parser.parse_args()

    model_name = interaction(args)
    if model_name is False:
        pass
    else:
        write_data_msg = {'valid': [], 'test': []}
        write_metrics_results = {'valid': [], 'test': []}
        if os.path.exists('../{}/samples'.format(model_name)):
            file_list = []
            for exp_file in range(len(os.listdir('../{}/samples'.format(model_name)))):
                target_name, target_ppl = getTheTargetFileName(
                    os.listdir('../{}/samples/exp_time_{}'.format(model_name, exp_file)))
                file_list.append(target_name)
            if len(file_list) > 0:
                print(' Valid samples metrics:\n')
                for idx, target_name in enumerate(file_list):
                    print(' exp_time_{}'.format(idx))
                    print('  file name : {}'.format(target_name))
                    file_path = os.path.join('..', model_name, 'samples', 'exp_time_{}'.format(idx), target_name)
                    data_msg, metrics_results = getMetricsMsg(file_path,
                                                              vocab=read_vocab(args.vocab_path),
                                                              word2vec=read_word2vec(args.word2vec_path),
                                                              model_path='lm_path')
                    if idx == 0: write_data_msg['valid'] = data_msg
                    write_metrics_results['valid'].append(metrics_results)

        if os.path.exists('../{}/test_samples'.format(model_name)):
            file_list = []
            for exp_file in range(len(os.listdir('../{}/test_samples'.format(model_name)))):
                target_name, target_ppl = getTheTargetFileName(
                    os.listdir('../{}/test_samples/exp_time_{}'.format(model_name, exp_file)))
                file_list.append(target_name)
            if len(file_list) > 0:
                print(' Test samples metrics:\n')
                for idx, target_name in enumerate(file_list):
                    print(' exp_time_{}'.format(idx))
                    print('  file name : {}'.format(target_name))
                    file_path = os.path.join('..', model_name, 'test_samples', 'exp_time_{}'.format(idx), target_name)
                    data_msg, metrics_results = getMetricsMsg(file_path,
                                                              vocab=read_vocab(args.vocab_path),
                                                              word2vec=read_word2vec(args.word2vec_path),
                                                              model_path='lm_path')
                    if idx == 0: write_data_msg['test'] = data_msg
                    write_metrics_results['test'].append(metrics_results)

        save_metrics(model_name, write_data_msg, write_metrics_results)
