#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = '领域关联性'
__author__ = 'pi'
__mtime__ = '5/11/2015-011'
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
              ┏┓      ┏┓
            ┏┛┻━━━┛┻┓
            ┃      ☃      ┃
            ┃  ┳┛  ┗┳  ┃
            ┃      ┻      ┃
            ┗━┓      ┏━┛
                ┃      ┗━━━┓
                ┃  神兽保佑    ┣┓
                ┃　永无BUG！   ┏┛
                ┗┓┓┏━┳┓┏┛
                  ┃┫┫  ┃┫┫
                  ┗┻┛  ┗┻┛
"""
from collections import defaultdict
from itertools import chain
from math import log
from pprint import pprint
from numpy import zeros, sum, sqrt, average, square, savetxt, loadtxt
from os import path, listdir, walk
from AMC.ReadAndWrite import get_corpus, get_corpus_all, get_word_id, get_id_word
from LDA.GlobalOption import GlobalOption

option = GlobalOption()


def TF_IDF(domain_corpus):
    '''
    从domain_corpus中得到所有(item， doc)对应的tf-idf
    :param domain_corpus:
    :return:
    '''
    all_word_set = set(chain.from_iterable(domain_corpus))
    N = len(domain_corpus)  # 文档数目
    M = len(all_word_set)  # 所有词的数目

    # idfs = get_idfs(domain_corpus)
    tfs = zeros([M, N])
    for doc_id, doc in enumerate(domain_corpus):
        for item_id in doc:
            tfs[item_id, doc_id] += 1
    dfs = sum(tfs, axis=1)

    W = zeros([M, N])
    for i in range(M):
        for j in range(N):
            if tfs[i, j] > 0:
                W[i, j] = (1 + log(tfs[i, j])) * log(N / dfs[i])
    return W


def DR_all(D_filename, load_flag=True):
    '''
    计算领域domain中所有term和Domain的相关性
    :param term:str
    :param domain_corpus:[[]]
    :return:
    '''
    dr_filename = path.join(path.dirname(D_filename), path.basename(D_filename).split('.')[0]) + '.dr'
    if load_flag and path.exists(dr_filename):
        return loadtxt(dr_filename)

    domain_corpus = get_corpus(D_filename)
    N = len(domain_corpus)  # 文档数目546
    # print(N)
    W = TF_IDF(domain_corpus)
    Wi_ave = average(W, axis=1).reshape((-1, 1))
    S = sqrt(sum(square(W - Wi_ave), axis=1) / N).reshape((-1, 1))

    disp = Wi_ave / S
    # print(disp)

    Wj_ave = average(W, axis=0)
    dev_j = sum(W - Wj_ave, axis=1).reshape((-1, 1))

    dr = disp * dev_j

    savetxt(dr_filename, dr, fmt='%f')
    return dr


# DR_all(path.join(option.CORPUS_NP_DIR, 'corpus_NP.docs'))


def DR(term, D_filename):
    vocab_filename = path.join(path.dirname(D_filename), path.basename(D_filename).split('.')[0]) + '.vocab'
    # print(vocab_filename)
    word2id = get_word_id(vocab_filename)
    # print(word2id)
    try:
        term_id = word2id[term]
        # print('term_id : %s' % term_id)
    except:
        return 0
    return DR_all(D_filename)[term_id]


def I(term, Din_filename, Douts_dir):
    dr_in = DR(term, Din_filename)
    # print('dr_in : %s' % dr_in)

    dr_out = 0
    domain_names = listdir(Douts_dir)
    Dout_filenames = [path.join(path.join(Douts_dir, domain_name), domain_name + '.docs') for domain_name in
                      domain_names]
    num_douts = len(domain_names)

    for Dout_filename in Dout_filenames:
        if DR(term, Dout_filename) == 0:
            num_douts -= 1
        else:
            # print(DR(term, Dout_filename))
            dr_out += DR(term, Dout_filename)
    # print('dr_out : %s' % dr_out)
    # print('num_douts : %s' % num_douts)

    if dr_out == 0:
        dr_out = 1
    else:
        dr_out /= num_douts
    return dr_in / dr_out


def cal_term_domain_relevance(term):
    Din_filename = path.join(option.CORPUS_NP_DIR, 'corpus_NP.docs')
    Douts_dir = option.LDA_NONELEC_DOMAINS1000
    return I(term, Din_filename, Douts_dir)


if __name__ == '__main__':
    # id2word = get_id_word(path.join(option.CORPUS_NP_DIR, path.split(option.CORPUS_NP_DIR)[1]) + '.vocab')
    id2word = get_id_word(path.join(option.LDA1000_ALARMCLOCK, path.split(option.LDA1000_ALARMCLOCK)[1]) + '.vocab')
    # print(id2word)
    term_relevance_dict = {}
    for term in id2word.values():
        relevance = cal_term_domain_relevance(term)
        term_relevance_dict[term] = relevance
        # print('%s, relevance=%s' % (term, relevance))
    pprint(sorted(term_relevance_dict.items(), key=lambda x: x[1], reverse=True))


def get_idfs(domain_corpus):
    '''
    从domain_corpus中得到idfs字典
    :param domain_corpus:
    :return:
    '''
    N = len(domain_corpus)
    dfs = defaultdict(int)
    for doc in domain_corpus:
        for item_id in doc:
            dfs[item_id] += 1
    return dict([(item_id, log(N / df)) for item_id, df in dfs.items()])
