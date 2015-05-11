#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'pi'
__mtime__ = '2/16/2015-016'
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
from os import listdir, makedirs
from os.path import join, exists


def domain_preprocess(domaFileName, STORE_DOMA_DIR_NAME):
    '''
    被amc_preprocess调用
    :param domaFileName:
    :param STORE_DOMA_DIR_NAME:
    :return:
    '''
    domaFilePre = domaFileName.split('\\')[-1].split('.')[0]
    domaDir = join(STORE_DOMA_DIR_NAME, domaFilePre)  # 每个domain存在一个目录下，每个domain包含docs, vocab
    # print(domaDir)
    if (not exists(domaDir)):
        # mkdir(domaDir)
        makedirs(domaDir)
    domaFile = open(domaFileName, encoding='utf-8', newline='\r\n', errors='ignore')

    # 建立字典id-word映射
    words = []
    for doc_line in domaFile:
        words += doc_line.strip().split(',')
    # print(len(words))
    vocab = list(set(words))
    vocab.remove('')
    # print(vocab)
    vocabFileName = join(domaDir, str(domaFilePre) + '.vocab')
    vocabFile = open(vocabFileName, 'w', encoding='utf-8')
    for vid, v in enumerate(vocab):
        # vocabFile.write(str(vid) + ':' + v + os.linesep)
        vocabFile.write(str(vid) + ':' + v + '\n')
    vocabFile.close()

    # 重新建立docs，用id表示word
    docFileName = join(domaDir, str(domaFilePre) + '.docs')
    # print(docFileName)
    docFile = open(docFileName, 'w', encoding='utf-8')

    # domaFile.close()
    # domaFile = open(domaFileName, encoding='utf-8', newline='\r\n', errors='ignore')
    domaFile.seek(0)
    for doc_line in domaFile:
        words_line = doc_line.strip().split(',')
        while '' in words_line:
            words_line.remove('')
        # print(words_line)
        wordIds = [vocab.index(word) for word in words_line]
        # print(wordIds)
        for wordId in wordIds:
            docFile.write(str(wordId) + " ")
        docFile.write('\n')
        # docFile.writelines(str(wordIds).replace(',', ' ').replace('[', '').replace(']', '') + '\n')

    domaFile.close()
    docFile.close()


def amc_preprocess(DIR, STORE_DOMA_DIR_NAME):
    '''
    将原始文件texts转换成docs+vocab
    :param DIR:
    :param STORE_DOMA_DIR_NAME:
    :return:
    '''
    domaFileNames = listdir(DIR)
    domaFileNames = [join(DIR, domaFileName) for domaFileName in domaFileNames]
    # print(domaFileNames)
    for domaFileName in domaFileNames:
        domain_preprocess(domaFileName, STORE_DOMA_DIR_NAME)


def docs_coopration(DOMA_DIR, TEST_DOMA_DIR):
    '''
    将多个docs合并到一个文件中，一行代表一个原始文件
    :param DOMA_DIR:
    :param TEST_DOMA_DIR:
    :return:
    '''
    allDocFileNames = listdir(DOMA_DIR)
    domainNames = set([docFileName.split('_')[0] for docFileName in allDocFileNames])
    if (not exists(TEST_DOMA_DIR)):
        makedirs(TEST_DOMA_DIR)

    # print(docFileNames)
    for domainName in domainNames:
        domainFileName = join(TEST_DOMA_DIR, domainName + '.csv')
        # print(domainFileName)
        domainFile = open(domainFileName, 'w', encoding='utf-8', errors='ignore')
        # docFileNames = [join(DOMA_DIR, docFileName) for docFileName in allDocFileNames if domainName in docFileName]
        docFileNames = [join(DOMA_DIR, docFileName) for docFileName in allDocFileNames if
                        docFileName.__contains__(domainName)]
        # print(docFileNames)

        for docFileName in docFileNames:
            # print(docFileName)
            docFile = open(docFileName, encoding='utf-8', newline='\r\n')
            lines = ''
            for line in docFile:
                lines += line.strip() + ','
            lines = lines.strip(",")
            # print(lines)
            domainFile.writelines(lines + '\n')


def get_dictionary(input_filename):
    '''
    从vocab文件（word_id:word）中获取字典并返回
    :return:
    '''
    d = dict()
    with open(input_filename) as input_file:
        for line_id, line in enumerate(input_file):
            d[line_id] = line.strip().split(':')[1]
    return d


def get_origin_texts(input_dir=r'E:\mine\java_workspace\AMC_master\Data\Input\100Reviews\Electronics',
                     output_dir=r'E:\mine\java_workspace\AMC_master\Data\Input\100Reviews_origin\Electronics'):
    '''
    将docs和vocab文件还原成原始文件
    :return:
    '''
    for domain_dir in listdir(input_dir):
        output_filename = join(output_dir, domain_dir) + '.txt'
        id_word_dict = get_dictionary(join(join(input_dir, domain_dir), domain_dir) + '.vocab')
        with open(join(join(input_dir, domain_dir), domain_dir) + '.docs') as docs_file, open(output_filename,
                                                                                              'w') as output_file:
            for line in docs_file:
                new_line = [id_word_dict[int(id)] for id in line.strip().split()]
                output_file.write(' '.join(new_line) + '\n')
