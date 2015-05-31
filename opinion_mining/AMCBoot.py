#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'LML_CH'
__mtime__ = '2015/5/9'
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
from __future__ import division
import math
from numpy.ma import sort,log
import re
import numpy
from scipy import spatial
from numpy import savetxt, loadtxt
import nltk
# from opinion_mining.AMC_preprocess import domain_preprocess
from Opinion_Mining.opinion_mining.AMC_preprocess import domain_preprocess

f = open(r'E:\python_workplace\Opinion_Mining\Data\English_stopwords.txt', encoding='utf-8')
stopwords = set(line.strip() for line in f.readlines())  # 读入停用词
lemitaion = nltk.WordNetLemmatizer()
f.close()
ignorechars = ''',:'.;!()#-./1234567890'''

def pre_proc(C):
    C = [w.replace(ignorechars, "") for w in C ]
    C = [lemitaion.lemmatize(w) for w in C if w not in stopwords and len(w) >= 3]
    C = [lemitaion.lemmatize(w, pos='v') for w in C if w not in stopwords and len(w) >= 3]
    C = [w for w in C if w not in stopwords and len(w) >= 3]
    return C

def KL_Measure(i, j):
    '''
    计算KL散度
    :return:
    '''
    KL1 = sum(i*(log(i/j).data))
    KL2 = sum(j*(log(j/i).data))
    D = (KL1 + KL2)/2
    return 1/(1+ math.e ** D )
    # return sum(kl_div(i,j))

def lemitate(w):
    w = w.replace(ignorechars, "")
    w = lemitaion.lemmatize(w)
    w = lemitaion.lemmatize(w, pos='v')
    return w

def getVocabulary():
    f1 = open(r'E:\python_workplace\Opinion_Mining\Data\Nokia 6610\Nokia6610.txt', 'w')
    f2 = open(r'E:\python_workplace\Opinion_Mining\Data\Nokia 6610\noun_prase.txt', 'w')
    f3 = open(r'E:\python_workplace\Opinion_Mining\Data\Nokia 6610\parse_result.txt', encoding='utf-8')
    CF = []
    NP = []
    flag = 1
    w1 = ''
    w2 = ''
    for line in f3:
        line = line.replace("*'", "")
        if line.startswith("result:"):
            NP = []
            temp = []
            f2.write('\n')   #note: remove the first one \n
            f1.write('\n')
        elif line.startswith("#"):
            if line.startswith("#nn"):
                line = re.match(r'.*\((.*)-\d*\'*,\s(.*)-\d*\'*\)$', line).groups()
                word = ' '.join([lemitate(line[1]),lemitate(line[0])])
                word = ' '.join([line[1],line[0]])
                NP.append(word)
                f2.write(word + ',')
        else:
            if line.split("\t")[7] == 'nn':
                w1 = line.split("\t")[1]
                flag = 0
            else:
                if flag == 0:
                    w2 = line.split("\t")[1]
                    w = ' '.join([w1,w2])
                    flag = 1
                else:
                    w = line.split("\t")[1]
                w = w.replace(ignorechars, "")
                if len(w)>2 and w not in stopwords:
                    w = lemitate(w)
                if len(w)>2 and w not in stopwords:
                    f1.write(w + ',')

    f1.close()
    f2.close()
    f3.close()
    domain_preprocess(r'E:\python_workplace\Opinion_Mining\Data\Nokia 6610\Nokia6610.txt',r'E:\eclipse_workplace\AMC\Data\Input\100Reviews\Electronics')

def get_CF():
    CF = []
    CO = []
    CF_N = []
    N = []
    NN = []
    temp = []
    root = ''
    f = open(r'E:\python_workplace\Opinion_Mining\Data\Nokia 6610\parse_result.txt', encoding='utf-8')
    p = open(r'E:\python_workplace\Opinion_Mining\Data\Nokia 6610\noun_phrase.txt',encoding='utf-8')
    NP = [line.strip().split(',') for line in p.readlines()]
    p.close()
    index = 0

    for line in f:
        line = line.replace("*'", "")
        if line.startswith("result:"):
            temp = []
            N = NP[index]
            index += 1
        elif line.startswith("#"):
            if re.match(r'.*\((.*)-\d*\'*,\s(.*)-\d*\'*\)$', line):
                line = re.match(r'.*\((.*)-\d*\'*,\s(.*)-\d*\'*\)$', line).groups()
            else:
                print(line)
            if line[0] == root:
                w = line[1]
                w = w.replace(ignorechars, "")
                # if w in N:
                flag = 0
                for token in N:
                    if token.__contains__(w) and token not in temp:
                        temp.append(token)
                        CF.append(token)
                        flag = 0
                    else:
                        flag = 1
                if flag and w in NN:
                    CF.append(w)
        elif line.split("\t")[7]=='root':
            root = line.split("\t")[1]

        #     w = ''
        #     if line.startswith("#nsubj") or line.startswith("#pobj") or line.startswith("#dobj"):
        #         line = re.match(r'.*\((.*)-\d*\'*,\s(.*)-\d*\'*\)$', line).groups()
        #         w = line[1]
        #     elif line.startswith("#det"):
        #         line = re.match(r'.*\((.*)-\d*\'*,\s(.*)-\d*\'*\)$', line).groups()
        #         w = line[0]
        #     w = w.replace(ignorechars, "")
        #     flag = 0
        #     for token in N:
        #         if token.__contains__(w) and token not in temp:
        #             temp.append(token)
        #             CF.append(token)
        #             flag = 0
        #         else:
        #             flag = 1
        #     if flag and w in NN and w not in temp:
        #         temp.append(w)
        #         CF.append(w)

        elif line.__contains__("NN"):
            word = line.split("\t")[1]
            NN.append(word)
            CF_N.append(word)
        elif line.__contains__("JJ") or line.__contains__("VB") :
            word = line.split("\t")[1]
            CO.append(word)

    # f = open(r'E:\python_workplace\hai2012\corpus\truefeature.txt', encoding='utf-8')
    # TF = []
    # for line in f.readlines():
    #     line.replace(', ',',')
    #     if ',' in line:
    #         tmp = line.split(',')
    #         for t in tmp:
    #             TF.append(t.strip())
    #     else:
    #         TF.append(line.strip())
    # CF = TF
    # CF = CF_N

    addition = ['at&t customer service', 'infrared', 'infrared', 'sprint plan', 'sprint customer service', 'sturdy', 'ringtone', 'background', 'screensaver', 'memory', 'menu options', 't-mobile reception', 't-zone', 't-zone', 't-mobile', 'customer rep', 'call', 'phone performance', 'look', 't-mobile', 'voice dialing', 'message', 'fm', 'operate', 'button', 'key', 'volume', 't-mobile', 'high speed internet', 'ringing tone', 'ring tone', 'game', 'button', 'size', 'size', 'key', 'vibrate setting', 'vibrate setting', 'voice dialing', 'voice dialing', 'picture', 'ringtone', 'key lock', 'ring tone', 'fm radio', 'weight', 'wallpaper', 'tune', 'size', 'size', 'key', 'pc cable', 'loud phone', 'size', 'application', 'pc suite', 'size', 'game', 'ringtone', 'ergonomics', 'size', 'size', 'volume', 'volume', 'size', 'weight', 'ringtone', 'volume', 'weight', 'pc sync', 'tone', 'wallpaper', 'application', 'message', 'picture sharing', 'mms', 'size', 'voice dialing', 'key', 'application', 'size', 'speakerphone', 'look', 'default ringtone', 't-mobile', 'ringtone', 'speakerphone', 'size', 'look', 'weight', 'browsing', 'game', 'battery life', 'voice dialing', 'command', 'button', 'key', 't-mobile', 't-mobile', 'size', 'earpiece', 'voice dialing', 'ringtone', 'gprs', 't-zone', 't-zone', 't-mobile service', 'rate plan', 'weight', 'signal']

    CF += addition
    return pre_proc(CO),pre_proc(CF)

def seed_mustlinks():
    f = open(r'E:\python_workplace\Opinion_Mining\Data\Nokia 6610\Nokia6610.knowl_mustlinks', encoding='utf-8')
    links = []
    for line in f:
        words = re.split("\s",line.strip())
        for word in words:
            links.append(word)
    S = ['phone','headphone']
    flag = 1
    while(flag):
        flag = 0
        for index, word in enumerate(links):
            if word in S:
                if index%2 == 0 and links[index+1] not in S:
                    S.append(links[index+1])
                    flag = 1
                elif index%2 == 1 and links[index-1] not in S:
                    S.append(links[index-1])
                    flag = 1
    f.close()
    return S

def get_pairwise():
    ntopic = 100
    # f = open(r'E:\python_workplace\hai2012\corpus\corpus_NP\corpus_NP.twords', encoding='utf-8')
    # tword_array = loadtxt(r'E:\python_workplace\hai2012\corpus\corpus_NP\corpus_NP.twdist')
    f = open(r'E:\python_workplace\Opinion_Mining\Data\Nokia 6610\Nokia6610.twords', encoding='utf-8')
    tword_array = loadtxt(r'E:\python_workplace\Opinion_Mining\Data\Nokia 6610\Nokia6610.twdist')
    tword_array = -sort(-tword_array,axis=1)
    tword_array = tword_array[:,0:100].transpose()
    wdict = {}
    for num, line in enumerate(f):
        if num == 0:
            pass  # 忽略标题
        else:
            words = re.split("\t",line.strip())
            dcount = 0
            for w in words:
                if w in wdict:
                    wdict[w].append((num-1,dcount))
                elif len(w)>1:
                    wdict[w] = [(num-1,dcount)]
                dcount += 1
    f.close()
    print (wdict)
    keys = [k for k in wdict.keys()]
    keys.sort()
    print (keys)
    # w_t = numpy.zeros([len(keys), ntopic])
    w_t = numpy.ones([len(keys), ntopic]) * 0.000001
    for i, k in enumerate(keys):
        for d in wdict[k]:
            w_t[i,d[1]] = tword_array[d[0]][d[1]]
    print(w_t)
    print(w_t.size)
    pairwise = spatial.distance.squareform(spatial.distance.pdist(w_t, metric = "cosine"))
    # pairwise = spatial.distance.squareform(spatial.distance.pdist(w_t, lambda i,j: KL_Measure(i, j)))

    pairwise_filename = r'../Data/pairwise.txt'
    savetxt(pairwise_filename, pairwise, fmt='%.8f')
    print (pairwise)
    print (pairwise.size)
    return keys, pairwise

keys,pairwise = get_pairwise()
def A(x, y):
    if x in keys:
        i = keys.index(x)
    else:
        # print(x)
        return 1
    if y in keys:
        j = keys.index(y)
    else :
        # print(y)
        return 1
    return pairwise[i,j]

def getCommonWords():
    '''
    调用DomainRelevace.py计算领域相关性低的词为common words
    outdomain的数据集太大，结果先手写设定
    :return:
    '''
    CommonWords = ['people','thing','year','hour','minute','time','motorola','samsung','s105','number','house','cell','night','number']
    return  CommonWords

def main():
    print ("result***********")
    threth_list = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99,1,2]
    # threth_list = [0]
    for threth in threth_list:
        print ("threth=", threth)
        CO,CF = get_CF()
        # CF = sum(C,[])
        print("##### CF，CO #####")
        print(CF)
        print(CO)
        # S = seed_mustlinks()
        S = ['phone','headphone']
        # print("##### S #####")
        # print(S)
        F = []
        O = []
        ffth = threth
        foth = threth
        ooth = threth
        flag = 0
        while (flag == 0):
            flag = 1
            for f in S:
                for cf in CF:
                    if  A(f, cf) <= ffth:
                        S.append(cf)
                        F.append(cf)
                        CF.remove(cf)
                        flag = 0
                for co in CO:
                    if A(f, co) <= foth:
                        O.append(co)
                        CO.remove(co)
                        flag = 0
            for o in O:
                for co in CO:
                    if A(o, co) <= ooth:
                        O.append(co)
                        CO.remove(co)
                        flag = 0
                for cf in CF:
                    if A(o, cf) <= foth:
                        S.append(cf)
                        F.append(cf)
                        CF.remove(cf)
                        flag = 0

        CommonWords = getCommonWords()
        F = [item for item in F if item not in CommonWords]
        print (F)
        print (O)

        f1 = open(r'E:\python_workplace\Opinion_Mining\Data\Nokia 6610\feature_amc.txt', 'w')
        for feature in F:
            f1.writelines(feature + '\n')
        f1.close()

        f = open(r'E:\python_workplace\Opinion_Mining\Data\Nokia 6610\true_feature.txt', encoding='utf-8')
        TF = []
        for line in f.readlines():
            line.replace(', ',',')
            if ',' in line:
                tmp = line.split(',')
                for t in tmp:
                    TF.append(t.strip())
            else:
                TF.append(line.strip())
        # print (TF)
        print (len(TF))
        print (len(F))
        TP = 0
        FP = 0
        # FN = 0
        test = []
        for cf in F:
            if cf in TF:
                TP += 1
                test.append(cf)
                TF.remove(cf)
            else:
                FP += 1
        FN = len(TF)
        print (test)
        print (TF)
        # for tf in TF:
        #     if tf not in F:
        #         FN += 1
        precision = TP/(TP+FP)
        recall = TP/(TP + FN)
        print (TP,FP,FN)
        print ('p=%f'% precision)
        print ('r=%f'% recall)
        f=(2*precision*recall)/(precision+recall)
        print ('F=%f' % f)

        # opinion word's extraction result
        print("opinion word:")
        p = open(r'E:\python_workplace\Opinion_Mining\Data\Nokia 6610\true_opinion.txt', encoding='utf-8')
        TO = []
        for line in p.readlines():
            line.replace(', ',',')
            if ',' in line:
                tmp = line.split(',')
                for t in tmp:
                    TO.append(t.strip())
            else:
                TO.append(line.strip())

        print (len(TO))
        print (len(O))
        TP = 0
        FP = 0
        test = []
        for co in O:
            if co in TO:
                TP += 1
                test.append(co)
                TO.remove(co)
            else:
                FP += 1
        FN = len(TO)
        if(TP):
            precision = TP/(TP+FP)
            recall = TP/(TP + FN)
            print (TP,FP,FN)
            print ('p=%f'% precision)
            print ('r=%f'% recall)
            f=(2*precision*recall)/(precision+recall)
            print ('F=%f' % f)

if __name__ == "__main__":
    # getVocabulary()
    # domain_preprocess(r'E:\python_workplace\Opinion Mining (LML)\Data\Nokia 6610\Nokia6610.txt',r'E:\eclipse_workplace\AMC\Data\Input\100Reviews\Electronics')
    main()