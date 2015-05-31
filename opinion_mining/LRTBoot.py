#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'LML_CH'
__mtime__ = '2015/1/27'
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
import re
import nltk
import math

f = open(r'E:\python_workplace\Opinion_Mining\Data\English_stopwords.txt', encoding='utf-8')
stopwords = set(line.strip() for line in f.readlines())  # 读入停用词
lemitaion = nltk.WordNetLemmatizer()
f.close()
ignorechars = ''',:'.;!()&#'''

def pre_proc1(C):
    # C = [review.decode(errors = 'ignore') for review in C]
    C = [[w.replace(ignorechars, "") for w in t ] for t in C]
    C = [[lemitaion.lemmatize(w) for w in t if w not in stopwords and len(w) >= 3] for t in C]
    C = [[lemitaion.lemmatize(w, pos='v') for w in t if w not in stopwords and len(w) >= 3] for t in C]
    C = [[w for w in t if w not in stopwords and len(w) >= 3] for t in C]
    return C

def pre_proc2(C):
    C = [w.replace(ignorechars, "") for w in C ]
    C = [lemitaion.lemmatize(w) for w in C if w not in stopwords and len(w) >= 3]
    C = [lemitaion.lemmatize(w, pos='v') for w in C if w not in stopwords and len(w) >= 3]
    C = [w for w in C if w not in stopwords and len(w) >= 3]
    return C

def load_review():
    C1 = []
    CO = []
    text = open(r'E:\python_workplace\Opinion_Mining\Data\Nokia 6610\Nokia 6610.txt', encoding='utf-8')
    f = open(r'E:\python_workplace\Opinion_Mining\Data\Nokia 6610\reviews.txt', 'w')
    # text = open(r'E:\python_workplace\Opinion_Mining\Data\Apex\Apex.txt', encoding='utf-8')
    # f = open(r'E:\python_workplace\Opinion_Mining\Data\Apex\init_reviews.txt', 'w')
    for line in text.readlines():
        if line.startswith('[t]') or line.startswith('*') or line.startswith('\r'):
            continue
        else:
            # print(line)
            C1.append(line.split("##")[1].strip())  # 取出评论句子
    text.close()
    # print C1
    words = []
    for review in C1:
        words.append(nltk.word_tokenize(review))
        f.write(review+'\n')
    for word in words:
        tagged_words = nltk.pos_tag(word)
        for tagged_word in tagged_words:
            if tagged_word[1] == "JJ" or tagged_word[1] == "VB" and len(tagged_word[0]) > 2:
                CO.append(tagged_word[0])
    # print words
    f.close()
    return pre_proc1(words),pre_proc2(CO)

def tagger(Words):
    print ("use the stanford tool")

def get_CF0():
    CF0 = []
    CF = []
    N = []
    f = open(r'E:\python_workplace\Opinion_Mining\Data\Nokia 6610\parse_result.txt', encoding='utf-8')
    for line in f.readlines():
        line = line.replace("*'", "").replace("'", "")
        # print line
        if line.startswith("result:"):
            N = []
            CF0 = []
        elif line.startswith("#nsubj") or line.startswith("#pobj") or line.startswith("#dobj")or line.startswith("#amod")or line.startswith("#nn"):
            line = re.match(r'.*\((.*)-\d*\'*,\s(.*)-\d*\'*\)$', line).groups()
            for i in [0, 1]:
                w = line[i]
                w = w.replace(ignorechars, "")
                if w in N and w not in CF0:
                    CF0.append(w)
                    CF.append(w)
        else:
            if line.__contains__("NN"):
                word = line.split("\t")[1]
                N.append(word)
    # for tagged_word in nltk.pos_tag(CF):
    #     if tagged_word[1] == "NN" or tagged_word[1] == "NNS" and len(tagged_word[0]) > 2:
    #         CF1.append(tagged_word[0])
    # print pre_proc2(CF)
#     return pre_proc2(CF)
#
def get_CF():
    CF = []
    CF_N = []
    N = []
    NN = []
    temp = []
    root = ''
    f = open(r'E:\python_workplace\Opinion_Mining\Data\Nokia 6610\parse_result.txt', encoding='utf-8')
    p = open(r'E:\python_workplace\Opinion_Mining\Data\Nokia 6610\noun_phrase.txt',encoding='utf-8')
    NP = [line.strip().split(',') for line in p.readlines()]
    print(NP)
    p.close()
    index = 0

    for line in f:
        line = line.replace("*'", "")
        if line.startswith("result:"):
            temp = []
            N = NP[index]
            index += 1
        elif line.startswith("#"):
            if line.startswith("#nsubj") or line.startswith("#pobj") or line.startswith("#dobj")or line.startswith("#amod"):
            # if re.match(r'.*\((.*)-\d*\'*,\s(.*)-\d*\'*\)$', line):
                line = re.match(r'.*\((.*)-\d*\'*,\s(.*)-\d*\'*\)$', line).groups()
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
        elif line.__contains__("NN"):
            word = line.split("\t")[1]
            NN.append(word)
            CF_N.append(word)

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

    # addition = ['at&t customer service', 'infrared', 'infrared', 'sprint plan', 'sprint customer service', 'sturdy', 'ringtone', 'background', 'screensaver', 'memory', 'menu options', 't-mobile reception', 't-zone', 't-zone', 't-mobile', 'customer rep', 'call', 'phone performance', 'look', 't-mobile', 'voice dialing', 'message', 'fm', 'operate', 'button', 'key', 'volume', 't-mobile', 'high speed internet', 'ringing tone', 'ring tone', 'game', 'button', 'size', 'size', 'key', 'vibrate setting', 'vibrate setting', 'voice dialing', 'voice dialing', 'picture', 'ringtone', 'key lock', 'ring tone', 'fm radio', 'weight', 'wallpaper', 'tune', 'size', 'size', 'key', 'pc cable', 'loud phone', 'size', 'application', 'pc suite', 'size', 'game', 'ringtone', 'ergonomics', 'size', 'size', 'volume', 'volume', 'size', 'weight', 'ringtone', 'volume', 'weight', 'pc sync', 'tone', 'wallpaper', 'application', 'message', 'picture sharing', 'mms', 'size', 'voice dialing', 'key', 'application', 'size', 'speakerphone', 'look', 'default ringtone', 't-mobile', 'ringtone', 'speakerphone', 'size', 'look', 'weight', 'browsing', 'game', 'battery life', 'voice dialing', 'command', 'button', 'key', 't-mobile', 't-mobile', 'size', 'earpiece', 'voice dialing', 'ringtone', 'gprs', 't-zone', 't-zone', 't-mobile service', 'rate plan', 'weight', 'signal']
    #
    # CF += addition
    return pre_proc2(CF)

def LL(p, k, n):  # log L
    return math.log(p ** k * (1 - p) ** (n - k))

def A(i, j, C):  # Association between i and j
    k1 = k2 = k3 = k4 = 0  # contingency table
    for review in C:
        if review.__contains__(i) and review.__contains__(j):
            k1 += 1
        elif review.__contains__(i) and review.__contains__(j) == 0:
            k2 += 1
        elif review.__contains__(i) == 0 and review.__contains__(j):
            k3 += 1
        else:
            k4 += 1
    n1 = k1 + k3
    # if (n1 == 0):
    #     print i, j
    n2 = k2 + k4
    if n1==0:
        p1 = 0
    else:
        p1 = k1 / n1
    if n2 == 0:
        p2 = 0
    else:
        p2 = k2 / n2
    p = (k1 + k2) / (n1 + n2)
    As = 2 * (LL(p1, k1, n1) + LL(p2, k2, n2) - LL(p, k1, n1) - LL(p, k2, n2))
    return As

def main():
    C0, CO0 = load_review()
    CF0 = get_CF()
    # f = open(r'E:\python_workplace\hai2012\corpus\truefeature.txt', encoding='utf-8')
    # TF = []
    # for line in f.readlines():
    #     if ' 'in line:
    #         line.replace(' ','')
    #     if ',' in line:
    #         tmp = line.split(',')
    #         for t in tmp:
    #             TF.append(t.strip())
    #     else:
    #         TF.append(line.strip())
    # CF = TF

    # 写入文件的C和CF  用来校验
    # f = open(r'E:\python_workplace\hai2012\corpus\foramc.txt', 'w')
    # f1 = open(r'E:\python_workplace\hai2012\corpus\corpus.txt', 'w')
    # f2 = open(r'E:\python_workplace\hai2012\corpus\cf.txt', 'w')
    # for review in C:
    #     for w in review:
    #         f1.write(w+'\n')
    #         f.write(w)
    #         f.write(',')
    #     f.write('\n')
    # f2.write(','.join(CF))

    print (C0)
    print (CF0)
    print (CO0)
    print ("result***********")
    thresh_list = [0,1,2,3,4,5,6,7,8,9,10]
    for thresh in thresh_list:
        print("thresh=", thresh)
        C, CO = load_review()
        CF = get_CF()
        O = []
        F = ['phone']
        ffth = thresh
        foth = thresh
        ooth = thresh
        flag = 0
        while (flag == 0):
            flag = 1
            for f in F:
                for cf in CF:
                    if A(f, cf, C) >= ffth:
                        # print(A(f, cf, C))
                        F.append(cf)
                        CF.remove(cf)
                        flag = 0
                for co in CO:
                    if A(f, co, C) >= foth:
                        O.append(co)
                        CO.remove(co)
                        flag = 0
            for o in O:
                for co in CO:
                    if A(o, co, C) >= ooth:
                        O.append(co)
                        CO.remove(co)
                        flag = 0
                for cf in CF:
                    if A(o, cf, C) >= foth:
                        F.append(cf)
                        CF.remove(cf)
                        flag = 0
        print (F)
        print (O)

        f1 = open(r'E:\python_workplace\Opinion_Mining\Data\Nokia 6610\feature_LRT.txt', 'w')
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
        f.close()
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
        precision = TP/(TP+FP)
        recall = TP/(TP + FN)
        print (TP,FP,FN)
        print ('p=%f'% precision)
        print ('r=%f'% recall)
        f=(2*precision*recall)/(precision+recall)
        print ('F=%f' % f)

if __name__ == "__main__":
    main()