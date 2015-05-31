#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
__title__ = ''
__author__ = 'LML_CH'
__mtime__ = '2015/2/5'
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
import re
import nltk
import numpy
from numpy import zeros
from scipy.linalg import svd
from scipy import spatial
#following needed for TFIDF
from math import log
from numpy import asarray, sum
from Opinion_Mining.opinion_mining.LRTBoot import load_review,get_CF

f = open(r'E:\python_workplace\Opinion_Mining\Data\English_stopwords.txt', encoding='utf-8')
stopwords = set(line.strip() for line in f.readlines())  # 读入停用词
lemitaion = nltk.WordNetLemmatizer()
f.close()
ignorechars = ''',:';!()&#'''

C1 = []
text = open(r'E:\python_workplace\Opinion_Mining\Data\Nokia 6610\Nokia 6610.txt', encoding='utf-8')
for line in text.readlines():
    if line.startswith('[t]') or line.startswith('*') or line.startswith('\r'):
        continue  # 忽略标题
    else:
        C1.append(line.split("##")[1].strip())  # 取出评论句子
text.close()

class LSA(object):
    def __init__(self, stopwords, ignorechars):
        # self.stopwords = stopwords
        # self.ignorechars = ignorechars
        self.wdict = {}
        self.dcount = 0
    def parse(self, doc):
        words = doc.split()
        # words = re.split( "[,,.:';!()\"/\-$&*#\s+]", doc)
        # words = [w.encode('utf-8').lower().translate(None, ignorechars) for w in words]
        # print(words)
        words = [lemitaion.lemmatize(w) for w in words if w not in stopwords and len(w) >= 3]
        # print(words)  #unicode
        # words = [w.encode('utf-8') for w in words]
        # print(words)  #str
        # words = [lemitaion.lemmatize(w, pos='VERB') for w in words if w not in stopwords and len(w) >= 3]
        for w in words:
            # w = w.encode('utf-8').lower().translate(None, ignorechars)
            w = w.lower().replace(ignorechars, "")
            if w in stopwords:
                continue
            elif w in self.wdict:
                self.wdict[w].append(self.dcount)
            else:
                self.wdict[w] = [self.dcount]
        self.dcount += 1
    def build(self):
        self.keys = [k for k in self.wdict.keys() if len(self.wdict[k]) > 1]
        self.keys.sort()
        # print self.keys
        self.A = numpy.zeros([len(self.keys), self.dcount])
        # self.A = numpy.ones([len(self.keys), self.dcount])/10000000000
        for i, k in enumerate(self.keys):
            # print self.keys[5]
            for d in self.wdict[k]:
                self.A[i,d] += 1
    def calc(self):
        print(self.A)
        self.U, self.S, self.Vt = svd(self.A)
    def TFIDF(self):
        WordsPerDoc = sum(self.A, axis=0)
        DocsPerWord = sum(asarray(self.A > 0, 'i'), axis=1)
        rows, cols = self.A.shape
        for i in range(rows):
            for j in range(cols):
                self.A[i,j] = (self.A[i,j] / (WordsPerDoc[j] + 0.1)) * log(float(cols) / (DocsPerWord[i] + 1))
    def printA(self):
        print ('Here is the count matrix')
        print (self.A)
    def printSVD(self):
        print ('Here are the singular values')
        print ('Here are the first 3 columns of the U matrix')
        print (-1*self.U[:, 0:3])
        print ('Here are the first 3 rows of the Vt matrix')
        print (-1*self.Vt[0:3, :])
        print (" here is the new matrix")
        K = 300
        self.Sd = numpy.eye(K)
        for i,k in enumerate(self.S) :
            if i<K:
                self.Sd[i][i] = k
        self.newA = numpy.dot(numpy.dot(self.U[:,0:K],self.Sd),self.Vt[0:K,:])
        print (self.newA)
    def cosimilarity(self):
        return spatial.distance.squareform(spatial.distance.pdist(self.newA, metric = "cosine"))
mylsa = LSA(stopwords, ignorechars)
for t in C1:
    mylsa.parse(t)
mylsa.build()
mylsa.TFIDF()
mylsa.printA()
mylsa.calc()
mylsa.printSVD()
pairwise = mylsa.cosimilarity()
print (pairwise)

def A(x, y, C):
    if x in mylsa.keys:
        i = mylsa.keys.index(x)
    else:
        return 1
    if y in mylsa.keys:
        j = mylsa.keys.index(y)
    else :
        return 1
    return pairwise[i,j]

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
    thresh_list = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.9,0.99,1]
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
                    if A(f, cf, C) <= ffth:
                        # print(A(f, cf, C))
                        F.append(cf)
                        CF.remove(cf)
                        flag = 0
                for co in CO:
                    if A(f, co, C) <= foth:
                        O.append(co)
                        CO.remove(co)
                        flag = 0
            for o in O:
                for co in CO:
                    if A(o, co, C) <= ooth:
                        O.append(co)
                        CO.remove(co)
                        flag = 0
                for cf in CF:
                    if A(o, cf, C) <= foth:
                        F.append(cf)
                        CF.remove(cf)
                        flag = 0
        print (F)
        print (O)

        f1 = open(r'E:\python_workplace\Opinion_Mining\Data\Nokia 6610\feature_LSA.txt', 'w')
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
        if(TP+FP):
            precision = TP/(TP+FP)
            recall = TP/(TP + FN)
            print (TP,FP,FN)
            print ('p=%f'% precision)
            print ('r=%f'% recall)
            f=(2*precision*recall)/(precision+recall)
            print ('F=%f' % f)
        else:
            precision = 0


if __name__ == "__main__":
    main()

