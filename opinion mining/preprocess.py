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
from numpy.ma import sort
import re
import numpy
from scipy import spatial
from numpy import savetxt, loadtxt
import nltk

f = open(r'E:\python_workplace\Opinion Mining (LML)\Data\English_stopwords.txt', encoding='utf-8')
stopwords = set(line.strip() for line in f.readlines())  # 读入停用词
lemitaion = nltk.WordNetLemmatizer()
f.close()
ignorechars = ''',:'.;!()#'''

def getVocabulary():
    from hai2012.AMC_preprocess import domain_preprocess
    f1 = open(r'E:\python_workplace\Opinion Mining (LML)\Data\Nokia 6610\Nokia6610.txt', 'w')
    f2 = open(r'E:\python_workplace\Opinion Mining (LML)\Data\Nokia 6610\noun_prase.txt', 'w')
    f3 = open(r'E:\python_workplace\Opinion Mining (LML)\Data\Nokia 6610\parse_result.txt', encoding='utf-8')
    C = []
    NP = []
    for line in f3:
        line = line.replace("*'", "")
        if line.startswith("result:"):
            C = []
            NP = []
        elif line.startswith("#"):
            if line.startswith("#nn"):
                line = re.match(r'.*\((.*)-\d*\'*,\s(.*)-\d*\'*\)$', line).groups()
                word = ' '.join([line[1],line[0]])
                NP.append(word)
                f2.write(word + ',')
        else:
            pass
        f2.write('\n')

    f1.close()
    f2.close()
    f3.close()
    domain_preprocess(r'E:\python_workplace\Opinion Mining (LML)\Data\Nokia 6610\Nokia6610.txt',r'E:\eclipse_workplace\AMC\Data\Input\100Reviews\Electronics')

pass