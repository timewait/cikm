# !/usr/bin/env python
# encoding:utf-8
from vocab import Vocab
from analyzer import SpaceTokenAnalyzer

if __name__ == '__main__':
    sentences = []
    with open("../data/cikm_english_train_20180516.txt", 'r') as f:
        for line in f:
            s = line.split("\t")
            sentences.append(s[1])
            sentences.append(s[3])        
    vocab = Vocab(sentences, analyzer=SpaceTokenAnalyzer(stopWords=[]))
    print vocab.wordsTable()
