# encoding: utf-8
import numpy as np
import pandas as pd
from analyzer import Analyzer, SpaceTokenAnalyzer

class Vocab(object):
    
    def __init__(self, sentences, analyzer=SpaceTokenAnalyzer()):
        if not isinstance(analyzer, Analyzer):
            raise ValueError("Need Analyzer type.")
        self.analyzer = analyzer
        
        if not isinstance(sentences, list):
            raise VaalueError("Need list parameter.")
        vocab_dict = {}
        for i, sentence in enumerate(sentences):
            vocab_dict[sentence] = dict((token, 1) for token in self.analyzer.tokenize(sentence))
        self.table = pd.DataFrame.from_records(vocab_dict).fillna(0).astype(int).T
        self.table.insert(0, "UNK", 0)
        self.table.insert(0, "EOS", 0)
        self.table.insert(0, "BOS", 0)
        self.table.insert(0, "PAD", 0)
    
    def size(self):
        return len(self.table.columns)
    
    def matrix(self):
        """
        获取句子向量表
        """
        return self.table.as_matrix()
    
    def __vector(self, sentence):
        if sentence in self.table.index:
            return self.table.loc[sentence,].as_matrix()
        else:
            record = {sentence: dict((token, 1) for token in self.analyzer.tokenize(sentence))}
            self.table = self.table.append(pd.DataFrame.from_records(record).T).fillna(0).astype(int)
            return self.table.loc[sentence,].as_matrix()

    def words(self):
        return self.table.columns.values
    
    def wordsTable(self):
        return "\n".join(["%s\t%s" % (w, i) for i, w in enumerate(self.words())])
    
    def loads(self):
        raise NotImplementedError()
        
    def vector(self, sentence):
        """
        返回传入句子对应的向量
        """
        if isinstance(sentence, str):
            return self.__vector(sentence)
        
        if isinstance(sentence, list):
            return np.array([self.__vector(s) for s in sentence])
    
    def __sentence(self, vector):
        return self.table.columns[vector].values
    
    def sentence(self, vector):
        """
        返回给定向量对应的单词
        """
        if not isinstance(vector, np.ndarray):
            raise ValueError("Need numpy.ndarray parameter.")
        if len(vector.shape) == 1:
            return self.__sentence(vector)
        
        if len(vector.shape) == 2:
            return np.array([self.__sentence(v) for v in vector])
        raise ValueError("vector's shape must be 1 or 2.")

    
