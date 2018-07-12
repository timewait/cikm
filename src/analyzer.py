# encoding:utf-8
import abc
import re
class Analyzer(object):
    __metaclass__  = abc.ABCMeta
    
    @abc.abstractmethod
    def tokenize(self, sentence):
        pass


class SpaceTokenAnalyzer(Analyzer):
    def __init__(self, stopWords=[]):
        self.stopWords = stopWords
        
    def tokenize(self, sentence):
        if not isinstance(sentence, str):
            raise Value("Need string parameter.")  
        tokens = sentence.split()
        return [ t.strip() for t in tokens if t.strip() not in self.stopWords ]
