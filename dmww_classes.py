#python
#import sys
from numpy import *
from random import *

################################################################
# The model
################################################################

##### world class #####
# gives the basics of the world in which learning takes plae

class World:
    def __init__(self,
                 n_words=4,
                 n_objects=4):
        self.n_words = 4
        self.n_objects = 4

    def show(self):
        print "n_words = " + str(self.n_words)
        print "n_objects = " + str(self.n_objects)

##### corpus class #####
# stores corpus for learning from

class Corpus:
    def __init__(self,
                 world=World(),
                 n_per_sent=2,
                 n_sents=12):
        self.world = world
        self.n_sents = n_sents
        self.n_per_sent = n_per_sent

    def sample_sents(self):
        self.sents = list()

        for s in range(self.n_sents):
            words = sample(range(self.world.n_words), self.n_per_sent)
            objs = words
            self.sents.append([[words],[objs]])

    def show(self):
        for s in self.sents:
            print "w: " + str(s[0]) + " o:" + str(s[1])

##### lexicon class is the main classs #####
class Lexicon:
    def __init__(self,
                 world=World(),
                 corpus=Corpus()):
        self.world = world
        self.corpus = corpus

    # get coocurrence counts
    def get_coocs(self):
        self.coocs = zeros((self.world.n_words,self.world.n_objects))

        for s in self.corpus.sents:
            for w in s[0]:
                for o in s[1]:
                    self.coocs[w,o] += 1
            
            
    def show_coocs(self):
        print self.coocs
        

