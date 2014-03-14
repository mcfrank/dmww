from numpy import *
from random import *
from dmww_classes import *

################################################################
# The model
################################################################

seed(1)

w = World(n_words=8,
           n_objs=8)
w.show()
            
c = Corpus(w,
           n_per_sent=3,
           n_sents=100)
c.sample_sents()
c.show()

print "*** coocurrence test ***"
l = CoocLexicon(w)
l.learn_lex(c)
l.show()

print "*** gibbs test ***"
p = Params(n_samps=500,
           alpha_r=.1,
           alpha_nr=10,
           empty_intent=.0001,
           no_ref_word=.000001,
           n_hypermoves=5)
p.show()


l = GibbsLexicon(c,p,
                 verbose=0,
                 hyper_inf=True)
l.learn_lex(c,p)
l.show()