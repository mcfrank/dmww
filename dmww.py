from numpy import *
from random import *
from dmww_classes import *

################################################################
# The model
################################################################

seed(1)

w = World()
w.show()
            
c = Corpus(w,n_per_sent=2)
c.sample_sents()
c.show()

print "*** coocurrence test ***"
l = CoocLexicon()
l.learn_lex(c)
l.show()

print "*** gibbs test ***"
p = Params(n_samps=100,
           alpha_r=.1,
           alpha_nr=10,
           empty_intent=.0001,
           no_ref_word=.000001,
           n_hypermoves=10)
p.show()


l = GibbsLexicon(c,p,verbose=1,hyper_inf=True)
l.learn_lex(c,p)