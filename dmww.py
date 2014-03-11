from numpy import *
from random import *
from dmww_classes import *

################################################################
# The model
################################################################

seed(1)

w = World()
w.show()
            
c = Corpus(w,n_per_sent=1)
c.sample_sents()
c.show()

print "*** coocurrence test ***"
l = CoocLexicon()
l.learn_lex(c)
l.show()

print "*** gibbs test ***"
p = Params(n_samps=10,
           alpha_r=.1,
           alpha_nr=10,
           empty_intent=.0001,
           no_ref_word=.000001,
           n_hypermoves=10)
p.show()

l = GibbsLexicon(c,p,verbose=1)
l.learn_lex(c,p)

### working example of crazy class thing
#
# class Thing(object):
#
#     def __init__(self):
#         self.blarg = 2
#
#     def change(self):
#         self.blarg += 1
#
#     def make_eq(self, t):
#         self.blarg = t.blarg
#
# foo = Thing()

## this does work:
# bar = Thing()
# bar.make_eq(foo)
# bar.change()
# print foo.blarg

## this oesn't work
# baz = Thing()
# baz.change()
# foo.blarg
