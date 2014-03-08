from numpy import *
from random import *
from dmww_classes import *

################################################################
# The model
################################################################

seed(1)

w = World()
w.show()
            
c = Corpus(w)
c.sampleSents()
c.show()

print "*** coocurrence test ***"
l = CoocLexicon()
l.learnLex(w,c)
l.show()

print "*** gibbs test ***"
l = GibbsLexicon(1)
p = Params()
l.learnLex(w,c,p)
l.show()
