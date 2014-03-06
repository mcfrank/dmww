from numpy import *
from random import *
from dmww_classes import *

################################################################
# The model
################################################################
     
w = World()
w.show()
            
c = Corpus(w)
c.sample_sents()
c.show()

l = CoocLexicon()
l.get_lex(w,c)
l.show()
