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

l = Lexicon(w,c)
l.get_coocs()
l.show_coocs()
