from numpy import *
from random import *
from sampling_helper import *
################################################################
# The model
################################################################

##### world class #####
# gives the basics of the world in which learning takes plae

class World:
    def __init__(self,
                 n_words=4,
                 n_objs=4):
        self.n_words = 4
        self.n_objs = 4

    def show(self):
        print "n_words = " + str(self.n_words)
        print "n_objs = " + str(self.n_objs)

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
            sent = list()
            words = sample(range(self.world.n_words), self.n_per_sent)
            objs = words
            sent.append(words)
            sent.append(objs)
            
            self.sents.append(sent)

    def show(self):
        for s in self.sents:
            print "w: " + str(s[0]) + " o: " + str(s[1])

##### lexicon class is the main classs #####
class Lexicon:
    def __init__(self):
        self.lex = None

    # generic show method, where visualization eventually goes
    def show(self):
        print "cooccurrence matrix:"
        print self.lex

##### CoocLexicon is a class of lexica based on co-occurrence #####
class CoocLexicon(Lexicon):
    
    # get coocurrence counts
    def get_lex(self, world, corpus):
        self.lex = zeros((world.n_objs,world.n_words))

        for s in corpus.sents:
            for w in s[0]:
                for o in s[1]:
                    self.lex[o,w] += 1

class Params:
    def __init__(self,
                n_samps = 100,
                alpha_nr = .1,
                alpha_r = .1,
                empty_intent = .001,
                no_ref_word = .001):
        self.n_samps = n_samps
        self.alpha_nr = alpha_nr
        self.alpha_r = alpha_r
        self.empty_intent = empty_intent
        self.no_ref_word = no_ref_word
               
##### CoocLexicon is a class of lexica learned by gibbs sampling #####
class GibbsLexicon(Lexicon):

    #########
    ## get_lex gets lexicon counts by gibbs sampling over the intended object/referring word
    ## the heart of this function is the loop over possible lexicons based on changing the scores
    ## this is technically a block gibbs over objects and words (indexed by j and k)
    def get_lex(self,
                world,
                corpus,
                params):

        win_score = array()
        lexicon = self.init_lex(corpus, world, params)
        
        for i in range(n_samps):
            for s in corpus.sents:
                prep_lex = prepareLexicon(corpus, lex, s)

                for j in range(len(s[0])+1):
                    for k in range(len(s[1])+1):
                        (scores[j,k], lexs[j][k]) = scoreLexicon(corpus, prep_lex, i, j, k)

                (j, k, win_score[s]) = chooseClass(scores)
                lex = lexs[j][k]
        #   [p(s) r(s) f(s)] = computeLexiconF(lex,gold_standard);


    #########
    ## init_lex initializes all of the lexicon bits and pieces, which include:
    ## - random guesses for intentions
    ## - counts for lexicon based on this
    ## - score caches for all words
    def init_lex(self,
                 corpus,
                 world,
                 params):

        # initialize the relevant variables
        self.non_ref = zeros((world.n_objs, world.n_words))
        self.ref = zeros((world.n_words))
        self.intent_obj = zeros(world.n_sents)
        self.ref_word = zeros(world.n_sents)

        # choose random word and object to be talked about in each sentence
        # or consider the null topic/object (the +1)
        for i in range(corpus.n_sents):
            self.intent_obj[i] = sample(range(len(corpus.sents[i][0])+1))
            self.ref_word[i] = sample(range(len(corpus.sents[i][1])+1))

        # initialize cached probabilities
        self.intent_obj_probs = nans((len(corpus.sents),
                                       max(map(lambda x: len(x[0]),
                                               corpus.sents))))
        self.ref_word_probs = nans((len(corpus.sents),
                                     max(map(lambda x: len(x[0]),
                                             corpus.sents))))

        # now update cache 
        for i in range(corpus.n_sents):
            o = len(corpus.sents[i][0])
            w = len(corpus.sents[i][1])
            self.oi[i] = range(o)
            self.wi[i] = range(w)

            # and cache word and object probabilities uniformly
            # 1 x o matrix with [uniform ... empty]
            # and 1 x w matrix again with [uniform ... empty]
            unif_o = log((1 - params.empty_intent)/o)
            unif_w = log((1 - params.no_ref_word)/w)
            self.intent_obj_probs[i] = [unif_o]*o + params.empty_intent
            self.ref_word_probs[i] = [unif_w]*o + params_no_ref_word

            # update lexicon dirichlets based on random init
            self.ref(corpus.sents[i][0][self.oi[i] == self.intent_obj[i]],
                    corpus.sents[i][1][self.wi[i] == self.ref_word[i]]) += 1
            self.non_ref(corpus.sents[i][1][self.wi[i] != self.ref_word[i]]) += 1

            self.intent_obj_prob[i] = self.intent_obj_probs[i](self.intent_obj[i])
            self.ref_worD_prob[i] = self.ref_word_prob[i](self.ref_word[i])

        # cache DM scores for lexicon
        for i in range(world.n_objs):
            self.ref_score[i] = scoreDM(self.ref[i,:], params.alpha_r)

        # cache non-ref DM score also
        self.nr_score = scoreDM(self.non_ref, lex.alpha_nr)

    #########
    ## prepare_lex subtracts out the current counts for this particular referential word and referred object, so that this can be done once and then counts can be added for each pairing quickly and independently via the gibbs loop. (It's just factoring out a step that would have to be done by each iteration of the block gibbs). 
    def prepare_lex(self,
                    corpus,
                    lex,
                    i):
        # cache old object and word
        old_o = corpus.sents[i][0][self.oi[i] == self.intent_obj[i]]
        old_w = corpus.sents[i][1][self.wi[i] == self.ref_word[i]]

        # now subtract their counts from the referential lexicon, but only if there was a referred object
        if old_o 
            self.ref[old_o, old_w] -= 1
            self.ref_score[old_o] = updateDMminus(self.ref_score[old_o],
                                                  self.ref[old_o,:],
                                                  params.alpha_r,
                                                  old_w)
            
        # and add back to the non-referential lexicon, again only if there's a referring word
        if old_w
            self.non_ref[old_w] += 1
            self.nr_score = updateDMplus(self.nr_score,
                                         self.non_ref,
                                         params.alpha_nr,
                                         old_w)
