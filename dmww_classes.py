from numpy import *
from random import *
from sampling_helper import *
from scipy.stats import gamma, beta
import sys
from corpus_helper import *

################################################################
# The model
################################################################

##### world class #####
# gives the basics of the world in which learning takes place

class World:
    def __init__(self,
                 n_words=4,
                 n_objs=4,
                 corpus=False):

        self.n_words = n_words
        self.n_objs = n_objs
        self.corpus = corpus

        if self.corpus != False:
            raw_corpus = loadtxt(corpus, delimiter=',', dtype=str)

            # create dictionary that maps word labels to numbers and object labels to numbers
            # first, get all unique words and objects
            all_words = list()
            all_objs = list()
            for i in range(0,shape(raw_corpus)[0]): 
                all_objs.extend(str.split(raw_corpus[i][0]))
                all_words.extend(str.split(raw_corpus[i][1]))

            # now, create two dictionaries
            u_objs = list(set(all_objs))
            self.n_objs = size(u_objs)
            self.objs_dict = list()
            for i in range(0,size(u_objs)): 
                self.objs_dict.append([u_objs[i], i])

            u_words = list(set(all_words))
            self.n_words = size(u_words)
            self.words_dict = list()
            for i in range(0,size(u_words)):
                self.words_dict.append([u_words[i], i])

            
    def show(self):
        print "n_objs = " + str(self.n_objs)
        print "n_words = " + str(self.n_words)


##### corpus class #####
# stores corpus for learning from

class Corpus:
    def __init__(self,
                 world=World(),
                 n_per_sent=2,
                 n_sents=12,
                 corpus=False):

        self.sents = list()
        self.world = world
        self.n_sents = n_sents
        self.n_per_sent = n_per_sent
        self.corpus = corpus
    
        # convert corpus from labels to numbers
        if self.corpus != False:
            raw_corpus = loadtxt(self.corpus, delimiter=',', dtype=str)
            self.sents = list()
            
            for s in range(shape(raw_corpus)[0]):
                sent = list()

                #add objs
                objs_labs = str.split(raw_corpus[s][0])
                objs = list()
                for i in range(size(objs_labs)):
                   objs.append(world.objs_dict[find(world.objs_dict, objs_labs[i])[0]][1])
                sent.append(array(objs))

                #add words
                word_labs = str.split(raw_corpus[s][1])
                words = list()
                for i in range(size(word_labs)):
                   words.append(world.words_dict[find(world.words_dict, word_labs[i])[0]][1])
                sent.append(array(words))

                self.sents.append(sent)

            self.n_sents = len(self.sents)

    def sample_sents(self):

        for s in range(self.n_sents):
            sent = list()
            words = array(sample(range(self.world.n_words), self.n_per_sent))
            objs = array(words)
            sent.append(words)
            sent.append(objs)

            self.sents.append(sent)

    def show(self):
        for s in self.sents:
            print "o: " + str(s[0]) + " w: " + str(s[1])


##### lexicon class is the main classs #####
class Lexicon:
    def __init__(self,world,
                 verbose=0):
        self.verbose = verbose
        self.ref = zeros((world.n_objs, world.n_words))

    # generic show method, where visualization eventually goes
    def show(self):
        print "cooccurrence matrix:"
        print self.ref
        if hasattr(self, 'non_ref'):
            print self.non_ref

    def show_top_match(self, corpus, world):
        if corpus.corpus != False:
            for o in range(world.n_objs):
                if max(self.ref[o,:]) > 0:
                    w = where(self.ref[o,:] == max(self.ref[o,:]))[0]
                    print "object: %s, word: %s" % (world.objs_dict[o][0], world.words_dict[w[0]][0])
        else:
            for o in range(world.n_objs):
                w = where(self.ref[o,:] == max(self.ref[o,:]))[0]
                print "object: %d, word: %d" % (o, w)


##### CoocLexicon is a class of lexica based on co-occurrence #####
class CoocLexicon(Lexicon):
    # get coocurrence counts
    def learn_lex(self, corpus):

        for s in corpus.sents:
            for o in s[0]:
                for w in s[1]:
                    self.ref[o, w] += 1


class Params:
    def __init__(self,
                 n_samps=100,
                 alpha_nr=.1,
                 alpha_r=.1,
                 empty_intent=.000001,
                 no_ref_word=.000001,
                 alpha_r_hp=1,
                 alpha_nr_hp=2,
                 intent_hp_a=1,
                 intent_hp_b=100,
                 n_hypermoves=5):

        # these are integers
        self.n_samps = int(n_samps)
        self.n_hypermoves = int(n_hypermoves)

        # cast these to floats to avoid weird type problems
        self.alpha_nr = float(alpha_nr)
        self.alpha_r = float(alpha_r)
        self.empty_intent = float(empty_intent)
        self.no_ref_word = float(no_ref_word)

        # hyper params
        self.alpha_r_hp = float(alpha_r_hp)
        self.alpha_nr_hp = float(alpha_nr_hp)
        self.intent_hp_a = float(intent_hp_a)
        self.intent_hp_b = float(intent_hp_b)

    # for debugging
    def show(self):
        for v in vars(self):
            print "%s: %2.4f" % (v, getattr(self, v))

    #########
    ## propose_hyperparams
    def propose_hyper_params(self, nps):

        for p in nps.__dict__.keys():
            exec ( "self.%s = nps.%s" % (p, p))

        self.alpha_nr = alter_0inf_var(nps.alpha_nr)
        self.alpha_r = alter_0inf_var(nps.alpha_r)
        self.empty_intent = alter_01_var(nps.empty_intent)
        self.no_ref_word = alter_01_var(nps.no_ref_word)

        return nps

##### GibbsLexicon is a class of lexica learned by gibbs sampling #####
class GibbsLexicon(Lexicon):
    ########
    ## initLex initializes all of the lexicon bits and pieces, which include:
    ## - random guesses for intentions
    ## - counts for lexicon based on this
    ## - score caches for all words
    def __init__(self,
                 corpus,
                 params,
                 verbose=0,
                 hyper_inf=True):


        # for debugging
        self.verbose = verbose
        self.hyper_inf = hyper_inf

        # initialize the relevant variables
        self.ref = zeros((corpus.world.n_objs, corpus.world.n_words))
        self.non_ref = zeros(corpus.world.n_words)
        self.intent_obj = zeros(corpus.n_sents, dtype=int)
        self.ref_word = zeros(corpus.n_sents, dtype=int)
        self.param_score = 0

        # choose random word and object to be talked about in each sentence
        # or consider the null topic/object (the +1)
        for i in range(corpus.n_sents):
            n_os = len(corpus.sents[i][0])
            n_ws = len(corpus.sents[i][1])

            self.intent_obj[i] = sample(range(n_os + 1), 1)[0]

            # if it's the null intention, it needs to be the null word
            if self.intent_obj[i] == n_os:
                self.ref_word[i] = n_ws  # null
            else:
                self.ref_word[i] = sample(range(n_ws + 1), 1)[0]

        # initialize cached probabilities
        self.intent_obj_probs = [None] * corpus.n_sents  # list
        self.ref_word_probs = [None] * corpus.n_sents  # list
        self.intent_obj_prob = zeros(corpus.n_sents)  # numpy array
        self.ref_word_prob = zeros(corpus.n_sents)  # numpy array
        self.ref_score = zeros(corpus.world.n_objs)

        # build object and word indices for quick indexing
        self.oi = map(lambda x: array(range(len(x[0]))), corpus.sents)
        self.wi = map(lambda x: array(range(len(x[1]))), corpus.sents)

        # now update all the scores
        self.score_full_lex(corpus, params, init=True)


    #########
    ## learnLex gets lexicon counts by gibbs sampling over the intended object/referring word
    ## the heart of this function is the loop over possible lexicons based on changing the scores
    ## this is technically a block gibbs over objects and words (indexed by j and k)
    def learn_lex(self,
                  corpus,
                  params):

        win_score = nans(params.n_samps)

        for s in range(params.n_samps):
            if self.verbose > 0:
                print "\n*************** sample %d ***************" % s
            else:
                if mod(s,80) == 0:
                    print "\n"
                else:
                    sys.stdout.write(".")

            for i in range(corpus.n_sents):
                if self.verbose > 1:
                    print "sent " + str(i) + " - " + str(corpus.sents[i]) + " :"

                n_os = len(corpus.sents[i][0])
                n_ws = len(corpus.sents[i][1])

                scores = neg_infs((n_os + 1, n_ws + 1))  # +1 for null

                for j in range(n_os + 1):  # +1 for null
                    for k in range(n_ws + 1):
                        if not (j == n_os and k < n_ws):  # condition to ensure non-ref intent gets non-ref word
                            if self.verbose > 1:
                                print "    j = " + str(j) + ", k = " + str(k)
                                print "    ref = " + str(self.intent_obj[i]) + "," + str(self.ref_word[i])
                                print self.ref
                                print self.non_ref

                            self.prep_lex(corpus, params, i)
                            scores[j, k] = self.score_lex_simple(corpus, params, i, j, k)

                            if self.verbose > 1:
                                print "r:" + str(self.ref)
                                print "nr: " + str(self.non_ref)
                                print "interim scores: %2.1f, %2.1f, %2.1f, " \
                                      "%2.1f, %2.1f, total: %2.1f" % (sum(self.ref_score),
                                                                      self.nr_score,
                                                                      sum(self.intent_obj_prob),
                                                                      sum(self.ref_word_prob),
                                                                      self.param_score,scores[j, k])

                (j, k, win_score[s]) = choose_class(scores)
                score = self.score_lex_simple(corpus, params, i, j, k)

                if self.verbose > 0:
                    print "r:" + str(self.ref)
                    print "nr: " + str(self.non_ref)
                    print "total counts: " + str(sum(sum(self.ref)) + sum(self.non_ref))
                    print "    *** scores: %2.1f, %2.1f, %2.1f, " \
                          "%2.1f, %2.1f, total: %2.1f ***" % (sum(self.ref_score),
                                                          self.nr_score,
                                                          sum(self.intent_obj_prob),
                                                          sum(self.ref_word_prob),
                                                          self.param_score,scores[j, k])

            if self.hyper_inf:
                params = self.hyper_param_inf(corpus, params, score)
                    #   [p(s) r(s) f(s)] = computeLexiconF(lex,gold_standard);


                    #########
                    ## scoreLexSimple - without any of the caching stuff
        self.params = params


    def score_lex_simple(self,
                         corpus,
                         params,
                         i, j, k):
        # reassign this j/k pair for this sentence
        new_o = corpus.sents[i][0][self.oi[i] == j]
        new_w = corpus.sents[i][1][self.wi[i] == k]

        self.intent_obj[i] = j
        self.ref_word[i] = k

        # update the probabilities
        self.intent_obj_prob[i] = self.intent_obj_probs[i][j]
        self.ref_word_prob[i] = self.ref_word_probs[i][k]

        # critical part: rescore and shift counts in ref lexicon
        self.ref[new_o, new_w] += 1
        self.non_ref[new_w] -= 1

        # score lexicon
        for o in range(corpus.world.n_objs):
            self.ref_score[o] = score_dm(self.ref[o, :], params.alpha_r)
        self.nr_score = score_dm(self.non_ref, params.alpha_nr)

        score = sum(self.intent_obj_prob) + sum(self.ref_word_prob) + self.nr_score + sum(
            self.ref_score) + self.param_score

        return score


    #########
    ## scoreFullLex - rescore everything
    ## - use this for setup in combo with initLex
    ## - important for hyperparameter inference
    def score_full_lex(self,
                       corpus,
                       params,
                       init=False):
        # set up the intent caching
        for i in range(corpus.n_sents):
            n_os = len(corpus.sents[i][0])
            n_ws = len(corpus.sents[i][1])

            # cache word and object probabilities uniformly
            # 1 x o matrix with [uniform ... empty]
            # and 1 x w matrix again with [uniform ... empty]
            unif_o = log((1 - params.empty_intent) / n_os)
            unif_w = log((1 - params.no_ref_word) / n_ws)

            self.intent_obj_probs[i] = [unif_o] * n_os + [log(params.empty_intent)]
            self.ref_word_probs[i] = [unif_w] * n_ws + [log(params.no_ref_word)]

            if init:
                # update lexicon dirichlets based on random init
                self.ref[corpus.sents[i][0][self.oi[i] == self.intent_obj[i]],
                         corpus.sents[i][1][self.wi[i] == self.ref_word[i]]] += 1

                # includes all words that are not the referential word
                self.non_ref[corpus.sents[i][1][self.wi[i] != self.ref_word[i]]] += 1

            # now set up the quick scoring probability caches
            self.intent_obj_prob[i] = self.intent_obj_probs[i][self.intent_obj[i]]
            self.ref_word_prob[i] = self.ref_word_probs[i][self.ref_word[i]]

        # cache DM scores for lexicon
        for i in range(corpus.world.n_objs):
            self.ref_score[i] = score_dm(self.ref[i, :], params.alpha_r)

        # cache non-ref DM score also
        self.nr_score = score_dm(self.non_ref, params.alpha_nr)

        # score hyperparameters (via hyper-hyperparameters)
        empty_intent_score = beta.logpdf(params.empty_intent, params.intent_hp_a, params.intent_hp_b)
        no_ref_word_score = beta.logpdf(params.no_ref_word, params.intent_hp_a, params.intent_hp_b)
        alpha_score = gamma.logpdf(params.alpha_r, params.alpha_r_hp) + gamma.logpdf(params.alpha_nr,
                                                                                     params.alpha_nr_hp)
        self.param_score = empty_intent_score + no_ref_word_score + alpha_score

        # debugging stuff
        if self.verbose > 1:
            print "-- score full lex"
            print "    intent obj: " + str(self.intent_obj)
            print "    intent obj prob: " + str(self.intent_obj_prob.round(1))
            print "    ref word: " + str(self.ref_word)
            print "    ref word prob: " + str(self.ref_word_prob.round(1))
            print "lex: " + str(self.ref)
            print "nr lex: " + str(self.non_ref)
            print "    ref score: " + str(self.ref_score)
            print "    nref score: " + str(self.nr_score)
            print "params: " + str(self.param_score)
            print "    empty: " + str(empty_intent_score)
            print "    no_ref: " + str(no_ref_word_score)
            print "    alpha: " + str(alpha_score)

        score = sum(self.intent_obj_prob) + sum(self.ref_word_prob)
        score += self.nr_score + sum(self.ref_score)
        score += self.param_score

        return score


    #########
    ## prepLex subtracts out the current counts for this particular referential word and referred object, so that this can be done once and then counts can be added for each pairing quickly and independently via the gibbs loop. (It's just factoring out a step that would have to be done by each iteration of the block gibbs).
    def prep_lex(self, corpus, params, i):
        # cache old object and word
        old_o = corpus.sents[i][0][self.oi[i] == self.intent_obj[i]]
        old_w = corpus.sents[i][1][self.wi[i] == self.ref_word[i]]

        # now subtract their counts from the referential lexicon,
        # but only if there was a referred object
        if old_o.size > 0:
            self.ref[old_o, old_w] -= 1
            self.ref_score[old_o] = update_dm_minus(self.ref_score[old_o],
                                                    self.ref[old_o, :][0],
                                                    params.alpha_r,
                                                    old_w)

        # and add back to the non-referential lexicon,
        # again only if there's a referring word
        if old_w.size > 0:
            self.non_ref[old_w] += 1
            self.nr_score = update_dm_plus(self.nr_score,
                                           self.non_ref,
                                           params.alpha_nr,
                                           old_w)

        if self.verbose > 1:
            print "-- prep lex"
            print "        old = " + str(old_o) + " " + str(old_w) + " "


    #########
    ## hyperParamInf implements hyperparameter inference
    def hyper_param_inf(self,
                        corpus,
                        params,
                        score):
        if self.verbose > 0:
            print "\n****** HP INFERENCE *******"

        for i in range(params.n_hypermoves):
            if self.verbose > 1:
                print "\n--- current params ---"
                params.show()
                print "hyper param score:" + str(score)

            new_params = Params()
            new_params.propose_hyper_params(params)
            new_score = self.score_full_lex(corpus, new_params)
            # print "* scoring"
            # params.show()

            if self.verbose > 1:
                print "--- new params ---"
                new_params.show()
                print "hyper param score:" + str(new_score)

            if random() < exp(new_score - score):
                params = new_params

                if self.verbose > 0:
                    print "    hp change! - old = %2.2f, new = %2.2f" % (score, new_score)

        # now rescore with the new parameters - redundant if you didn't swap, FIXME
        self.score_full_lex(corpus, params)

        return params


    # #########
    # ## scoreLex
    # def score_lex(self,
    #              corpus,
    #              params,
    #              i, j, k):
    #
    #     # reassign this j/k pair for this sentence
    #     new_o = corpus.sents[i][0][self.oi[i] == j]
    #     new_w = corpus.sents[i][1][self.wi[i] == k]
    #
    #     self.intent_obj[i] = j
    #     self.ref_word[i] = k
    #
    #     # update the probabilities
    #     self.intent_obj_prob[i] = self.intent_obj_probs[i][j]
    #     self.ref_word_prob[i] = self.ref_word_probs[i][k]
    #
    #     # critical part: rescore and shift counts in ref lexicon
    #     self.ref[new_o, new_w] += 1
    #     if new_o.size > 0 and new_w.size > 0:
    #         self.ref_score[new_o] = updateDMplus(self.ref_score[new_o], self.ref[new_o, :][0],
    #                                              params.alpha_r, new_w)
    #
    #     if self.verbose:
    #         print "-- score lex"
    #         print "    new o: " + str(new_o) + " , new w: " + str(new_w)
    #
    #     # and non-ref lexicon
    #     self.non_ref[new_w] -= 1
    #     if new_w.size > 0:
    #         self.nr_score = updateDMminus(self.nr_score, self.non_ref,
    #                                       params.alpha_nr, new_w)
    #
    #     score = sum(self.intent_obj_prob) + sum(self.ref_word_prob) + self.nr_score + sum(self.ref_score)
    #
    #     return score
