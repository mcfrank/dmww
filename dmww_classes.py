import numpy as np
from sampling_helper import *
from scipy.stats import gamma, beta
from scipy.misc import logsumexp
import sys as sys
from corpus_helper import *
import time
import copy
from random import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pylab

# todo:
# - fix intent hyperparameter inference
#   - figured out that empty intent hp inference messes everything up
#     model converges to never talking about anything
# - fix importance weights for particle filter

# done
# x consider removing word failure process

################################################################
# The model
################################################################

#################################################################
##### world class #####
## gives the basics of the world in which learning takes place
class World:
    def __init__(self,
                 n_words=4,
                 n_objs=4,
                 corpus=False):

        self.n_words = n_words
        self.n_objs = n_objs
        self.corpus = corpus

        if self.corpus:
            raw_corpus = loadtxt(corpus, delimiter=',', dtype=str)

            # create dictionary that maps word labels to numbers and object labels to numbers
            # first, get all unique words and objects
            all_words = list()
            all_objs = list()
            for i in range(0, shape(raw_corpus)[0]):
                all_objs.extend(str.split(raw_corpus[i][0]))
                all_words.extend(str.split(raw_corpus[i][1]))

            # now, create two dictionaries
            u_objs = list(set(all_objs))
            self.n_objs = size(u_objs)
            self.objs_dict = list()
            for i in range(0, size(u_objs)):
                self.objs_dict.append([u_objs[i], i])

            u_words = list(set(all_words))
            self.n_words = size(u_words)
            self.words_dict = list()
            for i in range(0, size(u_words)):
                self.words_dict.append([u_words[i], i])

    def show(self):
        print "n_objs = " + str(self.n_objs)
        print "n_words = " + str(self.n_words)


#################################################################
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
        self.gs = list()  # gold standard

        # convert corpus and gold standard from labels to numbers
        if self.corpus:
            # read in corpus
            raw_corpus = loadtxt(self.corpus, delimiter=',', dtype=str)
            self.sents = list()

            for s in range(shape(raw_corpus)[0]):
                sent = list()

                #add objs
                objs_labs = str.split(raw_corpus[s][0])
                objs = list()
                for i in range(size(objs_labs)):
                    objs.append(world.objs_dict[find(world.objs_dict, objs_labs[i])[0]][1])
                sent.append(np.array(objs))

                #add words
                word_labs = str.split(raw_corpus[s][1])
                words = list()
                for i in range(size(word_labs)):
                    words.append(world.words_dict[find(world.words_dict, word_labs[i])[0]][1])
                sent.append(np.array(words))

                self.sents.append(sent)

            self.n_sents = len(self.sents)

            # read in gold standard and convert with dict using world associated with corpus
            gs_raw = loadtxt('corpora/gold_standard.csv', delimiter=',', dtype=str)
            self.gs = list()

            for s in range(shape(gs_raw)[0]):
                sent = list()

                #add objs
                dict_io = find(world.objs_dict, gs_raw[s][0])
                if dict_io != -1:  #only include if in corpus
                    objs = world.objs_dict[dict_io[0]][1]
                    sent.append(np.array(objs))

                #add words
                dict_iw = find(world.words_dict, gs_raw[s][1])
                if dict_iw != -1:  #only include if in corpus
                    words = world.words_dict[dict_iw[0]][1]
                    sent.append(np.array(words))

                if ((dict_io != -1) & (dict_iw != -1)):
                    self.gs.append(sent)

        else:
            self.sample_sents()

        # keep track of number of words and objects in each sentence
        self.n_os = map(lambda x: len(x[0]), self.sents)
        self.n_ws = map(lambda x: len(x[1]), self.sents)


    ######
    ## sample_sents - for making artificial corpora
    ## note that the w/o mapping is always that the matching one is the same number
    def sample_sents(self):

        for s in range(self.n_sents):
            sent = list()
            words = np.array(sample(range(self.world.n_words), self.n_per_sent))
            objs = np.array(words)
            sent.append(words)
            sent.append(objs)

            self.sents.append(sent)

    def show(self):
        for s in self.sents:
            print "o: " + str(s[0]) + " w: " + str(s[1])

    def rep(self, n):
        self.n_sents *= n
        self.sents *= n
        self.n_os = map(lambda x: len(x[0]), self.sents)
        self.n_ws = map(lambda x: len(x[1]), self.sents)




#################################################################
##### Params class is the parameter set for the model
class Params:
    def __init__(self,
                 n_samps=100,
                 n_particles=1,
                 alpha_nr=.1,
                 alpha_r=.1,
                 empty_intent=.000001,
                 alpha_r_hp=1,
                 alpha_nr_hp=2,
                 intent_hp_a=1,
                 intent_hp_b=5,
                 n_hypermoves=5):

        # these are integers
        self.n_samps = int(n_samps)
        self.n_particles = int(n_particles)
        self.n_hypermoves = int(n_hypermoves)

        # cast these to floats to avoid weird type problems
        self.alpha_nr = float(alpha_nr)
        self.alpha_r = float(alpha_r)
        self.empty_intent = float(empty_intent)

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

        # choose between the three kinds of moves
        r = random()
        if r < .5:
            self.alpha_nr = alter_0inf_var(nps.alpha_nr)
        elif r < .5:
            self.alpha_r = alter_0inf_var(nps.alpha_r)
        # else:
            # self.empty_intent = alter_01_var(nps.empty_intent)

        return nps


#################################################################
##### Lexicon class is the main class for the model #####
class Lexicon:
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

            # settings
            self.hyper_inf = hyper_inf
            self.params = params
            self.inference_method = None

            # initialize the relevant variables
            self.ref = zeros((corpus.world.n_objs, corpus.world.n_words))
            self.non_ref = zeros(corpus.world.n_words)
            self.intent_obj = zeros(corpus.n_sents, dtype=int)
            self.ref_word = zeros(corpus.n_sents, dtype=int)
            self.param_score = 0.0

            # initialize cached probabilities
            self.intent_obj_probs = [None] * corpus.n_sents  # list
            self.intent_obj_prob = zeros(corpus.n_sents)  # numpy array
            self.ref_score = zeros(corpus.world.n_objs)
            self.nr_score = 0.0

            # build object and word indices for quick indexing
            self.oi = map(lambda x: np.array(range(len(x[0]))), corpus.sents)
            self.wi = map(lambda x: np.array(range(len(x[1]))), corpus.sents)

            # for gibbs, otherwise unused
            self.sample_scores = [None] * params.n_samps

            # for pf, otherwise unused
            self.particles = []

        #########
        ## learn_lex_cooc: get coocurrence counts
        def learn_lex_cooc(self, corpus):

            for s in corpus.sents:
                for o in s[0]:
                    for w in s[1]:
                        self.ref[o, w] += 1

        #########
        ## init_gibbs - randomly assigns and scores
        def init_gibbs(self, corpus, params):
            # choose random word and object to be talked about in each sentence
            # or consider the null object (the +1)
            for i in range(corpus.n_sents):
                n_os = len(corpus.sents[i][0])
                n_ws = len(corpus.sents[i][1])

                self.intent_obj[i] = sample(range(n_os + 1), 1)[0] # +1 for null
                self.ref_word[i] = sample(range(n_ws), 1)[0]

            # now update all the scores
            self.score_full_lex(corpus, params, init=True)

        #########
        ## plot_scores - plots scores
        def plot_scores(self):
            fig, ax = plt.subplots()
            ax.plot(np.arange(len(self.sample_scores)), self.sample_scores, '-')
            plt.xlabel('sample')
            plt.ylabel('sample score')
            plt.title('Lexicon score over time')
            return ax

        #########
        ## plot_lex plots lexicon nicely
        def plot_lex(self, world, colormap="Reds", certainwords=1):
            # certainwords plots all only non-zero probabilities

            # get data to plot
            if certainwords:
                hiwords = np.empty([0, 1], dtype=int)
                for word in range(world.n_words):
                    if np.count_nonzero(self.ref[:, word]):
                        hiwords = np.append(hiwords, word)
                self.ref_plot = self.ref[:, hiwords]
            else:
                self.ref_plot = self.ref
                hiwords = range(0, world.n_words)

            if hasattr(self, 'non_ref'):
                self.non_ref_plot = self.non_ref[:, hiwords]

                # get labels for plot, and sort data alphabetically
            if hasattr(world, 'words_dict'):

                #sort words
                wordlabs = [world.words_dict[hiwords[i]][0] for i in range(0, np.shape(hiwords)[0])]
                w_order = np.argsort(wordlabs)
                self.ref_plot = self.ref_plot[:, w_order]
                if hasattr(self, 'non_ref'):
                    self.non_ref_plot = self.non_ref_plot[:, w_order]
                wordlabs.sort()

                #sort objs
                objlabs = [world.objs_dict[i][0] for i in range(0, world.n_objs)]
                o_order = np.argsort(objlabs)
                self.ref_plot = self.ref_plot[o_order,]
                objlabs.sort()

            # set up plot
            if world.n_words < 30:
                fig = plt.figure(figsize=(10, 5))
            else:
                fig = plt.figure(figsize=(.5 * np.shape(hiwords)[0], .9 * world.n_objs))

            gs = gridspec.GridSpec(2, 1, height_ratios=[9, 1])
            fontsize = 1.5 * world.n_objs

            # plot referential lexicon
            ax1 = fig.add_subplot(gs[0])
            ax1.pcolormesh(self.ref_plot, cmap=colormap)

            #add word and obj ticks
            if hasattr(world, 'words_dict'):

                #word
                pylab.xticks(np.arange(np.shape(hiwords)[0]) + .5, wordlabs)
                plt.setp(plt.xticks()[1], rotation=90, fontsize=fontsize)

                #objss
                pylab.yticks(np.arange(world.n_objs) + .5, objlabs)
                plt.setp(plt.yticks()[1], fontsize=fontsize)

            else:
                ax1.set_xticks(np.arange(world.n_words) + .5)
                ax1.set_xticklabels(np.arange(world.n_words), fontsize=fontsize)
                ax1.set_yticks(np.arange(world.n_objs) + .5)
                ax1.set_yticklabels(np.arange(world.n_objs), fontsize=fontsize)

            ax1.set_ylabel("objects", fontsize=fontsize + 5)
            ax1.set_title('main lexicon', fontsize=fontsize + 10)

            # plot non-referential lexicon
            if hasattr(self, 'non_ref'):
                ax2 = fig.add_subplot(gs[1])
                ax2.pcolormesh(np.array([self.non_ref_plot]), cmap=colormap)

                #add words ticks
                if hasattr(world, 'words_dict'):
                    plt.setp(plt.xticks()[1], rotation=90, fontsize=fontsize)
                    pylab.xticks(np.arange(np.shape(hiwords)[0]) + .5, wordlabs)
                else:
                    ax2.set_xticks(np.arange(world.n_words) + .5)
                    ax2.set_xticklabels(np.arange(world.n_words), fontsize=fontsize)

                ax2.set_xlabel("words", fontsize=fontsize + 5)

                #add obj ticks
                ax2.set_yticks([])

                ax2.set_title('non-referential lexicon', fontsize=fontsize + 10)
                plt.tight_layout(pad=2)

            else:
                ax1.set_xlabel("words", fontsize=fontsize + 5)

        #########
        ##  get_f scores the lexicon based on some threshold
        def get_f(self, corpus, threshold, lex_eval="ref"):
            gs = squeeze(asarray(corpus.gs))
            gs_num_mappings = shape(gs)[0]

            if lex_eval == "ref":
                lex = self.ref
            else:
                lex = self.non_ref

            #threshold lexicon by normalizing across words (each row)
            row_sums = lex.sum(axis=1)
            lex = np.divide(lex, row_sums[:, newaxis])

            links = where(lex > threshold)
            obj_is = links[0]
            word_is = links[1]
            lex_num_mappings = size(obj_is)

            # compute precision, what portion of the target lex is composed of gold pairings
            p_count = 0
            for pair in range(lex_num_mappings):
                this_obj = obj_is[pair]
                this_word = word_is[pair]
                #loop over gold standard
                if size(where((gs[:, 0] == this_obj) & (gs[:, 1] == this_word))) > 0:
                    p_count = p_count
                    1

            if (lex_num_mappings == 0):  #special case
                precision = 0
            else:
                precision = float(p_count) / float(lex_num_mappings)

            # compute recall, how many of the total gold pairings are in the target lex
            recall = float(p_count) / float(gs_num_mappings)

            # now F is just the harmonic mean
            try:
                f = stats.hmean([precision, recall])
            except ValueError:
                print "Recall or precision 0, could not compute f score."
            else:
                print "precision: %2.2f " % precision
                print "recall: %2.2f" % recall
                print "f: %2.2f" % f

                #return (precision, recall, f)

        #########
        ## learnLex gets lexicon counts by gibbs sampling over the intended object/referring word
        ## the heart of this function is the loop over possible lexicons based on changing the scores
        ## this is technically a block gibbs over objects and words (indexed by j and k)
        def learn_lex_gibbs(self,
                      corpus,
                      params):

            # initialize for gibbs
            self.inference_method = "gibbs"
            self.init_gibbs(corpus, params)

            lexs = nans([corpus.world.n_objs, corpus.world.n_words, params.n_samps])
            start_time = time.clock()

            for s in range(params.n_samps):
                self.tick(s)  # keep track of samples

                for i in range(corpus.n_sents):
                    if self.verbose > 1:
                        print "\n********* sent " + str(i) + " - " + str(corpus.sents[i]) + " :"

                    # the important steps: prepare the lexicon for trying stuff in this setup
                    scores = neg_infs((corpus.n_os[i] + 1, corpus.n_ws[i]))  # +1 for null

                    for j in range(corpus.n_os[i] + 1):  # +1 for null
                        for k in range(corpus.n_ws[i]):
                            scores[j, k] = self.score_lex(corpus, params, i, j, k, self.verbose)

                    # now choose the class and reassign
                    (j, k, self.sample_scores[s]) = self.choose_class(scores)
                    self.score_lex(corpus, params, i, j, k, 0)
                    lexs[:, :, s] = copy.deepcopy(self.ref)

                if self.hyper_inf:
                    params = self.hyper_param_inf(corpus, params, self.sample_scores[s])
                    self.params = params

            # self.posterior_lex = self.get_posterior_lex(lexs)
            #   [p(s) r(s) f(s)] = computeLexiconF(lex,gold_standard);
            # print "\n"
            # self.show()
            # self.params.show()
            self.verbose = 2
            self.score_full_lex(corpus, params, init=False)
            print "\n *** average sample time: %2.3f sec" % ((time.clock() - start_time) / params.n_samps)


        #########
        ## learn_lex_pf implements a particle filter
        ## similar to the gibbs
        def learn_lex_pf(self,
                         corpus,
                         params,
                         resample):

            # start_time = time.clock()
            self.inference_method = "pf"

            for p in range(params.n_particles):
                self.particles.append(Particle(self, corpus, params))

            self.weights = log(np.divide([1.0],self.params.n_particles)) * self.params.n_particles
            # self.reweight_particles(0)

            for i in range(corpus.n_sents):
                self.tick(i)  # keep track of samples

                if self.verbose > 1:
                    print "\n********* sent " + str(i) + " - " + str(corpus.sents[i]) + " :"

                # the important steps: prepare the lexicon for trying stuff in this setup
                for p in self.particles:
                    p.prep_sent(corpus, params, i)
                    scores = neg_infs((corpus.n_os[i] + 1, corpus.n_ws[i]))  # +1 for null

                    for j in range(corpus.n_os[i] + 1):  # +1 for null
                        for k in range(corpus.n_ws[i]):
                            scores[j, k] = p.score_lex(corpus, params, i, j, k, self.verbose)

                    # now choose the class and reassign
                    (j, k, p.sample_scores[i]) = p.choose_class(scores)
                    p.score_lex(corpus, params, i, j, k, 0)

                # self.reweight_particles(i)

                # # resample when number of effective particles falls below N/2
                # if self.n_eff < self.params.n_particles / 2 and params.resample==True:
                #     self.systematic_resample_particles()

                # if self.hyper_inf:
                #     params = self.hyper_param_inf(corpus, params, self.sample_scores[s])
                #     self.params = params


        #########
        ## output_lex_pf does the evaluation
        def output_lex_pf(self,
                         corpus,
                         params):

            refs = np.zeros((params.n_particles, corpus.world.n_objs, corpus.world.n_words))
            for i, p in enumerate(self.particles):
                refs[i] = p.ref

            best = np.where(self.weights==max(self.weights))[0][0]

            print "\n**** BEST PARTICLE ****"
            self.particles[best].verbose = 2
            self.particles[best].score_full_lex(corpus, params, init=False)

            print "\n**** GRAND MEAN ****"
            self.ref = np.around(refs.mean(axis=0), decimals=2)
            print self.ref

            # TODO average nonref lexicons and populate
            # separate averaging and printing


        #########
        ## reweight_particles - do importance weights for all particles
        def reweight_particles(self, s):

            # get weights out
            for i in range(self.params.n_particles):
                # now get transition score for this move
                trans_score = self.particles[i].sample_scores[s] - self.particles[i].sample_scores[s-1]
                self.weights[i] = self.weights[i] + trans_score

            # normalize
            self.weights = self.weights - logsumexp(self.weights)

            self.n_eff = 1 / (sum(np.square(exp(self.weights))))

            if self.verbose >= 1:
                print "\n****** PARTICLE REWEIGHTING *******"
                print str(s) + ": particle weights:" + str(np.around(exp(self.weights),decimals=2))
                print "n: " + str(self.params.n_particles) + ", effective n: " + str(self.n_eff)

        #########
        ## do systematic resampling on particles
        ## from http://www.cs.ubc.ca/~arnaud/doucet_johansen_tutorialPF.pdf
        def systematic_resample_particles(self):

            # first generate random numbers, evenly spaced
            N = float(self.params.n_particles)
            u0 = random()/N
            u = np.linspace(u0, u0 + ((N-1)/N), N)
            cum_weights = np.cumsum(exp(self.weights))

            for i in range(self.params.n_particles):
                new_particle = np.where(u[i] < cum_weights)[0][0]
                # print new_particle
                self.particles[i].clone(self.particles[new_particle])

            self.weights = log(np.divide([1.0] * self.params.n_particles,N))

            if self.verbose >= 1:
                print "\n****** SYSTEMATIC RESAMPLING *******"
                print cum_weights
                print self.weights



        #########
        ## score_lex - a more cached version of the scoring functions
        def score_lex(self,
                      corpus,
                      params,
                      i, j, k,
                      verbose):

            # old object and word
            # we do this with indices so that if you give a bad index,
            # you just get an empty object/word
            old_o = corpus.sents[i][0][self.oi[i] == self.intent_obj[i]]
            old_w = corpus.sents[i][1][self.wi[i] == self.ref_word[i]]
            new_o = corpus.sents[i][0][self.oi[i] == j]
            new_w = corpus.sents[i][1][self.wi[i] == k]

            # now subtract their counts from the referential lexicon,
            # but only if there was a referred object
            # and add word back to the non-referential lexicon
            if old_o.size > 0:
                self.ref[old_o, old_w] -= 1
                self.ref_score[old_o] = update_dm_minus(self.ref_score[old_o],
                                                        self.ref[old_o, :][0],
                                                        params.alpha_r, old_w)
                self.non_ref[old_w] += 1
                self.nr_score = update_dm_plus(self.nr_score, self.non_ref,
                                               params.alpha_nr, old_w[0])

            # critical part: re-score and shift counts in ref lexicon
            # and non-ref lexicon, but only if there is a non-null object
            if new_o.size > 0:
                self.ref[new_o, new_w] += 1
                self.ref_score[new_o] = update_dm_plus(self.ref_score[new_o],
                                                       self.ref[new_o, :][0],
                                                       params.alpha_r, new_w)
                self.non_ref[new_w] -= 1
                self.nr_score = update_dm_minus(self.nr_score, self.non_ref,
                                                    params.alpha_nr, new_w[0])

            # now do the reassignment and update score
            self.intent_obj[i] = j
            self.ref_word[i] = k
            self.intent_obj_prob[i] = self.intent_obj_probs[i][j]
            score = self.update_score(i)

            if verbose > 1:
                print "\n--- score lex: %d, %d ---" % (j, k)
                print "old o: " + str(old_o) + ", old w: " + str(old_w)
                print "new o: " + str(new_o) + ", new w: " + str(new_w)

                print self.ref
                print " " + str(self.non_ref)
                print "counts: %d" % (sum(self.non_ref) + sum(self.ref))

                # note the only difference here is whether you score the full range of
                # intents - or only up to the sentence heard by pf
                if self.inference_method == "gibbs":
                    print "interim score: r %2.1f, nr %2.1f, i %2.1f, " \
                          "p %2.1f,  total: %2.1f" % (sum(self.ref_score),
                                                          self.nr_score,
                                                          sum(self.intent_obj_prob),
                                                          self.param_score,
                                                          score)
                elif self.inference_method == "pf":
                    print "interim score: r %2.1f, nr %2.1f, i %2.1f, " \
                          "p %2.1f,  total: %2.1f" % (sum(self.ref_score),
                                                          self.nr_score,
                                                          sum(self.intent_obj_prob[0:i+1]),
                                                          self.param_score,
                                                          score)
            return score

        #########
        ## score_full_lex - rescore everything
        ## - use this for setup in combo with initLex
        ## - important for hyperparameter inference
        def score_full_lex(self,
                           corpus,
                           params,
                           init=False):

            # set up the intent caching
            for i in range(corpus.n_sents):

                # cache word and object probabilities uniformly
                # 1 x o matrix with [uniform ... empty]
                # and 1 x w matrix again with [uniform ... empty]
                n_os = len(corpus.sents[i][0])
                if n_os > 0:
                    unif_o = log((1 - params.empty_intent) / n_os)
                else:
                    unif_o = [None] # protects against zero objects

                self.intent_obj_probs[i] = [unif_o] * n_os + [log(params.empty_intent)]

                if init:
                    # update lexicon dirichlets based on random init
                    io = self.oi[i] == self.intent_obj[i]
                    rw = self.wi[i] == self.ref_word[i]

                    if io.any():  # protect against nulls
                        self.ref[corpus.sents[i][0][io],corpus.sents[i][1][rw]] += 1

                    # includes all words that are not the referential word
                    self.non_ref[corpus.sents[i][1][self.wi[i] != self.ref_word[i]]] += 1

                    # now add the referential words for null objects
                    if not io.any():
                        self.non_ref[corpus.sents[i][1][self.wi[i] == self.ref_word[i]]] += 1


                # now set up the quick scoring probability caches
                self.intent_obj_prob[i] = self.intent_obj_probs[i][self.intent_obj[i]]

            # cache DM scores for lexicon
            for i in range(corpus.world.n_objs):
                self.ref_score[i] = score_dm(self.ref[i, :], params.alpha_r)

            # cache non-ref DM score also
            self.nr_score = score_dm(self.non_ref, params.alpha_nr)

            # score hyperparameters (via hyper-hyperparameters)
            empty_intent_score = beta.logpdf(params.empty_intent, params.intent_hp_a, params.intent_hp_b)
            alpha_score = gamma.logpdf(params.alpha_r, params.alpha_r_hp) + gamma.logpdf(params.alpha_nr,
                                                                                         params.alpha_nr_hp)
            self.param_score = empty_intent_score + alpha_score
            score = self.update_score(corpus.n_sents)

            # debugging stuff
            if self.verbose >= 1:
                print "\n--- score full lex ---"
                print self.ref
                print " " + str(self.non_ref)

                if self.verbose > 1:
                    print "counts: %d" % (sum(self.non_ref) + sum(self.ref))
                    print "    intent obj: " + str(self.intent_obj)
                    print "    ref word: " + str(self.ref_word)
                    print "    intent obj prob: " + str(self.intent_obj_prob.round(1))

                print "full score: r %2.1f, nr %2.1f, i %2.1f, " \
                          "p %2.1f,  total: %2.1f" % (sum(self.ref_score),
                                                      self.nr_score,
                                                      sum(self.intent_obj_prob),
                                                      self.param_score,
                                                      score)


            return score

        #########
        ## hyperParamInf implements hyperparameter inference
        def hyper_param_inf(self,
                            corpus,
                            params,
                            score):
            if self.verbose >= 1:
                print "\n****** HP INFERENCE *******"

            for i in range(params.n_hypermoves):
                if self.verbose > 1:
                    print "\n--- current params ---"
                    params.show()
                    print "hyper param score:" + str(score)
                    print "    a_nr: " + str(gamma.logpdf(params.alpha_r, params.alpha_r_hp))
                    print "    a_r: " + str(gamma.logpdf(params.alpha_nr, params.alpha_nr_hp))
                    print "    empty_i: " + str(beta.logpdf(params.empty_intent, params.intent_hp_a, params.intent_hp_b))

                new_params = Params()
                new_params.propose_hyper_params(params)
                new_score = self.score_full_lex(corpus, new_params)
                # print "* scoring"
                # params.show()

                if self.verbose > 1:
                    print "--- new params ---"
                    new_params.show()
                    print "hyper param score:" + str(new_score)
                    print "    a_nr: " + str(gamma.logpdf(new_params.alpha_r, new_params.alpha_r_hp))
                    print "    a_r: " + str(gamma.logpdf(new_params.alpha_nr, new_params.alpha_nr_hp))
                    print "    empty_i: " + str(beta.logpdf(new_params.empty_intent, new_params.intent_hp_a, new_params.intent_hp_b))

                if new_score - score > 0:
                    params = new_params
                elif random() < exp(new_score - score):
                    params = new_params

                    if self.verbose >= 1:
                        print "    hp change! - old = %2.2f, new = %2.2f" % (score, new_score)

            # now rescore with the new parameters - redundant if you didn't swap, FIXME
            self.score_full_lex(corpus, params)

            return params

        #########
        ## choose_class - does the selection step
        def choose_class(self, scores):
            new_scores = scores - scores.max()
            ps = exp(new_scores)
            ps = ps / sum(sum(ps))
            cum_ps = reshape(cumsum(ps), shape(ps))

            r = random()
            i = where(cum_ps > r)[0][0]
            j = where(cum_ps > r)[1][0]
            s = scores[i, j]

            # return tuple of indices for greater than
            if self.verbose >= 1:
                print "\n*** choosing %d, %d, score: %2.2f" % (i, j, s)

            return i, j, s

        #########
        ## little function to keep track of samples
        def tick(self, s):
            if self.verbose >= 1:
                print "\n*************** %d ***************" % s
            elif self.verbose > 0:
                print str(self.sample_scores[s-1])
            else:
                if mod(s, 80) == 0:
                    print "\n"
                else:
                    sys.stdout.write(".")

        ########
        ## scoring method
        def update_score(self, i):
            if self.inference_method == "gibbs":
                score = sum(self.intent_obj_prob) + self.nr_score + \
                        sum(self.ref_score) + self.param_score
            elif self.inference_method == "pf":
                # note +1 here to get the range right for the pf
                score = sum(self.intent_obj_prob[0:i + 1]) + self.nr_score + \
                        sum(self.ref_score) + self.param_score

            return score

        #########
        ## generic show method
        def show(self):
            print self.ref

            if hasattr(self, 'non_ref'):
                print "nr: " + str(self.non_ref)

        #########
        ## show_top_match: show nice matching entry
        def show_top_match(self, corpus, world):
            if corpus.corpus != False:
                for o in range(world.n_objs):
                    if max(self.ref[o, :]) > 0:
                        w = where(self.ref[o, :] == max(self.ref[o, :]))[0]
                        print "object: %s, word: %s" % (world.objs_dict[o][0], world.words_dict[w[0]][0])
            else:
                for o in range(world.n_objs):
                    w = where(self.ref[o, :] == max(self.ref[o, :]))[0]
                    print "object: %d, word: %d" % (o, w)


#################################################################
##### Particle class is for the particle filter
class Particle:
    #########
    ## initialize the particle
    def __init__(self, lex, corpus, params):
        # deepcopy the particular params we need from the base lexicon
        field_list = ["ref", "non_ref",
                      "intent_obj_probs", "intent_obj_prob",
                      "ref_score", "nr_score",
                      "oi", "wi", "verbose"]

        for p in lex.__dict__.keys():
            if p in field_list:
                exec ( "self.%s = copy.deepcopy(lex.%s)" % (p, p)) in locals(), globals()


        # bookkeeping
        self.inference_method = "pf"
        self.sample_scores = [0.0] * corpus.n_sents

        # initialize each object to have been null
        self.intent_obj = map(lambda x: len(x[0]), corpus.sents)
        self.ref_word = map(lambda x: len(x[1]), corpus.sents)

        # update all the scores
        self.score_full_lex(corpus, params, init=False)

    #########
    ## prep_sent - adds non-ref counts for current sentence
    def prep_sent(self,
                  corpus, params, i):

        ws = corpus.sents[i][1]
        for w in ws:
            self.non_ref[w] += 1
            self.nr_score = update_dm_plus(self.nr_score, self.non_ref,
                                           params.alpha_nr, w)  # index to make it float, not np.array

    #########
    ## clone - for resampling, takes a particle and deep-copies its internals
    def clone(self, particle):
        field_list = ["ref", "non_ref", "intent_obj_probs", "intent_obj_prob",
                      "ref_score","nr_score","oi","wi","sample_scores"]

        for p in particle.__dict__.keys():
            if p in field_list:
                exec ( "self.%s = copy.deepcopy(particle.%s)" % (p, p)) in locals(), globals()

    #########
    ## other inherited methods from Lexicon class
    score_lex = Lexicon.__dict__["score_lex"]

    choose_class = Lexicon.__dict__["choose_class"]

    score_full_lex = Lexicon.__dict__["score_full_lex"]

    update_score = Lexicon.__dict__["update_score"]