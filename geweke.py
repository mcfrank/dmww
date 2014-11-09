import scipy.stats
from dmww_classes import *


class GewekeTest:

    def __init__(self, diagnostics, m_marg, m_succ, alpha_r, alpha_nr, empty_intent,
                 n_objs, n_words, n_situations, max_obj_per_situation, max_words_per_sentence):

        self.diagnostics = diagnostics
        self.m_marg = m_marg
        self.m_succ = m_succ
        self.params = Params(n_samps=1,
                             alpha_nr=alpha_nr,
                             alpha_r=alpha_r,
                             empty_intent=empty_intent)
        self.n_situations = n_situations

        self.world = World(n_words, n_objs, corpus=False)
        self.situations = self.generate_situations(n_situations, max_obj_per_situation, max_words_per_sentence)
#        print self.situations

    @staticmethod
    def normalize(np_array):
        return np.divide(np_array, sum(np_array))

    @staticmethod
    def normalize2d(np_array):
        row_sums = np_array.sum(axis=1)
        return np_array / row_sums[:, np.newaxis]

    def sample_ref_from_prior(self):
        return np.random.dirichlet([self.params.alpha_r] * self.world.n_words, self.world.n_objs)

    def sample_nonref_from_prior(self):
        return self.normalize(np.random.dirichlet([self.params.alpha_nr] * self.world.n_words, 1)[0])

    def generate_situations(self, n_situations, max_obj_per_situation, max_words_per_sentence):
        # for each situation, sample a set of objects uniformly from all objects
        # number of objects is uniform between 1 and max objects per situation
        all_objs = range(self.world.n_objs)
        situation_objects = [sample(all_objs, choice(range(1, max_obj_per_situation+1))) for n in xrange(n_situations)]
        situation_words = [[None]*choice(range(1, max_words_per_sentence+1)) for n in xrange(n_situations)]
        return [[np.array(obj), np.array(words)] for obj, words in zip(situation_objects, situation_words)]

    def forward_sample(self, ref_lex, nonref_lex):

        corpus = Corpus(self.world)
        corpus.sents = []
        corpus.n_sents = self.n_situations

        for objects, dummy_words in self.situations:

            num_words = len(dummy_words)
            words = []

            # flip a coin to indicate if there are no referring words in sentence
            empty = scipy.stats.bernoulli.rvs(self.params.empty_intent)

            # sample a referential word
            if not empty:

                # sample an object uniformly from set of objects in situation
                intention = choice(objects)

                # sample a referential word conditioned on intention from referential lexicon
                norm_intent = self.normalize(ref_lex[intention])
                ref_word = np.where(np.random.multinomial(1, norm_intent) == 1)[0][0]
                words.append(ref_word)
                num_words -= 1

            # for all non-referential words, sample from non-referential lexicon
            for w in xrange(num_words):

                norm_nonref = self.normalize(nonref_lex)
                word = np.where(np.random.multinomial(1, norm_nonref) == 1)[0][0]
                words.append(word)

            corpus.sents.append([np.array(objects), np.array(words)])

        corpus.update()
#        print '\ngenerated corpus:'
#        print corpus.sents
#        corpus.show()

        return corpus

    def simulate_marginal(self):

        marg_stats = []
        for m in xrange(self.m_marg):

            # sample unobservables from prior
            ref_lex = self.sample_ref_from_prior()
            nonref_lex = self.sample_nonref_from_prior()

            # sample observables conditioned on unobservables
            forward_data = self.forward_sample(ref_lex, nonref_lex)

            # compute test statistic
            stats = [diagnostic(ref_lex, nonref_lex, forward_data) for diagnostic in self.diagnostics]
            marg_stats.append(stats)

        # compute the mean and variance of the test statistics
        means = [mean([s[d] for s in marg_stats]) for d in xrange(len(self.diagnostics))]
        vars = [var([s[d] for s in marg_stats]) for d in xrange(len(self.diagnostics))]
        print 'marg_stats', means, vars
#        return g, s
        return marg_stats

    def simulate_successive(self):

        # sample initial unobservables from prior
        ref_lex = self.sample_ref_from_prior()
        nonref_lex = self.sample_nonref_from_prior()

#        corpus = self.forward_sample(ref_lex, nonref_lex)

#        dummy_corpus = Corpus(self.world)
#        dummy_corpus.n_sents = self.n_situations
#        dummy_corpus.sents = self.situations
#        dummy_corpus.update()
#        dummy_corpus.show()
#        print corpus.sents[0]
#        lexicon = Lexicon(corpus, self.params, verbose=0, hyper_inf=False)
#        lexicon.inference_method = "gibbs"
#        lexicon.init_gibbs(corpus, self.params)

        succ_stats = []
        for m in xrange(self.m_succ):

            # resample observables conditioned on unobservables
            corpus = self.forward_sample(ref_lex, nonref_lex)

            # resample unobservables conditioned on observables
            lexicon = Lexicon(corpus, self.params, verbose=0, hyper_inf=False)
            lexicon.inference_method = "gibbs"
            lexicon.init_gibbs(corpus, self.params)
#            lexicon.gibbs_step(0, corpus, self.params)
            lexicon.learn_lex_gibbs(corpus, self.params)
            ref_lex = self.normalize2d(lexicon.ref)
            nonref_lex = self.normalize(lexicon.non_ref)

            # compute test statistic
            stats = [diagnostic(ref_lex, nonref_lex, corpus) for diagnostic in self.diagnostics]
            succ_stats.append(stats)

        # compute the mean and variance of the test statistics
        means = [mean([s[d] for s in succ_stats]) for d in xrange(len(self.diagnostics))]
        vars = [var([s[d] for s in succ_stats]) for d in xrange(len(self.diagnostics))]
        print 'succ_stats', means, vars
#        return g, s
        return succ_stats

    def run_test(self):

        marg_stats = self.simulate_marginal()
        succ_stats = self.simulate_successive()
#        diagnostic = (marg_mean - succ_mean) / sqrt((marg_var / self.m_marg)+(succ_var / self.m_succ))
#        return diagnostic

        for d in xrange(len(self.diagnostics)):
            _, ax = plt.subplots(2, 1, sharex=True)
            plt.sca(ax[0])
            parms = dict(bins=20)
            plt.hist([marg_stats[s][d] for s in xrange(self.m_marg)], **parms)
            plt.grid()
            plt.title('Ground truth')
            plt.sca(ax[1])
            plt.hist([succ_stats[s][d] for s in xrange(self.m_succ)], **parms)
            plt.grid()
            plt.title('Inferred')

#        marg_sorted = np.sort(marg_stats)
#        print marg_sorted
#        marg_cum = np.arange(len(marg_sorted))/float(len(marg_sorted))
#        print marg_cum
#        succ_sorted = np.sort(succ_stats)
#        print succ_sorted
#        succ_cum = np.arange(len(succ_sorted))/float(len(succ_sorted))
#        print succ_cum
#        plt.plot(marg_cum, succ_cum)
#        plt.xlabel('forward samples')
#        plt.ylabel('inferred samples')
#        plt.show()

#        marg_hist, marg_bin_edges = np.histogram(np.array(marg_stats), normed=True)
#        marg_cdf = np.cumsum(marg_hist)
#        print marg_cdf

#        succ_hist, succ_bin_edges = np.histogram(np.array(succ_stats), normed=True)
#        succ_cdf = np.cumsum(succ_hist)
#        print succ_cdf

#        fig, ax = plt.subplots()
#        plt.plot(marg_cdf, succ_cdf, '-')
#        plt.xlabel('forward samples')
#        plt.ylabel('inferred samples')
#        plt.title('')
#        return ax
        plt.show()


def stupid_func(ref_lex, nonref_lex, data):
    return sum(where(ref_lex > 0.5))


def top_left(ref_lex, nonref_lex, data):
    return GewekeTest.normalize2d(ref_lex)[0][0]


def full_grid(n_objs, n_words):
    return [lambda ref, nonref, data: ref[i][j] for i in xrange(n_objs) for j in xrange(n_words)]

#geweke = GewekeTest(diagnostics=full_grid(2, 2),
geweke = GewekeTest(diagnostics=[lambda ref, nonref, data: ref[0][0], lambda ref, nonref, data: ref[0][1],
                                 lambda ref, nonref, data: ref[1][0], lambda ref, nonref, data: ref[1][1]],
                    m_marg=100,
                    m_succ=100,
                    alpha_r=0.1,
                    alpha_nr=1.0,
                    empty_intent=0.01,
                    n_objs=2,
                    n_words=2,
                    n_situations=10,
                    max_obj_per_situation=2,
                    max_words_per_sentence=2)

geweke.run_test()