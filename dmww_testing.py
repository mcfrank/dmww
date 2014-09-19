import getopt, csv, itertools
from dmww_classes import *


class Simulation:

    def __init__(self, corpus_file, inference_algorithm, lexicon_params, results_writer):
        self.alg = inference_algorithm
        self.world = World(corpus=corpus_file)
        self.corpus = Corpus(world=self.world, corpus=corpus_file)
        self.gold = Corpus(world=self.world, corpus='corpora/gold_standard.csv')
        self.params = lexicon_params
        self.lexicon = Lexicon(self.corpus, self.params, verbose=0, hyper_inf=False)
#        self.filename = "simulations/%s_samp%s_ref%s_nonref%s_empint%s" % (self.alg,
#                                                                           self.params.n_samps,
#                                                                           self.params.n_particles,
#                                                                           self.params.alpha_r,
#                                                                           self.params.alpha_nr,
#                                                                           self.params.empty_intent)
        self.filename = 'simulations/' + str(id(self))
        self.write_file = open(self.filename + '.txt', 'a')
        self.results_writer = results_writer

    def learn_lexicon(self):
        if self.alg == 'gibbs':
            self.lexicon.learn_lex_gibbs(self.corpus, self.params)
        elif self.alg == 'pf':
            self.lexicon.learn_lex_pf(self.corpus, self.params, resample=False)
            self.lexicon.output_lex_pf(self.corpus, self.params)

    def maximize_score(self):
        scores = {}
        threshold_opts = [float(t)/100 for t in xrange(101)]
        for threshold in threshold_opts:
            score = self.lexicon.get_f(self.gold, threshold)
            if score:
                scores[threshold] = score
        best_threshold = max(scores, key=lambda x: scores[x][2])
        return best_threshold, scores[best_threshold]

    def run(self):

        self.write_file.write('Parameters:\n')
        self.write_file.write(str(self.lexicon.params))
        self.write_file.write('\n\n')

        self.learn_lexicon()
        threshold, (p, r, f) = self.maximize_score()
        ref = self.lexicon.ref

        self.write_file.write('Lexicon:\n')
        for obj in range(self.world.n_objs):
            row_sums = ref.sum(axis=1)
            links = where(ref[obj, :] / row_sums[obj] > threshold)
            for word in links[0]:
                self.write_file.write('o: %s, w: %s' % (self.world.objs_dict[obj][0], self.world.words_dict[word][0]))
                self.write_file.write('\n')
        self.write_file.write('\n\n')
        self.write_file.write('Precision: %s\nRecall: %s\nF-score: %s\nThreshold: %s' % (p, r, f, threshold))

        self.lexicon.plot_scores()
        plt.savefig(self.filename + '_scores.png')
        #self.lexicon.plot_lex(self.world)
        #plt.savefig(self.filename + "_lex.png")

        self.write_file.close()

        self.results_writer.writerow([str(id(self)), self.alg, self.params.n_samps, self.params.n_particles,
                                      self.params.alpha_r, self.params.alpha_nr, self.params.empty_intent,
                                      p, r, f, threshold])


def run_grid(alg_opts, n_opts, alpha_r_opts, alpha_nr_opts, empty_intent_opts):

    results_file = open('simulations/pilot_sims.csv', 'a')
    results_writer = csv.writer(results_file)
    results_writer.writerow(['id', 'alg', 'n_samps', 'n_particles',
                             'alpha_r', 'alpha_nr', 'empty_intent',
                             'precision', 'recall', 'f-score', 'threshold'])

    for alg, n, ar, anr, ei in itertools.product(alg_opts, n_opts, alpha_r_opts, alpha_nr_opts, empty_intent_opts):
        seed(1)
        params = Params(n_samps=n,
                        alpha_r=ar,
                        alpha_nr=anr,
                        empty_intent=ei,
                        n_particles=n)
        sim = Simulation('corpora/corpus.csv', alg, params, results_writer)
        sim.run()


def main():
    alg_opts = ['gibbs']
    n_opts = [1]
    alpha_r_opts = [0.1]
    alpha_nr_opts = [10]
    empty_intent_opts = [0.001, 0.01]
    run_grid(alg_opts, n_opts, alpha_r_opts, alpha_nr_opts, empty_intent_opts)

main()

# def main(argv):
#     inference_algorithm = 'gibbs'
#     n_samps = 1
#     n_particles = 10
#     alpha_r = 0.1
#     alpha_nr = 10
#     empty_intent = 0.01
#
#     usage = "usage: dmww_testing.py -a <inference algorithm: gibbs or pf> -n <number of samples/particles>" \
#             "--alpha-r <referential alpha> --alpha-nr <non-referential alpha> --empty-intent <empty intent probability>"
#
#     try:
#         opts, args = getopt.getopt(argv, "ha:n:", ["alpha-r=", "alpha-nr=", "empty-intent="])
#     except getopt.GetoptError:
#         print usage
#         sys.exit(2)
#     for opt, arg in opts:
#         if opt == '-h':
#             print usage
#             sys.exit()
#         elif opt == "-a":
#             if arg != 'pf' and arg != 'gibbs':
#                 print "Invalid inference algorithm, must be gibbs or pf."
#                 sys.exit(2)
#             inference_algorithm = arg
#         elif opt == "-n":
#             n_samps = arg
#             n_particles = arg
#         elif opt == "--alpha-r":
#             alpha_r = arg
#         elif opt == "--alpha-nr":
#             alpha_nr = arg
#         elif opt == "--empty-intent":
#             empty_intent = arg
#
#     params = Params(n_samps=n_samps,
#                     alpha_r=alpha_r,
#                     alpha_nr=alpha_nr,
#                     empty_intent=empty_intent,
#                     n_particles=n_particles)
#
#     sim = Simulation('corpora/corpus.csv', inference_algorithm, params)
#     sim.run()

#if __name__ == "__main__":
#    main(sys.argv[1:])