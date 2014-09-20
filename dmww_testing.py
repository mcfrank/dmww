import getopt, csv, itertools
from dmww_classes import *


class Simulation:

    def __init__(self, corpus_file, inference_algorithm, lexicon_params, data_writer):
        self.alg = inference_algorithm
        self.world = World(corpus=corpus_file)
        self.corpus = Corpus(world=self.world, corpus=corpus_file)
        self.gold = Corpus(world=self.world, corpus='corpora/gold_standard.csv')
        self.params = lexicon_params
        self.lexicon = Lexicon(self.corpus, self.params, verbose=0, hyper_inf=False)
        self.filename = 'simulations/' + str(id(self))
        self.data_writer = data_writer

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
        best_threshold = max(scores, key=lambda t: scores[t][2])
        return best_threshold, scores[best_threshold]

    def run(self):

        self.learn_lexicon()
        threshold, (p, r, f) = self.maximize_score()
        ref = self.lexicon.ref

        self.data_writer.writerow([str(id(self)), self.alg, self.params.n_samps, self.params.n_particles,
                                   self.params.alpha_r, self.params.alpha_nr, self.params.empty_intent,
                                   p, r, f, threshold])

        sim_file = open(self.filename + '.txt', 'a')
        sim_file.write('Algorithm:' + self.alg)
        sim_file.write('\n\n')

        sim_file.write('Parameters:\n')
        sim_file.write(str(self.lexicon.params))
        sim_file.write('\n\n')

        sim_file.write('Lexicon:\n')
        for obj in range(self.world.n_objs):
            row_sums = ref.sum(axis=1)
            links = where(ref[obj, :] / row_sums[obj] > threshold)
            for word in links[0]:
                sim_file.write('o: %s, w: %s' % (self.world.objs_dict[obj][0], self.world.words_dict[word][0]))
                sim_file.write('\n')
        sim_file.write('\n\n')

        sim_file.write('Precision: %s\nRecall: %s\nF-score: %s\nThreshold: %s' % (p, r, f, threshold))
        sim_file.close()

        self.lexicon.plot_scores()
        plt.savefig(self.filename + '_scores.png')
        plt.close()
        #self.lexicon.plot_lex(self.world)
        #plt.savefig(self.filename + "_lex.png")


def run_grid(data_file, alg, n, alpha_r_opts, alpha_nr_opts, empty_intent_opts):

    data_writer = csv.writer(data_file)
    data_writer.writerow(['id', 'alg', 'n_samps', 'n_particles',
                          'alpha_r', 'alpha_nr', 'empty_intent',
                          'precision', 'recall', 'f-score', 'threshold'])

    for ar, anr, ei in itertools.product(alpha_r_opts, alpha_nr_opts, empty_intent_opts):
        print '\nRunning simulation with algorithm %s and parameters n %s, alpha_r %s, alpha_nr %s, empty_intention %s'\
              % (alg, n, ar, anr, ei)
        seed(1)
        params = Params(n_samps=n,
                        alpha_r=ar,
                        alpha_nr=anr,
                        empty_intent=ei,
                        n_particles=n)
        sim = Simulation('corpora/corpus.csv', alg, params, data_writer)
        sim.run()


def main():

    alpha_r_opts = [0.1, 1.0, 10.0]
    alpha_nr_opts = [0.1, 1.0, 10.0]
    empty_intent_opts = [0.001, 0.01, 0.1]

    results_gibbs_n1 = open('simulations/pilot_sims_gibbs_n1.csv', 'a')
    run_grid(results_gibbs_n1, 'gibbs', 1, alpha_r_opts, alpha_nr_opts, empty_intent_opts)
    results_gibbs_n1.close()
    results_gibbs_n10 = open('simulations/pilot_sims_gibbs_n10.csv', 'a')
    run_grid(results_gibbs_n10, 'gibbs', 10, alpha_r_opts, alpha_nr_opts, empty_intent_opts)
    results_gibbs_n10.close()
    results_gibbs_n100 = open('simulations/pilot_sims_gibbs_n100.csv', 'a')
    run_grid(results_gibbs_n100, 'gibbs', 100, alpha_r_opts, alpha_nr_opts, empty_intent_opts)
    results_gibbs_n100.close()

    results_pf_n1 = open('simulations/pilot_sims_pf_n1.csv', 'a')
    run_grid(results_pf_n1, 'pf', 1, alpha_r_opts, alpha_nr_opts, empty_intent_opts)
    results_pf_n1.close()
    results_pf_n10 = open('simulations/pilot_sims_pf_n10.csv', 'a')
    run_grid(results_pf_n10, 'pf', 10, alpha_r_opts, alpha_nr_opts, empty_intent_opts)
    results_pf_n10.close()
    results_pf_n100 = open('simulations/pilot_sims_pf_n100.csv', 'a')
    run_grid(results_pf_n100, 'pf', 100, alpha_r_opts, alpha_nr_opts, empty_intent_opts)
    results_pf_n100.close()

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