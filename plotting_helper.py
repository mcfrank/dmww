# import stuff
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pylab


def lexplot(l, w, fontsize = 30, colormap = "Reds", certainwords = 1):
    # certainwords plots all only non-zero probabilities

    # get data to plot
    if certainwords:
        hiwords = np.empty([0,1], dtype=int)
        for word in range(w.n_words):
            if np.count_nonzero(l.ref[:, word]):
                hiwords = np.append(hiwords, word)
        l.ref_plot =  l.ref[:,hiwords]
    else:
        l.ref_plot =  l.ref
        hiwords = range(0, w.n_words)

    if hasattr(l, 'non_ref'):
        l.non_ref_plot =  l.non_ref[: , hiwords]

    # get labels for plot, and sort data alphabetically
    if hasattr(w, 'words_dict'):

        #word
        #wordlabs = list()
        #for i in range(0,np.shape(hiwords)[0]):
           # wordlabs.append(w.words_dict[hiwords[i]][0])

        wordlabs = [w.words_dict[hiwords[i]][0] for i in range(0,np.shape(hiwords)[0])]

        #sort words
        w_order = np.argsort(wordlabs)
        l.ref_plot = l.ref_plot[:,w_order]
        if hasattr(l, 'non_ref'):
            l.non_ref_plot = l.non_ref_plot[:,w_order]
        wordlabs.sort()

        #obj
        #objlabs = list()
        #for i in range(0,w.n_objs):
         #   objlabs.append(w.objs_dict[i][0])
        objlabs = [w.objs_dict[i][0] for i in range(0,w.n_objs)]


        #sort objs
        o_order = np.argsort(objlabs)
        l.ref_plot = l.ref_plot[o_order,]
        objlabs.sort()

    # set up plot
    if w.n_words < 30:
        fig = plt.figure(figsize=(10, 5))
    else:
        fig = plt.figure(figsize=(.5 * np.shape(hiwords)[0], .9 * w.n_objs))

    gs = gridspec.GridSpec(2, 1, height_ratios=[9,1])
    fontsize = 1.5 * w.n_objs

    # plot referential lexicon
    ax1 = fig.add_subplot(gs[0])
    ax1.pcolormesh(l.ref_plot, cmap = colormap)

    #add word and obj ticks
    if hasattr(w, 'words_dict'):

        #word
        pylab.xticks(np.arange(np.shape(hiwords)[0]) + .5, wordlabs)
        plt.setp(plt.xticks()[1], rotation=90, fontsize=fontsize)

        #objs
        pylab.yticks(np.arange(w.n_objs) + .5, objlabs)
        plt.setp(plt.yticks()[1], fontsize=fontsize)

    else:
        ax1.set_xticks(np.arange(w.n_words) + .5)
        ax1.set_xticklabels(np.arange(w.n_words), fontsize=fontsize)
        ax1.set_yticks(np.arange(w.n_objs) + .5)
        ax1.set_yticklabels(np.arange(w.n_objs), fontsize=fontsize)

    ax1.set_ylabel("objects", fontsize=fontsize + 5)
    ax1.set_title('main lexicon', fontsize=fontsize + 10)

    # plot non-referential lexicon
    if hasattr(l, 'non_ref'):
        ax2 = fig.add_subplot(gs[1])
        ax2.pcolormesh(np.array([l.non_ref_plot]), cmap = colormap)

        #add words ticks
        if hasattr(w, 'words_dict'):
            plt.setp(plt.xticks()[1], rotation=90, fontsize=fontsize)
            pylab.xticks(np.arange(np.shape(hiwords)[0]) + .5, wordlabs)
        else:
            ax2.set_xticks(np.arange(w.n_words) + .5)
            ax2.set_xticklabels(np.arange(w.n_words), fontsize=fontsize)

        ax2.set_xlabel("words", fontsize=fontsize + 5)

        #add obj ticks
        ax2.set_yticks([])

        ax2.set_title('non-referential lexicon', fontsize=fontsize + 10)
        plt.tight_layout(pad = 2)

    else:
        ax1.set_xlabel("words" , fontsize=fontsize + 5)


def scoreplot (scores):
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(scores)), scores, '-')
    plt.title('Sample Scores', )
    plt.xlabel('sample')
    plt.ylabel('sample score')

