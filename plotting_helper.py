# import stuff
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def lexplot(l, w):
    fig = plt.figure()
    gs = gridspec.GridSpec(2, 1, height_ratios=[8,1])
        
    ax1 = fig.add_subplot(gs[0])
    ax1.pcolormesh(l.ref, cmap = "Reds") 
    ax1.set_xticks(np.arange(w.n_words) + .5)
    ax1.set_xticklabels(np.arange(w.n_words))
    ax1.set_yticks(np.arange(w.n_objs) + .5)
    ax1.set_yticklabels(np.arange(w.n_objs))
    ax1.set_ylabel("objects")
    ax1.set_title('main lexicon', fontsize=14)

    if hasattr(l, 'non_ref'):
        ax2 = fig.add_subplot(gs[1])
        ax2.pcolormesh(np.array([l.non_ref]), cmap = "Reds") 
        ax2.set_xticks(np.arange(w.n_words) + .5)
        ax2.set_xticklabels(np.arange(w.n_words))
        ax2.set_yticks([])
        ax2.set_xlabel("words")
        ax2.set_title('non-referential lexicon', fontsize=14)
        plt.tight_layout(pad = 2)
    else:
        ax1.set_xlabel("words")


