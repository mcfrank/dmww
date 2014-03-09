# import stuff
import matplotlib.pyplot as plt
import prettyplotlib as ppl
import matplotlib.gridspec as gridspec

def lexplot(l, w, color_map = None, **kwargs):
    gs = gridspec.GridSpec(2, 1, height_ratios=[8,1])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    w_labels = "".join(map(str, range(0, w.n_words)))
    o_labels = "".join(map(str, range(0, w.n_objs)))

    #referential lexicon
    ppl.pcolormesh(fig, ax1, l.ref, xticklabels=w_labels, yticklabels=o_labels) #plt.ppl.pcolormesh(l.ref) 
    ax1.set_title('main lexicon', fontsize=14)
    ax1.set_ylabel("objects")

    #non-referential lexicon
    ppl.pcolormesh(fig, ax2, array([l.non_ref]), xticklabels=w_labels) #plt.ppl.pcolormesh(l.ref) 
    ax2.set_title('non-referential lexicon', fontsize=14)
    ax2.set_xlabel("words")
    ax2.set_yticks([])
    
    plt.tight_layout(pad = 2)
