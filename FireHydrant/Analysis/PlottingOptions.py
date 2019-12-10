#!/usr/bin/env python

"""matploblit plotting options"""

fill_opts = {
    'edgecolor': (0,0,0,0.3),
    'alpha': 0.8
}
error_opts = {
    'label':'Stat. Unc.',
    'hatch':'xxx',
    'facecolor':'none',
    'edgecolor':(0,0,0,.5),
    'linewidth': 0
}
data_err_opts = {
    'linestyle':'none',
    'marker': '.',
    'markersize': 10.,
    'color':'k',
    'elinewidth': 1,
    'emarker': '_'
}


def groupHandleLabel(ax):
    """
    group handle and labels in the same axes.
    `ax.legend(*groupHandleLabel(ax), prop={'size': 8,}, ncol=2)`

    :param matplotlib.pyplot.axes ax: axes
    :return: grouped handles and labels
    :rtype: tuple
    """
    from collections import defaultdict
    hl_ = defaultdict(list)
    for h, l in zip(*ax.get_legend_handles_labels()):
        hl_[l].append(h)
    l2 = hl_.keys()
    h2 = list()
    for h_ in hl_.values():
        h2.append(tuple(h_))
    return h2, l2


def make_ratio_plot(bkgh, datah, sigh=None, title=None, overflow='over'):
    import matplotlib.pyplot as plt
    from coffea import hist

    fig, (ax, rax) = plt.subplots(2,1,figsize=(8,8), gridspec_kw={'height_ratios': (4,1)}, sharex=True)
    fig.subplots_adjust(hspace=.07)
    hist.plot1d(bkgh, overlay='cat', ax=ax,
                clear=False, stack=True, overflow=overflow,
                line_opts=None, fill_opts=fill_opts, error_opts=error_opts)
    hist.plot1d(datah, overlay='cat', ax=ax,
                clear=False, overflow=overflow, error_opts=data_err_opts)
    if sigh:
        hist.plot1d(sigh, overlay='dataset', ax=ax, overflow=overflow, clear=False)
    ax.set_yscale('log')
    ax.autoscale(axis='both', tight=True)
    ax.set_xlabel(None)
    if sigh:
        ax.legend(*groupHandleLabel(ax), prop={'size': 8,}, ncol=3)
    else:
        leg=ax.legend()

    hist.plotratio(datah.sum('cat'), bkgh.sum('cat'),
                   ax=rax, overflow=overflow, error_opts=data_err_opts,
                   denom_fill_opts={}, guide_opts={}, unc='num')
    rax.set_ylabel('Data/MC')
    rax.set_ylim(0, 2)

    rax.set_xlabel(rax.get_xlabel(), x=1.0, ha="right")
    ax.set_ylabel(ax.get_ylabel(), y=1.0, ha="right")
    ax.set_title(title, x=0.0, ha="left")
    ax.text(1,1,'59.74/fb (13TeV)', ha='right', va='bottom', transform=ax.transAxes)

    return fig, (ax, rax)
