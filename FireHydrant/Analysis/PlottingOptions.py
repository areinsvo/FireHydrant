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

def SetROOTHistStyle():
    import ROOT
    ROOT.gROOT.SetBatch()

    # https://root.cern.ch/doc/master/classTStyle.html
    # title
    ROOT.gStyle.SetTitleAlign(13)
    ROOT.gStyle.SetTitleX(0.1)
    ROOT.gStyle.SetTitleY(0.96)
    ROOT.gStyle.SetTitleFont(62, 't')
    ROOT.gStyle.SetTitleFont(42, 'xyz')
    ROOT.gStyle.SetTitleFontSize(0.04)
    ROOT.gStyle.SetTitleSize(0.03, 'xyz')
    # stat
    ROOT.gStyle.SetStatFont(42)
    ROOT.gStyle.SetStatFontSize(0.02)
    ROOT.gStyle.SetStatW(0.15)
    ROOT.gStyle.SetStatX(0.9)
    ROOT.gStyle.SetStatY(0.9)
    # ticks
    ROOT.gStyle.SetPadTickX(1)
    ROOT.gStyle.SetPadTickY(1)
    ROOT.gStyle.SetTickLength(0.02, 'xyz')
    # labels
    ROOT.gStyle.SetLabelSize(0.03, 'xyz')


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


def make_ratio_plot(bkgh, datah, sigh=None, title=None, overflow='over', yscale='log'):
    import matplotlib.pyplot as plt
    import numpy as np
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
    ymin, ymax = ax.get_ylim()
    if yscale == 'linear': ymax = (ymax-ymin)*1.2 + ymin
    if yscale == 'log': ymax = 10**(((np.log10(ymax)-np.log10(ymin))*1.2) + np.log10(ymin))
    ax.set_ylim(ymin, ymax)
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


def make_mc_plot(bkgh, sigh=None, title=None, overflow='over', yscale='log'):
    import matplotlib.pyplot as plt
    import numpy as np
    from coffea import hist

    fig, ax = plt.subplots(1,1,figsize=(8,6))
    hist.plot1d(bkgh, overlay='cat', ax=ax,
                clear=False, stack=True, overflow=overflow,
                line_opts=None, fill_opts=fill_opts, error_opts=error_opts)
    if sigh:
        hist.plot1d(sigh, overlay='dataset', ax=ax, overflow=overflow, clear=False)
    ax.set_yscale(yscale)
    ax.autoscale(axis='both', tight=True)
    ymin, ymax = ax.get_ylim()
    if yscale == 'linear': ymax = (ymax-ymin)*1.2 + ymin
    if yscale == 'log': ymax = 10**(((np.log10(ymax)-np.log10(ymin))*1.2) + np.log10(ymin))
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(ax.get_xlabel(), x=1, ha='right')
    ax.set_ylabel(ax.get_ylabel(), y=1.0, ha="right")
    ax.set_title(title, x=0.0, ha="left")
    ax.text(1,1,'59.74/fb (13TeV)', ha='right', va='bottom', transform=ax.transAxes)
    if sigh:
        ax.legend(*groupHandleLabel(ax), prop={'size': 8,}, ncol=3)
    else:
        leg=ax.legend()

    return fig, ax


def make_signal_plot(sigh, title=None, overflow='over', yscale='log'):
    import matplotlib.pyplot as plt
    import numpy as np
    from coffea import hist

    fig, ax = plt.subplots(1,1,figsize=(8,6))
    hist.plot1d(sigh, overlay='dataset', ax=ax, overflow=overflow,)
    ax.set_yscale(yscale)
    ax.autoscale(axis='both', tight=True)
    ymin, ymax = ax.get_ylim()
    if yscale == 'linear': ymax = (ymax-ymin)*1.2 + ymin
    if yscale == 'log': ymax = 10**(((np.log10(ymax)-np.log10(ymin))*1.2) + np.log10(ymin))
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(ax.get_xlabel(), x=1, ha='right')
    ax.set_ylabel(ax.get_ylabel(), y=1.0, ha="right")
    ax.set_title(title, x=0.0, ha="left")
    ax.text(1,1,'59.74/fb (13TeV)', ha='right', va='bottom', transform=ax.transAxes)

    return fig, ax


def make_2d_hist(h, xaxis, title=None, zscale='linear', text_opts=None, **kwargs):
    """
    more kwargs see `coffea.hist.plot2d`
    (https://github.com/CoffeaTeam/coffea/blob/master/coffea/hist/plot.py#L388)
    """
    import matplotlib.pyplot as plt
    from coffea import hist

    fig, ax = plt.subplots(1,1,figsize=(8,6))
    if zscale=='log':
        from matplotlib.colors import LogNorm
        hist.plot2d(h, xaxis, ax=ax, patch_opts=dict(norm=LogNorm()), text_opts=text_opts)
    else:
        hist.plot2d(h, xaxis, ax=ax, text_opts=text_opts)

    if 'xlabel' in kwargs: ax.set_xlabel(kwargs['xlabel'])
    if 'ylabel' in kwargs: ax.set_ylabel(kwargs['ylabel'])
    ax.set_xlabel(ax.get_xlabel(), x=1, ha='right')
    ax.set_ylabel(ax.get_ylabel(), y=1, ha="right")

    if 'xlim' in kwargs: ax.set_xlim(*kwargs['xlim'])
    if 'ylim' in kwargs: ax.set_ylim(*kwargs['ylim'])

    ax.set_title(title, x=0.0, ha="left")
    ax.text(1,1,'59.74/fb (13TeV)', ha='right', va='bottom', transform=ax.transAxes)

    return fig, ax
