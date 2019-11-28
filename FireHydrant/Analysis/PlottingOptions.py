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
