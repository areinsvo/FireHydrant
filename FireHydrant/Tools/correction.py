#!/usr/bin/env python
"""correction functions need to be applied for the analysis
"""
import os
from os.path import join

import uproot
from coffea.lookup_tools.dense_lookup import dense_lookup


def get_pu_weights(nvtx):

    pufile_ = uproot.open(join(os.getenv('FH_BASE'), 'FireHydrant/Tools/store/puWeights_10x_56ifb.root'))
    sf_pu_cen = dense_lookup(pufile_['puWeights'].values, pufile_['puWeights'].edges)
    sf_pu_up  = dense_lookup(pufile_['puWeightsUp'].values, pufile_['puWeightsUp'].edges)
    sf_pu_down= dense_lookup(pufile_['puWeightsDown'].values, pufile_['puWeightsDown'].edges)

    return sf_pu_cen(nvtx), sf_pu_up(nvtx), sf_pu_down(nvtx)