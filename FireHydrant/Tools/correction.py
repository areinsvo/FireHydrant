#!/usr/bin/env python
"""correction functions need to be applied for the analysis
"""
import os
from os.path import join

import numpy as np
import uproot
from coffea.lookup_tools.dense_lookup import dense_lookup


def get_pu_weights(nvtx):

    pufile_ = uproot.open(join(os.getenv('FH_BASE'), 'FireHydrant/Tools/store/puWeights_10x_56ifb.root'))
    sf_pu_cen = dense_lookup(pufile_['puWeights'].values, pufile_['puWeights'].edges)
    sf_pu_up  = dense_lookup(pufile_['puWeightsUp'].values, pufile_['puWeightsUp'].edges)
    sf_pu_down= dense_lookup(pufile_['puWeightsDown'].values, pufile_['puWeightsDown'].edges)

    return sf_pu_cen(nvtx), sf_pu_up(nvtx), sf_pu_down(nvtx)


def get_ttbar_weight(pt):
    return np.exp(0.0615 - 0.0005 * np.clip(pt, 0, 800))


def get_nlo_weight(type, pt):
    kfactor = uproot.open(join(os.getenv('FH_BASE'), 'FireHydrant/Tools/store/kfactors.root'))
    sf_qcd = 1.
    sf_ewk = 1

    lo = dict(
        z="ZJets_LO/inv_pt",
        w="WJets_LO/inv_pt",
        a="GJets_LO/inv_pt_G",
    )

    nlo = dict(
        z="ZJets_012j_NLO/nominal",
        w="WJets_012j_NLO/nominal",
        a="GJets_1j_NLO/nominal_G",
    )

    LO = kfactor[lo[type]].values
    NLO = kfactor[nlo[type]].values
    EWK = kfactor[ewk[type]].values

    sf_qcd = NLO / LO
    sf_ewk = EWK / LO

    correction = dense_lookup(sf_qcd * sf_ewk, kfactor[nlo[type]].edges)

    return correction(pt)