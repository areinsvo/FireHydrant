#!/usr/bin/env python
"""Test dphi estimate
"""
import argparse
from contextlib import contextmanager

import awkward
import coffea.processor as processor
import numpy as np
import matplotlib.pyplot as plt
from coffea import hist
from coffea.analysis_objects import JaggedCandidateArray
from FireHydrant.Analysis.DatasetMapLoader import (DatasetMapLoader,
                                                   SigDatasetMapLoader)
from FireHydrant.Tools.correction import (get_nlo_weight_function,
                                          get_pu_weights_function,
                                          get_ttbar_weight)
from FireHydrant.Tools.metfilter import MetFilters
from FireHydrant.Tools.trigger import Triggers
from FireHydrant.Tools.uproothelpers import fromNestNestIndexArray

np.seterr(divide='ignore', invalid='ignore', over='ignore')

parser = argparse.ArgumentParser(description="leptonjet ABCD")
parser.add_argument("--sync", action='store_true', help="issue rsync command to sync plots folder to lxplus web server")
args = parser.parse_args()

import ROOT
ROOT.gROOT.SetBatch()


dml = DatasetMapLoader()
dataDS, dataMAP = dml.fetch('data')

@contextmanager
def _setIgnoreLevel(level):
    originalLevel = ROOT.gErrorIgnoreLevel
    ROOT.gErrorIgnoreLevel = level
    yield
    ROOT.gErrorIgnoreLevel = originalLevel

class LJPairDphi4Mu(processor.ProcessorABC):
    def __init__(self, category='00'):
        self.category = category
        dataset_axis = hist.Cat('dataset', 'dataset')
        self._accumulator = processor.dict_accumulator({
            'dphi': processor.column_accumulator(np.zeros(shape=(0,))),
        })

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, df):
        output = self.accumulator.identity()
        dataset = df['dataset']

        ## construct weights ##
        wgts = processor.Weights(df.size)

        triggermask = np.logical_or.reduce([df[t] for t in Triggers])
        wgts.add('trigger', triggermask)
        cosmicpairmask = df['cosmicveto_result']
        wgts.add('cosmicveto', cosmicpairmask)
        pvmask = df['metfilters_PrimaryVertexFilter']
        wgts.add('primaryvtx', pvmask)

        weight = wgts.weight()
        ########################

        leptonjets = JaggedCandidateArray.candidatesfromcounts(
            df['pfjet_p4'],
            px=df['pfjet_p4.fCoordinates.fX'].content,
            py=df['pfjet_p4.fCoordinates.fY'].content,
            pz=df['pfjet_p4.fCoordinates.fZ'].content,
            energy=df['pfjet_p4.fCoordinates.fT'].content,
            pfiso=df['pfjet_pfIsolationNoPU05'].content,
            mintkdist=df['pfjet_pfcands_minTwoTkDist'].content,
        )
        ljdautype = awkward.fromiter(df['pfjet_pfcand_type'])
        npfmu = (ljdautype==3).sum()
        ndsa = (ljdautype==8).sum()
        isegammajet = (npfmu==0)&(ndsa==0)
        ispfmujet = (npfmu>=2)&(ndsa==0)
        isdsajet = ndsa>0
        label = isegammajet.astype(int)*1+ispfmujet.astype(int)*2+isdsajet.astype(int)*3
        leptonjets.add_attributes(label=label, ndsa=ndsa)
        nmu = ((ljdautype==3)|(ljdautype==8)).sum()
        leptonjets.add_attributes(ismutype=(nmu>=2), iseltype=(nmu==0))
        ljdaucharge = awkward.fromiter(df['pfjet_pfcand_charge']).sum()
        leptonjets.add_attributes(qsum=ljdaucharge)
        leptonjets.add_attributes(isneutral=(leptonjets.iseltype | (leptonjets.ismutype&(leptonjets.qsum==0))))
        ## attribute: `nocosmic`
        ljdsamuFoundOppo = fromNestNestIndexArray(df['dsamuon_hasOppositeMuon'], awkward.fromiter(df['pfjet_pfcand_dsamuonIdx']))
        dtcscTime = fromNestNestIndexArray(df['dsamuon_timeDiffDTCSC'], awkward.fromiter(df['pfjet_pfcand_dsamuonIdx']))
        rpcTime = fromNestNestIndexArray(df['dsamuon_timeDiffRPC'], awkward.fromiter(df['pfjet_pfcand_dsamuonIdx']))
        if len(dtcscTime.flatten().flatten()):
            dtcscTime = dtcscTime[ljdsamuFoundOppo]
            rpcTime = rpcTime[ljdsamuFoundOppo]
        ljdsamuSubset = fromNestNestIndexArray(df['dsamuon_isSubsetFilteredCosmic1Leg'], awkward.fromiter(df['pfjet_pfcand_dsamuonIdx']))
        leptonjets.add_attributes(nocosmic=( ((dtcscTime<-20).sum()==0) & ((rpcTime<-7.5).sum()==0) & (ljdsamuSubset.sum()==0) ))
        leptonjets = leptonjets[(leptonjets.nocosmic)&(leptonjets.pt>30)&(leptonjets.mintkdist<30)]

        ## __ twoleptonjets__
        twoleptonjets = leptonjets.counts>=2
        dileptonjets = leptonjets[twoleptonjets]
        wgt = weight[twoleptonjets]

        if dileptonjets.size==0: return output
        lj0 = dileptonjets[dileptonjets.pt.argmax()]
        lj1 = dileptonjets[dileptonjets.pt.argsort()[:, 1:2]]

        ## channel def ##
        singleMuljEvents = dileptonjets.ismutype.sum()==1
        muljInLeading2Events = (lj0.ismutype | lj1.ismutype).flatten()
        channel_2mu2e = (singleMuljEvents&muljInLeading2Events).astype(int)*1

        doubleMuljEvents = dileptonjets.ismutype.sum()==2
        muljIsLeading2Events = (lj0.ismutype & lj1.ismutype).flatten()
        channel_4mu = doubleMuljEvents&muljIsLeading2Events
        ###########

        dphi = np.abs(lj0.p4.delta_phi(lj1.p4))

        if self.category == '00':
            channel_4mu = channel_4mu & ((lj0.isneutral & lj1.isneutral).flatten())
        elif self.category == '01':
            channel_4mu = channel_4mu & ((lj0.isneutral & (~lj1.isneutral)).flatten())
        elif self.category == '10':
            channel_4mu = channel_4mu & (((~lj0.isneutral) & lj1.isneutral).flatten())
        elif self.category == '11':
            channel_4mu = channel_4mu & (((~lj0.isneutral) & (~lj1.isneutral)).flatten())

        dphi = dphi[channel_4mu]
        wgt = wgt[channel_4mu]

        output['dphi'] += processor.column_accumulator(dphi[wgt.astype(bool)].flatten())

        return output

    def postprocess(self, accumulator):
        return accumulator



if __name__ == "__main__":
    import os
    import re
    from os.path import join, isdir, splitext
    from FireHydrant.Analysis.PlottingOptions import *

    reldir = splitext(__file__)[0].replace('_', '/')
    outdir = join(os.getenv('FH_BASE'), "Imgs", reldir)
    if not isdir(outdir): os.makedirs(outdir)

    SetROOTHistStyle()

    outputs = {}
    hists = {}

    CHOICES = {'0': 'neutral', '1': 'charged'}
    for c in ['00', '01', '10', '11']:
        outputs[c] = processor.run_uproot_job(dataDS,
                                        treename='ffNtuplizer/ffNtuple',
                                        processor_instance=LJPairDphi4Mu(category=c),
                                        executor=processor.futures_executor,
                                        executor_args=dict(workers=12, flatten=False),
                                        chunksize=500000,
                                        )

        hist_title = f'#Delta#phi(LJ0, LJ1), LJ0 {CHOICES[c[0]]} & LJ1 {CHOICES[c[1]]}'
        hists[c] = ROOT.TH1F(c, f'{hist_title};#Delta#phi;Counts', 20, 0, np.pi)

        print(f'--> Fill category {c}')
        for dphi_ in np.nditer([outputs[c]['dphi'].value,]):
            hists[c].Fill(dphi_)
            hists[c].SetMarkerStyle(20)
            hists[c].SetMarkerSize(0.6)
            hists[c].SetMarkerColor(4)

    print('--> Saving')
    c = ROOT.TCanvas('c', 'canvas', 700, 500)
    # c.SetLogy()
    for k, h in hists.items():
        h.Draw('E')
        c.Draw()
        with _setIgnoreLevel(ROOT.kError):
            c.SaveAs(f'{outdir}/ljpairDphi_chan4mu_{k}.png')
            c.SaveAs(f'{outdir}/ljpairDphi_chan4mu_{k}.pdf')
        c.Clear()

    print('--> Preserving')
    outrootf = ROOT.TFile(f'{outdir}/plots.root', 'RECREATE')
    outrootf.cd()
    for h in hists.values(): h.Write()
    outrootf.Close()


    # --------------------------------------------------------

    if args.sync:
        webserver = 'wsi@lxplus.cern.ch'
        if '/' in reldir:
            webdir = f'/eos/user/w/wsi/www/public/firehydrant/{reldir.rsplit("/", 1)[0]}'
            cmd = f'rsync -az --exclude ".*" --delete {outdir} --rsync-path="mkdir -p {webdir} && rsync" {webserver}:{webdir}'
        else:
            webdir = '/eos/user/w/wsi/www/public/firehydrant'
            cmd = f'rsync -az --exclude ".*" --delete {outdir} {webserver}:{webdir}'
        print(f"--> sync with: {webserver}:{webdir}")
        os.system(cmd)
