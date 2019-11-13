#!/usr/bin/env python
"""
leptonjet leading/subleading pT, eta
"""
import argparse

import awkward
import coffea.processor as processor
import numpy as np
import matplotlib.pyplot as plt
from coffea import hist
from coffea.analysis_objects import JaggedCandidateArray
from FireHydrant.Analysis.DatasetMapLoader import (DatasetMapLoader,
                                                   SigDatasetMapLoader)
from FireHydrant.Tools.metfilter import MetFilters
from FireHydrant.Tools.trigger import Triggers

np.seterr(divide='ignore', invalid='ignore', over='ignore')
plt.rcParams["savefig.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"


parser = argparse.ArgumentParser(description="leptonjet leading/subleading pt, eta")
parser.add_argument("--sync", action='store_true', help="issue rsync command to sync plots folder to lxplus web server")
args = parser.parse_args()

dml = DatasetMapLoader()
bkgDS, bkgMAP, bkgSCALE = dml.fetch('bkg')
dataDS, dataMAP = dml.fetch('data')

sdml = SigDatasetMapLoader()
sigDS, sigSCALE = sdml.fetch('simple')


"""Leptonjet leading/subleading pT, eta"""
class LeptonjetLeadSubleadProcessor(processor.ProcessorABC):
    def __init__(self, dphi_control=False, data_type='bkg'):
        self.dphi_control = dphi_control
        self.data_type = data_type

        dataset_axis = hist.Cat('dataset', 'dataset')
        pt_axis = hist.Bin('pt', '$p_T$ [GeV]', 100, 0, 200)
        eta_axis = hist.Bin('eta', 'eta', 100, -2.5, 2.5)
        channel_axis = hist.Bin('channel', 'channel', 3, 0, 3)

        self._accumulator = processor.dict_accumulator({
            'pt0': hist.Hist('Counts', dataset_axis, pt_axis, channel_axis),
            'pt1': hist.Hist('Counts', dataset_axis, pt_axis, channel_axis),
            'eta0': hist.Hist('Counts', dataset_axis, eta_axis, channel_axis),
            'eta1': hist.Hist('Counts', dataset_axis, eta_axis, channel_axis),
        })

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, df):
        output = self.accumulator.identity()
        if df.size==0: return output

        dataset = df['dataset']

        ## construct weights ##
        wgts = processor.Weights(df.size)
        if len(dataset)!=1:
            wgts.add('genw', df['weight'])

        triggermask = np.logical_or.reduce([df[t] for t in Triggers])
        wgts.add('trigger', triggermask)
        cosmicpairmask = df['cosmicveto_result']
        wgts.add('cosmicveto', cosmicpairmask)
        pvmask = df['metfilters_PrimaryVertexFilter']
        wgts.add('primaryvtx', pvmask)
        # ...bla bla, other weights goes here

        weight = wgts.weight()
        ########################


        leptonjets = JaggedCandidateArray.candidatesfromcounts(
            df['pfjet_p4'],
            px=df['pfjet_p4.fCoordinates.fX'],
            py=df['pfjet_p4.fCoordinates.fY'],
            pz=df['pfjet_p4.fCoordinates.fZ'],
            energy=df['pfjet_p4.fCoordinates.fT'],
            pfisoAll05=df['pfjet_pfIsolation05'],
            pfisoNopu05=df['pfjet_pfIsolationNoPU05'],
            pfisoDbeta=df['pfjet_pfiso'],
            ncands=df['pfjet_pfcands_n'],
        )
        ljdautype = awkward.fromiter(df['pfjet_pfcand_type'])
        npfmu = (ljdautype==3).sum()
        ndsa = (ljdautype==8).sum()
        isegammajet = (npfmu==0)&(ndsa==0)
        ispfmujet = (npfmu>=2)&(ndsa==0)
        isdsajet = ndsa>0
        label = isegammajet.astype(int)*1+ispfmujet.astype(int)*2+isdsajet.astype(int)*3
        leptonjets.add_attributes(label=label)
        nmu = ((ljdautype==3)|(ljdautype==8)).sum()
        leptonjets.add_attributes(ismutype=(nmu>=2), iseltype=(nmu==0))

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
        channel_4mu = (doubleMuljEvents&muljIsLeading2Events).astype(int)*2

        channel_ = channel_2mu2e + channel_4mu
        ###########

        isControl = (np.abs(lj0.p4.delta_phi(lj1.p4))<np.pi/2).flatten()

        ## __isControl__
        if self.dphi_control:
            leptonjets_ = dileptonjets[isControl]
            wgt = wgt[isControl]
            lj0 = lj0[isControl]
            lj1 = lj1[isControl]
            channel_ = channel_[isControl]
        else:
            leptonjets_ = dileptonjets
        if leptonjets_.size==0: return output

        output['pt0'].fill(dataset=dataset, pt=lj0.pt.flatten(), channel=channel_, weight=wgt)
        output['pt1'].fill(dataset=dataset, pt=lj1.pt.flatten(), channel=channel_, weight=wgt)
        output['eta0'].fill(dataset=dataset, eta=lj0.eta.flatten(), channel=channel_, weight=wgt)
        output['eta1'].fill(dataset=dataset, eta=lj1.eta.flatten(), channel=channel_, weight=wgt)

        return output

    def postprocess(self, accumulator):
        origidentity = list(accumulator)
        for k in origidentity:
            if self.data_type == 'bkg':
                accumulator[k].scale(bkgSCALE, axis='dataset')
                accumulator[k] = accumulator[k].group("dataset",
                                                    hist.Cat("cat", "datasets", sorting='integral'),
                                                    bkgMAP)
            if self.data_type == 'data':
                accumulator[k] = accumulator[k].group("dataset",
                                                    hist.Cat("cat", "datasets",),
                                                    dataMAP)
            if self.data_type == 'sig':
                accumulator[k].scale(sigSCALE, axis='dataset')
        return accumulator


def plot_leadSublead(histo, varname, datatype):

    h_2mu2e = histo.integrate('channel', slice(1,2))
    h_4mu = histo.integrate('channel', slice(2,3))

    fig, axes = plt.subplots(1,2,figsize=(14,5))
    if datatype == 'sig':
        hist.plot1d(h_2mu2e, overlay='dataset', ax=axes[0], density=True)
        hist.plot1d(h_4mu, overlay='dataset', ax=axes[1], density=True)
    if datatype == 'bkg':
        hist.plot1d(h_2mu2e, overlay='cat', ax=axes[0], density=True)
        hist.plot1d(h_4mu, overlay='cat', ax=axes[1], density=True)
    if datatype == 'data':
        hist.plot1d(h_2mu2e.sum('cat'), ax=axes[0], density=True)
        hist.plot1d(h_4mu.sum('cat'), ax=axes[1], density=True)

    axes[0].set_title(f'[2mu2e] leptonJet {varname}', x=0.0, ha="left")
    axes[1].set_title(f'[4mu] leptonJet {varname}', x=0.0, ha="left")
    for ax in axes:
        ax.text(1,1,'59.74/fb (13TeV)', ha='right', va='bottom', transform=ax.transAxes)
        ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
        ax.set_ylabel(ax.get_ylabel(), y=1.0, ha="right")
    return fig, axes


if __name__ == "__main__":
    import os
    from os.path import join, isdir

    outdir = join(os.getenv('FH_BASE'), "Imgs", __file__.split('.')[0])
    if not isdir(outdir): os.makedirs(outdir)

    outputs = {}
    outputs['bkg'] = processor.run_uproot_job(bkgDS,
                                  treename='ffNtuplizer/ffNtuple',
                                  processor_instance=LeptonjetLeadSubleadProcessor(data_type='bkg'),
                                  executor=processor.futures_executor,
                                  executor_args=dict(workers=12, flatten=True),
                                  chunksize=500000,
                                 )
    outputs['data'] = processor.run_uproot_job(dataDS,
                                  treename='ffNtuplizer/ffNtuple',
                                  processor_instance=LeptonjetLeadSubleadProcessor(dphi_control=True, data_type='data'),
                                  executor=processor.futures_executor,
                                  executor_args=dict(workers=12, flatten=True),
                                  chunksize=500000,
                                 )
    outputs['sig'] = processor.run_uproot_job(sigDS,
                                  treename='ffNtuplizer/ffNtuple',
                                  processor_instance=LeptonjetLeadSubleadProcessor(dphi_control=False, data_type='sig'),
                                  executor=processor.futures_executor,
                                  executor_args=dict(workers=12, flatten=True),
                                  chunksize=500000,
                                 )

    for t, output in outputs.items():
        for k, h in output.items():
            fig, axes = plot_leadSublead(h, k, t)
            fig.savefig(join(outdir, f"{k}__{t}.png"))
            fig.savefig(join(outdir, f"{k}__{t}.pdf"))
            plt.close(fig)

    if args.sync:
        webdir = 'wsi@lxplus.cern.ch:/eos/user/w/wsi/www/public/firehydrant'
        cmd = f'rsync -az --exclude ".*" --delete {outdir} {webdir}'
        print(f"--> sync with: {webdir}")
        os.system(cmd)
