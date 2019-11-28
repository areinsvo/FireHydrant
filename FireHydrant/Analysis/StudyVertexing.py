#!/usr/bin/env python
"""leptonjet vertexing
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
from FireHydrant.Tools.correction import (get_nlo_weight_function,
                                          get_pu_weights_function,
                                          get_ttbar_weight)
from FireHydrant.Tools.metfilter import MetFilters
from FireHydrant.Tools.trigger import Triggers

np.seterr(divide='ignore', invalid='ignore', over='ignore')
plt.rcParams["savefig.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"


parser = argparse.ArgumentParser(description="leptonjet vertexing")
parser.add_argument("--sync", action='store_true', help="issue rsync command to sync plots folder to lxplus web server")
args = parser.parse_args()

dml = DatasetMapLoader()
bkgDS, bkgMAP, bkgSCALE = dml.fetch('bkg')
dataDS, dataMAP = dml.fetch('data')

sdml = SigDatasetMapLoader()
sigDS_2mu2e, sigSCALE_2mu2e = sdml.fetch('2mu2e')
sigDS_4mu, sigSCALE_4mu = sdml.fetch('4mu')


class LeptonjetVertexProcessor(processor.ProcessorABC):
    def __init__(self, region='SR', data_type='sig-2mu2e'):
        self.region = region
        self.data_type = data_type

        dataset_axis = hist.Cat('dataset', 'dataset')
        channel_axis = hist.Bin('channel', 'channel', 3, 0, 3)
        bool_axis = hist.Bin('boolean', 'true/false', 2, 0, 2)
        vxy_axis = hist.Bin('vxy', 'vxy [cm]', 100, 0, 20)

        self._accumulator = processor.dict_accumulator({
            'vertexgood': hist.Hist('Frequency', dataset_axis, channel_axis, bool_axis),
            'vxy': hist.Hist('Counts', dataset_axis, channel_axis, vxy_axis),
        })
        self.pucorrs = get_pu_weights_function()
        ## NOT applied for now
        self.nlo_w = get_nlo_weight_function('w')
        self.nlo_z = get_nlo_weight_function('z')

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, df):
        output = self.accumulator.identity()
        if df.size==0: return output

        dataset = df['dataset']
        ## construct weights ##
        wgts = processor.Weights(df.size)
        if self.data_type!='data':
            wgts.add('genw', df['weight'])
            npv = df['trueInteractionNum']
            wgts.add('pileup', *(f(npv) for f in self.pucorrs))

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
            vx=df['pfjet_klmvtx.fCoordinates.fX'],
            vy=df['pfjet_klmvtx.fCoordinates.fY'],
            vz=df['pfjet_klmvtx.fCoordinates.fZ'],
            ncands=df['pfjet_pfcands_n'],
        )
        leptonjets.add_attributes(vxy=np.hypot(leptonjets.vx, leptonjets.vy))
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

        # leptonjets = leptonjets[((~leptonjets.iseltype)|(leptonjets.iseltype&(leptonjets.pt>40)))] # EGM-type lj pt > 40
        # leptonjets = leptonjets[((~leptonjets.ismutype)|(leptonjets.ismutype&(leptonjets.pt>30)))] # Mu-type lj pt > 30

        ## __ twoleptonjets__
        twoleptonjets = (leptonjets.counts>=2)&(leptonjets.ismutype.sum()>=1)
        dileptonjets = leptonjets[twoleptonjets]
        wgt = weight[twoleptonjets]

        ## __Mu-type pt0>40__
        # mask_ = dileptonjets[dileptonjets.ismutype].pt.max()>40
        # dileptonjets = dileptonjets[mask_]
        # wgt = wgt[mask_]

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
        if self.region=='CR':
            dileptonjets = dileptonjets[isControl]
            wgt = wgt[isControl]
            lj0 = lj0[isControl]
            lj1 = lj1[isControl]
            channel_ = channel_[isControl]
        elif self.region=='SR':
            dileptonjets = dileptonjets[~isControl]
            wgt = wgt[~isControl]
            lj0 = lj0[~isControl]
            lj1 = lj1[~isControl]
            channel_ = channel_[~isControl]
        else:
            dileptonjets = dileptonjets
        if dileptonjets.size==0: return output

        # pfmu-type only
        pfmujets = dileptonjets[dileptonjets.label==2]
        pfmujets_ones = pfmujets.pt.ones_like()
        output['vertexgood'].fill(dataset=dataset, boolean=~np.isnan(pfmujets.vx.flatten()), channel=(channel_*pfmujets_ones).flatten(), weight=(wgt*pfmujets_ones).flatten())
        output['vxy'].fill(dataset=dataset, vxy=pfmujets.vxy.flatten(), channel=(channel_*pfmujets_ones).flatten(), weight=(wgt*pfmujets_ones).flatten())

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
            if self.data_type == 'sig-2mu2e':
                accumulator[k].scale(sigSCALE_2mu2e, axis='dataset')
            if self.data_type == 'sig-4mu':
                accumulator[k].scale(sigSCALE_4mu, axis='dataset')

        return accumulator

def vertexingProb(output, channel, issig=True):
    if channel=='2mu2e':
        channelBin = slice(1,2)
    if channel=='4mu':
        channelBin = slice(2,3)
    h_ = output['vertexgood'].integrate('channel', channelBin)
    d_ = { k[0]: v[1]/np.sum(v) for k, v in h_.values().items() }

    if issig:
        from FireHydrant.Analysis.Utils import sigsort
        for k, v in sorted(d_.items(), key=lambda t:sigsort(t[0])):
            print(f'{k:25} {v*100:.3f}%')
    else:
        for k, v in d_.items():
            print(f'{k:25} {v*100:.3f}%')



if __name__ == "__main__":
    import os
    from os.path import join, isdir, splitext

    outdir = join(os.getenv('FH_BASE'), "Imgs", splitext(__file__)[0])
    if not isdir(outdir): os.makedirs(outdir)


    out_sig2mu2e = processor.run_uproot_job(sigDS_2mu2e,
                                  treename='ffNtuplizer/ffNtuple',
                                  processor_instance=LeptonjetVertexProcessor(data_type='sig-2mu2e', region='all'),
                                  executor=processor.futures_executor,
                                  executor_args=dict(workers=12, flatten=True),
                                  chunksize=500000,
                                 )

    out_sig4mu = processor.run_uproot_job(sigDS_4mu,
                                  treename='ffNtuplizer/ffNtuple',
                                  processor_instance=LeptonjetVertexProcessor(data_type='sig-4mu', region='all'),
                                  executor=processor.futures_executor,
                                  executor_args=dict(workers=12, flatten=True),
                                  chunksize=500000,
                                 )

    out_bkg = processor.run_uproot_job(bkgDS,
                                  treename='ffNtuplizer/ffNtuple',
                                  processor_instance=LeptonjetVertexProcessor(data_type='bkg', region='all'),
                                  executor=processor.futures_executor,
                                  executor_args=dict(workers=12, flatten=True),
                                  chunksize=500000,
                                 )

    out_data = processor.run_uproot_job(dataDS,
                                  treename='ffNtuplizer/ffNtuple',
                                  processor_instance=LeptonjetVertexProcessor(region='CR', data_type='data'),
                                  executor=processor.futures_executor,
                                  executor_args=dict(workers=12, flatten=True),
                                  chunksize=500000,
                                 )

    from FireHydrant.Analysis.PlottingOptions import *
    import re
    smallmxx = re.compile('mXX-(100|150|200)_\w+')

    ## CHANNEL - 2mu2e
    print('## CHANNEL - 2mu2e')
    vertexingProb(out_sig2mu2e, '2mu2e')
    vertexingProb(out_bkg, '2mu2e', issig=False)
    vertexingProb(out_data, '2mu2e', issig=False)

    fig, axes = plt.subplots(1,2,figsize=(16,6))
    fig.subplots_adjust(wspace=0.15)
    bkghist = out_bkg['vxy'].integrate('channel', slice(1,2))
    hist.plot1d(bkghist, overlay='cat', ax=axes[0], stack=True, overflow='over',
                line_opts=None, fill_opts=fill_opts, error_opts=error_opts)
    sighist = out_sig2mu2e['vxy'][smallmxx].integrate('channel', slice(1,2))
    hist.plot1d(sighist, overlay='dataset', ax=axes[0], overflow='over', clear=False)
    datahist = out_data['vxy'].integrate('channel', slice(1,2))
    hist.plot1d(datahist, overlay='cat', ax=axes[1], overflow='over', error_opts=data_err_opts)
    axes[0].set_title('[2mu2e|AR] leptonjet pfmu-type vxy', x=0.0, ha="left")
    axes[1].set_title('[2mu2e|CR] leptonjet pfmu-type vxy', x=0.0, ha="left")
    axes[0].legend(*groupHandleLabel(axes[0]), prop={'size': 8,}, ncol=3)

    for ax in axes:
        ax.set_yscale('log')
        ax.autoscale(axis='both', tight=True)
        ax.text(1,1,'59.74/fb (13TeV)', ha='right', va='bottom', transform=ax.transAxes)
        ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
        ax.set_ylabel(ax.get_ylabel(), y=1.0, ha="right")
        ax.vlines([3,], 0, 1, linestyles='dashed', transform=ax.get_xaxis_transform())

    fig.savefig(join(outdir, 'vxy_pfmu_2mu2e.png'))
    fig.savefig(join(outdir, 'vxy_pfmu_2mu2e.pdf'))
    plt.close(fig)

    ## CHANNEL - 4mu
    print('## CHANNEL - 4mu')
    vertexingProb(out_sig4mu, '4mu')
    vertexingProb(out_bkg, '4mu', issig=False)
    vertexingProb(out_data, '4mu', issig=False)

    fig, axes = plt.subplots(1,2,figsize=(16,6))
    fig.subplots_adjust(wspace=0.15)
    bkghist = out_bkg['vxy'].integrate('channel', slice(2,3))
    hist.plot1d(bkghist, overlay='cat', ax=axes[0], stack=True, overflow='over',
                line_opts=None, fill_opts=fill_opts, error_opts=error_opts)
    sighist = out_sig4mu['vxy'][smallmxx].integrate('channel', slice(2,3))
    hist.plot1d(sighist, overlay='dataset', ax=axes[0], overflow='over', clear=False)
    datahist = out_data['vxy'].integrate('channel', slice(2,3))
    hist.plot1d(datahist, overlay='cat', ax=axes[1], overflow='over', error_opts=data_err_opts)
    axes[0].set_title('[4mu|AR] leptonjet pfmu-type vxy', x=0.0, ha="left")
    axes[1].set_title('[4mu|CR] leptonjet pfmu-type vxy', x=0.0, ha="left")
    axes[0].legend(*groupHandleLabel(axes[0]), prop={'size': 8,}, ncol=3)

    for ax in axes:
        ax.set_yscale('log')
        ax.autoscale(axis='both', tight=True)
        ax.text(1,1,'59.74/fb (13TeV)', ha='right', va='bottom', transform=ax.transAxes)
        ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
        ax.set_ylabel(ax.get_ylabel(), y=1.0, ha="right")
        ax.vlines([3,], 0, 1, linestyles='dashed', transform=ax.get_xaxis_transform())

    fig.savefig(join(outdir, 'vxy_pfmu_4mu.png'))
    fig.savefig(join(outdir, 'vxy_pfmu_4mu.pdf'))
    plt.close(fig)

    if args.sync:
        webdir = 'wsi@lxplus.cern.ch:/eos/user/w/wsi/www/public/firehydrant'
        cmd = f'rsync -az --exclude ".*" --delete {outdir} {webdir}'
        print(f"--> sync with: {webdir}")
        os.system(cmd)
