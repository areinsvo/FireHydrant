#!/usr/bin/env python
"""For AN
cosmic removal for leptonjets
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
from FireHydrant.Tools.uproothelpers import fromNestNestIndexArray

np.seterr(divide='ignore', invalid='ignore', over='ignore')
plt.rcParams["savefig.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"


parser = argparse.ArgumentParser(description="[AN] leptonjet pair delta phi")
parser.add_argument("--sync", action='store_true', help="issue rsync command to sync plots folder to lxplus web server")
args = parser.parse_args()

# dml = DatasetMapLoader()
# bkgDS, bkgMAP, bkgSCALE = dml.fetch('bkg')
# dataDS, dataMAP = dml.fetch('data')

sdml = SigDatasetMapLoader()
sigDS, sigSCALE = sdml.fetch('all')

class LJCosmicProcessor(processor.ProcessorABC):
    def __init__(self):
        dataset_axis = hist.Cat('dataset', 'dataset')
        count_axis = hist.Bin('num', 'count', 20, 0, 20)
        frac_axis = hist.Bin('frac', 'fraction', 2, 0, 2)
        time_axis = hist.Bin('t', 'time [ns]', 100, -50, 50)
        self._accumulator = processor.dict_accumulator({
            'npairs': hist.Hist('Norm. Frequency', dataset_axis, count_axis),
            'ljdsaOppo': hist.Hist('Fraction', dataset_axis, frac_axis),
            'dtcscTime': hist.Hist("Norm. Frequency", dataset_axis, time_axis),
            'rpcTime': hist.Hist("Norm. Frequency", dataset_axis, time_axis),
            'ljdsaSubset': hist.Hist('Fraction', dataset_axis, frac_axis),
        })

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, df):
        output = self.accumulator.identity()
        dataset = df['dataset']

        parallelpairs = df['cosmicveto_parallelpairs']
        leptonjets = JaggedCandidateArray.candidatesfromcounts(
            df['pfjet_p4'],
            px=df['pfjet_p4.fCoordinates.fX'].content,
            py=df['pfjet_p4.fCoordinates.fY'].content,
            pz=df['pfjet_p4.fCoordinates.fZ'].content,
            energy=df['pfjet_p4.fCoordinates.fT'].content,
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
        ljdaucharge = awkward.fromiter(df['pfjet_pfcand_charge']).sum()
        leptonjets.add_attributes(qsum=ljdaucharge)
        leptonjets.add_attributes(isneutral=(leptonjets.iseltype | (leptonjets.ismutype&(leptonjets.qsum==0))))

        ljdsamuFoundOppo = fromNestNestIndexArray(df['dsamuon_hasOppositeMuon'], awkward.fromiter(df['pfjet_pfcand_dsamuonIdx']))
        dtcscTime = fromNestNestIndexArray(df['dsamuon_timeDiffDTCSC'], awkward.fromiter(df['pfjet_pfcand_dsamuonIdx']))[ljdsamuFoundOppo]
        rpcTime = fromNestNestIndexArray(df['dsamuon_timeDiffRPC'], awkward.fromiter(df['pfjet_pfcand_dsamuonIdx']))[ljdsamuFoundOppo]
        ljdsamuSubset = fromNestNestIndexArray(df['dsamuon_isSubsetFilteredCosmic1Leg'], awkward.fromiter(df['pfjet_pfcand_dsamuonIdx']))

        leptonjets = leptonjets[leptonjets.isneutral]
        ljdsamuFoundOppo = ljdsamuFoundOppo[leptonjets.isneutral]
        dtcscTime = dtcscTime[leptonjets.isneutral]
        rpcTime = rpcTime[leptonjets.isneutral]
        ljdsamuSubset = ljdsamuSubset[leptonjets.isneutral]

        ## __ twoleptonjets__
        twoleptonjets = leptonjets.counts>=2
        dileptonjets = leptonjets[twoleptonjets]
        parallelpairs = parallelpairs[twoleptonjets]
        ljdsamuFoundOppo = ljdsamuFoundOppo[twoleptonjets]
        dtcscTime = dtcscTime[twoleptonjets]
        rpcTime = rpcTime[twoleptonjets]
        ljdsamuSubset = ljdsamuSubset[twoleptonjets]

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

        output['npairs'].fill(dataset=dataset, num=parallelpairs[channel_>0])
        output['ljdsaOppo'].fill(dataset=dataset, frac=ljdsamuFoundOppo[channel_>0].flatten().flatten())
        output['dtcscTime'].fill(dataset=dataset, t=dtcscTime[channel_>0].flatten().flatten())
        output['rpcTime'].fill(dataset=dataset, t=rpcTime[channel_>0].flatten().flatten())
        output['ljdsaSubset'].fill(dataset=dataset, frac=ljdsamuSubset[channel_>0].flatten().flatten())

        return output

    def postprocess(self, accumulator):
        return accumulator


if __name__ == "__main__":
    import os
    from os.path import join, isdir, splitext
    from FireHydrant.Analysis.PlottingOptions import *

    reldir = splitext(__file__)[0].replace('_', '/')
    outdir = join(os.getenv('FH_BASE'), "Imgs", reldir)
    if not isdir(outdir): os.makedirs(outdir)

    import re

    output = processor.run_uproot_job(sigDS,
                                    treename='ffNtuplizer/ffNtuple',
                                    processor_instance=LJCosmicProcessor(),
                                    executor=processor.futures_executor,
                                    executor_args=dict(workers=12, flatten=False),
                                    chunksize=500000,
                                    )


    channel_2mu2e = re.compile('2mu2e.*_lxy-300')
    channel_4mu = re.compile('4mu.*_lxy-300')

    # ----------------------------------------------------------
    ## N parallel pair DSA in event

    fig, ax = plt.subplots(figsize=(8,6))
    hist.plot1d(output['npairs'][channel_2mu2e].sum('dataset'), ax=ax, overflow='none', density=True)
    hs_2mu2e = tuple(ax.get_legend_handles_labels()[0])
    hist.plot1d(output['npairs'][channel_4mu].sum('dataset'), ax=ax, overflow='none', density=True, clear=False)
    hs_ = ax.get_legend_handles_labels()[0]
    hs_4mu = tuple([h for h in hs_ if h not in hs_2mu2e])

    frac_2mu2e = output['npairs'][channel_2mu2e].sum('dataset').integrate('num', slice(10, 20)).values()[()]/output['npairs'][channel_2mu2e].sum('dataset', 'num').values()[()]
    frac_4mu   = output['npairs'][channel_4mu].sum('dataset').integrate('num', slice(10, 20)).values()[()]/output['npairs'][channel_4mu].sum('dataset', 'num').values()[()]

    ax.set_title(r'[signalMC|lxy300cm] Number of parallel pairs of DSAMu ($|cos(\alpha)|>0.99$)', x=0, ha='left')
    ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
    ax.set_ylabel(ax.get_ylabel()+'/1', y=1.0, ha="right")
    ax.set_yscale('log')
    ax.autoscale(axis='both', tight=True)
    ax.set_xticks(np.arange(0, 21, 2))
    ax.vlines([10,], 0, 1, linestyles='dashed', transform=ax.get_xaxis_transform())
    ax.legend([hs_4mu, hs_2mu2e], [f'4mu (N>10: {frac_4mu*100:.2f}%)', f'2mu2e (N>10: {frac_2mu2e*100:.2f}%)'])

    fig.savefig(join(outdir, 'numParallelPairs.png'))
    fig.savefig(join(outdir, 'numParallelPairs.pdf'))
    plt.close(fig)

    # ----------------------------------------------------------
    ## Fraction of DSA in leptonjets with anti-parallel companion

    fig, ax = plt.subplots(figsize=(8,6))
    hist.plot1d(output['ljdsaOppo'][channel_2mu2e].sum('dataset'), ax=ax, overflow='none', density=True)
    hs_2mu2e = tuple(ax.get_legend_handles_labels()[0])
    hist.plot1d(output['ljdsaOppo'][channel_4mu].sum('dataset'), ax=ax, overflow='none', density=True, clear=False)
    hs_ = ax.get_legend_handles_labels()[0]
    hs_4mu = tuple([h for h in hs_ if h not in hs_2mu2e])

    histval_2mu2e = output['ljdsaOppo'][channel_2mu2e].sum('dataset').values()[()]
    frac_2mu2e = histval_2mu2e[1]/histval_2mu2e.sum()
    histval_4mu = output['ljdsaOppo'][channel_4mu].sum('dataset').values()[()]
    frac_4mu = histval_4mu[1]/histval_4mu.sum()

    ax.set_title('[signalMC|lxy300cm] DSAMu in leptonjets\nhas anti-parallel companion ({})'.format(r'$cos(\alpha)<-0.99$'), x=0, ha='left')
    ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
    ax.set_ylabel(ax.get_ylabel(), y=1.0, ha="right")
    ax.set_yscale('log')
    ax.autoscale(axis='both', tight=True)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(['False', 'True'])
    ax.legend([hs_4mu, hs_2mu2e], [f'4mu ({frac_4mu*100:.2f}% found true)', f'2mu2e ({frac_2mu2e*100:.2f}% found true)'])

    fig.savefig(join(outdir, 'fracAntiparaDSA.png'))
    fig.savefig(join(outdir, 'fracAntiparaDSA.pdf'))
    plt.close(fig)

    # ----------------------------------------------------------
    ## DT/CSC timing of DSA in leptonjet with opposite companion

    fig, ax = plt.subplots(figsize=(8,6))
    hist.plot1d(output['dtcscTime'][channel_2mu2e].sum('dataset'), ax=ax, overflow='none', density=True)
    hs_2mu2e = tuple(ax.get_legend_handles_labels()[0])
    hist.plot1d(output['dtcscTime'][channel_4mu].sum('dataset'), ax=ax, overflow='none', density=True, clear=False)
    hs_ = ax.get_legend_handles_labels()[0]
    hs_4mu = tuple([h for h in hs_ if h not in hs_2mu2e])

    frac_2mu2e = output['dtcscTime'][channel_2mu2e].sum('dataset').integrate('t', slice(-50, -20)).values()[()]/output['dtcscTime'][channel_2mu2e].sum('dataset', 't').values()[()]
    frac_4mu   = output['dtcscTime'][channel_4mu].sum('dataset').integrate('t', slice(-50, -20)).values()[()]/output['dtcscTime'][channel_4mu].sum('dataset', 't').values()[()]

    ax.set_title('[signalMC|lxy300cm] DT/CSC timing of DSAMu with opposite companion\n({}) in leptonjets'.format(r'$cos(\alpha)<-0.99$'), x=0, ha='left')
    ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
    ax.set_ylabel(ax.get_ylabel()+'/1', y=1.0, ha="right")
    ax.autoscale(axis='both', tight=True)
    ax.vlines([-20,], 0, 1, linestyles='dashed', transform=ax.get_xaxis_transform())
    ax.legend([hs_4mu, hs_2mu2e], [f'4mu (t<-20: {frac_4mu*100:.2f}%)', f'2mu2e (t<-20: {frac_2mu2e*100:.2f}%)'])

    fig.savefig(join(outdir, 'dtcscTimingDSAOppo.png'))
    fig.savefig(join(outdir, 'dtcscTimingDSAOppo.pdf'))
    plt.close(fig)

    # ----------------------------------------------------------
    ## RPC timing of DSA in leptonjet with opposite companion

    fig, ax = plt.subplots(figsize=(8,6))
    hist.plot1d(output['rpcTime'][channel_2mu2e].sum('dataset'), ax=ax, overflow='none', density=True)
    hs_2mu2e = tuple(ax.get_legend_handles_labels()[0])
    hist.plot1d(output['rpcTime'][channel_4mu].sum('dataset'), ax=ax, overflow='none', density=True, clear=False)
    hs_ = ax.get_legend_handles_labels()[0]
    hs_4mu = tuple([h for h in hs_ if h not in hs_2mu2e])

    frac_2mu2e = output['rpcTime'][channel_2mu2e].sum('dataset').integrate('t', slice(-50, -7.5)).values()[()]/output['rpcTime'][channel_2mu2e].sum('dataset', 't').values()[()]
    frac_4mu   = output['rpcTime'][channel_4mu].sum('dataset').integrate('t', slice(-50, -7.5)).values()[()]/output['rpcTime'][channel_4mu].sum('dataset', 't').values()[()]

    ax.set_title('[signalMC|lxy300cm] RPC timing of DSAMu with opposite companion\n({}) in leptonjets'.format(r'$cos(\alpha)<-0.99$'), x=0, ha='left')
    ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
    ax.set_ylabel(ax.get_ylabel()+'/1', y=1.0, ha="right")
    ax.autoscale(axis='both', tight=True)
    ax.vlines([-7.5,], 0, 1, linestyles='dashed', transform=ax.get_xaxis_transform())
    ax.legend([hs_4mu, hs_2mu2e], [f'4mu (t<-7.5: {frac_4mu*100:.2f}%)', f'2mu2e (t<-7.5: {frac_2mu2e*100:.2f}%)'])

    fig.savefig(join(outdir, 'rpcTimingDSAOppo.png'))
    fig.savefig(join(outdir, 'rpcTimingDSAOppo.pdf'))
    plt.close(fig)

    # ----------------------------------------------------------
    ## Fraction of DSA in leptonjets subset of cosmicMuon1Leg

    fig, ax = plt.subplots(figsize=(8,6))
    hist.plot1d(output['ljdsaSubset'][channel_2mu2e].sum('dataset'), ax=ax, overflow='none', density=True)
    hs_2mu2e = tuple(ax.get_legend_handles_labels()[0])
    hist.plot1d(output['ljdsaSubset'][channel_4mu].sum('dataset'), ax=ax, overflow='none', density=True, clear=False)
    hs_ = ax.get_legend_handles_labels()[0]
    hs_4mu = tuple([h for h in hs_ if h not in hs_2mu2e])

    histval_2mu2e = output['ljdsaSubset'][channel_2mu2e].sum('dataset').values()[()]
    frac_2mu2e = histval_2mu2e[1]/histval_2mu2e.sum()
    histval_4mu = output['ljdsaSubset'][channel_4mu].sum('dataset').values()[()]
    frac_4mu = histval_4mu[1]/histval_4mu.sum()

    ax.set_title('[signalMC|lxy300cm] DSAMu in leptonjets with segments subset\nof filtered cosmicMuon1Leg', x=0, ha='left')
    ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
    ax.set_ylabel(ax.get_ylabel(), y=1.0, ha="right")
    ax.set_yscale('log')
    ax.autoscale(axis='both', tight=True)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(['False', 'True'])
    ax.legend([hs_4mu, hs_2mu2e], [f'4mu ({frac_4mu*100:.2f}% found true)', f'2mu2e ({frac_2mu2e*100:.2f}% found true)'])

    fig.savefig(join(outdir, 'fracSubsetCosmic1LegDSA.png'))
    fig.savefig(join(outdir, 'fracSubsetCosmic1LegDSA.pdf'))
    plt.close(fig)


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
