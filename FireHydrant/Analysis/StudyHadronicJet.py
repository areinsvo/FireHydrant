#!/usr/bin/env python
"""
leptonjet pT w/ hadronic jet multiplicity splitting of 4.
"""
import argparse

import awkward
import coffea.processor as processor
import matplotlib.pyplot as plt
import numpy as np
import uproot
from coffea import hist
from coffea.analysis_objects import JaggedCandidateArray
from FireHydrant.Analysis.DatasetMapLoader import DatasetMapLoader
from FireHydrant.Tools.correction import (get_nlo_weight_function,
                                          get_pu_weights_function,
                                          get_ttbar_weight)
from FireHydrant.Tools.metfilter import MetFilters
from FireHydrant.Tools.trigger import Triggers

np.seterr(divide='ignore', invalid='ignore', over='ignore')
plt.rcParams["savefig.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"


parser = argparse.ArgumentParser(description="produce leptonjet pt w/ hadronic jet mutliplicity splitting")
parser.add_argument("--sync", action='store_true', help="issue rsync command to sync plots folder to lxplus web server")
args = parser.parse_args()


dml = DatasetMapLoader()
bkgDS, bkgMAP, bkgSCALE = dml.fetch('bkg')
dataDS, dataMAP = dml.fetch('data')


"""Leptonjet hadronic jet splitting"""
class LeptonjetHadronicjetProcessor(processor.ProcessorABC):
    def __init__(self, dphi_control=False, data_type='bkg'):
        self.dphi_control = dphi_control
        self.data_type = data_type

        dataset_axis = hist.Cat('dataset', 'dataset')
        pt_axis = hist.Bin('pt', '$p_T$ [GeV]', 60, 0, 300)
        njet_axis = hist.Bin('njet', 'multiplicity', 10, 0, 10)
        channel_axis = hist.Bin('channel', 'channel', 3, 0, 3)

        self._accumulator = processor.dict_accumulator({
            'pt': hist.Hist('Counts', dataset_axis, pt_axis, channel_axis, njet_axis),
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

        ak4jets = JaggedCandidateArray.candidatesfromcounts(
            df['akjet_ak4PFJetsCHS_p4'],
            px=df['akjet_ak4PFJetsCHS_p4.fCoordinates.fX'],
            py=df['akjet_ak4PFJetsCHS_p4.fCoordinates.fY'],
            pz=df['akjet_ak4PFJetsCHS_p4.fCoordinates.fZ'],
            energy=df['akjet_ak4PFJetsCHS_p4.fCoordinates.fT'],
            jetid=df['akjet_ak4PFJetsCHS_jetid'],
        )
        ak4jets=ak4jets[ak4jets.jetid&(ak4jets.pt>20)&(np.abs(ak4jets.eta)<2.5)]

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
        ak4jets = ak4jets[twoleptonjets]
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
            dileptonjets = dileptonjets[isControl]
            ak4jets = ak4jets[isControl]
            wgt = wgt[isControl]
            lj0 = lj0[isControl]
            lj1 = lj1[isControl]
            channel_ = channel_[isControl]
        else:
            dileptonjets = dileptonjets
        if dileptonjets.size==0: return output

        ljones = dileptonjets.pt.ones_like()
        output['pt'].fill(dataset=dataset, pt=dileptonjets.pt.flatten(),
                          channel=(ljones*channel_).flatten(),
                          njet=(ljones*ak4jets.counts).flatten(),
                          weight=(ljones*wgt).flatten())

        return output

    def postprocess(self, accumulator):
        origidentity = list(accumulator)
        for k in origidentity:
            if self.data_type == 'bkg':
                accumulator[k].scale(bkgSCALE, axis='dataset')
                accumulator[k+'_cat'] = accumulator[k].group("dataset",
                                                    hist.Cat("cat", "datasets", sorting='integral'),
                                                    bkgMAP)
            if self.data_type == 'data':
                accumulator[k+'_cat'] = accumulator[k].group("dataset",
                                                    hist.Cat("cat", "datasets",),
                                                    dataMAP)
        return accumulator



def plot_datamc(outputs):
    bkg, data = outputs['bkg'], outputs['data']

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

    CHANNELS = ['2mu2e', '4mu']

    res = {}
    for i, chan in enumerate(CHANNELS, start=1):
        res[chan] = fig, (axes, raxes) = plt.subplots(2, 2, figsize=(16,8), gridspec_kw={"height_ratios": (4, 1)}, sharex=True)
        fig.subplots_adjust(hspace=.07, wspace=0.1)

        hist.plot1d(bkg['pt_cat'].integrate('channel', slice(i,i+1)).integrate('njet', slice(0,4)),
                    overlay='cat', ax=axes[0], stack=True, overflow='over',
                    line_opts=None, fill_opts=fill_opts, error_opts=error_opts
                    )
        hist.plot1d(data['pt_cat'].integrate('channel', slice(i,i+1)).integrate('njet', slice(0,4)),
                    overlay='cat', ax=axes[0], overflow='over', clear=False,
                    error_opts=data_err_opts
                    )

        hist.plot1d(bkg['pt_cat'].integrate('channel', slice(i,i+1)).integrate('njet', slice(4,10), overflow='over'),
                    overlay='cat', ax=axes[1], stack=True, overflow='over',
                    line_opts=None, fill_opts=fill_opts, error_opts=error_opts
                    )
        hist.plot1d(data['pt_cat'].integrate('channel', slice(i,i+1)).integrate('njet', slice(4,10), overflow='over'),
                    overlay='cat', ax=axes[1], overflow='over', clear=False,
                    error_opts=data_err_opts
                    )
        for ax in axes:
            ax.autoscale(axis='both', tight=True)
            ax.set_yscale('symlog')
            ax.set_xlabel(None)
            ax.set_ylabel(ax.get_ylabel(), y=1.0, ha="right")
            ax.text(1,1,'59.74/fb (13TeV)', ha='right', va='bottom', transform=ax.transAxes)
        axes[0].set_title(f'[{chan}|CR] leptonJets pT, N(AK4PFCHS)<4', x=0.0, ha="left")
        axes[1].set_title(f'[{chan}|CR] leptonJets pT, N(AK4PFCHS)>=4', x=0.0, ha="left")

        hist.plotratio(data['pt_cat'].integrate('channel', slice(i,i+1)).integrate('njet', slice(0,4)).sum('cat'),
                    bkg['pt_cat'].integrate('channel', slice(i,i+1)).integrate('njet', slice(0,4)).sum('cat'),
                    ax=raxes[0], overflow='over', error_opts=data_err_opts, unc='num',
                    denom_fill_opts={}, guide_opts={}
                    )
        hist.plotratio(data['pt_cat'].integrate('channel', slice(i,i+1)).integrate('njet', slice(4,10), overflow='over').sum('cat'),
                    bkg['pt_cat'].integrate('channel', slice(i,i+1)).integrate('njet', slice(4,10), overflow='over').sum('cat'),
                    ax=raxes[1], overflow='over', error_opts=data_err_opts, unc='num',
                    denom_fill_opts={}, guide_opts={}
                    )
        for rax in raxes:
            rax.set_ylabel('Data/MC')
            rax.set_ylim(0, 2)
            rax.set_xlabel(rax.get_xlabel(), x=1.0, ha="right")

    return res


if __name__ == "__main__":
    import os
    from os.path import join, isdir

    outdir = join(os.getenv('FH_BASE'), "Imgs", __file__.split('.')[0])
    if not isdir(outdir): os.makedirs(outdir)

    outputs = {}
    outputs['bkg'] = processor.run_uproot_job(bkgDS,
                                  treename='ffNtuplizer/ffNtuple',
                                  processor_instance=LeptonjetHadronicjetProcessor(dphi_control=True, data_type='bkg'),
                                  executor=processor.futures_executor,
                                  executor_args=dict(workers=12, flatten=True),
                                  chunksize=500000,
                                 )
    outputs['data'] = processor.run_uproot_job(dataDS,
                                  treename='ffNtuplizer/ffNtuple',
                                  processor_instance=LeptonjetHadronicjetProcessor(dphi_control=True, data_type='data'),
                                  executor=processor.futures_executor,
                                  executor_args=dict(workers=12, flatten=True),
                                  chunksize=500000,
                                 )

    plotsmap = plot_datamc(outputs)
    for chan in plotsmap:
        fig, (ax, rax) = plotsmap[chan]
        fig.savefig(join(outdir, f"{chan}_pt.png"))
        fig.savefig(join(outdir, f"{chan}_pt.pdf"))
        plt.close(fig)

    if args.sync:
        webdir = 'wsi@lxplus.cern.ch:/eos/user/w/wsi/www/public/firehydrant'
        cmd = f'rsync -az --exclude ".*" --delete {outdir} {webdir}'
        print(f"--> sync with: {webdir}")
        os.system(cmd)
