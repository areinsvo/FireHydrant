#!/usr/bin/env python

"""
make isolation plots of leptonjets
"""

import argparse

import awkward
import coffea.processor as processor
import matplotlib.pyplot as plt
import numpy as np
import uproot
from coffea import hist
from coffea.analysis_objects import JaggedCandidateArray
from FireHydrant.Analysis.DatasetMapLoader import (DatasetMapLoader,
                                                   SigDatasetMapLoader)
from FireHydrant.Tools.correction import (get_nlo_weight_function,
                                          get_pu_weights_function,
                                          get_ttbar_weight)
from FireHydrant.Tools.metfilter import MetFilters
from FireHydrant.Tools.trigger import Triggers
from matplotlib.colors import LogNorm

np.seterr(divide='ignore', invalid='ignore', over='ignore')
plt.rcParams["savefig.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"


parser = argparse.ArgumentParser(description="produce 2D leptonjet isolation map")
parser.add_argument("--sync", action='store_true', help="issue rsync command to sync plots folder to lxplus web server")
args = parser.parse_args()


dml = DatasetMapLoader()
bkgDS, bkgMAP, bkgSCALE = dml.fetch('bkg')
dataDS, dataMAP = dml.fetch('data')

sdml = SigDatasetMapLoader()
sigDS, sigSCALE = sdml.fetch('simple')


"""Leptonjet Isolation."""
class LeptonJetIsoProcessor(processor.ProcessorABC):
    def __init__(self, dphi_control=False, data_type='bkg'):
        self.dphi_control = dphi_control
        self.data_type = data_type
        dataset_axis = hist.Cat('dataset', 'dataset')
        lj0iso_axis = hist.Bin('lj0iso', 'iso value', 20, 0, 1)
        lj1iso_axis = hist.Bin('lj1iso', 'iso value', 20, 0, 1)
        channel_axis = hist.Cat('channel', 'channel')
        type_axis = hist.Cat('isotype', 'isotype')
        njet_axis = hist.Bin('njet', 'njet', 10, 0, 10)

        self._accumulator = processor.dict_accumulator({
            'ljpfiso': hist.Hist('Counts', dataset_axis, lj0iso_axis, lj1iso_axis, type_axis, channel_axis, njet_axis)
        })

        ## NOT applied for now
        self.pucorrs = get_pu_weights_function()
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
            leptonjets_ = dileptonjets[isControl]
            wgt = wgt[isControl]
            lj0 = lj0[isControl]
            lj1 = lj1[isControl]
            channel_ = channel_[isControl]
        else:
            leptonjets_ = dileptonjets
        if leptonjets_.size==0: return output

        output['ljpfiso'].fill(dataset=dataset, lj0iso=lj0[channel_==2].pfisoAll05.flatten(), lj1iso=lj1[channel_==2].pfisoAll05.flatten(), weight=wgt[channel_==2], channel='4mu', isotype='all05', njet=ak4jets.counts[channel_==2])
        output['ljpfiso'].fill(dataset=dataset, lj0iso=lj0[channel_==2].pfisoNopu05.flatten(), lj1iso=lj1[channel_==2].pfisoNopu05.flatten(), weight=wgt[channel_==2], channel='4mu', isotype='nopu05', njet=ak4jets.counts[channel_==2])
        output['ljpfiso'].fill(dataset=dataset, lj0iso=lj0[channel_==2].pfisoDbeta.flatten(), lj1iso=lj1[channel_==2].pfisoDbeta.flatten(), weight=wgt[channel_==2], channel='4mu', isotype='dbeta', njet=ak4jets.counts[channel_==2])

        ## 2mu2e
        dileptonjets_2mu2e = dileptonjets[channel_==1]
        egm_2mu2e = dileptonjets_2mu2e[dileptonjets_2mu2e.iseltype]
        egm_2mu2e = egm_2mu2e[egm_2mu2e.pt.argmax()]
        mu_2mu2e = dileptonjets_2mu2e[dileptonjets_2mu2e.ismutype]
        mu_2mu2e = mu_2mu2e[mu_2mu2e.pt.argmax()]
        output['ljpfiso'].fill(dataset=dataset, lj0iso=egm_2mu2e.pfisoAll05.flatten(), lj1iso=mu_2mu2e.pfisoAll05.flatten(), weight=wgt[channel_==1], channel='2mu2e', isotype='all05', njet=ak4jets.counts[channel_==1])
        output['ljpfiso'].fill(dataset=dataset, lj0iso=egm_2mu2e.pfisoNopu05.flatten(), lj1iso=mu_2mu2e.pfisoNopu05.flatten(), weight=wgt[channel_==1], channel='2mu2e', isotype='nopu05', njet=ak4jets.counts[channel_==1])
        output['ljpfiso'].fill(dataset=dataset, lj0iso=egm_2mu2e.pfisoDbeta.flatten(), lj1iso=mu_2mu2e.pfisoDbeta.flatten(), weight=wgt[channel_==1], channel='2mu2e', isotype='dbeta', njet=ak4jets.counts[channel_==1])


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
            if self.data_type == 'sig':
                accumulator[k].scale(sigSCALE, axis='dataset')
        return accumulator


def plot_iso_2d(output, isotype, njet='both', issignal=False):
    """
    make 2D plot with each leptonjet isolation value on XY axis. 2mu2e left/4mu right.

    :param hist.Hist output: histogram
    :param str isotype: all05/nopu05/dbeta
    :param str njet: both/sr/cr
    :param bool issignal: `output` from signal?
    :return: (fig, axes) / dictionary of (fig, ax)
    :rtype: tuple / dict
    """

    if issignal:
        res = {}
        for d in output['ljpfiso'].identifiers('dataset'):
            res[str(d)] = plt.subplots(1,1,figsize=(7,5))
            fig, ax = res[str(d)]
            histo = output['ljpfiso'].integrate('dataset', d).integrate('isotype', isotype)
            chan_ = str(d).split('/')[0]
            if njet == 'both':
                hist.plot2d(histo.integrate('channel', chan_).sum('njet'), xaxis='lj0iso', ax=ax,)
            elif njet == "sr":
                hist.plot2d(histo.integrate('channel', chan_).integrate('njet', slice(0, 4)), xaxis='lj0iso', ax=ax, )
            elif njet == 'cr':
                hist.plot2d(histo.integrate('channel', chan_).integrate('njet', slice(4, 10)), xaxis='lj0iso', ax=ax, )

            ax.set_title(f'[{chan_}] leptonjet isolation ({isotype})', x=0, ha="left")
            ax.text(1,1,'59.74/fb (13TeV)', ha='right', va='bottom', transform=ax.transAxes)
            if chan_=='2mu2e':
                ax.set_xlabel('EGM-type lj isolation value', x=1, ha='right')
                ax.set_ylabel('Mu-type lj isolation value', y=1, ha='right')
            if chan_=='4mu':
                ax.set_xlabel('Mu-type lj0 isolation value', x=1, ha='right')
                ax.set_ylabel('Mu-type lj1 isolation value', y=1, ha='right')
        return res

    else:
        fig, axes = plt.subplots(1,2,figsize=(14,5))
        histo = output['ljpfiso_cat'].sum('cat').integrate('isotype', isotype)
        if njet == "both":
            hist.plot2d(histo.integrate('channel', '2mu2e').sum('njet'), xaxis='lj0iso', ax=axes[0], patch_opts=dict(norm=LogNorm()))
            hist.plot2d(histo.integrate('channel', '4mu').sum('njet'), xaxis='lj0iso', ax=axes[1], patch_opts=dict(norm=LogNorm()))
        elif njet == "sr":
            hist.plot2d(histo.integrate('channel', '2mu2e').integrate('njet', slice(0, 4)), xaxis='lj0iso', ax=axes[0], patch_opts=dict(norm=LogNorm()))
            hist.plot2d(histo.integrate('channel', '4mu').integrate('njet', slice(0, 4)), xaxis='lj0iso', ax=axes[1], patch_opts=dict(norm=LogNorm()))
        elif njet == 'cr':
            hist.plot2d(histo.integrate('channel', '2mu2e').integrate('njet', slice(4, 10), overflow='over'), xaxis='lj0iso', ax=axes[0], patch_opts=dict(norm=LogNorm()))
            hist.plot2d(histo.integrate('channel', '4mu').integrate('njet', slice(4, 10), overflow='over'), xaxis='lj0iso', ax=axes[1], patch_opts=dict(norm=LogNorm()))

        axes[0].set_title(f'[2mu2e] leptonJet isolation ({isotype})', x=0.0, ha="left")
        axes[0].set_xlabel('EGM-type lj isolation value')
        axes[0].set_ylabel('Mu-type lj isolation value')
        axes[1].set_title(f'[4mu] leptonJet isolation ({isotype})', x=0.0, ha="left")
        axes[1].set_xlabel('Mu-type lj0 isolation value')
        axes[1].set_ylabel('Mu-type lj1 isolation value')
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
                                  processor_instance=LeptonJetIsoProcessor(),
                                  executor=processor.futures_executor,
                                  executor_args=dict(workers=12, flatten=True),
                                  chunksize=500000,
                                 )
    outputs['data'] = processor.run_uproot_job(dataDS,
                                  treename='ffNtuplizer/ffNtuple',
                                  processor_instance=LeptonJetIsoProcessor(dphi_control=True, data_type='data'),
                                  executor=processor.futures_executor,
                                  executor_args=dict(workers=12, flatten=True),
                                  chunksize=500000,
                                 )

    outputs['sig'] = processor.run_uproot_job(sigDS,
                                  treename='ffNtuplizer/ffNtuple',
                                  processor_instance=LeptonJetIsoProcessor(dphi_control=False, data_type='sig'),
                                  executor=processor.futures_executor,
                                  executor_args=dict(workers=12, flatten=True),
                                  chunksize=500000,
                                 )

    for iso in ['all05', 'nopu05', 'dbeta']:
        for njet in ['both', 'sr', 'cr']:
            fig, axes = plot_iso_2d(outputs['bkg'], iso, njet)
            fig.savefig(join(outdir, f"{iso}_njet-{njet}__bkg.png"))
            fig.savefig(join(outdir, f"{iso}_njet-{njet}__bkg.pdf"))
            plt.close(fig)

            fig, axes = plot_iso_2d(outputs['data'], iso, njet)
            fig.savefig(join(outdir, f"{iso}_njet-{njet}__data.png"))
            fig.savefig(join(outdir, f"{iso}_njet-{njet}__data.pdf"))
            plt.close(fig)

            figaxes = plot_iso_2d(outputs['sig'], iso, njet, issignal=True)
            for d, (fig, ax) in figaxes.items():
                param = str(d).replace('/', '_')
                fig.savefig(join(outdir, f"{iso}_njet-{njet}__{param}.png"))
                fig.savefig(join(outdir, f"{iso}_njet-{njet}__{param}.pdf"))
                plt.close(fig=fig)

    if args.sync:
        webdir = 'wsi@lxplus.cern.ch:/eos/user/w/wsi/www/public/firehydrant'
        cmd = f'rsync -az --exclude ".*" --delete {outdir} {webdir}'
        print(f"--> sync with: {webdir}")
        os.system(cmd)
