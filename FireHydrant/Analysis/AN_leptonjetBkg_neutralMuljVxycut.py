#!/usr/bin/env python
"""For AN
leptonjet+event kinematics for sig/bkg, all region,  w/ vxy cut, and neutrality for mulj
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


parser = argparse.ArgumentParser(description="[AN] bkg leptonjet, event kinematics")
parser.add_argument("--sync", action='store_true', help="issue rsync command to sync plots folder to lxplus web server")
args = parser.parse_args()

dml = DatasetMapLoader()
bkgDS, bkgMAP, bkgSCALE = dml.fetch('bkg')
# dataDS, dataMAP = dml.fetch('data')

sdml = SigDatasetMapLoader()
sigDS_2mu2e, sigSCALE_2mu2e = sdml.fetch('2mu2e')
sigDS_4mu, sigSCALE_4mu = sdml.fetch('4mu')

class LJBkgProcessor(processor.ProcessorABC):
    def __init__(self, data_type='bkg'):
        self.data_type = data_type

        dataset_axis = hist.Cat('dataset', 'dataset')
        pt_axis = hist.Bin('pt', '$p_T$ [GeV]', 100, 0, 200)
        ljmass_axis = hist.Bin('ljmass', 'mass [GeV]', 100, 0, 20)
        pairmass_axis = hist.Bin('pairmass', 'mass [GeV]', 100, 0, 200)
        vxy_axis = hist.Bin('vxy', 'vxy [cm]', 100, 0, 20)
        qsum_axis = hist.Bin('qsum', '$\sum$q', 2, 0, 2)
        dphi_axis = hist.Bin('dphi', '$\Delta\phi$', 50, 0, np.pi)
        channel_axis = hist.Bin('channel', 'channel', 3, 0, 3)

        self._accumulator = processor.dict_accumulator({
            'lj0pt': hist.Hist('Counts/2GeV', dataset_axis, pt_axis, channel_axis),
            'lj1pt': hist.Hist('Counts/2GeV', dataset_axis, pt_axis, channel_axis),
            'muljmass': hist.Hist('Counts/0.2GeV', dataset_axis, ljmass_axis, channel_axis),
            'muljvxy': hist.Hist('Counts/0.2cm', dataset_axis, vxy_axis, channel_axis),
            'muljqsum': hist.Hist('Counts', dataset_axis, qsum_axis, channel_axis),
            'ljpairmass': hist.Hist('Counts/2GeV', dataset_axis, pairmass_axis, channel_axis),
            'ljpairdphi': hist.Hist('Counts/$\pi$/50', dataset_axis, dphi_axis, channel_axis),
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
        ljdaucharge = awkward.fromiter(df['pfjet_pfcand_charge']).sum()
        leptonjets.add_attributes(qsum=ljdaucharge)
        leptonjets.add_attributes(isneutral=(leptonjets.iseltype | (leptonjets.ismutype&(leptonjets.qsum==0))))
        leptonjets.add_attributes(displaced=((leptonjets.vxy>=5)|(np.isnan(leptonjets.vxy)&leptonjets.ismutype))) # non-vertex treated as displaced too
        leptonjets = leptonjets[leptonjets.isneutral]

        ## __ twoleptonjets__ AND >=1 displaced
        twoleptonjets = (leptonjets.counts>=2)&(leptonjets.ismutype.sum()>=1)&(leptonjets.displaced.sum()>=1)
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

        output['lj0pt'].fill(dataset=dataset, pt=lj0.pt.flatten(), channel=channel_, weight=wgt)
        output['lj1pt'].fill(dataset=dataset, pt=lj1.pt.flatten(), channel=channel_, weight=wgt)

        mulj = dileptonjets[dileptonjets.ismutype]
        muljones = mulj.pt.ones_like()
        output['muljmass'].fill(dataset=dataset, ljmass=mulj.mass.flatten(), channel=(channel_*muljones).flatten(), weight=(wgt*muljones).flatten())
        output['muljvxy'].fill(dataset=dataset, vxy=mulj.vxy.flatten(), channel=(channel_*muljones).flatten(), weight=(wgt*muljones).flatten())
        output['muljqsum'].fill(dataset=dataset, qsum=mulj.isneutral.flatten(), channel=(channel_*muljones).flatten(), weight=(wgt*muljones).flatten())

        output['ljpairmass'].fill(dataset=dataset, pairmass=(lj0.p4+lj1.p4).mass.flatten(), channel=channel_, weight=wgt)
        output['ljpairdphi'].fill(dataset=dataset, dphi=(np.abs(lj0.p4.delta_phi(lj1.p4))).flatten(), channel=channel_, weight=wgt)

        return output

    def postprocess(self, accumulator):
        origidentity = list(accumulator)
        for k in origidentity:
            if self.data_type == 'bkg':
                accumulator[k].scale(bkgSCALE, axis='dataset')
                accumulator[k] = accumulator[k].group("dataset",
                                                    hist.Cat("cat", "datasets", sorting='integral'),
                                                    bkgMAP)
            # if self.data_type == 'data':
            #     accumulator[k] = accumulator[k].group("dataset",
            #                                         hist.Cat("cat", "datasets",),
            #                                         dataMAP)
            if self.data_type == 'sig-2mu2e':
                accumulator[k].scale(sigSCALE_2mu2e, axis='dataset')
            if self.data_type == 'sig-4mu':
                accumulator[k].scale(sigSCALE_4mu, axis='dataset')

        return accumulator



if __name__ == "__main__":
    import os
    from os.path import join, isdir, splitext
    from FireHydrant.Analysis.PlottingOptions import *

    reldir = splitext(__file__)[0].replace('_', '/')
    outdir = join(os.getenv('FH_BASE'), "Imgs", reldir)
    if not isdir(outdir): os.makedirs(outdir)

    out_sig2mu2e = processor.run_uproot_job(sigDS_2mu2e,
                                  treename='ffNtuplizer/ffNtuple',
                                  processor_instance=LJBkgProcessor(data_type='sig-2mu2e'),
                                  executor=processor.futures_executor,
                                  executor_args=dict(workers=12, flatten=True),
                                  chunksize=500000,
                                 )

    out_sig4mu = processor.run_uproot_job(sigDS_4mu,
                                  treename='ffNtuplizer/ffNtuple',
                                  processor_instance=LJBkgProcessor(data_type='sig-4mu'),
                                  executor=processor.futures_executor,
                                  executor_args=dict(workers=12, flatten=True),
                                  chunksize=500000,
                                 )

    out_bkg = processor.run_uproot_job(bkgDS,
                                  treename='ffNtuplizer/ffNtuple',
                                  processor_instance=LJBkgProcessor(data_type='bkg'),
                                  executor=processor.futures_executor,
                                  executor_args=dict(workers=12, flatten=True),
                                  chunksize=500000,
                                 )

    import re
    smallmxx = re.compile('mXX-(100|150|200)_\w+')

    ## CHANNEL - 2mu2e
    print('## CHANNEL - 2mu2e')

    # leading lj pt
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    bkghist = out_bkg['lj0pt'].integrate('channel', slice(1,2))
    hist.plot1d(bkghist, overlay='cat', ax=ax, stack=True, overflow='over',
                line_opts=None, fill_opts=fill_opts, error_opts=error_opts)
    sighist = out_sig2mu2e['lj0pt'][smallmxx].integrate('channel', slice(1,2))
    hist.plot1d(sighist, overlay='dataset', ax=ax, overflow='over', clear=False)

    ax.set_title('[$2\mu 2e$|MC] leading leptonjet $p_T$', x=0.0, ha="left")
    ax.legend(*groupHandleLabel(ax), prop={'size': 8,}, ncol=3)
    ax.set_yscale('log')
    ax.autoscale(axis='both', tight=True)
    ax.text(1,1,'59.74/fb (13TeV)', ha='right', va='bottom', transform=ax.transAxes)
    ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
    ax.set_ylabel(ax.get_ylabel(), y=1.0, ha="right")

    fig.savefig(join(outdir, 'lj0pt_2mu2e_neutralvxycut.png'))
    fig.savefig(join(outdir, 'lj0pt_2mu2e_neutralvxycut.pdf'))
    plt.close(fig)

    # subleading lj pt
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    bkghist = out_bkg['lj1pt'].integrate('channel', slice(1,2))
    hist.plot1d(bkghist, overlay='cat', ax=ax, stack=True, overflow='over',
                line_opts=None, fill_opts=fill_opts, error_opts=error_opts)
    sighist = out_sig2mu2e['lj1pt'][smallmxx].integrate('channel', slice(1,2))
    hist.plot1d(sighist, overlay='dataset', ax=ax, overflow='over', clear=False)

    ax.set_title('[$2\mu 2e$|MC] subleading leptonjet $p_T$', x=0.0, ha="left")
    ax.legend(*groupHandleLabel(ax), prop={'size': 8,}, ncol=3)
    ax.set_yscale('log')
    ax.autoscale(axis='both', tight=True)
    ax.text(1,1,'59.74/fb (13TeV)', ha='right', va='bottom', transform=ax.transAxes)
    ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
    ax.set_ylabel(ax.get_ylabel(), y=1.0, ha="right")

    fig.savefig(join(outdir, 'lj1pt_2mu2e_neutralvxycut.png'))
    fig.savefig(join(outdir, 'lj1pt_2mu2e_neutralvxycut.pdf'))
    plt.close(fig)

    # muon-type lj mass
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    bkghist = out_bkg['muljmass'].integrate('channel', slice(1,2))
    hist.plot1d(bkghist, overlay='cat', ax=ax, stack=True, overflow='over',
                line_opts=None, fill_opts=fill_opts, error_opts=error_opts)
    sighist = out_sig2mu2e['muljmass'][smallmxx].integrate('channel', slice(1,2))
    hist.plot1d(sighist, overlay='dataset', ax=ax, overflow='over', clear=False)

    ax.set_title('[$2\mu 2e$|MC] muon-type leptonjet mass', x=0.0, ha="left")
    ax.legend(*groupHandleLabel(ax), prop={'size': 8,}, ncol=3)
    ax.set_yscale('log')
    ax.autoscale(axis='both', tight=True)
    ax.text(1,1,'59.74/fb (13TeV)', ha='right', va='bottom', transform=ax.transAxes)
    ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
    ax.set_ylabel(ax.get_ylabel(), y=1.0, ha="right")

    fig.savefig(join(outdir, 'muljmass_2mu2e_neutralvxycut.png'))
    fig.savefig(join(outdir, 'muljmass_2mu2e_neutralvxycut.pdf'))
    plt.close(fig)

    # muon-type lj vxy
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    bkghist = out_bkg['muljvxy'].integrate('channel', slice(1,2))
    hist.plot1d(bkghist, overlay='cat', ax=ax, stack=True, overflow='over',
                line_opts=None, fill_opts=fill_opts, error_opts=error_opts)
    sighist = out_sig2mu2e['muljvxy'][smallmxx].integrate('channel', slice(1,2))
    hist.plot1d(sighist, overlay='dataset', ax=ax, overflow='over', clear=False)

    ax.set_title('[$2\mu 2e$|MC] muon-type leptonjet vxy', x=0.0, ha="left")
    ax.legend(*groupHandleLabel(ax), prop={'size': 8,}, ncol=3)
    ax.set_yscale('log')
    ax.autoscale(axis='both', tight=True)
    ax.text(1,1,'59.74/fb (13TeV)', ha='right', va='bottom', transform=ax.transAxes)
    ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
    ax.set_ylabel(ax.get_ylabel(), y=1.0, ha="right")

    fig.savefig(join(outdir, 'muljvxy_2mu2e_neutralvxycut.png'))
    fig.savefig(join(outdir, 'muljvxy_2mu2e_neutralvxycut.pdf'))
    plt.close(fig)

    # muon-type lj qsum
    fig, axes = plt.subplots(1,2,figsize=(16,6))
    fig.subplots_adjust(wspace=0.15)
    bkghist = out_bkg['muljqsum'].integrate('channel', slice(1,2))
    hist.plot1d(bkghist, overlay='cat', ax=axes[0], stack=True, overflow='none',
                line_opts=None, fill_opts=fill_opts, error_opts=error_opts,)
    sighist = out_sig2mu2e['muljqsum'].sum('dataset').integrate('channel', slice(1,2))
    hist.plot1d(sighist, ax=axes[1], overflow='none', density=True)

    axes[0].set_title('[$2\mu 2e$|BackgroundMC] muon-type leptonjet charge sum', x=0.0, ha="left")
    axes[1].set_title('[$2\mu 2e$|SignalMC] muon-type leptonjet charge sum', x=0.0, ha="left")
    for ax in axes:
        ax.autoscale(axis='both', tight=True)
        ax.text(1,1,'59.74/fb (13TeV)', ha='right', va='bottom', transform=ax.transAxes)
        ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
        ax.set_xticks([0.5, 1.5])
        ax.set_xticklabels([r'qsum$\neq$0', 'qsum=0'])
    axes[0].set_ylabel(ax.get_ylabel(), y=1.0, ha="right")
    axes[1].set_ylabel('Norm. '+ax.get_ylabel(), y=1.0, ha="right")

    fig.savefig(join(outdir, 'muljqsum_2mu2e_neutralvxycut.png'))
    fig.savefig(join(outdir, 'muljqsum_2mu2e_neutralvxycut.pdf'))
    plt.close(fig)

    # lj pair mass
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    bkghist = out_bkg['ljpairmass'].integrate('channel', slice(1,2))
    hist.plot1d(bkghist, overlay='cat', ax=ax, stack=True, overflow='over',
                line_opts=None, fill_opts=fill_opts, error_opts=error_opts)
    sighist = out_sig2mu2e['ljpairmass'][smallmxx].integrate('channel', slice(1,2))
    hist.plot1d(sighist, overlay='dataset', ax=ax, overflow='over', clear=False)

    ax.set_title('[$2\mu 2e$|MC] leptonjet pair invariant mass', x=0.0, ha="left")
    ax.legend(*groupHandleLabel(ax), prop={'size': 8,}, ncol=3)
    ax.set_yscale('log')
    ax.autoscale(axis='both', tight=True)
    ax.text(1,1,'59.74/fb (13TeV)', ha='right', va='bottom', transform=ax.transAxes)
    ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
    ax.set_ylabel(ax.get_ylabel(), y=1.0, ha="right")

    fig.savefig(join(outdir, 'ljpairmass_2mu2e_neutralvxycut.png'))
    fig.savefig(join(outdir, 'ljpairmass_2mu2e_neutralvxycut.pdf'))
    plt.close(fig)

    # lj pair dphi
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    bkghist = out_bkg['ljpairdphi'].integrate('channel', slice(1,2))
    hist.plot1d(bkghist, overlay='cat', ax=ax, stack=True, overflow='over',
                line_opts=None, fill_opts=fill_opts, error_opts=error_opts)
    sighist = out_sig2mu2e['ljpairdphi'][smallmxx].integrate('channel', slice(1,2))
    hist.plot1d(sighist, overlay='dataset', ax=ax, overflow='over', clear=False)

    ax.set_title('[$2\mu 2e$|MC] leptonjet pair $\Delta\phi$', x=0.0, ha="left")
    ax.legend(*groupHandleLabel(ax), prop={'size': 8,}, ncol=3)
    ax.set_yscale('log')
    ax.autoscale(axis='both', tight=True)
    ax.text(1,1,'59.74/fb (13TeV)', ha='right', va='bottom', transform=ax.transAxes)
    ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
    ax.set_ylabel(ax.get_ylabel(), y=1.0, ha="right")

    fig.savefig(join(outdir, 'ljpairdphi_2mu2e_neutralvxycut.png'))
    fig.savefig(join(outdir, 'ljpairdphi_2mu2e_neutralvxycut.pdf'))
    plt.close(fig)



    ##############################################################


    ## CHANNEL - 4mu
    print('## CHANNEL - 4mu')

    # leading lj pt
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    bkghist = out_bkg['lj0pt'].integrate('channel', slice(2,3))
    hist.plot1d(bkghist, overlay='cat', ax=ax, stack=True, overflow='over',
                line_opts=None, fill_opts=fill_opts, error_opts=error_opts)
    sighist = out_sig4mu['lj0pt'][smallmxx].integrate('channel', slice(2,3))
    hist.plot1d(sighist, overlay='dataset', ax=ax, overflow='over', clear=False)

    ax.set_title('[$4\mu$|MC] leading leptonjet $p_T$', x=0.0, ha="left")
    ax.legend(*groupHandleLabel(ax), prop={'size': 8,}, ncol=3)
    ax.set_yscale('log')
    ax.autoscale(axis='both', tight=True)
    ax.text(1,1,'59.74/fb (13TeV)', ha='right', va='bottom', transform=ax.transAxes)
    ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
    ax.set_ylabel(ax.get_ylabel(), y=1.0, ha="right")

    fig.savefig(join(outdir, 'lj0pt_4mu_neutralvxycut.png'))
    fig.savefig(join(outdir, 'lj0pt_4mu_neutralvxycut.pdf'))
    plt.close(fig)

    # subleading lj pt
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    bkghist = out_bkg['lj1pt'].integrate('channel', slice(2,3))
    hist.plot1d(bkghist, overlay='cat', ax=ax, stack=True, overflow='over',
                line_opts=None, fill_opts=fill_opts, error_opts=error_opts)
    sighist = out_sig4mu['lj1pt'][smallmxx].integrate('channel', slice(2,3))
    hist.plot1d(sighist, overlay='dataset', ax=ax, overflow='over', clear=False)

    ax.set_title('[$4\mu$|MC] subleading leptonjet $p_T$', x=0.0, ha="left")
    ax.legend(*groupHandleLabel(ax), prop={'size': 8,}, ncol=3)
    ax.set_yscale('log')
    ax.autoscale(axis='both', tight=True)
    ax.text(1,1,'59.74/fb (13TeV)', ha='right', va='bottom', transform=ax.transAxes)
    ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
    ax.set_ylabel(ax.get_ylabel(), y=1.0, ha="right")

    fig.savefig(join(outdir, 'lj1pt_4mu_neutralvxycut.png'))
    fig.savefig(join(outdir, 'lj1pt_4mu_neutralvxycut.pdf'))
    plt.close(fig)

    # muon-type lj mass
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    bkghist = out_bkg['muljmass'].integrate('channel', slice(2,3))
    hist.plot1d(bkghist, overlay='cat', ax=ax, stack=True, overflow='over',
                line_opts=None, fill_opts=fill_opts, error_opts=error_opts)
    sighist = out_sig4mu['muljmass'][smallmxx].integrate('channel', slice(2,3))
    hist.plot1d(sighist, overlay='dataset', ax=ax, overflow='over', clear=False)

    ax.set_title('[$4\mu$|MC] muon-type leptonjet mass', x=0.0, ha="left")
    ax.legend(*groupHandleLabel(ax), prop={'size': 8,}, ncol=3)
    ax.set_yscale('log')
    ax.autoscale(axis='both', tight=True)
    ax.text(1,1,'59.74/fb (13TeV)', ha='right', va='bottom', transform=ax.transAxes)
    ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
    ax.set_ylabel(ax.get_ylabel(), y=1.0, ha="right")

    fig.savefig(join(outdir, 'muljmass_4mu_neutralvxycut.png'))
    fig.savefig(join(outdir, 'muljmass_4mu_neutralvxycut.pdf'))
    plt.close(fig)

    # muon-type lj vxy
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    bkghist = out_bkg['muljvxy'].integrate('channel', slice(2,3))
    hist.plot1d(bkghist, overlay='cat', ax=ax, stack=True, overflow='over',
                line_opts=None, fill_opts=fill_opts, error_opts=error_opts)
    sighist = out_sig4mu['muljvxy'][smallmxx].integrate('channel', slice(2,3))
    hist.plot1d(sighist, overlay='dataset', ax=ax, overflow='over', clear=False)

    ax.set_title('[$4\mu$|MC] muon-type leptonjet vxy', x=0.0, ha="left")
    ax.legend(*groupHandleLabel(ax), prop={'size': 8,}, ncol=3)
    ax.set_yscale('log')
    ax.autoscale(axis='both', tight=True)
    ax.text(1,1,'59.74/fb (13TeV)', ha='right', va='bottom', transform=ax.transAxes)
    ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
    ax.set_ylabel(ax.get_ylabel(), y=1.0, ha="right")

    fig.savefig(join(outdir, 'muljvxy_4mu_neutralvxycut.png'))
    fig.savefig(join(outdir, 'muljvxy_4mu_neutralvxycut.pdf'))
    plt.close(fig)

    # muon-type lj qsum
    fig, axes = plt.subplots(1,2,figsize=(16,6))
    fig.subplots_adjust(wspace=0.15)
    bkghist = out_bkg['muljqsum'].integrate('channel', slice(2,3))
    hist.plot1d(bkghist, overlay='cat', ax=axes[0], stack=True, overflow='none',
                line_opts=None, fill_opts=fill_opts, error_opts=error_opts)
    sighist = out_sig4mu['muljqsum'].sum('dataset').integrate('channel', slice(2,3))
    hist.plot1d(sighist, ax=axes[1], overflow='none', density=True, )

    axes[0].set_title('[$4\mu$|BackgroundMC] muon-type leptonjet charge sum', x=0.0, ha="left")
    axes[1].set_title('[$4\mu$|SignalMC] muon-type leptonjet charge sum', x=0.0, ha="left")
    for ax in axes:
        ax.autoscale(axis='both', tight=True)
        ax.text(1,1,'59.74/fb (13TeV)', ha='right', va='bottom', transform=ax.transAxes)
        ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
        ax.set_xticks([0.5, 1.5])
        ax.set_xticklabels([r'qsum$\neq$0', 'qsum=0'])
    axes[0].set_ylabel(ax.get_ylabel(), y=1.0, ha="right")
    axes[1].set_ylabel('Norm. '+ax.get_ylabel(), y=1.0, ha="right")

    fig.savefig(join(outdir, 'muljqsum_4mu_neutralvxycut.png'))
    fig.savefig(join(outdir, 'muljqsum_4mu_neutralvxycut.pdf'))
    plt.close(fig)

    # lj pair mass
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    bkghist = out_bkg['ljpairmass'].integrate('channel', slice(2,3))
    hist.plot1d(bkghist, overlay='cat', ax=ax, stack=True, overflow='over',
                line_opts=None, fill_opts=fill_opts, error_opts=error_opts)
    sighist = out_sig4mu['ljpairmass'][smallmxx].integrate('channel', slice(2,3))
    hist.plot1d(sighist, overlay='dataset', ax=ax, overflow='over', clear=False)

    ax.set_title('[$4\mu$|MC] leptonjet pair invariant mass', x=0.0, ha="left")
    ax.legend(*groupHandleLabel(ax), prop={'size': 8,}, ncol=3)
    ax.set_yscale('log')
    ax.autoscale(axis='both', tight=True)
    ax.text(1,1,'59.74/fb (13TeV)', ha='right', va='bottom', transform=ax.transAxes)
    ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
    ax.set_ylabel(ax.get_ylabel(), y=1.0, ha="right")

    fig.savefig(join(outdir, 'ljpairmass_4mu_neutralvxycut.png'))
    fig.savefig(join(outdir, 'ljpairmass_4mu_neutralvxycut.pdf'))
    plt.close(fig)

    # lj pair dphi
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    bkghist = out_bkg['ljpairdphi'].integrate('channel', slice(2,3))
    hist.plot1d(bkghist, overlay='cat', ax=ax, stack=True, overflow='over',
                line_opts=None, fill_opts=fill_opts, error_opts=error_opts)
    sighist = out_sig4mu['ljpairdphi'][smallmxx].integrate('channel', slice(2,3))
    hist.plot1d(sighist, overlay='dataset', ax=ax, overflow='over', clear=False)

    ax.set_title('[$4\mu$|MC] leptonjet pair $\Delta\phi$', x=0.0, ha="left")
    ax.legend(*groupHandleLabel(ax), prop={'size': 8,}, ncol=3)
    ax.set_yscale('log')
    ax.autoscale(axis='both', tight=True)
    ax.text(1,1,'59.74/fb (13TeV)', ha='right', va='bottom', transform=ax.transAxes)
    ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
    ax.set_ylabel(ax.get_ylabel(), y=1.0, ha="right")

    fig.savefig(join(outdir, 'ljpairdphi_4mu_neutralvxycut.png'))
    fig.savefig(join(outdir, 'ljpairdphi_4mu_neutralvxycut.pdf'))
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
        ## copy to AN image folder
        an_dir = '/uscms_data/d3/wsi/lpcdm/AN-18-125/image'
        if isdir(an_dir):
            cmd = f'cp {outdir}/*.pdf {an_dir}'
            print(f'--> copy to AN folder: {an_dir}')
            os.system(cmd)
