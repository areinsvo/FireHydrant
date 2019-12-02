#!/usr/bin/env python
"""
leptonjet leading/subleading pT
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


parser = argparse.ArgumentParser(description="leptonjet leading/subleading pt")
parser.add_argument("--sync", action='store_true', help="issue rsync command to sync plots folder to lxplus web server")
args = parser.parse_args()

dml = DatasetMapLoader()
bkgDS, bkgMAP, bkgSCALE = dml.fetch('bkg')
dataDS, dataMAP = dml.fetch('data')

sdml = SigDatasetMapLoader()
sigDS_2mu2e, sigSCALE_2mu2e = sdml.fetch('2mu2e')
sigDS_4mu, sigSCALE_4mu = sdml.fetch('4mu')


"""Leptonjet leading/subleading pT, eta"""
class LeptonjetLeadSubleadProcessor(processor.ProcessorABC):
    def __init__(self, region='SR', data_type='bkg'):
        self.region = region
        self.data_type = data_type

        dataset_axis = hist.Cat('dataset', 'dataset')
        pt_axis = hist.Bin('pt', '$p_T$ [GeV]', 100, 0, 200)
        invm_axis = hist.Bin('invm', 'mass [GeV]', 100, 0, 200)
        mass_axis = hist.Bin('mass', 'mass [GeV]', 100, 0, 200)
        channel_axis = hist.Bin('channel', 'channel', 3, 0, 3)

        self._accumulator = processor.dict_accumulator({
            'pt0': hist.Hist('Counts', dataset_axis, pt_axis, channel_axis),
            'pt1': hist.Hist('Counts', dataset_axis, pt_axis, channel_axis),
            'ptegm': hist.Hist('Counts', dataset_axis, pt_axis, channel_axis), # leading EGM-type for 2mu2e channel
            'ptmu': hist.Hist('Counts', dataset_axis, pt_axis, channel_axis), # leading mu-type for 2mu2e channel
            'invm': hist.Hist('Counts', dataset_axis, invm_axis, channel_axis),
            'massmu': hist.Hist('Counts', dataset_axis, mass_axis, channel_axis), # mass of mu-type leptonjet
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
        ljdaucharge = awkward.fromiter(df['pfjet_pfcand_charge']).sum()
        leptonjets.add_attributes(qsum=ljdaucharge)
        leptonjets.add_attributes(isneutral=(leptonjets.iseltype | (leptonjets.ismutype&(leptonjets.qsum==0))))
        leptonjets = leptonjets[leptonjets.isneutral]

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

        output['pt0'].fill(dataset=dataset, pt=lj0.pt.flatten(), channel=channel_, weight=wgt)
        output['pt1'].fill(dataset=dataset, pt=lj1.pt.flatten(), channel=channel_, weight=wgt)
        output['invm'].fill(dataset=dataset, invm=(lj0.p4+lj1.p4).mass.flatten(), channel=channel_, weight=wgt)

        egm_lj = dileptonjets[dileptonjets.iseltype]
        egm_lj = egm_lj[egm_lj.pt.argmax()]
        output['ptegm'].fill(dataset=dataset, pt=egm_lj.pt.flatten(), channel=channel_[egm_lj.counts!=0], weight=wgt[egm_lj.counts!=0])

        mu_lj = dileptonjets[dileptonjets.ismutype]
        mu_lj = mu_lj[mu_lj.pt.argmax()] # leading
        output['ptmu'].fill(dataset=dataset, pt=mu_lj.pt.flatten(), channel=channel_[mu_lj.counts!=0], weight=wgt[mu_lj.counts!=0])
        output['massmu'].fill(dataset=dataset, mass=mu_lj.mass.flatten(), channel=channel_[mu_lj.counts!=0], weight=wgt[mu_lj.counts!=0])

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


def filterSigDS(origds):
    """
    filter a complete signal DS map with some condition

    :param dict origds: original signal dataset map
    :return: filtered signal dataset dict
    :rtype: dict
    """
    res = {}
    for k in origds:
        param = k.split('_')
        mxx = float(param[0].split('-')[-1])
        if mxx > 300: continue
        res[k] = origds[k]
    return res



if __name__ == "__main__":
    import os
    from os.path import join, isdir
    from FireHydrant.Analysis.PlottingOptions import *

    outdir = join(os.getenv('FH_BASE'), "Imgs", __file__.split('.')[0])
    if not isdir(outdir): os.makedirs(outdir)

    outputs = {}
    outputs['bkg'] = processor.run_uproot_job(bkgDS,
                                  treename='ffNtuplizer/ffNtuple',
                                  processor_instance=LeptonjetLeadSubleadProcessor(region='SR', data_type='bkg'),
                                  executor=processor.futures_executor,
                                  executor_args=dict(workers=12, flatten=True),
                                  chunksize=500000,
                                 )

    outputs['sig-2mu2e'] = processor.run_uproot_job(filterSigDS(sigDS_2mu2e),
                                  treename='ffNtuplizer/ffNtuple',
                                  processor_instance=LeptonjetLeadSubleadProcessor(region='SR', data_type='sig-2mu2e'),
                                  executor=processor.futures_executor,
                                  executor_args=dict(workers=12, flatten=True),
                                  chunksize=500000,
                                 )

    outputs['sig-4mu'] = processor.run_uproot_job(filterSigDS(sigDS_4mu),
                                  treename='ffNtuplizer/ffNtuple',
                                  processor_instance=LeptonjetLeadSubleadProcessor(region='SR', data_type='sig-4mu'),
                                  executor=processor.futures_executor,
                                  executor_args=dict(workers=12, flatten=True),
                                  chunksize=500000,
                                 )

    outputs['data'] = processor.run_uproot_job(dataDS,
                                  treename='ffNtuplizer/ffNtuple',
                                  processor_instance=LeptonjetLeadSubleadProcessor(region='CR', data_type='data'),
                                  executor=processor.futures_executor,
                                  executor_args=dict(workers=12, flatten=True),
                                  chunksize=500000,
                                 )

    ## CHANNEL - 2mu2e
    #### leading
    fig, axes = plt.subplots(1,2,figsize=(16,6))
    fig.subplots_adjust(wspace=0.15)

    bkgpt0 = outputs['bkg']['pt0'].integrate('channel', slice(1,2))
    hist.plot1d(bkgpt0, overlay='cat', ax=axes[0], stack=True, overflow='over',
                line_opts=None, fill_opts=fill_opts, error_opts=error_opts)

    sigpt0 = outputs['sig-2mu2e']['pt0'].integrate('channel', slice(1,2))
    hist.plot1d(sigpt0, overlay='dataset', ax=axes[0], overflow='over', clear=False)

    datapt0 = outputs['data']['pt0'].integrate('channel', slice(1,2))
    hist.plot1d(datapt0, overlay='cat', ax=axes[1], overflow='over', error_opts=data_err_opts)

    axes[0].set_title('[2mu2e|SR] leptonjet leading pT', x=0.0, ha="left")
    axes[1].set_title('[2mu2e|CR] leptonjet leading pT', x=0.0, ha="left")

    axes[0].legend(*groupHandleLabel(axes[0]), prop={'size': 8,}, ncol=3)

    for ax in axes:
        ax.set_yscale('log')
        ax.autoscale(axis='both', tight=True)
        ax.text(1,1,'59.74/fb (13TeV)', ha='right', va='bottom', transform=ax.transAxes)
        ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
        ax.set_ylabel(ax.get_ylabel(), y=1.0, ha="right")

    fig.savefig(join(outdir, 'pt0_2mu2e.png'))
    fig.savefig(join(outdir, 'pt0_2mu2e.pdf'))
    plt.close(fig)

    #### subleading
    fig, axes = plt.subplots(1,2,figsize=(16,6))
    fig.subplots_adjust(wspace=0.15)

    bkgpt1 = outputs['bkg']['pt1'].integrate('channel', slice(1,2))
    hist.plot1d(bkgpt1, overlay='cat', ax=axes[0], stack=True, overflow='over',
                line_opts=None, fill_opts=fill_opts, error_opts=error_opts)

    sigpt1 = outputs['sig-2mu2e']['pt1'].integrate('channel', slice(1,2))
    hist.plot1d(sigpt1, overlay='dataset', ax=axes[0], overflow='over', clear=False)

    datapt1 = outputs['data']['pt1'].integrate('channel', slice(1,2))
    hist.plot1d(datapt1, overlay='cat', ax=axes[1], overflow='over', error_opts=data_err_opts)

    axes[0].set_title('[2mu2e|SR] leptonjet subleading pT', x=0.0, ha="left")
    axes[1].set_title('[2mu2e|CR] leptonjet subleading pT', x=0.0, ha="left")

    axes[0].legend(*groupHandleLabel(axes[0]), prop={'size': 8,}, ncol=3)

    for ax in axes:
        ax.set_yscale('log')
        ax.autoscale(axis='both', tight=True)
        ax.text(1,1,'59.74/fb (13TeV)', ha='right', va='bottom', transform=ax.transAxes)
        ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
        ax.set_ylabel(ax.get_ylabel(), y=1.0, ha="right")

    fig.savefig(join(outdir, 'pt1_2mu2e.png'))
    fig.savefig(join(outdir, 'pt1_2mu2e.pdf'))
    plt.close(fig)

    #### invM(lj0, lj1)
    fig, axes = plt.subplots(1,2,figsize=(16,6))
    fig.subplots_adjust(wspace=0.15)

    bkginvm = outputs['bkg']['invm'].integrate('channel', slice(1,2))
    hist.plot1d(bkginvm, overlay='cat', ax=axes[0], stack=True, overflow='over',
                line_opts=None, fill_opts=fill_opts, error_opts=error_opts)

    siginvm = outputs['sig-2mu2e']['invm'].integrate('channel', slice(1,2))
    hist.plot1d(siginvm, overlay='dataset', ax=axes[0], overflow='over', clear=False)

    datainvm = outputs['data']['invm'].integrate('channel', slice(1,2))
    hist.plot1d(datainvm, overlay='cat', ax=axes[1], overflow='over', error_opts=data_err_opts)

    axes[0].set_title('[2mu2e|SR] leptonjet pair invM', x=0.0, ha="left")
    axes[1].set_title('[2mu2e|CR] leptonjet pair invM', x=0.0, ha="left")

    axes[0].legend(*groupHandleLabel(axes[0]), prop={'size': 8,}, ncol=3)

    for ax in axes:
        # ax.set_yscale('log')
        ax.autoscale(axis='both', tight=True)
        left, right = ax.get_xlim()
        ax.set_xticks(np.arange(left, right+10, 10))
        ax.text(1,1,'59.74/fb (13TeV)', ha='right', va='bottom', transform=ax.transAxes)
        ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
        ax.set_ylabel(ax.get_ylabel(), y=1.0, ha="right")

    fig.savefig(join(outdir, 'invm_2mu2e.png'))
    fig.savefig(join(outdir, 'invm_2mu2e.pdf'))
    plt.close(fig)

    #### EGM-type
    fig, axes = plt.subplots(1,2,figsize=(16,6))
    fig.subplots_adjust(wspace=0.15)

    bkgptegm = outputs['bkg']['ptegm'].integrate('channel', slice(1,2))
    hist.plot1d(bkgptegm, overlay='cat', ax=axes[0], stack=True, overflow='over',
                line_opts=None, fill_opts=fill_opts, error_opts=error_opts)

    sigptegm = outputs['sig-2mu2e']['ptegm'].integrate('channel', slice(1,2))
    hist.plot1d(sigptegm, overlay='dataset', ax=axes[0], overflow='over', clear=False)

    dataptegm = outputs['data']['ptegm'].integrate('channel', slice(1,2))
    hist.plot1d(dataptegm, overlay='cat', ax=axes[1], overflow='over', error_opts=data_err_opts)

    axes[0].set_title('[2mu2e|SR] EGM-type leptonjet pT', x=0.0, ha="left")
    axes[1].set_title('[2mu2e|CR] EGM-type leptonjet pT', x=0.0, ha="left")

    axes[0].legend(*groupHandleLabel(axes[0]), prop={'size': 8,}, ncol=3)

    for ax in axes:
        ax.set_yscale('log')
        ax.autoscale(axis='both', tight=True)
        ax.text(1,1,'59.74/fb (13TeV)', ha='right', va='bottom', transform=ax.transAxes)
        ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
        ax.set_ylabel(ax.get_ylabel(), y=1.0, ha="right")
        ax.vlines(40, 1e-6, 1e4, linestyles='dashed')

    fig.savefig(join(outdir, 'ptegm_2mu2e.png'))
    fig.savefig(join(outdir, 'ptegm_2mu2e.pdf'))
    plt.close(fig)

    #### mu-type, pt
    fig, axes = plt.subplots(1,2,figsize=(16,6))
    fig.subplots_adjust(wspace=0.15)

    bkgptmu = outputs['bkg']['ptmu'].integrate('channel', slice(1,2))
    hist.plot1d(bkgptmu, overlay='cat', ax=axes[0], stack=True, overflow='over',
                line_opts=None, fill_opts=fill_opts, error_opts=error_opts)

    sigptmu = outputs['sig-2mu2e']['ptmu'].integrate('channel', slice(1,2))
    hist.plot1d(sigptmu, overlay='dataset', ax=axes[0], overflow='over', clear=False)

    dataptmu = outputs['data']['ptmu'].integrate('channel', slice(1,2))
    hist.plot1d(dataptmu, overlay='cat', ax=axes[1], overflow='over', error_opts=data_err_opts)

    axes[0].set_title('[2mu2e|SR] mu-type leptonjet pT', x=0.0, ha="left")
    axes[1].set_title('[2mu2e|CR] mu-type leptonjet pT', x=0.0, ha="left")

    axes[0].legend(*groupHandleLabel(axes[0]), prop={'size': 8,}, ncol=3)

    for ax in axes:
        ax.set_yscale('log')
        ax.autoscale(axis='both', tight=True)
        ax.text(1,1,'59.74/fb (13TeV)', ha='right', va='bottom', transform=ax.transAxes)
        ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
        ax.set_ylabel(ax.get_ylabel(), y=1.0, ha="right")
        ax.vlines(40, 1e-6, 1e4, linestyles='dashed')

    fig.savefig(join(outdir, 'ptmu_2mu2e.png'))
    fig.savefig(join(outdir, 'ptmu_2mu2e.pdf'))
    plt.close(fig)

    #### mu-type, mass
    fig, axes = plt.subplots(1,2,figsize=(16,6))
    fig.subplots_adjust(wspace=0.15)

    bkgmassmu = outputs['bkg']['massmu'].integrate('channel', slice(1,2))
    hist.plot1d(bkgmassmu, overlay='cat', ax=axes[0], stack=True, overflow='over',
                line_opts=None, fill_opts=fill_opts, error_opts=error_opts)

    sigmassmu = outputs['sig-2mu2e']['massmu'].integrate('channel', slice(1,2))
    hist.plot1d(sigptmu, overlay='dataset', ax=axes[0], overflow='over', clear=False)

    datamassmu = outputs['data']['massmu'].integrate('channel', slice(1,2))
    hist.plot1d(datamassmu, overlay='cat', ax=axes[1], overflow='over', error_opts=data_err_opts)

    axes[0].set_title('[2mu2e|SR] mu-type leptonjet mass', x=0.0, ha="left")
    axes[1].set_title('[2mu2e|CR] mu-type leptonjet mass', x=0.0, ha="left")

    axes[0].legend(*groupHandleLabel(axes[0]), prop={'size': 8,}, ncol=3)

    for ax in axes:
        # ax.set_yscale('log')
        left, right = ax.get_xlim()
        ax.set_xticks(np.arange(left, right+10, 10))
        ax.autoscale(axis='both', tight=True)
        ax.text(1,1,'59.74/fb (13TeV)', ha='right', va='bottom', transform=ax.transAxes)
        ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
        ax.set_ylabel(ax.get_ylabel(), y=1.0, ha="right")

    fig.savefig(join(outdir, 'massmu_2mu2e.png'))
    fig.savefig(join(outdir, 'massmu_2mu2e.pdf'))
    plt.close(fig)


    ## CHANNEL - 4mu
    #### leading
    fig, axes = plt.subplots(1,2,figsize=(16,6))
    fig.subplots_adjust(wspace=0.15)

    bkgpt0 = outputs['bkg']['pt0'].integrate('channel', slice(2,3))
    hist.plot1d(bkgpt0, overlay='cat', ax=axes[0], stack=True, overflow='over',
                line_opts=None, fill_opts=fill_opts, error_opts=error_opts)

    sigpt0 = outputs['sig-4mu']['pt0'].integrate('channel', slice(2,3))
    hist.plot1d(sigpt0, overlay='dataset', ax=axes[0], overflow='over', clear=False)

    datapt0 = outputs['data']['pt0'].integrate('channel', slice(2,3))
    hist.plot1d(datapt0, overlay='cat', ax=axes[1], overflow='over', error_opts=data_err_opts)

    axes[0].set_title('[4mu|SR] leptonjet leading pT', x=0.0, ha="left")
    axes[1].set_title('[4mu|CR] leptonjet leading pT', x=0.0, ha="left")

    axes[0].legend(*groupHandleLabel(axes[0]), prop={'size': 8,}, ncol=3)

    for ax in axes:
        ax.set_yscale('log')
        ax.autoscale(axis='both', tight=True)
        ax.text(1,1,'59.74/fb (13TeV)', ha='right', va='bottom', transform=ax.transAxes)
        ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
        ax.set_ylabel(ax.get_ylabel(), y=1.0, ha="right")

    fig.savefig(join(outdir, 'pt0_4mu.png'))
    fig.savefig(join(outdir, 'pt0_4mu.pdf'))
    plt.close(fig)

    #### subleading
    fig, axes = plt.subplots(1,2,figsize=(16,6))
    fig.subplots_adjust(wspace=0.15)

    bkgpt1 = outputs['bkg']['pt1'].integrate('channel', slice(2,3))
    hist.plot1d(bkgpt1, overlay='cat', ax=axes[0], stack=True, overflow='over',
                line_opts=None, fill_opts=fill_opts, error_opts=error_opts)

    sigpt1 = outputs['sig-4mu']['pt1'].integrate('channel', slice(2,3))
    hist.plot1d(sigpt1, overlay='dataset', ax=axes[0], overflow='over', clear=False)

    datapt1 = outputs['data']['pt1'].integrate('channel', slice(2,3))
    hist.plot1d(datapt1, overlay='cat', ax=axes[1], overflow='over', error_opts=data_err_opts)

    axes[0].set_title('[4mu|SR] leptonjet subleading pT', x=0.0, ha="left")
    axes[1].set_title('[4mu|CR] leptonjet subleading pT', x=0.0, ha="left")

    axes[0].legend(*groupHandleLabel(axes[0]), prop={'size': 8,}, ncol=3)

    for ax in axes:
        ax.set_yscale('log')
        ax.autoscale(axis='both', tight=True)
        ax.text(1,1,'59.74/fb (13TeV)', ha='right', va='bottom', transform=ax.transAxes)
        ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
        ax.set_ylabel(ax.get_ylabel(), y=1.0, ha="right")

    fig.savefig(join(outdir, 'pt1_4mu.png'))
    fig.savefig(join(outdir, 'pt1_4mu.pdf'))
    plt.close(fig)

    #### invM(lj0, lj1)
    fig, axes = plt.subplots(1,2,figsize=(16,6))
    fig.subplots_adjust(wspace=0.15)

    bkginvm = outputs['bkg']['invm'].integrate('channel', slice(2,3))
    hist.plot1d(bkginvm, overlay='cat', ax=axes[0], stack=True, overflow='over',
                line_opts=None, fill_opts=fill_opts, error_opts=error_opts)

    siginvm = outputs['sig-4mu']['invm'].integrate('channel', slice(2,3))
    hist.plot1d(siginvm, overlay='dataset', ax=axes[0], overflow='over', clear=False)

    datainvm = outputs['data']['invm'].integrate('channel', slice(2,3))
    hist.plot1d(datainvm, overlay='cat', ax=axes[1], overflow='over', error_opts=data_err_opts)

    axes[0].set_title('[4mu|SR] leptonjet pair invM', x=0.0, ha="left")
    axes[1].set_title('[4mu|CR] leptonjet pair invM', x=0.0, ha="left")

    axes[0].legend(*groupHandleLabel(axes[0]), prop={'size': 8,}, ncol=3)

    for ax in axes:
        ax.set_yscale('log')
        ax.autoscale(axis='both', tight=True)
        ax.text(1,1,'59.74/fb (13TeV)', ha='right', va='bottom', transform=ax.transAxes)
        ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
        ax.set_ylabel(ax.get_ylabel(), y=1.0, ha="right")

    fig.savefig(join(outdir, 'invm_4mu.png'))
    fig.savefig(join(outdir, 'invm_4mu.pdf'))
    plt.close(fig)


    if args.sync:
        webdir = 'wsi@lxplus.cern.ch:/eos/user/w/wsi/www/public/firehydrant'
        cmd = f'rsync -az --exclude ".*" --delete {outdir} {webdir}'
        print(f"--> sync with: {webdir}")
        os.system(cmd)
