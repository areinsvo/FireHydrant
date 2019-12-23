#!/usr/bin/env python
"""For AN
leptonjet pair delta phi
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

dml = DatasetMapLoader()
bkgDS, bkgMAP, bkgSCALE = dml.fetch('bkg')
dataDS, dataMAP = dml.fetch('data')

sdml = SigDatasetMapLoader()
sigDS, sigSCALE = sdml.fetch('all')
sigDS_2mu2e, sigSCALE_2mu2e = sdml.fetch('2mu2e')
sigDS_4mu, sigSCALE_4mu = sdml.fetch('4mu')

class LJPairDphiProcessor(processor.ProcessorABC):
    def __init__(self):
        dataset_axis = hist.Cat('dataset', 'dataset')
        dphi_axis = hist.Bin('dphi', '$\Delta\phi$', 50, 0, np.pi)
        self._accumulator = processor.dict_accumulator({
            'dphi': hist.Hist('Norm. Frequency', dataset_axis, dphi_axis),
        })

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, df):
        output = self.accumulator.identity()
        dataset = df['dataset']

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
        leptonjets = leptonjets[leptonjets.isneutral]

        ## __ twoleptonjets__
        twoleptonjets = leptonjets.counts>=2
        dileptonjets = leptonjets[twoleptonjets]

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

        output['dphi'].fill(dataset=dataset, dphi=np.abs(lj0.p4.delta_phi(lj1.p4)[channel_>0].flatten()), )

        return output

    def postprocess(self, accumulator):
        return accumulator


class LJPairDphiProcessorBkg(processor.ProcessorABC):
    def __init__(self):
        dataset_axis = hist.Cat('dataset', 'dataset')
        dphi_axis = hist.Bin('dphi', '$\Delta\phi$', 50, 0, np.pi)
        self._accumulator = processor.dict_accumulator({
            'dphi': hist.Hist('Counts', dataset_axis, dphi_axis),
        })

        self.pucorrs = get_pu_weights_function()
        ## NOT applied for now
        self.nlo_w = get_nlo_weight_function('w')
        self.nlo_z = get_nlo_weight_function('z')

        self.data_type = 'bkg'

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, df):
        output = self.accumulator.identity()
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
        leptonjets = leptonjets[leptonjets.isneutral]

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

        output['dphi'].fill(dataset=dataset, dphi=np.abs(lj0.p4.delta_phi(lj1.p4)[channel_>0].flatten()), weight=wgt[channel_>0])

        return output

    def postprocess(self, accumulator):
        origidentity = list(accumulator)
        for k in origidentity:
            accumulator[k].scale(bkgSCALE, axis='dataset')
            accumulator[k] = accumulator[k].group("dataset",
                                                hist.Cat("cat", "datasets", sorting='integral'),
                                                bkgMAP)
        return accumulator


class LJPairDphiProcessorTotal(processor.ProcessorABC):
    def __init__(self, data_type='bkg'):
        dataset_axis = hist.Cat('dataset', 'dataset')
        dphi_axis = hist.Bin('dphi', '$\Delta\phi$', 20, 0, np.pi)
        channel_axis = hist.Bin('channel', 'channel', 3, 0, 3)
        self._accumulator = processor.dict_accumulator({
            'dphi-neu': hist.Hist('Counts', dataset_axis, dphi_axis, channel_axis),
            'dphi-cha': hist.Hist('Counts', dataset_axis, dphi_axis, channel_axis),
            'dphi-0mucha': hist.Hist('Counts', dataset_axis, dphi_axis, channel_axis),
            'dphi-1mucha': hist.Hist('Counts', dataset_axis, dphi_axis, channel_axis),
            'dphi-01mucha': hist.Hist('Counts', dataset_axis, dphi_axis, channel_axis),
        })

        self.pucorrs = get_pu_weights_function()
        ## NOT applied for now
        self.nlo_w = get_nlo_weight_function('w')
        self.nlo_z = get_nlo_weight_function('z')

        self.data_type = data_type

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, df):
        output = self.accumulator.identity()
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
            px=df['pfjet_p4.fCoordinates.fX'].content,
            py=df['pfjet_p4.fCoordinates.fY'].content,
            pz=df['pfjet_p4.fCoordinates.fZ'].content,
            energy=df['pfjet_p4.fCoordinates.fT'].content,
            mintkdist=df['pfjet_pfcands_minTwoTkDist'].content,
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
        leptonjets.add_attributes(mucharged=(leptonjets.iseltype | (leptonjets.ismutype&(leptonjets.qsum!=0))))
        ljdsamuSubset = fromNestNestIndexArray(df['dsamuon_isSubsetFilteredCosmic1Leg'], awkward.fromiter(df['pfjet_pfcand_dsamuonIdx']))
        leptonjets.add_attributes(nocosmic=(ljdsamuSubset.sum()==0))
        leptonjets = leptonjets[(leptonjets.nocosmic)&(leptonjets.pt>30)&(leptonjets.mintkdist<50)]

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

        ljBothNeutral = ((lj0.isneutral)&(lj1.isneutral)).flatten()
        ljBothCharged = ((~lj0.isneutral).sum()+(~lj1.isneutral).sum())==((lj0.ismutype).sum()+(lj1.ismutype).sum())
        lj4mu0cha = (channel_==2)&((lj0.mucharged&lj1.isneutral).flatten())
        lj4mu1cha = (channel_==2)&((lj0.isneutral&(lj1.mucharged)).flatten())
        lj4mu0or1cha = (channel_==2)&((lj0.isneutral&(lj1.mucharged)).flatten()|(lj0.mucharged&lj1.isneutral).flatten())
        output['dphi-neu'].fill(dataset=dataset, dphi=np.abs(lj0.p4.delta_phi(lj1.p4)[ljBothNeutral].flatten()), channel=channel_[ljBothNeutral], weight=wgt[ljBothNeutral])
        output['dphi-cha'].fill(dataset=dataset, dphi=np.abs(lj0.p4.delta_phi(lj1.p4)[ljBothCharged].flatten()), channel=channel_[ljBothCharged], weight=wgt[ljBothCharged])
        output['dphi-0mucha'].fill(dataset=dataset, dphi=np.abs(lj0.p4.delta_phi(lj1.p4)[lj4mu0cha].flatten()), channel=channel_[lj4mu0cha], weight=wgt[lj4mu0cha])
        output['dphi-1mucha'].fill(dataset=dataset, dphi=np.abs(lj0.p4.delta_phi(lj1.p4)[lj4mu1cha].flatten()), channel=channel_[lj4mu1cha], weight=wgt[lj4mu1cha])
        output['dphi-01mucha'].fill(dataset=dataset, dphi=np.abs(lj0.p4.delta_phi(lj1.p4)[lj4mu0or1cha].flatten()), channel=channel_[lj4mu0or1cha], weight=wgt[lj4mu0or1cha])

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




if __name__ == "__main__":
    import os
    from os.path import join, isdir, splitext
    from FireHydrant.Analysis.PlottingOptions import *

    reldir = splitext(__file__)[0].replace('_', '/')
    outdir = join(os.getenv('FH_BASE'), "Imgs", reldir)
    if not isdir(outdir): os.makedirs(outdir)

    import re

    # output = processor.run_uproot_job(sigDS,
    #                                 treename='ffNtuplizer/ffNtuple',
    #                                 processor_instance=LJPairDphiProcessor(),
    #                                 executor=processor.futures_executor,
    #                                 executor_args=dict(workers=12, flatten=False),
    #                                 chunksize=500000,
    #                                 )

    # outputbkg = processor.run_uproot_job(bkgDS,
    #                                 treename='ffNtuplizer/ffNtuple',
    #                                 processor_instance=LJPairDphiProcessorBkg(),
    #                                 executor=processor.futures_executor,
    #                                 executor_args=dict(workers=12, flatten=False),
    #                                 chunksize=500000,
    #                                 )

    # ----------------------------------------------------------

    channel_2mu2e = re.compile('2mu2e.*$')
    channel_4mu = re.compile('4mu.*$')

    # fig, ax = plt.subplots(figsize=(8,6))
    # hist.plot1d(output['dphi'][channel_2mu2e].sum('dataset'), ax=ax, overflow='none', density=True)
    # hs_2mu2e = tuple(ax.get_legend_handles_labels()[0])
    # hist.plot1d(output['dphi'][channel_4mu].sum('dataset'), ax=ax, overflow='none', density=True, clear=False)
    # hs_ = ax.get_legend_handles_labels()[0]
    # hs_4mu = tuple([h for h in hs_ if h not in hs_2mu2e])

    # frac_2mu2e = output['dphi'][channel_2mu2e].sum('dataset').integrate('dphi', slice(0, np.pi/2)).values()[()]/output['dphi'][channel_2mu2e].sum('dataset', 'dphi').values()[()]
    # frac_4mu   = output['dphi'][channel_4mu].sum('dataset').integrate('dphi', slice(0, np.pi/2)).values()[()]/output['dphi'][channel_4mu].sum('dataset', 'dphi').values()[()]

    # ax.set_title('[signalMC|lxy300cm] leptonjet pair $\Delta\phi$', x=0, ha='left')
    # ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
    # ax.set_ylabel(ax.get_ylabel()+'/$\pi$/50', y=1.0, ha="right")
    # ax.set_yscale('log')
    # ax.autoscale(axis='both', tight=True)
    # ax.vlines([np.pi/2,], 0, 1, linestyles='dashed', colors='tab:gray', transform=ax.get_xaxis_transform())
    # ax.legend([hs_4mu, hs_2mu2e], [f'4mu (<$\pi$/2: {frac_4mu*100:.2f}%)', f'2mu2e (<$\pi$/2: {frac_2mu2e*100:.2f}%)'])

    # fig.savefig(join(outdir, 'ljpairDphi.png'))
    # fig.savefig(join(outdir, 'ljpairDphi.pdf'))
    # plt.close(fig)

    # ----------------------------------------------------------

    # fig, ax = plt.subplots(figsize=(8,6))
    # hist.plot1d(outputbkg['dphi'], ax=ax, overlay='cat', stack=True, line_opts=None, fill_opts=fill_opts, error_opts=error_opts)
    # ax.set_title('[BackgroundMC] leptonjet pair $\Delta\phi$', x=0, ha='left')
    # ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
    # ax.set_ylabel(ax.get_ylabel()+'/$\pi$/50', y=1.0, ha="right")
    # ax.set_yscale('log')
    # ax.autoscale(axis='both', tight=True)
    # ax.text(1,1,'59.74/fb (13TeV)', ha='right', va='bottom', transform=ax.transAxes)
    # ax.vlines([np.pi/2,], 0, 1, linestyles='dashed', colors='tab:gray', transform=ax.get_xaxis_transform())

    # fig.savefig(join(outdir, 'ljpairDphi_bkg.png'))
    # fig.savefig(join(outdir, 'ljpairDphi_bkg.pdf'))
    # plt.close(fig)

    # ----------------------------------------------------------

    output_2mu2e = processor.run_uproot_job(sigDS_2mu2e,
                                    treename='ffNtuplizer/ffNtuple',
                                    processor_instance=LJPairDphiProcessorTotal(data_type='sig-2mu2e'),
                                    executor=processor.futures_executor,
                                    executor_args=dict(workers=12, flatten=False),
                                    chunksize=500000,
                                    )

    output_4mu = processor.run_uproot_job(sigDS_4mu,
                                    treename='ffNtuplizer/ffNtuple',
                                    processor_instance=LJPairDphiProcessorTotal(data_type='sig-4mu'),
                                    executor=processor.futures_executor,
                                    executor_args=dict(workers=12, flatten=False),
                                    chunksize=500000,
                                    )

    output_bkg = processor.run_uproot_job(bkgDS,
                                    treename='ffNtuplizer/ffNtuple',
                                    processor_instance=LJPairDphiProcessorTotal(data_type='bkg'),
                                    executor=processor.futures_executor,
                                    executor_args=dict(workers=12, flatten=False),
                                    chunksize=500000,
                                    )

    output_data = processor.run_uproot_job(dataDS,
                                    treename='ffNtuplizer/ffNtuple',
                                    processor_instance=LJPairDphiProcessorTotal(data_type='data'),
                                    executor=processor.futures_executor,
                                    executor_args=dict(workers=12, flatten=False),
                                    chunksize=500000,
                                    )

    sampleSig = re.compile('mXX-150_mA-0p25_lxy-300|mXX-500_mA-1p2_lxy-300|mXX-800_mA-5_lxy-300')

    fig, (ax, rax) = make_ratio_plot(output_bkg['dphi-neu'].integrate('channel', slice(1,2)),
                                     output_data['dphi-neu'].integrate('channel', slice(1,2)),
                                     sigh=output_2mu2e['dphi-neu'][sampleSig].integrate('channel', slice(1,2)),
                                     title='[$2\mu 2e$, lj0,1 sumq=0] leptonJets pair $\Delta\phi$',
                                     overflow='none')
    ax.vlines([np.pi/2,], 0, 1, linestyles='dashed', colors='tab:gray', transform=ax.get_xaxis_transform())

    fig.savefig(join(outdir, 'ljpairDphi-total-neulj_2mu2e.png'))
    fig.savefig(join(outdir, 'ljpairDphi-total-neulj_2mu2e.pdf'))
    plt.close(fig)

    fig, (ax, rax) = make_ratio_plot(output_bkg['dphi-cha'].integrate('channel', slice(1,2)),
                                     output_data['dphi-cha'].integrate('channel', slice(1,2)),
                                     sigh=output_2mu2e['dphi-cha'][sampleSig].integrate('channel', slice(1,2)),
                                     title='[$2\mu 2e$, ~(lj0,1 sumq=0)] leptonJets pair $\Delta\phi$', overflow='none')
    ax.vlines([np.pi/2,], 0, 1, linestyles='dashed', colors='tab:gray', transform=ax.get_xaxis_transform())

    print("## ljpairDphi-total-chalj_2mu2e")
    sigh = output_2mu2e['dphi-cha'].integrate('dataset', 'mXX-500_mA-1p2_lxy-300').integrate('channel', slice(1,2))
    bkgh = output_bkg['dphi-cha'].sum('cat').integrate('channel', slice(1,2))
    print('sig:', sigh.values()[()])
    print('bkg:', bkgh.values()[()])
    print('sig/bkg:', sigh.values()[()]/bkgh.values()[()])

    fig.savefig(join(outdir, 'ljpairDphi-total-chalj_2mu2e.png'))
    fig.savefig(join(outdir, 'ljpairDphi-total-chalj_2mu2e.pdf'))
    plt.close(fig)

    fig, (ax, rax) = make_ratio_plot(output_bkg['dphi-neu'].integrate('channel', slice(2,3)),
                                     output_data['dphi-neu'].integrate('channel', slice(2,3)),
                                     sigh=output_4mu['dphi-neu'][sampleSig].integrate('channel', slice(2,3)),
                                     title='[$4\mu$, lj0,1 sumq=0] leptonJets pair $\Delta\phi$', overflow='none')
    ax.vlines([np.pi/2,], 0, 1, linestyles='dashed', colors='tab:gray', transform=ax.get_xaxis_transform())

    fig.savefig(join(outdir, 'ljpairDphi-total-neulj_4mu.png'))
    fig.savefig(join(outdir, 'ljpairDphi-total-neulj_4mu.pdf'))
    plt.close(fig)

    fig, (ax, rax) = make_ratio_plot(output_bkg['dphi-cha'].integrate('channel', slice(2,3)),
                                     output_data['dphi-cha'].integrate('channel', slice(2,3)),
                                     sigh=output_4mu['dphi-cha'][sampleSig].integrate('channel', slice(2,3)),
                                     title='[$4\mu$, ~(lj0,1 sumq=0)] leptonJets pair $\Delta\phi$', overflow='none')
    ax.vlines([np.pi/2,], 0, 1, linestyles='dashed', colors='tab:gray', transform=ax.get_xaxis_transform())

    fig.savefig(join(outdir, 'ljpairDphi-total-chalj_4mu.png'))
    fig.savefig(join(outdir, 'ljpairDphi-total-chalj_4mu.pdf'))
    plt.close(fig)

    fig, (ax, rax) = make_ratio_plot(output_bkg['dphi-0mucha'].integrate('channel', slice(2,3)),
                                     output_data['dphi-0mucha'].integrate('channel', slice(2,3)),
                                     sigh=output_4mu['dphi-0mucha'][sampleSig].integrate('channel', slice(2,3)),
                                     title='[$4\mu$, lj0 cha,lj1 neu)] leptonJets pair $\Delta\phi$', overflow='none')
    ax.vlines([np.pi/2,], 0, 1, linestyles='dashed', colors='tab:gray', transform=ax.get_xaxis_transform())

    fig.savefig(join(outdir, 'ljpairDphi-total-0muchalj_4mu.png'))
    fig.savefig(join(outdir, 'ljpairDphi-total-0muchalj_4mu.pdf'))
    plt.close(fig)

    fig, (ax, rax) = make_ratio_plot(output_bkg['dphi-1mucha'].integrate('channel', slice(2,3)),
                                     output_data['dphi-1mucha'].integrate('channel', slice(2,3)),
                                     sigh=output_4mu['dphi-1mucha'][sampleSig].integrate('channel', slice(2,3)),
                                     title='[$4\mu$, lj0 neu, lj1 cha)] leptonJets pair $\Delta\phi$', overflow='none')
    ax.vlines([np.pi/2,], 0, 1, linestyles='dashed', colors='tab:gray', transform=ax.get_xaxis_transform())

    fig.savefig(join(outdir, 'ljpairDphi-total-1muchalj_4mu.png'))
    fig.savefig(join(outdir, 'ljpairDphi-total-1muchalj_4mu.pdf'))
    plt.close(fig)

    fig, (ax, rax) = make_ratio_plot(output_bkg['dphi-01mucha'].integrate('channel', slice(2,3)),
                                     output_data['dphi-01mucha'].integrate('channel', slice(2,3)),
                                     sigh=output_4mu['dphi-01mucha'][sampleSig].integrate('channel', slice(2,3)),
                                     title='[$4\mu$, lj0|lj1 cha)] leptonJets pair $\Delta\phi$', overflow='none')
    ax.vlines([np.pi/2,], 0, 1, linestyles='dashed', colors='tab:gray', transform=ax.get_xaxis_transform())

    print("## ljpairDphi-total-01muchalj_4mu")
    sigh = output_4mu['dphi-01mucha'].integrate('dataset', 'mXX-500_mA-1p2_lxy-300').integrate('channel', slice(2,3))
    bkgh = output_bkg['dphi-01mucha'].sum('cat').integrate('channel', slice(2,3))
    print('sig:', sigh.values()[()])
    print('bkg:', bkgh.values()[()])
    print('sig/bkg:', sigh.values()[()]/bkgh.values()[()])

    fig.savefig(join(outdir, 'ljpairDphi-total-01muchalj_4mu.png'))
    fig.savefig(join(outdir, 'ljpairDphi-total-01muchalj_4mu.pdf'))
    plt.close(fig)

    # ----------------------------------------------------------

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
