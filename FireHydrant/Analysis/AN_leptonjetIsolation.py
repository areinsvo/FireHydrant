#!/usr/bin/env python
"""For AN
leptonjet isolation
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


class LjTkIsoProcessorSig(processor.ProcessorABC):
    def __init__(self):
        dataset_axis = hist.Cat('dataset', 'dataset')
        sumpt_axis = hist.Bin('sumpt', '$\sum p_T$ [GeV]', 50, 0, 50)
        iso_axis = hist.Bin('iso', 'Isolation', 50, 0, 1)
        self._accumulator = processor.dict_accumulator({
            'sumpt': hist.Hist('Counts', dataset_axis, sumpt_axis),
            'pfiso': hist.Hist('Counts', dataset_axis, iso_axis),
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
            sumtkpt=df['pfjet_tkPtSum05'].content,
            pfiso=df['pfjet_pfIsolationNoPU05'].content,
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
        ljdsamuSubset = fromNestNestIndexArray(df['dsamuon_isSubsetFilteredCosmic1Leg'], awkward.fromiter(df['pfjet_pfcand_dsamuonIdx']))
        leptonjets.add_attributes(nocosmic=(ljdsamuSubset.sum()==0))
        leptonjets = leptonjets[(leptonjets.nocosmic)&(leptonjets.isneutral)]

        ## __ twoleptonjets__
        twoleptonjets = leptonjets.counts>=2
        dileptonjets = leptonjets[twoleptonjets]
        # wgt = weight[twoleptonjets]

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

        ljones = dileptonjets.pt.ones_like()
        output['sumpt'].fill(dataset=dataset, sumpt=dileptonjets.sumtkpt.flatten(), )
        output['pfiso'].fill(dataset=dataset, iso=dileptonjets.pfiso.flatten(), )

        return output

    def postprocess(self, accumulator):
        return accumulator


class LjTkIsoProcessor(processor.ProcessorABC):
    def __init__(self, data_type='bkg', bothNeutral=True):
        dataset_axis = hist.Cat('dataset', 'dataset')
        sumpt_axis = hist.Bin('sumpt', '$\sum p_T$ [GeV]', 50, 0, 50)
        iso_axis = hist.Bin('iso', 'Isolation', 50, 0, 1)
        channel_axis = hist.Bin('channel', 'channel', 3, 0, 3)
        self._accumulator = processor.dict_accumulator({
            'sumpt': hist.Hist('Counts', dataset_axis, sumpt_axis, channel_axis),
            'pfiso': hist.Hist('Counts', dataset_axis, iso_axis, channel_axis),
            'isodbeta': hist.Hist('Counts', dataset_axis, iso_axis, channel_axis),
            'minpfiso': hist.Hist('Counts', dataset_axis, iso_axis, channel_axis),
            'maxpfiso': hist.Hist('Counts', dataset_axis, iso_axis, channel_axis),
        })

        self.pucorrs = get_pu_weights_function()
        ## NOT applied for now
        self.nlo_w = get_nlo_weight_function('w')
        self.nlo_z = get_nlo_weight_function('z')

        self.data_type = data_type
        self.bothNeutral = bothNeutral

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
            sumtkpt=df['pfjet_tkPtSum05'].content,
            pfiso=df['pfjet_pfIsolationNoPU05'].content,
            isodbeta=df['pfjet_pfiso'].content,
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
        ljdsamuSubset = fromNestNestIndexArray(df['dsamuon_isSubsetFilteredCosmic1Leg'], awkward.fromiter(df['pfjet_pfcand_dsamuonIdx']))
        leptonjets.add_attributes(nocosmic=(ljdsamuSubset.sum()==0))
        leptonjets = leptonjets[(leptonjets.nocosmic)&(leptonjets.pt>30)]

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

        # isControl = (np.abs(lj0.p4.delta_phi(lj1.p4))<np.pi/2).flatten()
        # if self.data_type!='data':
        #     dileptonjets = dileptonjets[isControl]
        #     channel_ = channel_[isControl]
        #     wgt = wgt[isControl]

        mask_ = (lj0.isneutral&lj1.isneutral).flatten()
        if self.bothNeutral is False:
            mask_ = ~mask_
            mask_ = ((channel_==2)&((~lj0.isneutral&(~lj1.isneutral)).flatten())) | ((channel_==1)&mask_)

        channel_ = channel_[mask_]
        wgt = wgt[mask_]
        dileptonjets = dileptonjets[mask_]

        minpfiso = (lj0.pfiso>lj1.pfiso).astype(int)*lj1.pfiso + (lj0.pfiso<lj1.pfiso).astype(int)*lj0.pfiso
        output['minpfiso'].fill(dataset=dataset, iso=minpfiso[mask_].flatten(), channel=channel_, weight=wgt)
        maxpfiso = (lj0.pfiso>lj1.pfiso).astype(int)*lj0.pfiso + (lj0.pfiso<lj1.pfiso).astype(int)*lj1.pfiso
        output['maxpfiso'].fill(dataset=dataset, iso=maxpfiso[mask_].flatten(), channel=channel_, weight=wgt)

        ljones = dileptonjets.pt.ones_like()
        output['sumpt'].fill(dataset=dataset, sumpt=dileptonjets.sumtkpt.flatten(), channel=(channel_*ljones).flatten(), weight=(wgt*ljones).flatten())
        output['pfiso'].fill(dataset=dataset, iso=dileptonjets.pfiso.flatten(), channel=(channel_*ljones).flatten(), weight=(wgt*ljones).flatten())
        output['isodbeta'].fill(dataset=dataset, iso=dileptonjets.isodbeta.flatten(), channel=(channel_*ljones).flatten(), weight=(wgt*ljones).flatten())

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
    import re
    from os.path import join, isdir, splitext
    from FireHydrant.Analysis.PlottingOptions import *

    reldir = splitext(__file__)[0].replace('_', '/')
    outdir = join(os.getenv('FH_BASE'), "Imgs", reldir)
    if not isdir(outdir): os.makedirs(outdir)

    output_2mu2e = processor.run_uproot_job(sigDS_2mu2e,
                                    treename='ffNtuplizer/ffNtuple',
                                    processor_instance=LjTkIsoProcessor(data_type='sig-2mu2e'),
                                    executor=processor.futures_executor,
                                    executor_args=dict(workers=12, flatten=False),
                                    chunksize=500000,
                                    )

    output_4mu = processor.run_uproot_job(sigDS_4mu,
                                    treename='ffNtuplizer/ffNtuple',
                                    processor_instance=LjTkIsoProcessor(data_type='sig-4mu'),
                                    executor=processor.futures_executor,
                                    executor_args=dict(workers=12, flatten=False),
                                    chunksize=500000,
                                    )

    output_bkg = processor.run_uproot_job(bkgDS,
                                    treename='ffNtuplizer/ffNtuple',
                                    processor_instance=LjTkIsoProcessor(data_type='bkg'),
                                    executor=processor.futures_executor,
                                    executor_args=dict(workers=12, flatten=False),
                                    chunksize=500000,
                                    )

    output_data = processor.run_uproot_job(dataDS,
                                    treename='ffNtuplizer/ffNtuple',
                                    processor_instance=LjTkIsoProcessor(data_type='data'),
                                    executor=processor.futures_executor,
                                    executor_args=dict(workers=12, flatten=False),
                                    chunksize=500000,
                                    )

    sampleSig = re.compile('mXX-150_mA-0p25_lxy-300|mXX-500_mA-1p2_lxy-300|mXX-800_mA-5_lxy-300')

    ## sum track pt
    # fig, (ax, rax) = make_ratio_plot(output_bkg['sumpt'].integrate('channel', slice(1,2)),
    #                                  output_data['sumpt'].integrate('channel', slice(1,2)),
    #                                  title='[$2\mu 2e$,data-CR] leptonjet track $\sum p_T$')
    # fig.savefig(join(outdir, 'ljiso-tkptsum_2mu2e.png'))
    # fig.savefig(join(outdir, 'ljiso-tkptsum_2mu2e.pdf'))
    # plt.close(fig)

    # fig, (ax, rax) = make_ratio_plot(output_bkg['sumpt'].integrate('channel', slice(2,3)),
    #                                  output_data['sumpt'].integrate('channel', slice(2,3)),
    #                                  title='[$4\mu$,data-CR] leptonjet track $\sum p_T$')
    # fig.savefig(join(outdir, 'ljiso-tkptsum_4mu.png'))
    # fig.savefig(join(outdir, 'ljiso-tkptsum_4mu.pdf'))
    # plt.close(fig)

    ## pfiso05 noPU
    # fig, (ax, rax) = make_ratio_plot(output_bkg['pfiso'].integrate('channel', slice(1,2)),
    #                                  output_data['pfiso'].integrate('channel', slice(1,2)),
    #                                  sigh=output_2mu2e['pfiso'][sampleSig].integrate('channel', slice(1,2)),
    #                                  title='[$2\mu 2e$,data-CR] leptonjet pfiso05-noPU', overflow='none')
    # fig.savefig(join(outdir, 'ljiso-pfiso05nopu_2mu2e.png'))
    # fig.savefig(join(outdir, 'ljiso-pfiso05nopu_2mu2e.pdf'))
    # plt.close(fig)

    # fig, (ax, rax) = make_ratio_plot(output_bkg['pfiso'].integrate('channel', slice(2,3)),
    #                                  output_data['pfiso'].integrate('channel', slice(2,3)),
    #                                  sigh=output_4mu['pfiso'][sampleSig].integrate('channel', slice(2,3)),
    #                                  title='[$4\mu$,data-CR] leptonjet pfiso05-noPU', overflow='none')
    # fig.savefig(join(outdir, 'ljiso-pfiso05nopu_4mu.png'))
    # fig.savefig(join(outdir, 'ljiso-pfiso05nopu_4mu.pdf'))
    # plt.close(fig)

    ## min pfiso05 noPU
    fig, (ax, rax) = make_ratio_plot(output_bkg['minpfiso'].integrate('channel', slice(1,2)),
                                     output_data['minpfiso'].integrate('channel', slice(1,2)),
                                     sigh=output_2mu2e['minpfiso'][sampleSig].integrate('channel', slice(1,2)),
                                     title='[$2\mu 2e$,data-CR] leptonjet min pfiso05-noPU', overflow='none')
    fig.savefig(join(outdir, 'ljiso-minpfiso05nopu_2mu2e.png'))
    fig.savefig(join(outdir, 'ljiso-minpfiso05nopu_2mu2e.pdf'))
    plt.close(fig)

    fig, (ax, rax) = make_ratio_plot(output_bkg['minpfiso'].integrate('channel', slice(2,3)),
                                     output_data['minpfiso'].integrate('channel', slice(2,3)),
                                     sigh=output_4mu['minpfiso'][sampleSig].integrate('channel', slice(2,3)),
                                     title='[$4\mu$,data-CR] leptonjet min pfiso05-noPU', overflow='none')
    fig.savefig(join(outdir, 'ljiso-minpfiso05nopu_4mu.png'))
    fig.savefig(join(outdir, 'ljiso-minpfiso05nopu_4mu.pdf'))
    plt.close(fig)

    ## max pfiso05 noPU
    # fig, (ax, rax) = make_ratio_plot(output_bkg['maxpfiso'].integrate('channel', slice(1,2)),
    #                                  output_data['maxpfiso'].integrate('channel', slice(1,2)),
    #                                  sigh=output_2mu2e['maxpfiso'][sampleSig].integrate('channel', slice(1,2)),
    #                                  title='[$2\mu 2e$,data-CR] leptonjet max pfiso05-noPU', overflow='none')
    # fig.savefig(join(outdir, 'ljiso-maxpfiso05nopu_2mu2e.png'))
    # fig.savefig(join(outdir, 'ljiso-maxpfiso05nopu_2mu2e.pdf'))
    # plt.close(fig)

    # fig, (ax, rax) = make_ratio_plot(output_bkg['maxpfiso'].integrate('channel', slice(2,3)),
    #                                  output_data['maxpfiso'].integrate('channel', slice(2,3)),
    #                                  sigh=output_4mu['maxpfiso'][sampleSig].integrate('channel', slice(2,3)),
    #                                  title='[$4\mu$,data-CR] leptonjet max pfiso05-noPU', overflow='none')
    # fig.savefig(join(outdir, 'ljiso-maxpfiso05nopu_4mu.png'))
    # fig.savefig(join(outdir, 'ljiso-maxpfiso05nopu_4mu.pdf'))
    # plt.close(fig)


    ## pfiso dbeta
    # fig, (ax, rax) = make_ratio_plot(output_bkg['isodbeta'].integrate('channel', slice(1,2)),
    #                                  output_data['isodbeta'].integrate('channel', slice(1,2)),
    #                                  sigh=output_2mu2e['isodbeta'][sampleSig].integrate('channel', slice(1,2)),
    #                                  title='[$2\mu 2e$,data-CR] leptonjet pfiso-dbeta', overflow='none')
    # fig.savefig(join(outdir, 'ljiso-pfisdbeta_2mu2e.png'))
    # fig.savefig(join(outdir, 'ljiso-pfisdbeta_2mu2e.pdf'))
    # plt.close(fig)

    # fig, (ax, rax) = make_ratio_plot(output_bkg['isodbeta'].integrate('channel', slice(2,3)),
    #                                  output_data['isodbeta'].integrate('channel', slice(2,3)),
    #                                  sigh=output_4mu['isodbeta'][sampleSig].integrate('channel', slice(2,3)),
    #                                  title='[$4\mu$,data-CR] leptonjet pfiso-dbeta', overflow='none')
    # fig.savefig(join(outdir, 'ljiso-pfisodbeta_4mu.png'))
    # fig.savefig(join(outdir, 'ljiso-pfisodbeta_4mu.pdf'))
    # plt.close(fig)

    # ----------------------------------------------------------

    output_2mu2e = processor.run_uproot_job(sigDS_2mu2e,
                                    treename='ffNtuplizer/ffNtuple',
                                    processor_instance=LjTkIsoProcessor(data_type='sig-2mu2e', bothNeutral=False),
                                    executor=processor.futures_executor,
                                    executor_args=dict(workers=12, flatten=False),
                                    chunksize=500000,
                                    )

    output_4mu = processor.run_uproot_job(sigDS_4mu,
                                    treename='ffNtuplizer/ffNtuple',
                                    processor_instance=LjTkIsoProcessor(data_type='sig-4mu', bothNeutral=False),
                                    executor=processor.futures_executor,
                                    executor_args=dict(workers=12, flatten=False),
                                    chunksize=500000,
                                    )

    output_bkg = processor.run_uproot_job(bkgDS,
                                    treename='ffNtuplizer/ffNtuple',
                                    processor_instance=LjTkIsoProcessor(data_type='bkg', bothNeutral=False),
                                    executor=processor.futures_executor,
                                    executor_args=dict(workers=12, flatten=False),
                                    chunksize=500000,
                                    )

    output_data = processor.run_uproot_job(dataDS,
                                    treename='ffNtuplizer/ffNtuple',
                                    processor_instance=LjTkIsoProcessor(data_type='data', bothNeutral=False),
                                    executor=processor.futures_executor,
                                    executor_args=dict(workers=12, flatten=False),
                                    chunksize=500000,
                                    )

    ## min pfiso05 noPU
    fig, (ax, rax) = make_ratio_plot(output_bkg['minpfiso'].integrate('channel', slice(1,2)),
                                     output_data['minpfiso'].integrate('channel', slice(1,2)),
                                     sigh=output_2mu2e['minpfiso'][sampleSig].integrate('channel', slice(1,2)),
                                     title='[$2\mu 2e$] leptonjet(~neutral) min pfiso05-noPU', overflow='none')
    fig.savefig(join(outdir, 'chaljiso-minpfiso05nopu_2mu2e.png'))
    fig.savefig(join(outdir, 'chaljiso-minpfiso05nopu_2mu2e.pdf'))
    plt.close(fig)

    fig, (ax, rax) = make_ratio_plot(output_bkg['minpfiso'].integrate('channel', slice(2,3)),
                                     output_data['minpfiso'].integrate('channel', slice(2,3)),
                                     sigh=output_4mu['minpfiso'][sampleSig].integrate('channel', slice(2,3)),
                                     title='[$4\mu$] leptonjet(~neutral) min pfiso05-noPU', overflow='none')
    fig.savefig(join(outdir, 'chaljiso-minpfiso05nopu_4mu.png'))
    fig.savefig(join(outdir, 'chaljiso-minpfiso05nopu_4mu.pdf'))
    plt.close(fig)

    # ----------------------------------------------------------


    # output_sig = processor.run_uproot_job(sigDS,
    #                                 treename='ffNtuplizer/ffNtuple',
    #                                 processor_instance=LjTkIsoProcessorSig(),
    #                                 executor=processor.futures_executor,
    #                                 executor_args=dict(workers=12, flatten=False),
    #                                 chunksize=500000,
    #                                 )
    # channel_2mu2e = re.compile('2mu2e.*$')
    # channel_4mu = re.compile('4mu.*$')

    # fig, ax = plt.subplots(figsize=(8,6))
    # hist.plot1d(output_sig['sumpt'][channel_2mu2e].sum('dataset'), ax=ax, overflow='over', density=True)
    # hs_2mu2e = tuple(ax.get_legend_handles_labels()[0])
    # hist.plot1d(output_sig['sumpt'][channel_4mu].sum('dataset'), ax=ax, overflow='over', density=True, clear=False)
    # hs_ = ax.get_legend_handles_labels()[0]
    # hs_4mu = tuple([h for h in hs_ if h not in hs_2mu2e])

    # ax.set_title('[signalMC|lxy300cm] leptonjet track $\sum p_T$', x=0, ha='left')
    # ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
    # ax.set_ylabel(ax.get_ylabel()+'/1GeV', y=1.0, ha="right")
    # ax.set_yscale('log')
    # ax.autoscale(axis='both', tight=True)
    # ax.legend([hs_4mu, hs_2mu2e], ['$4\mu$', '$2\mu 2e$'])

    # fig.savefig(join(outdir, 'ljiso-tkptsum_sig.png'))
    # fig.savefig(join(outdir, 'ljiso-tkptsum_sig.pdf'))
    # plt.close(fig)


    # fig, ax = plt.subplots(figsize=(8,6))
    # hist.plot1d(output_sig['pfiso'][channel_2mu2e].sum('dataset'), ax=ax, overflow='over', density=True)
    # hs_2mu2e = tuple(ax.get_legend_handles_labels()[0])
    # hist.plot1d(output_sig['pfiso'][channel_4mu].sum('dataset'), ax=ax, overflow='over', density=True, clear=False)
    # hs_ = ax.get_legend_handles_labels()[0]
    # hs_4mu = tuple([h for h in hs_ if h not in hs_2mu2e])

    # ax.set_title('[signalMC|lxy300cm] leptonjet pfiso05-nopu', x=0, ha='left')
    # ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
    # ax.set_ylabel(ax.get_ylabel()+'/1GeV', y=1.0, ha="right")
    # ax.set_yscale('log')
    # ax.autoscale(axis='both', tight=True)
    # ax.legend([hs_4mu, hs_2mu2e], ['$4\mu$', '$2\mu 2e$'])

    # fig.savefig(join(outdir, 'ljiso-pfiso05nopu_sig.png'))
    # fig.savefig(join(outdir, 'ljiso-pfiso05nopu_sig.pdf'))
    # plt.close(fig)

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
