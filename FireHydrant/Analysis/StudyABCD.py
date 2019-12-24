#!/usr/bin/env python
"""ABCD hist2D
+ channel 4mu
    - x: min pfiso
    - y: both neutral/charged
+ channel 2mu2e
    - x: min pfiso
    - y: #jets=0/>0
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

parser = argparse.ArgumentParser(description="leptonjet ABCD")
parser.add_argument("--sync", action='store_true', help="issue rsync command to sync plots folder to lxplus web server")
args = parser.parse_args()


dml = DatasetMapLoader()
# dataDS, dataMAP = dml.fetch('data')
bkgDS, bkgMAP, bkgSCALE = dml.fetch('bkg')

sdml = SigDatasetMapLoader()
sigDS_2mu2e, sigSCALE_2mu2e = sdml.fetch('2mu2e')
sigDS_4mu, sigSCALE_4mu = sdml.fetch('4mu')



class LjABCDProcessor(processor.ProcessorABC):
    def __init__(self, data_type='bkg'):
        self.data_type = data_type

        dataset_axis = hist.Cat('dataset', 'dataset')
        iso_axis = hist.Bin('iso', 'min pfIso', 50, 0, 0.5)
        bin_axis = hist.Bin('val', 'binary value', 3, 0, 3)
        self._accumulator = processor.dict_accumulator({
            'chan-4mu': hist.Hist('Counts', dataset_axis, iso_axis, bin_axis),
            'chan-2mu2e': hist.Hist('Counts', dataset_axis, iso_axis, bin_axis),
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

        weight = wgts.weight()
        ########################

        ak4jets = JaggedCandidateArray.candidatesfromcounts(
            df['akjet_ak4PFJetsCHS_p4'],
            px=df['akjet_ak4PFJetsCHS_p4.fCoordinates.fX'].content,
            py=df['akjet_ak4PFJetsCHS_p4.fCoordinates.fY'].content,
            pz=df['akjet_ak4PFJetsCHS_p4.fCoordinates.fZ'].content,
            energy=df['akjet_ak4PFJetsCHS_p4.fCoordinates.fT'].content,
            jetid=df['akjet_ak4PFJetsCHS_jetid'].content,
        )
        ak4jets=ak4jets[ak4jets.jetid&(ak4jets.pt>30)&(np.abs(ak4jets.eta)<2.4)]

        leptonjets = JaggedCandidateArray.candidatesfromcounts(
            df['pfjet_p4'],
            px=df['pfjet_p4.fCoordinates.fX'].content,
            py=df['pfjet_p4.fCoordinates.fY'].content,
            pz=df['pfjet_p4.fCoordinates.fZ'].content,
            energy=df['pfjet_p4.fCoordinates.fT'].content,
            sumtkpt=df['pfjet_tkPtSum05'].content,
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
        leptonjets.add_attributes(mucharged=(leptonjets.iseltype | (leptonjets.ismutype&(leptonjets.qsum!=0))))
        ljdsamuSubset = fromNestNestIndexArray(df['dsamuon_isSubsetFilteredCosmic1Leg'], awkward.fromiter(df['pfjet_pfcand_dsamuonIdx']))
        leptonjets.add_attributes(nocosmic=(ljdsamuSubset.sum()==0))
        leptonjets = leptonjets[(leptonjets.nocosmic)&(leptonjets.pt>30)&(leptonjets.mintkdist<50)]

        ## __ twoleptonjets__
        twoleptonjets = leptonjets.counts>=2
        dileptonjets = leptonjets[twoleptonjets]
        ak4jets = ak4jets[twoleptonjets]
        wgt = weight[twoleptonjets]

        if dileptonjets.size==0: return output
        lj0 = dileptonjets[dileptonjets.pt.argmax()]
        lj1 = dileptonjets[dileptonjets.pt.argsort()[:, 1:2]]

        ak4jets = ak4jets[ak4jets.pt>(lj0.pt.flatten())]
        ak4jetCounts = (ak4jets.counts>0).astype(int)
        minpfiso = ((lj0.pfiso>lj1.pfiso).astype(int)*lj1.pfiso + (lj0.pfiso<lj1.pfiso).astype(int)*lj0.pfiso).flatten()
        ljneutrality = ((lj0.isneutral&lj1.isneutral).astype(int)*1+(lj0.mucharged&lj1.mucharged).astype(int)*2).flatten()

        ## channel def ##
        #### 2mu2e
        singleMuljEvents = dileptonjets.ismutype.sum()==1
        muljInLeading2Events = (lj0.ismutype | lj1.ismutype).flatten()
        channel_2mu2e = singleMuljEvents&muljInLeading2Events

        output['chan-2mu2e'].fill(dataset=dataset, iso=minpfiso[channel_2mu2e], val=ak4jetCounts[channel_2mu2e], weight=wgt[channel_2mu2e])

        #### 4mu
        doubleMuljEvents = dileptonjets.ismutype.sum()==2
        muljIsLeading2Events = (lj0.ismutype & lj1.ismutype).flatten()
        channel_4mu = doubleMuljEvents&muljIsLeading2Events

        output['chan-4mu'].fill(dataset=dataset, iso=minpfiso[channel_4mu], val=ljneutrality[channel_4mu], weight=wgt[channel_4mu])

        ###########


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
                                    processor_instance=LjABCDProcessor(data_type='sig-2mu2e'),
                                    executor=processor.futures_executor,
                                    executor_args=dict(workers=12, flatten=False),
                                    chunksize=500000,
                                    )

    output_4mu = processor.run_uproot_job(sigDS_4mu,
                                    treename='ffNtuplizer/ffNtuple',
                                    processor_instance=LjABCDProcessor(data_type='sig-4mu'),
                                    executor=processor.futures_executor,
                                    executor_args=dict(workers=12, flatten=False),
                                    chunksize=500000,
                                    )

    output_bkg = processor.run_uproot_job(bkgDS,
                                    treename='ffNtuplizer/ffNtuple',
                                    processor_instance=LjABCDProcessor(data_type='bkg'),
                                    executor=processor.futures_executor,
                                    executor_args=dict(workers=12, flatten=False),
                                    chunksize=500000,
                                    )

    signalPts = [
        'mXX-150_mA-0p25_lxy-300',
        'mXX-500_mA-1p2_lxy-300',
        'mXX-800_mA-5_lxy-300',
    ]

    for p in signalPts:
        htitle = f'[2$\mu$2e|{p}] #AK4Jet vs. leptonjet min pfiso'
        fig, ax = make_2d_hist(output_2mu2e['chan-2mu2e'].integrate('dataset', p), 'iso',
                               title=htitle, zscale='log', ylim=[0,2], ylabel='#AK4Jets', xoverflow='over')
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(['0', '>0'])

        fig.savefig(join(outdir, f'{p}__2mu2e.png'))
        fig.savefig(join(outdir, f'{p}__2mu2e.pdf'))
        plt.close(fig)

        fig, ax = make_2d_hist(output_2mu2e['chan-2mu2e'].integrate('dataset', p).rebin('iso', hist.Bin('iso', 'min pfiso', np.array([0, 0.12, 1]))), 'iso',
                               title=htitle, zscale='log', text_opts=dict(), ylim=[0,2], ylabel='#AK4Jets', xoverflow='over')
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(['0', '>0'])
        ax.set_xticks([0, 0.12, 1])

        fig.savefig(join(outdir, f'{p}__2mu2e_4bins.png'))
        fig.savefig(join(outdir, f'{p}__2mu2e_4bins.pdf'))
        plt.close(fig)

        # ------------------------------------------

        htitle = f'[4$\mu$|{p}] leptonjet neutrality vs. min pfiso'
        fig, ax = make_2d_hist(output_4mu['chan-4mu'].integrate('dataset', p), 'iso',
                               title=htitle, zscale='log', ylim=[1,3], ylabel='Neutrality', xoverflow='over')
        ax.set_yticks([1.5, 2.5])
        ax.set_yticklabels(['Both neutral', 'Both charged'])

        fig.savefig(join(outdir, f'{p}__4mu.png'))
        fig.savefig(join(outdir, f'{p}__4mu.pdf'))
        plt.close(fig)

        fig, ax = make_2d_hist(output_4mu['chan-4mu'].integrate('dataset', p).rebin('iso', hist.Bin('iso', 'min pfiso', np.array([0, 0.12, 1]))), 'iso',
                               title=htitle, zscale='log', text_opts=dict(), ylim=[1,3], ylabel='Neutrality', xoverflow='over')
        ax.set_yticks([1.5, 2.5])
        ax.set_yticklabels(['Both neutral', 'Both charged'])
        ax.set_xticks([0, 0.12, 1])

        fig.savefig(join(outdir, f'{p}__4mu_4bins.png'))
        fig.savefig(join(outdir, f'{p}__4mu_4bins.pdf'))
        plt.close(fig)


    htitle = '[2$\mu$2e|bkgmc] #AK4Jet vs. leptonjet min pfiso'
    fig, ax = make_2d_hist(output_bkg['chan-2mu2e'].sum('cat'), 'iso',
                           title=htitle, zscale='log', ylim=[0,2], ylabel='#AK4Jets', xoverflow='over')
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(['0', '>0'])
    fig.savefig(join(outdir, 'bkgmc__2mu2e.png'))
    fig.savefig(join(outdir, 'bkgmc__2mu2e.pdf'))
    plt.close(fig)

    fig, ax = make_2d_hist(output_bkg['chan-2mu2e'].sum('cat').rebin('iso', hist.Bin('iso', 'min pfiso', np.array([0, 0.12, 1]))), 'iso',
                           title=htitle, zscale='log', text_opts=dict(), ylim=[0,2], ylabel='#AK4Jets', xoverflow='over')
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(['0', '>0'])
    ax.set_xticks([0, 0.12, 1])
    fig.savefig(join(outdir, 'bkgmc__2mu2e_4bins.png'))
    fig.savefig(join(outdir, 'bkgmc__2mu2e_4bins.pdf'))
    plt.close(fig)


    htitle = '[4$\mu$|bkgmc] leptonjet neutrality vs. min pfiso'
    fig, ax = make_2d_hist(output_bkg['chan-4mu'].sum('cat'), 'iso',
                           title=htitle, zscale='log', ylim=[1,3], ylabel='Neutrality', xoverflow='over')
    ax.set_yticks([1.5, 2.5])
    ax.set_yticklabels(['Both neutral', 'Both charged'])
    fig.savefig(join(outdir, 'bkgmc__4mu.png'))
    fig.savefig(join(outdir, 'bkgmc__4mu.pdf'))
    plt.close(fig)

    fig, ax = make_2d_hist(output_bkg['chan-4mu'].sum('cat').rebin('iso', hist.Bin('iso', 'min pfiso', np.array([0, 0.12, 1]))), 'iso',
                           title=htitle, zscale='log', text_opts=dict(), ylim=[1,3], ylabel='Neutrality', xoverflow='over')
    ax.set_yticks([1.5, 2.5])
    ax.set_yticklabels(['Both neutral', 'Both charged'])
    ax.set_xticks([0, 0.12, 1])
    fig.savefig(join(outdir, 'bkgmc__4mu_4bins.png'))
    fig.savefig(join(outdir, 'bkgmc__4mu_4bins.pdf'))
    plt.close(fig)
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
        ## copy to AN image folder
        # an_dir = '/uscms_data/d3/wsi/lpcdm/AN-18-125/image'
        # if isdir(an_dir):
        #     cmd = f'cp {outdir}/*.pdf {an_dir}'
        #     print(f'--> copy to AN folder: {an_dir}')
        #     os.system(cmd)
