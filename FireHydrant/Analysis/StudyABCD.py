#!/usr/bin/env python
"""ABCD scatter
- x: dphi of leptonjet pair
- y: min pfiso05NoPU of leading two leptonjets
"""
import argparse
from contextlib import contextmanager

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

# import ROOT
# ROOT.gROOT.SetBatch()


dml = DatasetMapLoader()
dataDS, dataMAP = dml.fetch('data')

# @contextmanager
# def _setIgnoreLevel(level):
#     originalLevel = ROOT.gErrorIgnoreLevel
#     ROOT.gErrorIgnoreLevel = level
#     yield
#     ROOT.gErrorIgnoreLevel = originalLevel


class LjABCDProcessor(processor.ProcessorABC):
    def __init__(self, bothNeutral=True):
        self.bothNeutral = bothNeutral
        dataset_axis = hist.Cat('dataset', 'dataset')
        self._accumulator = processor.dict_accumulator({
            'miniso': processor.column_accumulator(np.zeros(shape=(0,))),
            'dphi': processor.column_accumulator(np.zeros(shape=(0,))),
            'channel': processor.column_accumulator(np.zeros(shape=(0,))),
            'categ': processor.column_accumulator(np.zeros(shape=(0,))),
        })

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, df):
        output = self.accumulator.identity()
        dataset = df['dataset']

        ## construct weights ##
        wgts = processor.Weights(df.size)

        triggermask = np.logical_or.reduce([df[t] for t in Triggers])
        wgts.add('trigger', triggermask)
        cosmicpairmask = df['cosmicveto_result']
        wgts.add('cosmicveto', cosmicpairmask)
        pvmask = df['metfilters_PrimaryVertexFilter']
        wgts.add('primaryvtx', pvmask)

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
        leptonjets.add_attributes(label=label, ndsa=ndsa)
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

        minpfiso = (lj0.pfiso>lj1.pfiso).astype(int)*lj1.pfiso + (lj0.pfiso<lj1.pfiso).astype(int)*lj0.pfiso
        dphi = np.abs(lj0.p4.delta_phi(lj1.p4))
        categ = lj0.ndsa*10 + lj1.ndsa

        mask_ = (lj0.isneutral&lj1.isneutral).flatten()
        if self.bothNeutral is False:
            mask_ = ~mask_
            mask_ = ((channel_==2)&((~lj0.isneutral&(~lj1.isneutral)).flatten())) | ((channel_==1)&mask_)

        channel_ = channel_[mask_]
        wgt = wgt[mask_]
        minpfiso = minpfiso[mask_]
        dphi = dphi[mask_]
        categ = categ[mask_]

        output['miniso'] += processor.column_accumulator(minpfiso[wgt.astype(bool)].flatten())
        output['dphi'] += processor.column_accumulator(dphi[wgt.astype(bool)].flatten())
        output['channel'] += processor.column_accumulator(channel_[wgt.astype(bool)].flatten())
        output['categ'] += processor.column_accumulator(categ[wgt.astype(bool)].flatten())

        return output

    def postprocess(self, accumulator):
        return accumulator


if __name__ == "__main__":
    import os
    import re
    from os.path import join, isdir, splitext
    from FireHydrant.Analysis.PlottingOptions import *

    reldir = splitext(__file__)[0].replace('_', '/')
    outdir = join(os.getenv('FH_BASE'), "Imgs", reldir)
    if not isdir(outdir): os.makedirs(outdir)

    print('[both neutral]')
    output = processor.run_uproot_job(dataDS,
                                    treename='ffNtuplizer/ffNtuple',
                                    processor_instance=LjABCDProcessor(bothNeutral=True),
                                    executor=processor.futures_executor,
                                    executor_args=dict(workers=12, flatten=False),
                                    chunksize=500000,
                                    )
    chan_ = output['channel'].value
    miniso_ = output['miniso'].value
    dphi_ = output['dphi'].value
    categ_ = output['categ'].value

    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(dphi_[chan_==1.], miniso_[chan_==1.], s=4, marker='o', c='blue')
    ax.set_title('[$2\mu 2e$] leptonjet pair $\Delta\phi$ vs min(pfiso05)', x=0, ha='left')
    ax.set_xlabel('$\Delta\phi$', x=1, ha='right')
    ax.set_ylabel('min(pfiso)', y=1, ha='right')
    ax.text(1,1,'59.74/fb (13TeV)', ha='right', va='bottom', transform=ax.transAxes)
    ax.autoscale(axis='both', tight=True)
    ax.set_xlim(0, np.pi)
    ax.set_ylim(0, 0.6)

    fig.savefig(join(outdir, 'dphi-miniso-neulj_2mu2e.png'))
    fig.savefig(join(outdir, 'dphi-miniso-neulj_2mu2e.pdf'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(dphi_[chan_==2.], miniso_[chan_==2.], s=4, marker='o', c='blue')
    ax.set_title('[$4\mu$] leptonjet pair $\Delta\phi$ vs min(pfiso05)', x=0, ha='left')
    ax.set_xlabel('$\Delta\phi$', x=1, ha='right')
    ax.set_ylabel('min(pfiso)', y=1, ha='right')
    ax.text(1,1,'59.74/fb (13TeV)', ha='right', va='bottom', transform=ax.transAxes)
    ax.autoscale(axis='both', tight=True)
    ax.set_xlim(0, np.pi)
    ax.set_ylim(0, 0.6)

    fig.savefig(join(outdir, 'dphi-miniso-neulj_4mu.png'))
    fig.savefig(join(outdir, 'dphi-miniso-neulj_4mu.pdf'))
    plt.close(fig)


    categories = {
        '2mu2e': [
            (chan_==1.)&(categ_==0),
            (chan_==1.)&((categ_==1)|(categ_==10)),
            (chan_==1.)&((categ_==2)|(categ_==20)),
        ],
        '4mu': [
            (chan_==2.)&(categ_==0),
            (chan_==2.)&((categ_==1)|(categ_==10)),
            (chan_==2.)&((categ_==2)|(categ_==20)),
            (chan_==2.)&(categ_==11),
            (chan_==2.)&((categ_==12)|(categ_==21)),
            (chan_==2.)&(categ_==22),
        ]
    }

    for i, c in enumerate(categories['2mu2e']):
        fig, ax = plt.subplots(figsize=(8,6))
        ax.scatter(dphi_[c], miniso_[c], s=4, marker='o', c='blue')
        ax.set_title(f'[$2\mu 2e$, categ{i}] leptonjet pair $\Delta\phi$ vs min(pfiso05)', x=0, ha='left')
        ax.set_xlabel('$\Delta\phi$', x=1, ha='right')
        ax.set_ylabel('min(pfiso)', y=1, ha='right')
        text = 'category definition:\n0: muonlj.nDSA==0\n1: muonlj.nDSA==1\n2: muonlj.nDSA==2'
        ax.text(0.99, 0.99, text, ha='right', va='top', transform=ax.transAxes)
        ax.text(1,1,'59.74/fb (13TeV)', ha='right', va='bottom', transform=ax.transAxes)
        ax.autoscale(axis='both', tight=True)
        ax.set_xlim(0, np.pi)
        ax.set_ylim(0, 0.6)

        fig.savefig(join(outdir, f'dphi-miniso-neulj-categ{i}_2mu2e.png'))
        fig.savefig(join(outdir, f'dphi-miniso-neulj-categ{i}_2mu2e.pdf'))
        plt.close(fig)

    for i, c in enumerate(categories['4mu']):
        fig, ax = plt.subplots(figsize=(8,6))
        ax.scatter(dphi_[c], miniso_[c], s=4, marker='o', c='blue')
        ax.set_title(f'[$4\mu$, categ{i}] leptonjet pair $\Delta\phi$ vs min(pfiso05)', x=0, ha='left')
        ax.set_xlabel('$\Delta\phi$', x=1, ha='right')
        ax.set_ylabel('min(pfiso)', y=1, ha='right')
        text = '\n'.join([
            'category definition:',
            '0: 0 muonlj.nDSA>0',
            '1: 1 muonlj.nDSA==1',
            '2: 1 muonlj.nDSA==2',
            '3: 2 muonlj.nDSA==1',
            '4: (1 muonlj.nDSA==1)&(1muonlj.nDSA==2)',
            '5: 2 muonlj.nDSA==2'
        ])
        ax.text(0.99, 0.99, text, ha='right', va='top', transform=ax.transAxes)
        ax.text(1,1,'59.74/fb (13TeV)', ha='right', va='bottom', transform=ax.transAxes)
        ax.autoscale(axis='both', tight=True)
        ax.set_xlim(0, np.pi)
        ax.set_ylim(0, 0.6)

        fig.savefig(join(outdir, f'dphi-miniso-neulj-categ{i}_4mu.png'))
        fig.savefig(join(outdir, f'dphi-miniso-neulj-categ{i}_4mu.pdf'))
        plt.close(fig)


    # --------------------------------------------------------

    print('[both charged]')
    output = processor.run_uproot_job(dataDS,
                                    treename='ffNtuplizer/ffNtuple',
                                    processor_instance=LjABCDProcessor(bothNeutral=False),
                                    executor=processor.futures_executor,
                                    executor_args=dict(workers=12, flatten=False),
                                    chunksize=500000,
                                    )
    chan_ = output['channel'].value
    miniso_ = output['miniso'].value
    dphi_ = output['dphi'].value
    categ_ = output['categ'].value

    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(dphi_[chan_==1.], miniso_[chan_==1.], s=4, marker='o', c='blue')
    ax.set_title('[$2\mu 2e$] leptonjet pair $\Delta\phi$ vs min(pfiso05)', x=0, ha='left')
    ax.set_xlabel('$\Delta\phi$', x=1, ha='right')
    ax.set_ylabel('min(pfiso)', y=1, ha='right')
    ax.text(1,1,'59.74/fb (13TeV)', ha='right', va='bottom', transform=ax.transAxes)
    ax.autoscale(axis='both', tight=True)
    ax.set_xlim(0, np.pi)
    ax.set_ylim(0, 0.6)

    fig.savefig(join(outdir, 'dphi-miniso-chalj_2mu2e.png'))
    fig.savefig(join(outdir, 'dphi-miniso-chalj_2mu2e.pdf'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8,6))
    ax.scatter(dphi_[chan_==2.], miniso_[chan_==2.], s=4, marker='o', c='blue')
    ax.set_title('[$4\mu$] leptonjet pair $\Delta\phi$ vs min(pfiso05)', x=0, ha='left')
    ax.set_xlabel('$\Delta\phi$', x=1, ha='right')
    ax.set_ylabel('min(pfiso)', y=1, ha='right')
    ax.text(1,1,'59.74/fb (13TeV)', ha='right', va='bottom', transform=ax.transAxes)
    ax.autoscale(axis='both', tight=True)
    ax.set_xlim(0, np.pi)
    ax.set_ylim(0, 0.6)

    fig.savefig(join(outdir, 'dphi-miniso-chalj_4mu.png'))
    fig.savefig(join(outdir, 'dphi-miniso-chalj_4mu.pdf'))
    plt.close(fig)

    categories = {
        '2mu2e': [
            (chan_==1.)&(categ_==0),
            (chan_==1.)&((categ_==1)|(categ_==10)),
            (chan_==1.)&((categ_==2)|(categ_==20)),
        ],
        '4mu': [
            (chan_==2.)&(categ_==0),
            (chan_==2.)&((categ_==1)|(categ_==10)),
            (chan_==2.)&((categ_==2)|(categ_==20)),
            (chan_==2.)&(categ_==11),
            (chan_==2.)&((categ_==12)|(categ_==21)),
            (chan_==2.)&(categ_==22),
        ]
    }

    for i, c in enumerate(categories['2mu2e']):
        fig, ax = plt.subplots(figsize=(8,6))
        ax.scatter(dphi_[c], miniso_[c], s=4, marker='o', c='blue')
        ax.set_title(f'[$2\mu 2e$, categ{i}] leptonjet pair $\Delta\phi$ vs min(pfiso05)', x=0, ha='left')
        ax.set_xlabel('$\Delta\phi$', x=1, ha='right')
        ax.set_ylabel('min(pfiso)', y=1, ha='right')
        text = 'category definition:\n0: muonlj.nDSA==0\n1: muonlj.nDSA==1\n2: muonlj.nDSA==2'
        ax.text(0.99, 0.99, text, ha='right', va='top', transform=ax.transAxes)
        ax.text(1,1,'59.74/fb (13TeV)', ha='right', va='bottom', transform=ax.transAxes)
        ax.autoscale(axis='both', tight=True)
        ax.set_xlim(0, np.pi)
        ax.set_ylim(0, 0.6)

        fig.savefig(join(outdir, f'dphi-miniso-chalj-categ{i}_2mu2e.png'))
        fig.savefig(join(outdir, f'dphi-miniso-chalj-categ{i}_2mu2e.pdf'))
        plt.close(fig)

    for i, c in enumerate(categories['4mu']):
        fig, ax = plt.subplots(figsize=(8,6))
        ax.scatter(dphi_[c], miniso_[c], s=4, marker='o', c='blue')
        ax.set_title(f'[$4\mu$, categ{i}] leptonjet pair $\Delta\phi$ vs min(pfiso05)', x=0, ha='left')
        ax.set_xlabel('$\Delta\phi$', x=1, ha='right')
        ax.set_ylabel('min(pfiso)', y=1, ha='right')
        text = '\n'.join([
            'category definition:',
            '0: 0 muonlj.nDSA>0',
            '1: 1 muonlj.nDSA==1',
            '2: 1 muonlj.nDSA==2',
            '3: 2 muonlj.nDSA==1',
            '4: (1 muonlj.nDSA==1)&(1muonlj.nDSA==2)',
            '5: 2 muonlj.nDSA==2'
        ])
        ax.text(0.99, 0.99, text, ha='right', va='top', transform=ax.transAxes)
        ax.text(1,1,'59.74/fb (13TeV)', ha='right', va='bottom', transform=ax.transAxes)
        ax.autoscale(axis='both', tight=True)
        ax.set_xlim(0, np.pi)
        ax.set_ylim(0, 0.6)

        fig.savefig(join(outdir, f'dphi-miniso-chalj-categ{i}_4mu.png'))
        fig.savefig(join(outdir, f'dphi-miniso-chalj-categ{i}_4mu.pdf'))
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
