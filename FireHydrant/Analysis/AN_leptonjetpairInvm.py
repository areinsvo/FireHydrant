#!/usr/bin/env python
"""For AN
leptonjet pair invariant mass
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


parser = argparse.ArgumentParser(description="[AN] leptonjet pair invariant mass")
parser.add_argument("--sync", action='store_true', help="issue rsync command to sync plots folder to lxplus web server")
args = parser.parse_args()

# dml = DatasetMapLoader()
# bkgDS, bkgMAP, bkgSCALE = dml.fetch('bkg')
# dataDS, dataMAP = dml.fetch('data')

sdml = SigDatasetMapLoader()
sigDS, sigSCALE = sdml.fetch('all')


class LJPairInvMProcessor(processor.ProcessorABC):
    def __init__(self):
        dataset_axis = hist.Cat('dataset', 'dataset')
        ms_axis = hist.Bin('mass_s', 'mass [GeV]', 50, 0, 200)
        mm_axis = hist.Bin('mass_m', 'mass [GeV]', 50, 250, 750)
        ml_axis = hist.Bin('mass_l', 'mass [GeV]', 50, 500, 1500)
        self._accumulator = processor.dict_accumulator({
            'invm_s': hist.Hist('Norm. Frequency', dataset_axis, ms_axis),
            'invm_m': hist.Hist('Norm. Frequency', dataset_axis, mm_axis),
            'invm_l': hist.Hist('Norm. Frequency', dataset_axis, ml_axis),
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

        output['invm_s'].fill(dataset=dataset, mass_s=(lj0+lj1).p4.mass[channel_>0].flatten())
        output['invm_m'].fill(dataset=dataset, mass_m=(lj0+lj1).p4.mass[channel_>0].flatten())
        output['invm_l'].fill(dataset=dataset, mass_l=(lj0+lj1).p4.mass[channel_>0].flatten())

        return output

    def postprocess(self, accumulator):
        return accumulator


if __name__ == "__main__":
    import os
    from os.path import join, isdir, splitext

    reldir = splitext(__file__)[0].replace('_', '/')
    outdir = join(os.getenv('FH_BASE'), "Imgs", reldir)
    if not isdir(outdir): os.makedirs(outdir)

    import re

    output = processor.run_uproot_job(sigDS,
                                    treename='ffNtuplizer/ffNtuple',
                                    processor_instance=LJPairInvMProcessor(),
                                    executor=processor.futures_executor,
                                    executor_args=dict(workers=12, flatten=False),
                                    chunksize=500000,
                                    )


    # ----------------------------------------------------------
    mxx100 = re.compile('\w+/mXX-100_mA-(0p25|5)_lxy-300')

    fig, ax = plt.subplots(figsize=(8,6))
    hist.plot1d(output['invm_s'][mxx100], overlay='dataset', ax=ax, overflow='none', density=True)
    ax.set_title('[signalMC|lxy300cm] leptonjet pair invariant mass', x=0, ha='left')
    ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
    ax.set_ylabel(ax.get_ylabel()+'/4', y=1.0, ha="right")

    fig.savefig(join(outdir, 'ljpairInvm-mxx100.png'))
    fig.savefig(join(outdir, 'ljpairInvm-mxx100.pdf'))
    plt.close(fig)

    # ----------------------------------------------------------
    mxx500 = re.compile('\w+/mXX-500_mA-(0p25|5)_lxy-300')

    fig, ax = plt.subplots(figsize=(8,6))
    hist.plot1d(output['invm_m'][mxx500], overlay='dataset', ax=ax, overflow='none', density=True)
    ax.set_title('[signalMC|lxy300cm] leptonjet pair invariant mass', x=0, ha='left')
    ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
    ax.set_ylabel(ax.get_ylabel()+'/10', y=1.0, ha="right")

    fig.savefig(join(outdir, 'ljpairInvm-mxx500.png'))
    fig.savefig(join(outdir, 'ljpairInvm-mxx500.pdf'))
    plt.close(fig)

    # ----------------------------------------------------------
    mxx1000 = re.compile('\w+/mXX-1000_mA-(0p25|5)_lxy-300')

    fig, ax = plt.subplots(figsize=(8,6))
    hist.plot1d(output['invm_l'][mxx1000], overlay='dataset', ax=ax, overflow='none', density=True)
    ax.set_title('[signalMC|lxy300cm] leptonjet pair invariant mass', x=0, ha='left')
    ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
    ax.set_ylabel(ax.get_ylabel()+'/20', y=1.0, ha="right")

    fig.savefig(join(outdir, 'ljpairInvm-mxx1000.png'))
    fig.savefig(join(outdir, 'ljpairInvm-mxx1000.pdf'))
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
