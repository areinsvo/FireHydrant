#!/usr/bin/env python
"""GenJet"""

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


parser = argparse.ArgumentParser(description="[AN] about hadronic jets")
parser.add_argument("--sync", action='store_true', help="issue rsync command to sync plots folder to lxplus web server")
args = parser.parse_args()

sdml = SigDatasetMapLoader()
sigDS_2mu2e, sigSCALE_2mu2e = sdml.fetch('2mu2e')
sigDS_4mu, sigSCALE_4mu = sdml.fetch('4mu')

class GenJetProcessor(processor.ProcessorABC):
    def __init__(self, data_type = 'sig-2mu2e'):
        self.data_type = data_type

        dataset_axis = hist.Cat('dataset', 'dataset')
        count_axis = hist.Bin('cnt', 'Number of Jets', 10, 0, 10)
        self._accumulator = processor.dict_accumulator({
            'njets': hist.Hist('Counts', dataset_axis, count_axis),
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

        genjets = JaggedCandidateArray.candidatesfromcounts(
            df['genjet_p4'],
            px=df['genjet_p4.fCoordinates.fX'].content,
            py=df['genjet_p4.fCoordinates.fY'].content,
            pz=df['genjet_p4.fCoordinates.fZ'].content,
            energy=df['genjet_p4.fCoordinates.fT'].content,
        )
        genparticles = JaggedCandidateArray.candidatesfromcounts(
            df['gen_p4'],
            px=df['gen_p4.fCoordinates.fX'].content,
            py=df['gen_p4.fCoordinates.fY'].content,
            pz=df['gen_p4.fCoordinates.fZ'].content,
            energy=df['gen_p4.fCoordinates.fT'].content,
            pid=df['gen_pid'].content,
        )
        darkphotons = genparticles[genparticles.pid==32]
        dpptmax = darkphotons.pt.max()

        mask_ = genjets.match(darkphotons, deltaRCut=0.4)
        genjets = genjets[~mask_]

        output['njets'].fill(dataset=dataset, cnt=genjets[genjets.pt>dpptmax].counts, weight=weight,)

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
                                  processor_instance=GenJetProcessor(data_type='sig-2mu2e'),
                                  executor=processor.futures_executor,
                                  executor_args=dict(workers=12, flatten=False),
                                  chunksize=500000,
                                 )
    out_sig4mu = processor.run_uproot_job(sigDS_4mu,
                                  treename='ffNtuplizer/ffNtuple',
                                  processor_instance=GenJetProcessor(data_type='sig-4mu'),
                                  executor=processor.futures_executor,
                                  executor_args=dict(workers=12, flatten=False),
                                  chunksize=500000,
                                 )

    import re
    longdecay = re.compile('^.*_lxy-300$')
    sampleSig = re.compile('mXX-150_mA-0p25_lxy-300|mXX-500_mA-1p2_lxy-300|mXX-800_mA-5_lxy-300')

    fig, ax = make_signal_plot(out_sig2mu2e['njets'][sampleSig],
                               title='[$2\mu 2e$] Gen jets multiplicity')

    fig.savefig(join(outdir, 'NGenJets-2mu2e.png'))
    fig.savefig(join(outdir, 'NGenJets-2mu2e.pdf'))
    plt.close(fig)

    fig, ax = make_signal_plot(out_sig4mu['njets'][sampleSig],
                               title='[$4\mu$] Gen jets multiplicity')

    fig.savefig(join(outdir, 'NGenJets-4mu.png'))
    fig.savefig(join(outdir, 'NGenJets-4mu.pdf'))
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
