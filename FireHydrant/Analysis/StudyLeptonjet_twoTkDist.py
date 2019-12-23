#!/usr/bin/env python
"""
leptonjet, any two track distance, min/max
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
from FireHydrant.Tools.uproothelpers import fromNestNestIndexArray

np.seterr(divide='ignore', invalid='ignore', over='ignore')
plt.rcParams["savefig.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"


parser = argparse.ArgumentParser(description="leptonjets any two track distance")
parser.add_argument("--sync", action='store_true', help="issue rsync command to sync plots folder to lxplus web server")
args = parser.parse_args()


sdml = SigDatasetMapLoader()
sigDS_2mu2e, sigSCALE_2mu2e = sdml.fetch('2mu2e')
sigDS_4mu, sigSCALE_4mu = sdml.fetch('4mu')

dml = DatasetMapLoader()
bkgDS, bkgMAP, bkgSCALE = dml.fetch('bkg')
dataDS, dataMAP = dml.fetch('data')

class LeptonjetTkProcessor(processor.ProcessorABC):
    def __init__(self, data_type='sig-2mu2e', lj_type='neutral'):
        self.data_type = data_type
        self.lj_type = lj_type

        dataset_axis = hist.Cat('dataset', 'dataset')
        channel_axis = hist.Bin('channel', 'channel', 3, 0, 3)
        dist_axis = hist.Bin('dist', 'two track min distance [cm]', 50, 0, 200)
        self._accumulator = processor.dict_accumulator({
            'mindist': hist.Hist('Counts', dataset_axis, dist_axis, channel_axis),
            'maxdist': hist.Hist('Counts', dataset_axis, dist_axis, channel_axis),
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
            px=df['pfjet_p4.fCoordinates.fX'].content,
            py=df['pfjet_p4.fCoordinates.fY'].content,
            pz=df['pfjet_p4.fCoordinates.fZ'].content,
            energy=df['pfjet_p4.fCoordinates.fT'].content,
            pfiso=df['pfjet_pfIsolationNoPU05'].content,
            mintkdist=df['pfjet_pfcands_minTwoTkDist'].content,
            maxtkdist=df['pfjet_pfcands_maxTwoTkDist'].content,
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
        if self.lj_type == 'neutral':
            leptonjets = leptonjets[(leptonjets.pt>30)&(leptonjets.isneutral)]
        else:
            leptonjets = leptonjets[(leptonjets.pt>30)&(leptonjets.mucharged)]

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

        ljones = dileptonjets.pt.ones_like()
        output['mindist'].fill(dataset=dataset, dist=dileptonjets.mintkdist.flatten(), weight=(wgt*ljones).flatten(), channel=(channel_*ljones).flatten())
        output['maxdist'].fill(dataset=dataset, dist=dileptonjets.maxtkdist.flatten(), weight=(wgt*ljones).flatten(), channel=(channel_*ljones).flatten())

        return output

    def postprocess(self, accumulator):
        origidentity = list(accumulator)
        for k in origidentity:
            if self.data_type == 'sig-2mu2e':
                accumulator[k].scale(sigSCALE_2mu2e, axis='dataset')
            if self.data_type == 'sig-4mu':
                accumulator[k].scale(sigSCALE_4mu, axis='dataset')
            if self.data_type == 'bkg':
                accumulator[k].scale(bkgSCALE, axis='dataset')
                accumulator[k] = accumulator[k].group("dataset",
                                                    hist.Cat("cat", "datasets", sorting='integral'),
                                                    bkgMAP)
            if self.data_type == 'data':
                accumulator[k] = accumulator[k].group("dataset",
                                                    hist.Cat("cat", "datasets",),
                                                    dataMAP)
        return accumulator


if __name__ == "__main__":
    import os
    from os.path import join, isdir, splitext
    from FireHydrant.Analysis.PlottingOptions import *

    reldir = splitext(__file__)[0].replace('_', '/')
    outdir = join(os.getenv('FH_BASE'), "Imgs", reldir)
    if not isdir(outdir): os.makedirs(outdir)

    output_2mu2e = processor.run_uproot_job(sigDS_2mu2e,
                                  treename='ffNtuplizer/ffNtuple',
                                  processor_instance=LeptonjetTkProcessor(data_type='sig-2mu2e'),
                                  executor=processor.futures_executor,
                                  executor_args=dict(workers=12, flatten=False),
                                  chunksize=500000,
                                 )
    output_4mu = processor.run_uproot_job(sigDS_4mu,
                                  treename='ffNtuplizer/ffNtuple',
                                  processor_instance=LeptonjetTkProcessor(data_type='sig-4mu'),
                                  executor=processor.futures_executor,
                                  executor_args=dict(workers=12, flatten=False),
                                  chunksize=500000,
                                 )
    output_bkg = processor.run_uproot_job(bkgDS,
                                    treename='ffNtuplizer/ffNtuple',
                                    processor_instance=LeptonjetTkProcessor(data_type='bkg'),
                                    executor=processor.futures_executor,
                                    executor_args=dict(workers=12, flatten=False),
                                    chunksize=500000,
                                    )
    output_data = processor.run_uproot_job(dataDS,
                                    treename='ffNtuplizer/ffNtuple',
                                    processor_instance=LeptonjetTkProcessor(data_type='data'),
                                    executor=processor.futures_executor,
                                    executor_args=dict(workers=12, flatten=False),
                                    chunksize=500000,
                                    )

    import re
    longdecay = re.compile('^.*_lxy-300$')
    sampleSig = re.compile('mXX-150_mA-0p25_lxy-300|mXX-500_mA-1p2_lxy-300|mXX-800_mA-5_lxy-300')

    fig, (ax, rax) = make_ratio_plot(output_bkg['mindist'].integrate('channel', slice(1,2)),
                                     output_data['mindist'].integrate('channel', slice(1,2)),
                                     sigh=output_2mu2e['mindist'][sampleSig].integrate('channel', slice(1,2)),
                                     title='[$2\mu 2e$, neutral] leptonJets min two track distance',
                                     overflow='over')

    fig.savefig(join(outdir, 'ljMinTwoTkDist-neulj_2mu2e.png'))
    fig.savefig(join(outdir, 'ljMinTwoTkDist-neulj_2mu2e.pdf'))
    plt.close(fig)

    fig, (ax, rax) = make_ratio_plot(output_bkg['mindist'].integrate('channel', slice(2,3)),
                                     output_data['mindist'].integrate('channel', slice(2,3)),
                                     sigh=output_4mu['mindist'][sampleSig].integrate('channel', slice(2,3)),
                                     title='[$4\mu$, neutral] leptonJets min two track distance', overflow='over')

    fig.savefig(join(outdir, 'ljMinTwoTkDist-neulj_4mu.png'))
    fig.savefig(join(outdir, 'ljMinTwoTkDist-neulj_4mu.pdf'))
    plt.close(fig)

    fig, (ax, rax) = make_ratio_plot(output_bkg['maxdist'].integrate('channel', slice(1,2)),
                                     output_data['maxdist'].integrate('channel', slice(1,2)),
                                     sigh=output_2mu2e['maxdist'][sampleSig].integrate('channel', slice(1,2)),
                                     title='[$2\mu 2e$, neutral] leptonJets max two track distance',
                                     overflow='over')

    fig.savefig(join(outdir, 'ljMaxTwoTkDist-neulj_2mu2e.png'))
    fig.savefig(join(outdir, 'ljMaxTwoTkDist-neulj_2mu2e.pdf'))
    plt.close(fig)

    fig, (ax, rax) = make_ratio_plot(output_bkg['maxdist'].integrate('channel', slice(2,3)),
                                     output_data['maxdist'].integrate('channel', slice(2,3)),
                                     sigh=output_4mu['maxdist'][sampleSig].integrate('channel', slice(2,3)),
                                     title='[$4\mu$, neutral] leptonJets max two track distance', overflow='over')

    fig.savefig(join(outdir, 'ljMaxTwoTkDist-neulj_4mu.png'))
    fig.savefig(join(outdir, 'ljMaxTwoTkDist-neulj_4mu.pdf'))
    plt.close(fig)




    output_2mu2e = processor.run_uproot_job(sigDS_2mu2e,
                                  treename='ffNtuplizer/ffNtuple',
                                  processor_instance=LeptonjetTkProcessor(data_type='sig-2mu2e', lj_type='charged'),
                                  executor=processor.futures_executor,
                                  executor_args=dict(workers=12, flatten=False),
                                  chunksize=500000,
                                 )
    output_4mu = processor.run_uproot_job(sigDS_4mu,
                                  treename='ffNtuplizer/ffNtuple',
                                  processor_instance=LeptonjetTkProcessor(data_type='sig-4mu', lj_type='charged'),
                                  executor=processor.futures_executor,
                                  executor_args=dict(workers=12, flatten=False),
                                  chunksize=500000,
                                 )
    output_bkg = processor.run_uproot_job(bkgDS,
                                    treename='ffNtuplizer/ffNtuple',
                                    processor_instance=LeptonjetTkProcessor(data_type='bkg', lj_type='charged'),
                                    executor=processor.futures_executor,
                                    executor_args=dict(workers=12, flatten=False),
                                    chunksize=500000,
                                    )
    output_data = processor.run_uproot_job(dataDS,
                                    treename='ffNtuplizer/ffNtuple',
                                    processor_instance=LeptonjetTkProcessor(data_type='data', lj_type='charged'),
                                    executor=processor.futures_executor,
                                    executor_args=dict(workers=12, flatten=False),
                                    chunksize=500000,
                                    )

    fig, (ax, rax) = make_ratio_plot(output_bkg['mindist'].integrate('channel', slice(1,2)),
                                     output_data['mindist'].integrate('channel', slice(1,2)),
                                     sigh=output_2mu2e['mindist'][sampleSig].integrate('channel', slice(1,2)),
                                     title='[$2\mu 2e$, charged] leptonJets min two track distance',
                                     overflow='over')

    fig.savefig(join(outdir, 'ljMinTwoTkDist-chalj_2mu2e.png'))
    fig.savefig(join(outdir, 'ljMinTwoTkDist-chalj_2mu2e.pdf'))
    plt.close(fig)

    fig, (ax, rax) = make_ratio_plot(output_bkg['mindist'].integrate('channel', slice(2,3)),
                                     output_data['mindist'].integrate('channel', slice(2,3)),
                                     sigh=output_4mu['mindist'][sampleSig].integrate('channel', slice(2,3)),
                                     title='[$4\mu$, charged] leptonJets min two track distance', overflow='over')

    fig.savefig(join(outdir, 'ljMinTwoTkDist-chalj_4mu.png'))
    fig.savefig(join(outdir, 'ljMinTwoTkDist-chalj_4mu.pdf'))
    plt.close(fig)

    fig, (ax, rax) = make_ratio_plot(output_bkg['maxdist'].integrate('channel', slice(1,2)),
                                     output_data['maxdist'].integrate('channel', slice(1,2)),
                                     sigh=output_2mu2e['maxdist'][sampleSig].integrate('channel', slice(1,2)),
                                     title='[$2\mu 2e$, charged] leptonJets max two track distance',
                                     overflow='over')

    fig.savefig(join(outdir, 'ljMaxTwoTkDist-chalj_2mu2e.png'))
    fig.savefig(join(outdir, 'ljMaxTwoTkDist-chalj_2mu2e.pdf'))
    plt.close(fig)

    fig, (ax, rax) = make_ratio_plot(output_bkg['maxdist'].integrate('channel', slice(2,3)),
                                     output_data['maxdist'].integrate('channel', slice(2,3)),
                                     sigh=output_4mu['maxdist'][sampleSig].integrate('channel', slice(2,3)),
                                     title='[$4\mu$, charged] leptonJets max two track distance', overflow='over')

    fig.savefig(join(outdir, 'ljMaxTwoTkDist-chalj_4mu.png'))
    fig.savefig(join(outdir, 'ljMaxTwoTkDist-chalj_4mu.pdf'))
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
