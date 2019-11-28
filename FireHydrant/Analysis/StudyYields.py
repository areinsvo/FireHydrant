#!/usr/bin/env python
"""
print yields table
"""
import argparse
import itertools

import awkward
import coffea.processor as processor
import numpy as np
from coffea import hist
from coffea.analysis_objects import JaggedCandidateArray
from FireHydrant.Analysis.DatasetMapLoader import (DatasetMapLoader,
                                                   SigDatasetMapLoader)
from FireHydrant.Tools.correction import (get_nlo_weight_function,
                                          get_pu_weights_function,
                                          get_ttbar_weight)
from FireHydrant.Tools.metfilter import MetFilters
from FireHydrant.Tools.trigger import Triggers
from FireHydrant.Analysis.Utils import sigsort

parser = argparse.ArgumentParser(description="print sigmc/bkgmc yields")
parser.add_argument("--sync", action='store_true', help="issue rsync command to sync plots folder to lxplus web server")
args = parser.parse_args()

sdml = SigDatasetMapLoader()
sigDS_2mu2e, sigSCALE_2mu2e = sdml.fetch('2mu2e')
sigDS_4mu, sigSCALE_4mu = sdml.fetch('4mu')

dml = DatasetMapLoader()
bkgDS, bkgMAP, bkgSCALE = dml.fetch('bkg')
dataDS, dataMAP = dml.fetch('data')


"""event yields"""
class LeptonjetProcessor(processor.ProcessorABC):
    def __init__(self, data_type='bkg', region='SR'):
        self.data_type = data_type
        self.region = region

        dataset_axis = hist.Cat('dataset', 'dataset')
        channel_axis = hist.Bin('channel', 'channel', 3, 0, 3)
        count_axis = hist.Bin('cnt', 'event count', 10, 0, 10)
        dphi_axis = hist.Bin('dphi', '$\Delta\phi$', 20, 0, np.pi)

        self._accumulator = processor.dict_accumulator({
            'count': hist.Hist('Counts', dataset_axis, count_axis, channel_axis),
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

        ak4jets = JaggedCandidateArray.candidatesfromcounts(
            df['akjet_ak4PFJetsCHS_p4'],
            px=df['akjet_ak4PFJetsCHS_p4.fCoordinates.fX'],
            py=df['akjet_ak4PFJetsCHS_p4.fCoordinates.fY'],
            pz=df['akjet_ak4PFJetsCHS_p4.fCoordinates.fZ'],
            energy=df['akjet_ak4PFJetsCHS_p4.fCoordinates.fT'],
            jetid=df['akjet_ak4PFJetsCHS_jetid'],
            deepcsv=df['hftagscore_DeepCSV_b'],
            hadfrac=df['akjet_ak4PFJetsCHS_hadronEnergyFraction'],
        )
        deepcsv_tight = np.bitwise_and(ak4jets.deepcsv, 1<<2)==(1<<2)
        ak4jets.add_attributes(deepcsvTight=deepcsv_tight)
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
        leptonjets.add_attributes(label=label, ndsa=ndsa)
        nmu = ((ljdautype==3)|(ljdautype==8)).sum()
        leptonjets.add_attributes(ismutype=(nmu>=2), iseltype=(nmu==0))

        ## __twoleptonjets__
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


        cuts = [
            np.ones_like(wgt).astype(bool),                        # all
            (np.abs(lj0.p4.delta_phi(lj1.p4))>np.pi/2).flatten(),  # dphi > pi/2
            ak4jets.counts<4,                                      # N(jets) < 4
            ak4jets[(ak4jets.pt>30)&(np.abs(ak4jets.eta)<2.4)&ak4jets.deepcsvTight].counts==0, # N(tightB)==0
            (~channel_2mu2e.astype(bool)) | (channel_2mu2e.astype(bool)&(((lj0.iseltype)&(lj0.pt>40)) | ((lj1.iseltype)&(lj1.pt>40))).flatten() ), # EGMpt0>40
            ( (lj0.ismutype&(lj0.pt>40)) | ((~lj0.ismutype)&(lj1.ismutype&(lj1.pt>40))) ).flatten(), # Mupt0>40
            ( (~(channel_==2)) | (channel_==2)&((lj1.pt>30).flatten()) ), # Mupt1>30
            ((lj0.label==3)|(lj1.label==3)).flatten(),             # >=1 dsalj
            ((lj0.ndsa==2)|(lj1.ndsa==2)).flatten(),               # >=1 dsalj(2dsa)
        ]

        if self.region == 'CR':
            cuts[1] = ~cuts[1]

        for i, c in enumerate(itertools.accumulate(cuts, np.logical_and)):
            output['count'].fill(dataset=dataset, cnt=np.ones_like(wgt[c])*i, weight=wgt[c], channel=channel_[c])


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
    from collections import OrderedDict
    import pandas as pd

    outdir = join(os.getenv('FH_BASE'), "Imgs", splitext(__file__)[0])
    if not isdir(outdir): os.makedirs(outdir)

    YieldsDf = {}

    ## SR

    CUTNAMES = dict(enumerate([
        '>=2lj',
        'dphi>pi/2',
        'Njets<4',
        'NtightB==0',
        'EGM0pt>40',
        'Mu0pt>40',
        'Mu1pt>30',
        '>=1DSALJ',
        '>=1DSALJ(2dsa)',
    ]))

    out_sig2mu2e = processor.run_uproot_job(sigDS_2mu2e,
                                  treename='ffNtuplizer/ffNtuple',
                                  processor_instance=LeptonjetProcessor(data_type='sig-2mu2e'),
                                  executor=processor.futures_executor,
                                  executor_args=dict(workers=12, flatten=True),
                                  chunksize=500000,
                                 )

    out_sig4mu = processor.run_uproot_job(sigDS_4mu,
                                  treename='ffNtuplizer/ffNtuple',
                                  processor_instance=LeptonjetProcessor(data_type='sig-4mu'),
                                  executor=processor.futures_executor,
                                  executor_args=dict(workers=12, flatten=True),
                                  chunksize=500000,
                                 )

    out_bkg = processor.run_uproot_job(bkgDS,
                                  treename='ffNtuplizer/ffNtuple',
                                  processor_instance=LeptonjetProcessor(data_type='bkg'),
                                  executor=processor.futures_executor,
                                  executor_args=dict(workers=12, flatten=True),
                                  chunksize=500000,
                                 )

    ## CHANNEL - 2mu2e
    outputs = OrderedDict()
    h_ = out_sig2mu2e['count'].integrate('channel', slice(1,2))
    d_ = { k[0]: v for k, v in h_.values().items() }
    outputs.update( sorted(d_.items(), key=lambda t: sigsort(t[0])) )

    h_ = out_bkg['count'].integrate('channel', slice(1,2))
    outputs.update( { k[0]: v for k, v in h_.values().items() } )

    df_ = pd.DataFrame(outputs).transpose()
    for k, n in CUTNAMES.items():
        df_.rename(columns={k: n}, inplace=True)
    YieldsDf['2mu2e-SR'] = df_


    ## CHANNEL - 4mu
    outputs = OrderedDict()
    h_ = out_sig4mu['count'].integrate('channel', slice(2, 3))
    d_ = { k[0]: v for k, v in h_.values().items() }
    outputs.update( sorted(d_.items(), key=lambda t: sigsort(t[0])) )

    h_ = out_bkg['count'].integrate('channel', slice(2, 3))
    outputs.update( { k[0]: v for k, v in h_.values().items() } )

    df_ = pd.DataFrame(outputs).transpose()
    for k, n in CUTNAMES.items():
        df_.rename(columns={k: n}, inplace=True)
    YieldsDf['4mu-SR'] = df_

    ## CR

    CUTNAMES = dict(enumerate([
        '>=2lj',
        'dphi<pi/2',
        'Njets<4',
        'NtightB==0',
        'EGM0pt>40',
        'Mu0pt>40',
        'Mu1pt>30',
        '>=1DSALJ',
        '>=1DSALJ(2dsa)',
    ]))

    out_sig2mu2e = processor.run_uproot_job(sigDS_2mu2e,
                                  treename='ffNtuplizer/ffNtuple',
                                  processor_instance=LeptonjetProcessor(data_type='sig-2mu2e', region='CR'),
                                  executor=processor.futures_executor,
                                  executor_args=dict(workers=12, flatten=True),
                                  chunksize=500000,
                                 )

    out_sig4mu = processor.run_uproot_job(sigDS_4mu,
                                  treename='ffNtuplizer/ffNtuple',
                                  processor_instance=LeptonjetProcessor(data_type='sig-4mu', region='CR'),
                                  executor=processor.futures_executor,
                                  executor_args=dict(workers=12, flatten=True),
                                  chunksize=500000,
                                 )

    out_bkg = processor.run_uproot_job(bkgDS,
                                  treename='ffNtuplizer/ffNtuple',
                                  processor_instance=LeptonjetProcessor(data_type='bkg', region='CR'),
                                  executor=processor.futures_executor,
                                  executor_args=dict(workers=12, flatten=True),
                                  chunksize=500000,
                                 )

    out_data = processor.run_uproot_job(dataDS,
                                  treename='ffNtuplizer/ffNtuple',
                                  processor_instance=LeptonjetProcessor(data_type='data', region='CR'),
                                  executor=processor.futures_executor,
                                  executor_args=dict(workers=12, flatten=True),
                                  chunksize=500000,
                                 )

    ## CHANNEL - 2mu2e
    outputs = OrderedDict()
    h_ = out_sig2mu2e['count'].integrate('channel', slice(1,2))
    d_ = { k[0]: v for k, v in h_.values().items() }
    outputs.update( sorted(d_.items(), key=lambda t: sigsort(t[0])) )

    h_ = out_bkg['count'].integrate('channel', slice(1,2))
    outputs.update( { k[0]: v for k, v in h_.values().items() } )

    h_ = out_data['count'].integrate('channel', slice(1,2))
    outputs.update( { k[0]: v for k, v in h_.values().items() } )

    df_ = pd.DataFrame(outputs).transpose()
    for k, n in CUTNAMES.items():
        df_.rename(columns={k: n}, inplace=True)
    YieldsDf['2mu2e-CR'] = df_


    ## CHANNEL - 4mu
    outputs = OrderedDict()
    h_ = out_sig4mu['count'].integrate('channel', slice(2, 3))
    d_ = { k[0]: v for k, v in h_.values().items() }
    outputs.update( sorted(d_.items(), key=lambda t: sigsort(t[0])) )

    h_ = out_bkg['count'].integrate('channel', slice(2, 3))
    outputs.update( { k[0]: v for k, v in h_.values().items() } )

    h_ = out_data['count'].integrate('channel', slice(2, 3))
    outputs.update( { k[0]: v for k, v in h_.values().items() } )

    df_ = pd.DataFrame(outputs).transpose()
    for k, n in CUTNAMES.items():
        df_.rename(columns={k: n}, inplace=True)
    YieldsDf['4mu-CR'] = df_


    with open(f'{outdir}/readme.txt', 'w') as outf:
        outf.write('SIGNAL REGION'.center(100, '_')+'\n')
        outf.write('+'*50+'\n')
        outf.write('CHANNEL - 2mu2e'.center(50, ' ')+'\n')
        outf.write('+'*50+'\n')
        outf.write(YieldsDf['2mu2e-SR'].to_string())
        outf.write('\n'*3)
        outf.write('+'*50+'\n')
        outf.write('CHANNEL - 4mu'.center(50, ' ')+'\n')
        outf.write('+'*50+'\n')
        outf.write(YieldsDf['4mu-SR'].to_string())
        outf.write('\n'*5)

        outf.write('CONTROL REGION'.center(100, '_')+'\n')
        outf.write('+'*50+'\n')
        outf.write('CHANNEL - 2mu2e'.center(50, ' ')+'\n')
        outf.write('+'*50+'\n')
        outf.write(YieldsDf['2mu2e-CR'].to_string())
        outf.write('\n'*3)
        outf.write('+'*50+'\n')
        outf.write('CHANNEL - 4mu'.center(50, ' ')+'\n')
        outf.write('+'*50+'\n')
        outf.write(YieldsDf['4mu-CR'].to_string())

    print(open(f'{outdir}/readme.txt').read())

    if args.sync:
        webdir = 'wsi@lxplus.cern.ch:/eos/user/w/wsi/www/public/firehydrant'
        cmd = f'rsync -az --exclude ".*" --delete {outdir} {webdir}'
        print(f"--> sync with: {webdir}")
        os.system(cmd)
