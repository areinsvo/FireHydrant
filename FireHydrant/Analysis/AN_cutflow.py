#!/usr/bin/env python
"""For AN
cutflow table
"""
import argparse
import itertools

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


parser = argparse.ArgumentParser(description="[AN] making cutflow tables")
parser.add_argument("--sync", action='store_true', help="issue rsync command to sync plots folder to lxplus web server")
args = parser.parse_args()

dml = DatasetMapLoader()
bkgDS, bkgMAP, bkgSCALE = dml.fetch('bkg')
dataDS, dataMAP = dml.fetch('data')

# sdml = SigDatasetMapLoader()
# sigDS_2mu2e, sigSCALE_2mu2e = sdml.fetch('2mu2e')

class CutflowProcessor(processor.ProcessorABC):
    def __init__(self, data_type='bkg', region='SR', enforceNeutral=True):
        self.data_type = data_type
        self.region = region
        self.enforceNeutral = enforceNeutral

        dataset_axis = hist.Cat('dataset', 'dataset')
        channel_axis = hist.Bin('channel', 'channel', 3, 0, 3)
        count_axis = hist.Bin('cnt', 'event count', 5, 0, 5)

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
            px=df['akjet_ak4PFJetsCHS_p4.fCoordinates.fX'].content,
            py=df['akjet_ak4PFJetsCHS_p4.fCoordinates.fY'].content,
            pz=df['akjet_ak4PFJetsCHS_p4.fCoordinates.fZ'].content,
            energy=df['akjet_ak4PFJetsCHS_p4.fCoordinates.fT'].content,
            jetid=df['akjet_ak4PFJetsCHS_jetid'].content,
            deepcsv=df['hftagscore_DeepCSV_b'].content,
        )
        deepcsv_tight = np.bitwise_and(ak4jets.deepcsv, 1<<2)==(1<<2)
        ak4jets.add_attributes(deepcsvTight=deepcsv_tight)
        ak4jets=ak4jets[ak4jets.jetid&(ak4jets.pt>20)&(np.abs(ak4jets.eta)<2.5)]

        leptonjets = JaggedCandidateArray.candidatesfromcounts(
            df['pfjet_p4'],
            px=df['pfjet_p4.fCoordinates.fX'].content,
            py=df['pfjet_p4.fCoordinates.fY'].content,
            pz=df['pfjet_p4.fCoordinates.fZ'].content,
            energy=df['pfjet_p4.fCoordinates.fT'].content,
            lxy=df['pfjet_klmvtx_lxy'].content,
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
        leptonjets.add_attributes(displaced=((np.abs(leptonjets.lxy)>=5)|(np.isnan(leptonjets.lxy)&leptonjets.ismutype))) # non-vertex treated as displaced too
        ljdsamuSubset = fromNestNestIndexArray(df['dsamuon_isSubsetFilteredCosmic1Leg'], awkward.fromiter(df['pfjet_pfcand_dsamuonIdx']))
        leptonjets.add_attributes(nocosmic=(ljdsamuSubset.sum()==0))

        leptonjets = leptonjets[(leptonjets.nocosmic)&(leptonjets.pt>30)]

        ## __twoleptonjets__ AND >=1 displaced
        twoleptonjets = (leptonjets.counts>=2)&(leptonjets.ismutype.sum()>=1)&(leptonjets.displaced.sum()>=1)
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
            ((lj0.isneutral)&(lj1.isneutral)).flatten(), # both 'neutral'
            (np.abs(lj0.p4.delta_phi(lj1.p4)) > np.pi / 2).flatten(),  # dphi > pi/2
            (~channel_2mu2e.astype(bool)) | (channel_2mu2e.astype(bool)&(((lj0.iseltype)&(lj0.pt>60)) | ((lj1.iseltype)&(lj1.pt>60))).flatten() ), # EGMpt0>60
            ak4jets.counts<3,                                      # N(jets) < 4
            ak4jets[(ak4jets.pt > 30) & (np.abs(ak4jets.eta) < 2.4) & ak4jets.deepcsvTight].counts == 0,  # N(tightB)==0
        ]

        if self.region == 'CR':
            cuts[1] = ~cuts[1]
        if self.enforceNeutral == False:
            cuts[0] = ~cuts[0]

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
            # if self.data_type == 'sig-2mu2e':
            #     accumulator[k].scale(sigSCALE_2mu2e, axis='dataset')
            # if self.data_type == 'sig-4mu':
            #     accumulator[k].scale(sigSCALE_4mu, axis='dataset')

        return accumulator


if __name__ == "__main__":
    import os
    from os.path import join, isdir, splitext
    from collections import OrderedDict
    import pandas as pd

    reldir = splitext(__file__)[0].replace('_', '/')
    outdir = join(os.getenv('FH_BASE'), "Imgs", reldir)
    if not isdir(outdir): os.makedirs(outdir)

    YieldsDf = {}

    ## SR
    CUTNAMES = dict(enumerate([
        'lj0,1 sumq0',
        'dphi>pi/2',
        'EGM0pt>60',
        'Njets<4',
        'NtightB==0',
    ]))

    out_bkg = processor.run_uproot_job(bkgDS,
                                  treename='ffNtuplizer/ffNtuple',
                                  processor_instance=CutflowProcessor(data_type='bkg', region='SR', enforceNeutral=True),
                                  executor=processor.futures_executor,
                                  executor_args=dict(workers=12, flatten=False),
                                  chunksize=500000,
                                 )
    # --- CHANNEL - 2mu2e
    outputs = OrderedDict()
    h_ = out_bkg['count'].integrate('channel', slice(1,2))
    outputs.update( { k[0]: v for k, v in h_.values().items() } )
    df_ = pd.DataFrame(outputs)
    for k, n in CUTNAMES.items():
        df_.rename(index={k: n}, inplace=True)
    YieldsDf['2mu2e-SR-OS'] = df_

    # --- CHANNEL - 4mu
    outputs = OrderedDict()
    h_ = out_bkg['count'].integrate('channel', slice(2,3))
    outputs.update( { k[0]: v for k, v in h_.values().items() } )
    df_ = pd.DataFrame(outputs)
    for k, n in CUTNAMES.items():
        df_.rename(index={k: n}, inplace=True)
    YieldsDf['4mu-SR-OS'] = df_


    ## CR
    CUTNAMES = dict(enumerate([
        'lj0,1 sumq0',
        'dphi<pi/2',
        'EGM0pt>60',
        'Njets<4',
        'NtightB==0',
    ]))

    out_bkg = processor.run_uproot_job(bkgDS,
                                  treename='ffNtuplizer/ffNtuple',
                                  processor_instance=CutflowProcessor(data_type='bkg', region='CR', enforceNeutral=True),
                                  executor=processor.futures_executor,
                                  executor_args=dict(workers=12, flatten=False),
                                  chunksize=500000,
                                 )

    out_data = processor.run_uproot_job(dataDS,
                                  treename='ffNtuplizer/ffNtuple',
                                  processor_instance=CutflowProcessor(data_type='data', region='CR', enforceNeutral=True),
                                  executor=processor.futures_executor,
                                  executor_args=dict(workers=12, flatten=False),
                                  chunksize=500000,
                                 )

    # --- CHANNEL - 2mu2e
    outputs = OrderedDict()
    h_ = out_bkg['count'].integrate('channel', slice(1,2))
    outputs.update( { k[0]: v for k, v in h_.values().items() } )

    h_ = out_data['count'].integrate('channel', slice(1,2))
    outputs.update( { k[0]: v for k, v in h_.values().items() } )

    df_ = pd.DataFrame(outputs)
    for k, n in CUTNAMES.items():
        df_.rename(index={k: n}, inplace=True)
    YieldsDf['2mu2e-CR-OS'] = df_

    # --- CHANNEL - 4mu
    outputs = OrderedDict()
    h_ = out_bkg['count'].integrate('channel', slice(2,3))
    outputs.update( { k[0]: v for k, v in h_.values().items() } )

    h_ = out_data['count'].integrate('channel', slice(2,3))
    outputs.update( { k[0]: v for k, v in h_.values().items() } )

    df_ = pd.DataFrame(outputs)
    for k, n in CUTNAMES.items():
        df_.rename(index={k: n}, inplace=True)
    YieldsDf['4mu-CR-OS'] = df_

    # ----------------------------------------------------------
    # SS
    # ----------------------------------------------------------

    ## SR
    CUTNAMES = dict(enumerate([
        '~lj0,1 sumq0',
        'dphi>pi/2',
        'EGM0pt>60',
        'Njets<4',
        'NtightB==0',
    ]))

    out_bkg = processor.run_uproot_job(bkgDS,
                                  treename='ffNtuplizer/ffNtuple',
                                  processor_instance=CutflowProcessor(data_type='bkg', region='SR', enforceNeutral=False),
                                  executor=processor.futures_executor,
                                  executor_args=dict(workers=12, flatten=False),
                                  chunksize=500000,
                                 )

    # --- CHANNEL - 2mu2e
    outputs = OrderedDict()
    h_ = out_bkg['count'].integrate('channel', slice(1,2))
    outputs.update( { k[0]: v for k, v in h_.values().items() } )
    df_ = pd.DataFrame(outputs)
    for k, n in CUTNAMES.items():
        df_.rename(index={k: n}, inplace=True)
    YieldsDf['2mu2e-SR-SS'] = df_

    # --- CHANNEL - 4mu
    outputs = OrderedDict()
    h_ = out_bkg['count'].integrate('channel', slice(2,3))
    outputs.update( { k[0]: v for k, v in h_.values().items() } )
    df_ = pd.DataFrame(outputs)
    for k, n in CUTNAMES.items():
        df_.rename(index={k: n}, inplace=True)
    YieldsDf['4mu-SR-SS'] = df_

    ## CR
    CUTNAMES = dict(enumerate([
        '~lj0,1 sumq0',
        'dphi<pi/2',
        'EGM0pt>60',
        'Njets<4',
        'NtightB==0',
    ]))

    out_bkg = processor.run_uproot_job(bkgDS,
                                  treename='ffNtuplizer/ffNtuple',
                                  processor_instance=CutflowProcessor(data_type='bkg', region='CR', enforceNeutral=False),
                                  executor=processor.futures_executor,
                                  executor_args=dict(workers=12, flatten=False),
                                  chunksize=500000,
                                 )

    out_data = processor.run_uproot_job(dataDS,
                                  treename='ffNtuplizer/ffNtuple',
                                  processor_instance=CutflowProcessor(data_type='data', region='CR', enforceNeutral=False),
                                  executor=processor.futures_executor,
                                  executor_args=dict(workers=12, flatten=False),
                                  chunksize=500000,
                                 )

    # --- CHANNEL - 2mu2e
    outputs = OrderedDict()
    h_ = out_bkg['count'].integrate('channel', slice(1,2))
    outputs.update( { k[0]: v for k, v in h_.values().items() } )

    h_ = out_data['count'].integrate('channel', slice(1,2))
    outputs.update( { k[0]: v for k, v in h_.values().items() } )

    df_ = pd.DataFrame(outputs)
    for k, n in CUTNAMES.items():
        df_.rename(index={k: n}, inplace=True)
    YieldsDf['2mu2e-CR-SS'] = df_

    # --- CHANNEL - 4mu
    outputs = OrderedDict()
    h_ = out_bkg['count'].integrate('channel', slice(2,3))
    outputs.update( { k[0]: v for k, v in h_.values().items() } )

    h_ = out_data['count'].integrate('channel', slice(2,3))
    outputs.update( { k[0]: v for k, v in h_.values().items() } )

    df_ = pd.DataFrame(outputs)
    for k, n in CUTNAMES.items():
        df_.rename(index={k: n}, inplace=True)
    YieldsDf['4mu-CR-SS'] = df_

    # -------------------------------------------------------------------
    with open(f'{outdir}/readme.txt', 'w') as outf:
        outf.write('SIGNAL REGION'.center(100, '_')+'\n')
        outf.write('+'*50+'\n')
        outf.write('CHANNEL - 2mu2e'.center(50, ' ')+'\n')
        outf.write('+'*50+'\n')

        outf.write('>'*25+'   neutral muon-type lj only\n')
        outf.write(YieldsDf['2mu2e-SR-OS'].to_string())
        outf.write('\n'*3)
        outf.write('>'*25+'   ~(neutral muon-type lj only)\n')
        outf.write(YieldsDf['2mu2e-SR-SS'].to_string())
        outf.write('\n'*3)

        outf.write('+'*50+'\n')
        outf.write('CHANNEL - 4mu'.center(50, ' ')+'\n')
        outf.write('+'*50+'\n')

        outf.write('>'*25+'   neutral muon-type lj only\n')
        outf.write(YieldsDf['4mu-SR-OS'].to_string())
        outf.write('\n'*3)
        outf.write('>'*25+'   ~(neutral muon-type lj only)\n')
        outf.write(YieldsDf['4mu-SR-SS'].to_string())
        outf.write('\n'*3)


        outf.write('\n'*5)

        outf.write('CONTROL REGION'.center(100, '_')+'\n')
        outf.write('+'*50+'\n')
        outf.write('CHANNEL - 2mu2e'.center(50, ' ')+'\n')
        outf.write('+'*50+'\n')

        outf.write('>'*25+'   neutral muon-type lj only\n')
        outf.write(YieldsDf['2mu2e-CR-OS'].to_string())
        outf.write('\n'*3)
        outf.write('>'*25+'   ~(neutral muon-type lj only)\n')
        outf.write(YieldsDf['2mu2e-CR-SS'].to_string())
        outf.write('\n'*3)

        outf.write('+'*50+'\n')
        outf.write('CHANNEL - 4mu'.center(50, ' ')+'\n')
        outf.write('+'*50+'\n')

        outf.write('>'*25+'   neutral muon-type lj only\n')
        outf.write(YieldsDf['4mu-CR-OS'].to_string())
        outf.write('\n'*3)
        outf.write('>'*25+'   ~(neutral muon-type lj only)\n')
        outf.write(YieldsDf['4mu-CR-SS'].to_string())
        outf.write('\n'*3)

    print(open(f'{outdir}/readme.txt').read())


    for k, df in YieldsDf.items():
        print(k)
        print(df.to_csv())
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
