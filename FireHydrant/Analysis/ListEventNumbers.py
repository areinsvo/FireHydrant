#!/usr/bin/env python
"""
List interesting/filtered event numbers
"""
import awkward
import coffea.processor as processor
import numpy as np
from coffea import hist
from coffea.analysis_objects import JaggedCandidateArray
from FireHydrant.Analysis.DatasetMapLoader import DatasetMapLoader
from FireHydrant.Tools.correction import (get_nlo_weight_function,
                                          get_pu_weights_function,
                                          get_ttbar_weight)
from FireHydrant.Tools.metfilter import MetFilters
from FireHydrant.Tools.trigger import Triggers

dml = DatasetMapLoader()
bkgDS, bkgMAP, bkgSCALE = dml.fetch('bkg')
dataDS, dataMAP = dml.fetch('data')


"""list events"""
class LeptonjetEventDrawer(processor.ProcessorABC):
    def __init__(self, data_type='data'):
        self.data_type = data_type
        self._accumulator = processor.dict_accumulator({
            'run_1': processor.column_accumulator(np.zeros(shape=(0,))),
            'lumi_1': processor.column_accumulator(np.zeros(shape=(0,))),
            'event_1': processor.column_accumulator(np.zeros(shape=(0,))),
            'run_2': processor.column_accumulator(np.zeros(shape=(0,))),
            'lumi_2': processor.column_accumulator(np.zeros(shape=(0,))),
            'event_2': processor.column_accumulator(np.zeros(shape=(0,))),
            'era_1': processor.column_accumulator(np.zeros(shape=(0,))),
            'era_2': processor.column_accumulator(np.zeros(shape=(0,))),
        })

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, df):
        output = self.accumulator.identity()
        if df.size==0: return output

        dataset = df['dataset']
        run = df['run']
        lumi = df['lumi']
        event = df['event']

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
        leptonjets.add_attributes(label=label)
        nmu = ((ljdautype==3)|(ljdautype==8)).sum()
        leptonjets.add_attributes(ismutype=(nmu>=2), iseltype=(nmu==0))

        ## __twoleptonjets__
        twoleptonjets = leptonjets.counts>=2
        dileptonjets = leptonjets[twoleptonjets]
        ak4jets = ak4jets[twoleptonjets]
        wgt = weight[twoleptonjets]

        run = run[twoleptonjets]
        lumi = lumi[twoleptonjets]
        event = event[twoleptonjets]

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
            wgt.astype(bool),                        # all, non-zero weighted events
            (np.abs(lj0.p4.delta_phi(lj1.p4))>np.pi/2).flatten(),  # dphi > pi/2
            ak4jets.counts<4,                                      # N(jets) < 4
            ak4jets[(ak4jets.pt>30)&(np.abs(ak4jets.eta)<2.4)&ak4jets.deepcsvTight].counts==0, # N(tightB)==0
            (~channel_2mu2e.astype(bool)) | (channel_2mu2e.astype(bool)&(((lj0.iseltype)&(lj0.pt>40)) | ((lj1.iseltype)&(lj1.pt>40))).flatten() ), # EGMpt0>40
            ( (lj0.ismutype&(lj0.pt>40)) | ((~lj0.ismutype)&(lj1.ismutype&(lj1.pt>40))) ).flatten(), # Mupt0>40
            ( (~(channel_==2)) | (channel_==2)&((lj1.pt>30).flatten()) ), # Mupt1>30
        ]
        if self.data_type == 'data':
            cuts[1] = ~cuts[1]

        totcut = np.logical_and.reduce(cuts)

        output['run_1']   += processor.column_accumulator(run[totcut&(channel_==1)])
        output['lumi_1']  += processor.column_accumulator(lumi[totcut&(channel_==1)])
        output['event_1'] += processor.column_accumulator(event[totcut&(channel_==1)])
        output['run_2']   += processor.column_accumulator(run[totcut&(channel_==2)])
        output['lumi_2']  += processor.column_accumulator(lumi[totcut&(channel_==2)])
        output['event_2'] += processor.column_accumulator(event[totcut&(channel_==2)])

        if self.data_type == 'data':
            era = np.ones_like(run) * list('ABCD').index(dataset)
            output['era_1']   += processor.column_accumulator(era[totcut&(channel_==1)])
            output['era_2']   += processor.column_accumulator(era[totcut&(channel_==2)])

        return output

    def postprocess(self, accumulator):
        return accumulator


if __name__ == "__main__":
    import pandas as pd

    out_ = processor.run_uproot_job(dataDS,
                                  treename='ffNtuplizer/ffNtuple',
                                  processor_instance=LeptonjetEventDrawer(data_type='data'),
                                  executor=processor.futures_executor,
                                  executor_args=dict(workers=12, flatten=True),
                                  chunksize=500000,
                                 )

    df_4mu = pd.DataFrame(
        [
            out_['run_2'].value,
            out_['lumi_2'].value,
            out_['event_2'].value,
            out_['era_2'].value,
        ],
        index=['run', 'lumi', 'event', 'era'],
        dtype='Int64',
    ).transpose()
    df_4mu.sort_values(by=['run', 'lumi', 'event'], inplace=True)

    print(df_4mu)

    print('\n+++ ABC')
    for row in df_4mu.query('era!=3').itertuples(index=False):
        print(f'{row.run}:{row.lumi}:{row.event}')
    print('\n+++ D')
    for row in df_4mu.query('era==3').itertuples(index=False):
        print(f'{row.run}:{row.lumi}:{row.event}')
