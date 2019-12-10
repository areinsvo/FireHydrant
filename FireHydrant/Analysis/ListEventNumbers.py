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
from FireHydrant.Tools.uproothelpers import fromNestNestIndexArray

dml = DatasetMapLoader()
# bkgDS, bkgMAP, bkgSCALE = dml.fetch('bkg')
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
            vx=df['pfjet_klmvtx.fCoordinates.fX'].content,
            vy=df['pfjet_klmvtx.fCoordinates.fY'].content,
            vz=df['pfjet_klmvtx.fCoordinates.fZ'].content,
        )
        ## attribute: `label, ismutype, iseltype`
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
        ## attribute: `isneutral`
        ljdaucharge = awkward.fromiter(df['pfjet_pfcand_charge']).sum()
        leptonjets.add_attributes(qsum=ljdaucharge)
        leptonjets.add_attributes(isneutral=(leptonjets.iseltype | (leptonjets.ismutype&(leptonjets.qsum==0))))
        ## attribute: `displaced`
        leptonjets.add_attributes(vxy=np.hypot(leptonjets.vx, leptonjets.vy))
        leptonjets.add_attributes(displaced=((leptonjets.vxy>=5)|(np.isnan(leptonjets.vxy)&leptonjets.ismutype))) # non-vertex treated as displaced too
        ## attribute: `nocosmic`
        ljdsamuFoundOppo = fromNestNestIndexArray(df['dsamuon_hasOppositeMuon'], awkward.fromiter(df['pfjet_pfcand_dsamuonIdx']))
        dtcscTime = fromNestNestIndexArray(df['dsamuon_timeDiffDTCSC'], awkward.fromiter(df['pfjet_pfcand_dsamuonIdx']))
        rpcTime = fromNestNestIndexArray(df['dsamuon_timeDiffRPC'], awkward.fromiter(df['pfjet_pfcand_dsamuonIdx']))
        if len(dtcscTime.flatten().flatten()):
            dtcscTime = dtcscTime[ljdsamuFoundOppo]
            rpcTime = rpcTime[ljdsamuFoundOppo]
        ljdsamuSubset = fromNestNestIndexArray(df['dsamuon_isSubsetFilteredCosmic1Leg'], awkward.fromiter(df['pfjet_pfcand_dsamuonIdx']))
        leptonjets.add_attributes(nocosmic=( ((dtcscTime<-20).sum()==0) & ((rpcTime<-7.5).sum()==0) & (ljdsamuSubset.sum()==0) ))

        leptonjets = leptonjets[(leptonjets.nocosmic)&(leptonjets.pt>30)]

        ## __twoleptonjets__
        twoleptonjets = (leptonjets.counts>=2)&(leptonjets.ismutype.sum()>=1)&(leptonjets.displaced.sum()>=1)
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
            ((lj0.isneutral)&(lj1.isneutral)).flatten(), # both 'neutral'
            (np.abs(lj0.p4.delta_phi(lj1.p4)) > np.pi / 2).flatten(),  # dphi > pi/2
            (~channel_2mu2e.astype(bool)) | (channel_2mu2e.astype(bool)&(((lj0.iseltype)&(lj0.pt>60)) | ((lj1.iseltype)&(lj1.pt>60))).flatten() ), # EGMpt0>60
            ak4jets.counts<4,                                      # N(jets) < 4
            ak4jets[(ak4jets.pt > 30) & (np.abs(ak4jets.eta) < 2.4) & ak4jets.deepcsvTight].counts == 0,  # N(tightB)==0
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
                                  executor_args=dict(workers=12, flatten=False),
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

    print('4mu'.center(80, '_'))
    print(df_4mu)

    print('\n+++ ABC')
    for row in df_4mu.query('era!=3').itertuples(index=False):
        print(f'{row.run}:{row.lumi}:{row.event}')
    print('\n+++ D')
    for row in df_4mu.query('era==3').itertuples(index=False):
        print(f'{row.run}:{row.lumi}:{row.event}')


    df_2mu2e = pd.DataFrame(
        [
            out_['run_1'].value,
            out_['lumi_1'].value,
            out_['event_1'].value,
            out_['era_1'].value,
        ],
        index=['run', 'lumi', 'event', 'era'],
        dtype='Int64',
    ).transpose()
    df_2mu2e.sort_values(by=['run', 'lumi', 'event'], inplace=True)

    print('2mu2e'.center(80, '_'))
    print(df_2mu2e)

    print('\n+++ ABC')
    for row in df_2mu2e.query('era!=3').itertuples(index=False):
        print(f'{row.run}:{row.lumi}:{row.event}')
    print('\n+++ D')
    for row in df_2mu2e.query('era==3').itertuples(index=False):
        print(f'{row.run}:{row.lumi}:{row.event}')
