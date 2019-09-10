#!/usr/bin/env python
"""collect feature variables from ffNtuples for BDT training
"""
from coffea.analysis_objects import JaggedCandidateArray
from coffea.processor import column_accumulator
import coffea.processor as processor

import numpy as np
np.seterr(divide='ignore', invalid='ignore', over='ignore')

from FireHydrant.Tools.uproothelpers import NestNestObjArrayToJagged
from FireHydrant.Tools.trigger import Triggers
from FireHydrant.Tools.metfilter import MetFilters


class SignalLeptonJetsFeatureHarvester(processor.ProcessorABC):
    def __init__(self):

        self._accumulator = processor.dict_accumulator({
            'pt': column_accumulator(np.zeros(shape=(0,))),
            'eta': column_accumulator(np.zeros(shape=(0,))),
            'nef': column_accumulator(np.zeros(shape=(0,))),
            'maxd0': column_accumulator(np.zeros(shape=(0,))),
            'mind0': column_accumulator(np.zeros(shape=(0,))),
            'maxd0sig': column_accumulator(np.zeros(shape=(0,))),
            'mind0sig': column_accumulator(np.zeros(shape=(0,))),
            'tkiso05': column_accumulator(np.zeros(shape=(0,))),
            'pfiso05': column_accumulator(np.zeros(shape=(0,))),
            'tkiso06': column_accumulator(np.zeros(shape=(0,))),
            'pfiso06': column_accumulator(np.zeros(shape=(0,))),
            'tkiso07': column_accumulator(np.zeros(shape=(0,))),
            'pfiso07': column_accumulator(np.zeros(shape=(0,))),
            'spreadpt': column_accumulator(np.zeros(shape=(0,))),
            'spreaddr': column_accumulator(np.zeros(shape=(0,))),
            'lamb': column_accumulator(np.zeros(shape=(0,))),
            'epsi': column_accumulator(np.zeros(shape=(0,))),
            'ecfe1': column_accumulator(np.zeros(shape=(0,))),
            'ecfe2': column_accumulator(np.zeros(shape=(0,))),
            'ecfe3': column_accumulator(np.zeros(shape=(0,))),
            'label': column_accumulator(np.zeros(shape=(0,))),
        })

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, df):
        output = self.accumulator.identity()

        absd0 = np.abs(NestNestObjArrayToJagged(df['pfjet_pfcand_tkD0'])).fillna(0)
        d0sig = NestNestObjArrayToJagged(df['pfjet_pfcand_tkD0Sig']).fillna(0)

        leptonjets = JaggedCandidateArray.candidatesfromcounts(
            df['pfjet_p4'],
            px=df['pfjet_p4.fCoordinates.fX'],
            py=df['pfjet_p4.fCoordinates.fY'],
            pz=df['pfjet_p4.fCoordinates.fZ'],
            energy=df['pfjet_p4.fCoordinates.fT'],
            nef=(df['pfjet_neutralEmE']+df['pfjet_neutralHadronE'])/df['pfjet_p4.fCoordinates.fT'],
            maxd0=absd0.max().content,
            mind0=absd0.min().content,
            maxd0sig=d0sig.max().content,
            mind0sig=d0sig.min().content,
            tkiso05=df['pfjet_tkIsolation05'],
            pfiso05=df['pfjet_pfIsolation05'],
            tkiso06=df['pfjet_tkIsolation06'],
            pfiso06=df['pfjet_pfIsolation06'],
            tkiso07=df['pfjet_tkIsolation07'],
            pfiso07=df['pfjet_pfIsolation07'],
            spreadpt=df['pfjet_ptDistribution'],
            spreaddr=df['pfjet_dRSpread'],
            lamb=df['pfjet_subjet_lambda'],
            epsi=df['pfjet_subjet_epsilon'],
            ecf1=df['pfjet_subjet_ecf1'],
            ecf2=df['pfjet_subjet_ecf2'],
            ecf3=df['pfjet_subjet_ecf3'],
        )
        genparticles = JaggedCandidateArray.candidatesfromcounts(
            df['gen_p4'],
            px=df['gen_p4.fCoordinates.fX'],
            py=df['gen_p4.fCoordinates.fY'],
            pz=df['gen_p4.fCoordinates.fZ'],
            energy=df['gen_p4.fCoordinates.fT'],
            pid=df['gen_pid']
        )
        darkphotons = genparticles[genparticles.pid==32]
        matchmask = leptonjets.match(darkphotons, deltaRCut=0.3)

        metfiltermask = np.logical_and.reduce([df[mf] for mf in MetFilters])
        triggermask = np.logical_or.reduce([df[tp] for tp in Triggers])

        leptonjets = leptonjets[metfiltermask&triggermask]
        matchmask  = matchmask[metfiltermask&triggermask]

        output['pt']       += column_accumulator(leptonjets.pt.flatten())
        output['eta']      += column_accumulator(leptonjets.eta.flatten())
        output['nef']      += column_accumulator(leptonjets.nef.flatten())
        output['maxd0']    += column_accumulator(leptonjets.maxd0.flatten())
        output['mind0']    += column_accumulator(leptonjets.mind0.flatten())
        output['maxd0sig'] += column_accumulator(leptonjets.maxd0sig.flatten())
        output['mind0sig'] += column_accumulator(leptonjets.mind0sig.flatten())
        output['tkiso05']  += column_accumulator(leptonjets.tkiso05.flatten())
        output['pfiso05']  += column_accumulator(leptonjets.pfiso05.flatten())
        output['tkiso06']  += column_accumulator(leptonjets.tkiso06.flatten())
        output['pfiso06']  += column_accumulator(leptonjets.pfiso06.flatten())
        output['tkiso07']  += column_accumulator(leptonjets.tkiso07.flatten())
        output['pfiso07']  += column_accumulator(leptonjets.pfiso07.flatten())
        output['spreadpt'] += column_accumulator(leptonjets.spreadpt.flatten())
        output['spreaddr'] += column_accumulator(leptonjets.spreaddr.flatten())
        output['lamb']     += column_accumulator(leptonjets.lamb.flatten())
        output['epsi']     += column_accumulator(leptonjets.epsi.flatten())
        output['ecfe1']    += column_accumulator(leptonjets.ecf1.flatten())
        output['ecfe2']    += column_accumulator(leptonjets.ecf2.flatten())
        output['ecfe3']    += column_accumulator(leptonjets.ecf3.flatten())
        output['label']    += column_accumulator(matchmask.flatten())

        return output

    def postprocess(self, accumulator):
        return accumulator


class BackgroundLeptonJetsFeatureHarvester(processor.ProcessorABC):
    def __init__(self):

        self._accumulator = processor.dict_accumulator({
            'pt': column_accumulator(np.zeros(shape=(0,))),
            'eta': column_accumulator(np.zeros(shape=(0,))),
            'nef': column_accumulator(np.zeros(shape=(0,))),
            'maxd0': column_accumulator(np.zeros(shape=(0,))),
            'mind0': column_accumulator(np.zeros(shape=(0,))),
            'maxd0sig': column_accumulator(np.zeros(shape=(0,))),
            'mind0sig': column_accumulator(np.zeros(shape=(0,))),
            'tkiso05': column_accumulator(np.zeros(shape=(0,))),
            'pfiso05': column_accumulator(np.zeros(shape=(0,))),
            'tkiso06': column_accumulator(np.zeros(shape=(0,))),
            'pfiso06': column_accumulator(np.zeros(shape=(0,))),
            'tkiso07': column_accumulator(np.zeros(shape=(0,))),
            'pfiso07': column_accumulator(np.zeros(shape=(0,))),
            'spreadpt': column_accumulator(np.zeros(shape=(0,))),
            'spreaddr': column_accumulator(np.zeros(shape=(0,))),
            'lamb': column_accumulator(np.zeros(shape=(0,))),
            'epsi': column_accumulator(np.zeros(shape=(0,))),
            'ecfe1': column_accumulator(np.zeros(shape=(0,))),
            'ecfe2': column_accumulator(np.zeros(shape=(0,))),
            'ecfe3': column_accumulator(np.zeros(shape=(0,))),
            'label': column_accumulator(np.zeros(shape=(0,))),
        })

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, df):
        output = self.accumulator.identity()

        absd0 = np.abs(NestNestObjArrayToJagged(df['pfjet_pfcand_tkD0'])).fillna(0)
        d0sig = NestNestObjArrayToJagged(df['pfjet_pfcand_tkD0Sig']).fillna(0)

        leptonjets = JaggedCandidateArray.candidatesfromcounts(
            df['pfjet_p4'],
            px=df['pfjet_p4.fCoordinates.fX'],
            py=df['pfjet_p4.fCoordinates.fY'],
            pz=df['pfjet_p4.fCoordinates.fZ'],
            energy=df['pfjet_p4.fCoordinates.fT'],
            nef=(df['pfjet_neutralEmE']+df['pfjet_neutralHadronE'])/df['pfjet_p4.fCoordinates.fT'],
            maxd0=absd0.max().content,
            mind0=absd0.min().content,
            maxd0sig=d0sig.max().content,
            mind0sig=d0sig.min().content,
            tkiso05=df['pfjet_tkIsolation05'],
            pfiso05=df['pfjet_pfIsolation05'],
            tkiso06=df['pfjet_tkIsolation06'],
            pfiso06=df['pfjet_pfIsolation06'],
            tkiso07=df['pfjet_tkIsolation07'],
            pfiso07=df['pfjet_pfIsolation07'],
            spreadpt=df['pfjet_ptDistribution'],
            spreaddr=df['pfjet_dRSpread'],
            lamb=df['pfjet_subjet_lambda'],
            epsi=df['pfjet_subjet_epsilon'],
            ecf1=df['pfjet_subjet_ecf1'],
            ecf2=df['pfjet_subjet_ecf2'],
            ecf3=df['pfjet_subjet_ecf3'],
        )


        metfiltermask = np.logical_and.reduce([df[mf] for mf in MetFilters])
        triggermask = np.logical_or.reduce([df[tp] for tp in Triggers])

        leptonjets = leptonjets[metfiltermask&triggermask]

        output['pt']       += column_accumulator(leptonjets.pt.flatten())
        output['eta']      += column_accumulator(leptonjets.eta.flatten())
        output['nef']      += column_accumulator(leptonjets.nef.flatten())
        output['maxd0']    += column_accumulator(leptonjets.maxd0.flatten())
        output['mind0']    += column_accumulator(leptonjets.mind0.flatten())
        output['maxd0sig'] += column_accumulator(leptonjets.maxd0sig.flatten())
        output['mind0sig'] += column_accumulator(leptonjets.mind0sig.flatten())
        output['tkiso05']  += column_accumulator(leptonjets.tkiso05.flatten())
        output['pfiso05']  += column_accumulator(leptonjets.pfiso05.flatten())
        output['tkiso06']  += column_accumulator(leptonjets.tkiso06.flatten())
        output['pfiso06']  += column_accumulator(leptonjets.pfiso06.flatten())
        output['tkiso07']  += column_accumulator(leptonjets.tkiso07.flatten())
        output['pfiso07']  += column_accumulator(leptonjets.pfiso07.flatten())
        output['spreadpt'] += column_accumulator(leptonjets.spreadpt.flatten())
        output['spreaddr'] += column_accumulator(leptonjets.spreaddr.flatten())
        output['lamb']     += column_accumulator(leptonjets.lamb.flatten())
        output['epsi']     += column_accumulator(leptonjets.epsi.flatten())
        output['ecfe1']    += column_accumulator(leptonjets.ecf1.flatten())
        output['ecfe2']    += column_accumulator(leptonjets.ecf2.flatten())
        output['ecfe3']    += column_accumulator(leptonjets.ecf3.flatten())
        output['label']    += column_accumulator(leptonjets.pt.zeros_like().flatten())

        return output

    def postprocess(self, accumulator):
        return accumulator


if __name__ == "__main__":

    ## prepare datasets
    import os
    import json
    from os.path import join

    dataset4mu_   = json.load(open(join(os.getenv('FH_BASE'), 'Notebooks/MC/Samples/signal_4mu.json')))
    dataset2mu2e_ = json.load(open(join(os.getenv('FH_BASE'), 'Notebooks/MC/Samples/signal_2mu2e.json')))
    datasetsignal = {}
    datasetsignal.update({
        f'4mu/{k}': dict(files=v, treename='ffNtuplizer/ffNtuple')
        for k, v in dataset4mu_.items()
    })
    datasetsignal.update({
        f'2mu2e/{k}': dict(files=v, treename='ffNtuplizer/ffNtuple')
        for k, v in dataset2mu2e_.items()
    })

    datasetbkg_ = json.load(open(join(os.getenv('FH_BASE'),
                                'Notebooks/MC/Samples/backgrounds_nonempty.json')))
    datasetbackgrounds = {}
    for group in datasetbkg_:
        for tag in datasetbkg_[group]:
            files = datasetbkg_[group][tag]
            datasetbackgrounds[tag] = {'files': files, 'treename': 'ffNtuples/ffNtuple'}
            if tag=='TTJets': datasetbackgrounds[tag]['treename'] = 'ffNtuplizer/ffNtuple'


    import time
    starttime = time.time()
    print("Start harvesting at:", time.ctime())

    ## collect from signal
    print("Collecting from signal ...")
    output_s = processor.run_uproot_job(datasetsignal,
                                  treename=None,
                                  processor_instance=SignalLeptonJetsFeatureHarvester(),
                                  executor=processor.futures_executor,
                                  executor_args=dict(workers=12, flatten=True),
                                  chunksize=500000,
                                 )
    print("... done.")

    ## collect from backgrounds
    print("Collecting from backgrounds ...")
    output_b = processor.run_uproot_job(datasetbackgrounds,
                                  treename=None,
                                  processor_instance=BackgroundLeptonJetsFeatureHarvester(),
                                  executor=processor.futures_executor,
                                  executor_args=dict(workers=12, flatten=True),
                                  chunksize=500000,
                                 )
    print("... done.")

    ## merge together
    output = output_s + output_b

    ## save as dataframe
    import pandas as pd
    df = pd.DataFrame({k: v.value for k, v in output.items()})
    df.fillna(0, inplace=True) # zero-padding
    print(df.tail())

    filename_ = 'trainingdatasplit.h5'

    ## saving to disk
    df.query("nef>=0.999").to_hdf(filename_, key='notrack')
    df.query("nef< 0.999").to_hdf(filename_, key='tracked')
    print(f"Saving dataframe as '{filename_}' with key 'notrack' and 'tracked'.")
    print("--> took {} s".format(time.time() - starttime))
    print(f"To load:\n\t`df = pd.read_hdf('{filename_}', 'notrack')` or\n\t`df = pd.read_hdf('{filename_}', 'tracked')`")