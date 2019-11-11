#!/usr/bin/env python

import json
import os
from os.path import join, basename

DATASET_BKG = join(os.getenv("FH_BASE"), "FireHydrant/Samples/latest/skimmed_backgrounds.json")
DATASET_DATA = join(os.getenv("FH_BASE"), "FireHydrant/Samples/latest/skimmed_control_data2018.json")
SCALE_BKG = join(os.getenv("FH_BASE"), "FireHydrant/Samples/latest/skimmed_backgrounds_scale.json")

DATASET_SIG_4MU = join(os.getenv('FH_BASE'), 'FireHydrant/Samples/latest/signal_4mu.json')
DATASET_SIG_2MU2E = join(os.getenv('FH_BASE'), 'FireHydrant/Samples/latest/signal_2mu2e.json')
SCALE_SIG_4MU = join(os.getenv('FH_BASE'), 'FireHydrant/Samples/latest/signal_4mu_scale.json')
SCALE_SIG_2MU2E = join(os.getenv('FH_BASE'), 'FireHydrant/Samples/latest/signal_2mu2e_scale.json')

LUMI = 59.74 * 1e3


class DatasetMapLoader:
    def __init__(self, debug=True):
        self.fBkg = json.load(open(DATASET_BKG))
        self.fData = json.load(open(DATASET_DATA))
        self.fBkgScale = json.load(open(SCALE_BKG))
        self.fLumi = LUMI
        if debug:
            print("DatasetMapLoader loading...")
            print('+'*50)
            print("@ Backgrounds -->", basename(DATASET_BKG))
            print("@ Data        -->", basename(DATASET_DATA))
            print("@ bkg scale   -->", basename(SCALE_BKG))
            print(f"@ Lumi set as --> {LUMI}/pb")
            print('+'*50)

    def get_mapping(self, which):
        mapping = {}
        if which == "bkg":
            mapping = {k: list(self.fBkg[k]) for k in self.fBkg}
        if which == "data":
            mapping = {"data": list("ABCD")}
        if which == "both":
            mapping = {k: list(self.fBkg[k]) for k in self.fBkg}
            mapping.update({"data": list("ABCD")})
        return mapping


    def get_datasets(self, which):
        datasets = {}
        if which == "bkg":
            datasets = {tag: self.fBkg[group][tag] for group in self.fBkg for tag in self.fBkg[group]}
        if which == "data":
            datasets = self.fData
        if which == "both":
            datasets = {tag: self.fBkg[group][tag] for group in self.fBkg for tag in self.fBkg[group]}
            datasets.update(self.fData)
        return datasets

    def get_bkgscales(self):
        scales = {tag: self.fBkgScale[group][tag] for group in self.fBkgScale for tag in self.fBkgScale[group]}
        for k in scales:
            scales[k] *= self.fLumi
        return scales

    def fetch(self, which="both"):
        if which == "both" or which == "bkg":
            return self.get_datasets(which), self.get_mapping(which), self.get_bkgscales()
        elif which == "data":
            return self.get_datasets(which), self.get_mapping(which)
        else:
            raise ValueError("`which` can only be bkg/data/both.")



class SigDatasetMapLoader:
    def __init__(self, debug=True):
        self.fSig4mu = json.load(open(DATASET_SIG_4MU))
        self.fSig2mu2e = json.load(open(DATASET_SIG_2MU2E))
        self.fScale4mu = json.load(open(SCALE_SIG_4MU))
        self.fScale2mu2e = json.load(open(SCALE_SIG_2MU2E))
        self.fLumi = LUMI
        if debug:
            print("SigDatasetMapLoader loading...")
            print("+"*80)
            print("@ 4mu   / scale -->", basename(DATASET_SIG_4MU), '/', basename(SCALE_SIG_4MU))
            print("@ 2mu2e / scale -->", basename(DATASET_SIG_2MU2E), '/', basename(SCALE_SIG_2MU2E))
            print(f"@ Lumi set as   --> {LUMI}/pb")
            print("+"*80)

    def get_datasets(self, which):
        res = {}
        if which == "all":
            res.update({f'4mu/{k}': self.fSig4mu[k] for k in self.fSig4mu})
            res.update({f'2mu2e/{k}': self.fSig2mu2e[k] for k in self.fSig2mu2e})
        elif which == "4mu":
            res.update(self.fSig4mu)
        elif which == "2mu2e":
            res.update(self.fSig4mu)
        elif which == "simple":
            res.update({
                '4mu/mXX-100_mA-5_lxy-300': self.fSig4mu['mXX-100_mA-5_lxy-300'],
                '4mu/mXX-1000_mA-0p25_lxy-300': self.fSig4mu['mXX-1000_mA-0p25_lxy-300'],
                '2mu2e/mXX-100_mA-5_lxy-300': self.fSig2mu2e['mXX-100_mA-5_lxy-300'],
                '2mu2e/mXX-1000_mA-0p25_lxy-300': self.fSig2mu2e['mXX-1000_mA-0p25_lxy-300'],

                '4mu/mXX-100_mA-5_lxy-0p3': self.fSig4mu['mXX-100_mA-5_lxy-0p3'],
                '4mu/mXX-1000_mA-0p25_lxy-0p3': self.fSig4mu['mXX-1000_mA-0p25_lxy-0p3'],
                '2mu2e/mXX-100_mA-5_lxy-0p3': self.fSig2mu2e['mXX-100_mA-5_lxy-0p3'],
                '2mu2e/mXX-1000_mA-0p25_lxy-0p3': self.fSig2mu2e['mXX-1000_mA-0p25_lxy-0p3'],
            })
        else:
            raise ValueError("`which` can only be all/4mu/2mu2e/simple.")
        return res

    def get_scales(self, which):
        res = {}
        if which == "all":
            res.update({f'4mu/{k}': self.fScale4mu[k] for k in self.fScale4mu})
            res.update({f'2mu2e/{k}': self.fScale2mu2e[k] for k in self.fScale2mu2e})
        elif which == "4mu":
            res.update(self.fScale4mu)
        elif which == "2mu2e":
            res.update(self.fScale2mu2e)
        elif which == "simple":
            res.update({
                '4mu/mXX-100_mA-5_lxy-300': self.fScale4mu['mXX-100_mA-5_lxy-300'],
                '4mu/mXX-1000_mA-0p25_lxy-300': self.fScale4mu['mXX-1000_mA-0p25_lxy-300'],
                '2mu2e/mXX-100_mA-5_lxy-300':self.fScale2mu2e['mXX-100_mA-5_lxy-300'],
                '2mu2e/mXX-1000_mA-0p25_lxy-300': self.fScale2mu2e['mXX-1000_mA-0p25_lxy-300'],

                '4mu/mXX-100_mA-5_lxy-0p3': self.fScale4mu['mXX-100_mA-5_lxy-0p3'],
                '4mu/mXX-1000_mA-0p25_lxy-0p3': self.fScale4mu['mXX-1000_mA-0p25_lxy-0p3'],
                '2mu2e/mXX-100_mA-5_lxy-0p3': self.fScale2mu2e['mXX-100_mA-5_lxy-0p3'],
                '2mu2e/mXX-1000_mA-0p25_lxy-0p3': self.fScale2mu2e['mXX-1000_mA-0p25_lxy-0p3'],
            })
        else:
            raise ValueError("`which` can only be all/4mu/2mu2e/simple.")

        res = {k: self.fLumi/1e3 * res[k] for k in res}

        return res

    def fetch(self, which="all"):
        return self.get_datasets(which), self.get_scales(which)



if __name__ == "__main__":

    dml = DatasetMapLoader()
    print(dml.fetch())
    print(dml.fetch("bkg"))
    print(dml.fetch('data'))

    sdml = SigDatasetMapLoader()
    print(sdml.fetch())
    print(sdml.fetch("4mu"))
    print(sdml.fetch("2mu2e"))
    print(sdml.fetch("simple"))