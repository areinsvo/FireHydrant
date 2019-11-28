#!/usr/bin/env python
"""
leptonjet, muon timing
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
from FireHydrant.Analysis.StudyLeadingSubleading import (filterSigDS,
                                                         groupHandleLabel)
from FireHydrant.Tools.correction import (get_nlo_weight_function,
                                          get_pu_weights_function,
                                          get_ttbar_weight)
from FireHydrant.Tools.metfilter import MetFilters
from FireHydrant.Tools.trigger import Triggers
from FireHydrant.Tools.uproothelpers import fromNestNestIndexArray
from matplotlib.ticker import LogLocator, SymmetricalLogLocator

np.seterr(divide='ignore', invalid='ignore', over='ignore')
plt.rcParams["savefig.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"


parser = argparse.ArgumentParser(description="mu-type leptonjets timing")
parser.add_argument("--sync", action='store_true', help="issue rsync command to sync plots folder to lxplus web server")
args = parser.parse_args()


sdml = SigDatasetMapLoader()
sigDS_2mu2e, sigSCALE_2mu2e = sdml.fetch('2mu2e')
sigDS_4mu, sigSCALE_4mu = sdml.fetch('4mu')

dml = DatasetMapLoader()
bkgDS, bkgMAP, bkgSCALE = dml.fetch('bkg')
dataDS, dataMAP = dml.fetch('data')

"""muon-type leptonjet timing"""
class MuonTimingProcessor(processor.ProcessorABC):
    def __init__(self, data_type='bkg', region='SR'):
        self.data_type = data_type
        self.region = region

        dataset_axis = hist.Cat('dataset', 'dataset')
        channel_axis = hist.Bin('channel', 'channel', 3, 0, 3)
        count_axis = hist.Bin('cnt', 'count', 3, 0, 3)
        time_axis = hist.Bin('t', 'timing(ns)', 100, -50, 50)
        self._accumulator = processor.dict_accumulator({
            'ndsa':  hist.Hist('Counts', dataset_axis, count_axis, channel_axis),
            'mutiming': hist.Hist('Counts', dataset_axis, time_axis, channel_axis),
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
            hadfrac=df['akjet_ak4PFJetsCHS_hadronEnergyFraction'].content,
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
            ncands=df['pfjet_pfcands_n'].content,
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

        leptonjets.add_attributes(muontiming=awkward.fromiter(df['pfjet_pfcand_muonTime']).mean())

        ## __ twoleptonjets__
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
        ]

        if self.region == 'CR':
            cuts[1] = ~cuts[1]

        totcut = np.logical_and.reduce(cuts)

        dileptonjets = dileptonjets[totcut]
        wgt = wgt[totcut]
        channel_ = channel_[totcut]

        ljmu = dileptonjets[dileptonjets.ismutype]
        ljmuones = ljmu.pt.ones_like()

        output['ndsa'].fill(dataset=dataset, cnt=ljmu.ndsa.flatten(), weight=(wgt*ljmuones).flatten(), channel=(channel_*ljmuones).flatten())
        output['mutiming'].fill(dataset=dataset, t=ljmu.muontiming.flatten(), weight=(wgt*ljmuones).flatten(), channel=(channel_*ljmuones).flatten())

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

    reldir = splitext(__file__)[0].replace('_', '/')
    outdir = join(os.getenv('FH_BASE'), "Imgs", reldir)
    if not isdir(outdir): os.makedirs(outdir)

    fill_opts = {
        'edgecolor': (0,0,0,0.3),
        'alpha': 0.8
    }
    error_opts = {
        'label':'Stat. Unc.',
        'hatch':'xxx',
        'facecolor':'none',
        'edgecolor':(0,0,0,.5),
        'linewidth': 0
    }
    data_err_opts = {
        'linestyle':'none',
        'marker': '.',
        'markersize': 10.,
        'color':'k',
        'elinewidth': 1,
        'emarker': '_'
    }

    outputs = {}
    outputs['data'] = processor.run_uproot_job(dataDS,
                                  treename='ffNtuplizer/ffNtuple',
                                  processor_instance=MuonTimingProcessor(region='CR', data_type='data'),
                                  executor=processor.futures_executor,
                                  executor_args=dict(workers=12, flatten=False),
                                  chunksize=500000,
                                 )


    ## CHANNEL - 2mu2e
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    h = outputs['data']['ndsa'].integrate('channel', slice(1,2))
    hist.plot1d(h, overlay='cat', ax=ax, overflow='over', error_opts=data_err_opts)
    ax.set_title('[2mu2e|CR] mu-type leptonjet N(dsa)', x=0.0, ha="left")
    ax.set_yscale('symlog')
    ax.autoscale(axis='both', tight=True)
    ax.text(1,1,'59.74/fb (13TeV)', ha='right', va='bottom', transform=ax.transAxes)
    ax.get_yaxis().set_major_locator(SymmetricalLogLocator(base=10., linthresh=1, subs=range(1,10)))
    ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
    ax.set_ylabel(ax.get_ylabel(), y=1.0, ha="right")
    fig.savefig(join(outdir, 'ndsa_CR_2mu2e.png'))
    fig.savefig(join(outdir, 'ndsa_CR_2mu2e.pdf'))
    plt.close(fig)

    fig, ax = plt.subplots(1,1,figsize=(8,6))
    h = outputs['data']['mutiming'].integrate('channel', slice(1,2))
    hist.plot1d(h, overlay='cat', ax=ax, overflow='over', error_opts=data_err_opts)
    ax.set_title('[2mu2e|CR] mu-type leptonjet mean timing', x=0.0, ha="left")
    ax.autoscale(axis='both', tight=True)
    ax.text(1,1,'59.74/fb (13TeV)', ha='right', va='bottom', transform=ax.transAxes)
    ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
    ax.set_ylabel(ax.get_ylabel(), y=1.0, ha="right")
    fig.savefig(join(outdir, 'mutiming_CR_2mu2e.png'))
    fig.savefig(join(outdir, 'mutiming_CR_2mu2e.pdf'))
    plt.close(fig)


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
