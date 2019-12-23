#!/usr/bin/env python
"""For AN
About hadronic jets
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
from FireHydrant.Tools.uproothelpers import fromNestNestIndexArray

np.seterr(divide='ignore', invalid='ignore', over='ignore')
plt.rcParams["savefig.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"


parser = argparse.ArgumentParser(description="[AN] about hadronic jets")
parser.add_argument("--sync", action='store_true', help="issue rsync command to sync plots folder to lxplus web server")
args = parser.parse_args()

dml = DatasetMapLoader()
bkgDS, bkgMAP, bkgSCALE = dml.fetch('bkg')
# dataDS, dataMAP = dml.fetch('data')

sdml = SigDatasetMapLoader()
sigDS_2mu2e, sigSCALE_2mu2e = sdml.fetch('2mu2e')
sigDS_4mu, sigSCALE_4mu = sdml.fetch('4mu')


class HadronicJetProcessor(processor.ProcessorABC):
    def __init__(self, data_type='bkg'):
        self.data_type = data_type

        dataset_axis = hist.Cat('dataset', 'dataset')
        count_axis = hist.Bin('cnt', 'Number of Jets', 10, 0, 10)
        channel_axis = hist.Bin('channel', 'channel', 3, 0, 3)
        self._accumulator = processor.dict_accumulator({
            'njets': hist.Hist('Counts', dataset_axis, count_axis, channel_axis),
            'ntightb': hist.Hist('Counts', dataset_axis, count_axis, channel_axis),
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
            muefrac=df['akjet_ak4PFJetsCHS_muonEnergyFraction'].content,
            chaemefrac=df['akjet_ak4PFJetsCHS_chaEmEnergyFraction'].content,
            emefrac=df['akjet_ak4PFJetsCHS_emEnergyFraction'].content,
            hadfrac=df['akjet_ak4PFJetsCHS_hadronEnergyFraction'].content,
            chahadfrac=df['akjet_ak4PFJetsCHS_chaHadEnergyFraction'].content,
            deepcsv=df['hftagscore_DeepCSV_b'].content,
        )
        deepcsv_tight = np.bitwise_and(ak4jets.deepcsv, 1<<2)==(1<<2)
        ak4jets.add_attributes(deepcsvTight=deepcsv_tight,)
        ak4jets=ak4jets[ak4jets.jetid&(ak4jets.pt>30)&(np.abs(ak4jets.eta)<2.4)]

        leptonjets = JaggedCandidateArray.candidatesfromcounts(
            df['pfjet_p4'],
            px=df['pfjet_p4.fCoordinates.fX'].content,
            py=df['pfjet_p4.fCoordinates.fY'].content,
            pz=df['pfjet_p4.fCoordinates.fZ'].content,
            energy=df['pfjet_p4.fCoordinates.fT'].content,
            vx=df['pfjet_klmvtx.fCoordinates.fX'].content,
            vy=df['pfjet_klmvtx.fCoordinates.fY'].content,
            vz=df['pfjet_klmvtx.fCoordinates.fZ'].content,
            mintkdist=df['pfjet_pfcands_minTwoTkDist'].content,
        )
        leptonjets.add_attributes(vxy=np.hypot(leptonjets.vx, leptonjets.vy))
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
        leptonjets.add_attributes(displaced=((leptonjets.vxy>=5)|(np.isnan(leptonjets.vxy)&leptonjets.ismutype))) # non-vertex treated as displaced too
        ljdsamuSubset = fromNestNestIndexArray(df['dsamuon_isSubsetFilteredCosmic1Leg'], awkward.fromiter(df['pfjet_pfcand_dsamuonIdx']))
        leptonjets.add_attributes(nocosmic=(ljdsamuSubset.sum()==0))

        leptonjets = leptonjets[(leptonjets.isneutral)&(leptonjets.nocosmic)&(leptonjets.pt>30)&(leptonjets.mintkdist<50)]

        # mask_ = ak4jets.match(leptonjets, deltaRCut=0.4)
        # ak4jets = ak4jets[~mask_]

        ## __ twoleptonjets__
        twoleptonjets = (leptonjets.counts>=2)&(leptonjets.ismutype.sum()>=1)
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

        ak4jets = ak4jets[ak4jets.pt>(lj0.pt.flatten())]

        output['njets'].fill(dataset=dataset, cnt=ak4jets.counts, weight=wgt, channel=channel_)
        if ak4jets.flatten().size !=0:
            ak4jets = ak4jets[(ak4jets.pt>30)&(np.abs(ak4jets.eta)<2.4)&(ak4jets.deepcsvTight)]
        output['ntightb'].fill(dataset=dataset, cnt=ak4jets.counts, weight=wgt, channel=channel_)

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
                                  processor_instance=HadronicJetProcessor(data_type='sig-2mu2e'),
                                  executor=processor.futures_executor,
                                  executor_args=dict(workers=12, flatten=False),
                                  chunksize=500000,
                                 )
    out_sig4mu = processor.run_uproot_job(sigDS_4mu,
                                  treename='ffNtuplizer/ffNtuple',
                                  processor_instance=HadronicJetProcessor(data_type='sig-4mu'),
                                  executor=processor.futures_executor,
                                  executor_args=dict(workers=12, flatten=False),
                                  chunksize=500000,
                                 )

    out_bkg = processor.run_uproot_job(bkgDS,
                                  treename='ffNtuplizer/ffNtuple',
                                  processor_instance=HadronicJetProcessor(data_type='bkg'),
                                  executor=processor.futures_executor,
                                  executor_args=dict(workers=12, flatten=False),
                                  chunksize=500000,
                                 )

    import re
    longdecay = re.compile('^.*_lxy-300$')
    sampleSig = re.compile('mXX-150_mA-0p25_lxy-300|mXX-500_mA-1p2_lxy-300|mXX-800_mA-5_lxy-300')

    # N(AK4PFCHS)
    fig, ax = make_mc_plot(out_bkg['njets'].integrate('channel', slice(1,2)),
                           sigh=out_sig2mu2e['njets'][sampleSig].integrate('channel', slice(1,2)),
                           title='[$2\mu 2e$] Hadronic jets multiplicity')
    ax.vlines([1,], 0, 1, linestyles='dashed', colors='tab:gray', transform=ax.get_xaxis_transform())

    # fig, axes = plt.subplots(1,2,figsize=(16,6))
    # fig.subplots_adjust(wspace=0.15)
    # bkghist = out_bkg['njets'].integrate('channel', slice(1,2))
    # hist.plot1d(bkghist, overlay='cat', ax=axes[0], stack=True, overflow='over',
    #             line_opts=None, fill_opts=fill_opts, error_opts=error_opts,)
    # sighist = out_sig2mu2e['njets'][longdecay].sum('dataset').integrate('channel', slice(1,2))
    # hist.plot1d(sighist, ax=axes[1], overflow='over', density=True)

    # axes[0].set_title('[$2\mu 2e$|BackgroundMC] AK4PFCHS\n(jetId, $p_T$>20, |$\eta$|<2.5) multiplicity', x=0.0, ha="left")
    # axes[1].set_title('[$2\mu 2e$|SignalMC] AK4PFCHS\n(jetId, $p_T$>20, |$\eta$|<2.5) multiplicity', x=0.0, ha="left")
    # axes[0].text(1,1,'59.74/fb (13TeV)', ha='right', va='bottom', transform=axes[0].transAxes)
    # axes[0].set_yscale('log')
    # for ax in axes:
    #     ax.autoscale(axis='both', tight=True)
    #     ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
    #     ax.vlines([4,], 0, 1, linestyles='dashed', colors='tab:gray', transform=ax.get_xaxis_transform())

    # axes[0].set_ylabel(ax.get_ylabel(), y=1.0, ha="right")
    # axes[1].set_ylabel('Norm. '+ax.get_ylabel(), y=1.0, ha="right")

    # hdlSig = tuple(axes[1].get_legend_handles_labels()[0])
    # fracSig = sighist.integrate('cnt', slice(4,10)).values()[()]/sighist.sum('cnt').values()[()]
    # axes[1].legend([hdlSig,], [f'N$\geqslant$4: {fracSig*100:.2f}%',])

    print("## NAK4PFCHS-2mu2e")
    sigh = out_sig2mu2e['njets'].integrate('dataset', 'mXX-500_mA-1p2_lxy-300').integrate('channel', slice(1,2))
    bkgh = out_bkg['njets'].sum('cat').integrate('channel', slice(1,2))
    print('sig:', sigh.values()[()])
    print('bkg:', bkgh.values()[()])
    print('sig/bkg > 0:', sigh.values()[()][1:].sum()/bkgh.values()[()][1:].sum())

    fig.savefig(join(outdir, 'NAK4PFCHS-2mu2e.png'))
    fig.savefig(join(outdir, 'NAK4PFCHS-2mu2e.pdf'))
    plt.close(fig)

    fig, ax = make_mc_plot(out_bkg['njets'].integrate('channel', slice(2,3)),
                           sigh=out_sig4mu['njets'][sampleSig].integrate('channel', slice(2,3)),
                           title='[$4\mu$] Hadronic jets multiplicity')

    fig.savefig(join(outdir, 'NAK4PFCHS-4mu.png'))
    fig.savefig(join(outdir, 'NAK4PFCHS-4mu.pdf'))
    plt.close(fig)


    # N(tight b)
    # fig, axes = plt.subplots(1,2,figsize=(16,6))
    # fig.subplots_adjust(wspace=0.15)
    # bkghist = out_bkg['ntightb'].integrate('channel', slice(1,2))
    # hist.plot1d(bkghist, overlay='cat', ax=axes[0], stack=True, overflow='over',
    #             line_opts=None, fill_opts=fill_opts, error_opts=error_opts,)
    # sighist = out_sig2mu2e['ntightb'][longdecay].sum('dataset').integrate('channel', slice(1,2))
    # hist.plot1d(sighist, ax=axes[1], overflow='over', density=True)

    # axes[0].set_title('[$2\mu 2e$|BackgroundMC] DeepCSV tight AK4PFCHS\n(jetId, $p_T$>30, |$\eta$|<2.5) multiplicity', x=0.0, ha="left")
    # axes[1].set_title('[$2\mu 2e$|SignalMC] DeepCSV tight AK4PFCHS\n(jetId, $p_T$>30, |$\eta$|<2.5) multiplicity', x=0.0, ha="left")
    # axes[0].text(1,1,'59.74/fb (13TeV)', ha='right', va='bottom', transform=axes[0].transAxes)
    # axes[0].set_yscale('log')
    # for ax in axes:
    #     ax.autoscale(axis='both', tight=True)
    #     ax.set_xlim([0, 5])
    #     ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
    #     ax.vlines([1,], 0, 1, linestyles='dashed', colors='tab:gray', transform=ax.get_xaxis_transform())

    # axes[0].set_ylabel(ax.get_ylabel(), y=1.0, ha="right")
    # axes[1].set_ylabel('Norm. '+ax.get_ylabel(), y=1.0, ha="right")

    # hdlSig = tuple(axes[1].get_legend_handles_labels()[0])
    # fracSig = sighist.integrate('cnt', slice(1,10)).values()[()]/sighist.sum('cnt').values()[()]
    # axes[1].legend([hdlSig,], [f'N$\geqslant$1: {fracSig*100:.2f}%',])

    # fig.savefig(join(outdir, 'NTightB-2mu2e.png'))
    # fig.savefig(join(outdir, 'NTightB-2mu2e.pdf'))
    # plt.close(fig)


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
        ## copy to AN image folder
        an_dir = '/uscms_data/d3/wsi/lpcdm/AN-18-125/image'
        if isdir(an_dir):
            cmd = f'cp {outdir}/*.pdf {an_dir}'
            print(f'--> copy to AN folder: {an_dir}')
            os.system(cmd)
