#!/usr/bin/env python
"""deltaPhi in ABCD
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

parser = argparse.ArgumentParser(description="leptonjet deltaPhi in ABCD")
parser.add_argument("--sync", action='store_true', help="issue rsync command to sync plots folder to lxplus web server")
args = parser.parse_args()


dml = DatasetMapLoader()
dataDS, dataMAP = dml.fetch('data')
bkgDS, bkgMAP, bkgSCALE = dml.fetch('bkg')

sdml = SigDatasetMapLoader()
sigDS_2mu2e, sigSCALE_2mu2e = sdml.fetch('2mu2e')
sigDS_4mu, sigSCALE_4mu = sdml.fetch('4mu')



class LjDphiABCDProcessor(processor.ProcessorABC):
    def __init__(self, data_type='bkg'):
        self.data_type = data_type

        dataset_axis = hist.Cat('dataset', 'dataset')
        dphi_axis = hist.Bin('dphi', '$\Delta\phi$', 8, 0, np.pi)
        categ_axis = hist.Bin('categ', 'categ', 4, 1, 5)
        self._accumulator = processor.dict_accumulator({
            'chan-4mu': hist.Hist(  'Counts', dataset_axis, dphi_axis, categ_axis),
            'chan-2mu2e': hist.Hist('Counts', dataset_axis, dphi_axis, categ_axis),
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

        weight = wgts.weight()
        ########################

        ak4jets = JaggedCandidateArray.candidatesfromcounts(
            df['akjet_ak4PFJetsCHS_p4'],
            px=df['akjet_ak4PFJetsCHS_p4.fCoordinates.fX'].content,
            py=df['akjet_ak4PFJetsCHS_p4.fCoordinates.fY'].content,
            pz=df['akjet_ak4PFJetsCHS_p4.fCoordinates.fZ'].content,
            energy=df['akjet_ak4PFJetsCHS_p4.fCoordinates.fT'].content,
            jetid=df['akjet_ak4PFJetsCHS_jetid'].content,
        )
        ak4jets=ak4jets[ak4jets.jetid&(ak4jets.pt>30)&(np.abs(ak4jets.eta)<2.4)]

        leptonjets = JaggedCandidateArray.candidatesfromcounts(
            df['pfjet_p4'],
            px=df['pfjet_p4.fCoordinates.fX'].content,
            py=df['pfjet_p4.fCoordinates.fY'].content,
            pz=df['pfjet_p4.fCoordinates.fZ'].content,
            energy=df['pfjet_p4.fCoordinates.fT'].content,
            sumtkpt=df['pfjet_tkPtSum05'].content,
            pfiso=df['pfjet_pfIsolationNoPU05'].content,
            mintkdist=df['pfjet_pfcands_minTwoTkDist'].content,
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
        ljdsamuSubset = fromNestNestIndexArray(df['dsamuon_isSubsetFilteredCosmic1Leg'], awkward.fromiter(df['pfjet_pfcand_dsamuonIdx']))
        leptonjets.add_attributes(nocosmic=(ljdsamuSubset.sum()==0))
        leptonjets = leptonjets[(leptonjets.nocosmic)&(leptonjets.pt>30)&(leptonjets.mintkdist<50)]

        ## __ twoleptonjets__
        twoleptonjets = leptonjets.counts>=2
        dileptonjets = leptonjets[twoleptonjets]
        ak4jets = ak4jets[twoleptonjets]
        wgt = weight[twoleptonjets]

        if dileptonjets.size==0: return output
        lj0 = dileptonjets[dileptonjets.pt.argmax()]
        lj1 = dileptonjets[dileptonjets.pt.argsort()[:, 1:2]]

        ak4jets = ak4jets[ak4jets.pt>(lj0.pt.flatten())]
        ak4jetCounts = (ak4jets.counts>0).astype(int)
        minpfiso = ((lj0.pfiso>lj1.pfiso).astype(int)*lj1.pfiso + (lj0.pfiso<lj1.pfiso).astype(int)*lj0.pfiso).flatten()
        ljneutrality = ((lj0.isneutral&lj1.isneutral).astype(int)*1+(lj0.mucharged&lj1.mucharged).astype(int)*2).flatten()
        ljdphi = np.abs(lj0.p4.delta_phi(lj1.p4)).flatten()

        ## channel def ##
        #### 2mu2e
        singleMuljEvents = dileptonjets.ismutype.sum()==1
        muljInLeading2Events = (lj0.ismutype | lj1.ismutype).flatten()
        channel_2mu2e = singleMuljEvents&muljInLeading2Events

        minpfiso_ = minpfiso[channel_2mu2e]
        ak4jetCounts_ = ak4jetCounts[channel_2mu2e]
        wgt_ = wgt[channel_2mu2e]
        ljdphi_ = ljdphi[channel_2mu2e]
        categ_ = ((minpfiso_<0.12)&(ak4jetCounts_==0)).astype(int)*1 +\
                 ((minpfiso_<0.12)&(ak4jetCounts_==1)).astype(int)*2 +\
                 ((minpfiso_>=0.12)&(ak4jetCounts_==1)).astype(int)*3 +\
                 ((minpfiso_>=0.12)&(ak4jetCounts_==0)).astype(int)*4
        if self.data_type == 'data':
            dataMask = (categ_==1)&(ljdphi_>np.pi/2)
            ljdphi_ = ljdphi_[~dataMask]
            categ_ = categ_[~dataMask]
            wgt_ = wgt_[~dataMask]

        output['chan-2mu2e'].fill(dataset=dataset, dphi=ljdphi_, categ=categ_, weight=wgt_)

        #### 4mu
        doubleMuljEvents = dileptonjets.ismutype.sum()==2
        muljIsLeading2Events = (lj0.ismutype & lj1.ismutype).flatten()
        channel_4mu = doubleMuljEvents&muljIsLeading2Events

        minpfiso_ = minpfiso[channel_4mu]
        ljneutrality_ = ljneutrality[channel_4mu]
        wgt_ = wgt[channel_4mu]
        ljdphi_ = ljdphi[channel_4mu]
        categ_ = ((minpfiso_<0.12)&(ljneutrality_==1)).astype(int)*1 +\
                 ((minpfiso_<0.12)&(ljneutrality_==2)).astype(int)*2 +\
                 ((minpfiso_>=0.12)&(ljneutrality_==2)).astype(int)*3 +\
                 ((minpfiso_>=0.12)&(ljneutrality_==1)).astype(int)*4

        output['chan-4mu'].fill(dataset=dataset, dphi=ljdphi_, categ=categ_, weight=wgt_)

        ###########

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
    import re
    from os.path import join, isdir, splitext
    from FireHydrant.Analysis.PlottingOptions import *

    reldir = splitext(__file__)[0].replace('_', '/')
    outdir = join(os.getenv('FH_BASE'), "Imgs", reldir)
    if not isdir(outdir): os.makedirs(outdir)


    output_2mu2e = processor.run_uproot_job(sigDS_2mu2e,
                                    treename='ffNtuplizer/ffNtuple',
                                    processor_instance=LjDphiABCDProcessor(data_type='sig-2mu2e'),
                                    executor=processor.futures_executor,
                                    executor_args=dict(workers=12, flatten=False),
                                    chunksize=500000,
                                    )

    output_4mu = processor.run_uproot_job(sigDS_4mu,
                                    treename='ffNtuplizer/ffNtuple',
                                    processor_instance=LjDphiABCDProcessor(data_type='sig-4mu'),
                                    executor=processor.futures_executor,
                                    executor_args=dict(workers=12, flatten=False),
                                    chunksize=500000,
                                    )

    output_bkg = processor.run_uproot_job(bkgDS,
                                    treename='ffNtuplizer/ffNtuple',
                                    processor_instance=LjDphiABCDProcessor(data_type='bkg'),
                                    executor=processor.futures_executor,
                                    executor_args=dict(workers=12, flatten=False),
                                    chunksize=500000,
                                    )

    output_data = processor.run_uproot_job(dataDS,
                                    treename='ffNtuplizer/ffNtuple',
                                    processor_instance=LjDphiABCDProcessor(data_type='data'),
                                    executor=processor.futures_executor,
                                    executor_args=dict(workers=12, flatten=False),
                                    chunksize=500000,
                                    )

    sampleSig = re.compile('mXX-150_mA-0p25_lxy-300|mXX-500_mA-1p2_lxy-300|mXX-800_mA-5_lxy-300')


    for i, rg in enumerate(list('ABCD'), start=1):
        binslice = slice(i, i+1)
        fig, (ax, rax) = make_ratio_plot(output_bkg['chan-2mu2e'].integrate('categ', binslice),
                                         output_data['chan-2mu2e'].integrate('categ', binslice),
                                         sigh=output_2mu2e['chan-2mu2e'][sampleSig].integrate('categ', binslice),
                                         title=f'[2$\mu$2e, region {rg}] leptonJets pair $\Delta\phi$',
                                         overflow='none')
        ax.vlines([np.pi/2,], 0, 1, linestyles='dashed', colors='tab:gray', transform=ax.get_xaxis_transform())

        if i!=1:
            print(f'## 2mu2e categ {i}')
            sigh = output_2mu2e['chan-2mu2e'].integrate('dataset', 'mXX-500_mA-1p2_lxy-300').integrate('categ', binslice)
            bkgh = output_bkg['chan-2mu2e'].sum('cat').integrate('categ', binslice)
            print('sig/bkg:', sigh.values()[()]/bkgh.values()[()])


        fig.savefig(join(outdir, f'ljpairDphi-2mu2e_region{rg}.png'))
        fig.savefig(join(outdir, f'ljpairDphi-2mu2e_region{rg}.pdf'))
        plt.close(fig)

        fig, (ax, rax) = make_ratio_plot(output_bkg['chan-4mu'].integrate('categ', binslice),
                                         output_data['chan-4mu'].integrate('categ', binslice),
                                         sigh=output_4mu['chan-4mu'][sampleSig].integrate('categ', binslice),
                                         title=f'[4$\mu$, region {rg}] leptonJets pair $\Delta\phi$',
                                         overflow='none')
        ax.vlines([np.pi/2,], 0, 1, linestyles='dashed', colors='tab:gray', transform=ax.get_xaxis_transform())

        if i==4:
            print('## 4mu categ 4')
            sigh = output_4mu['chan-4mu'].integrate('dataset', 'mXX-500_mA-1p2_lxy-300').integrate('categ', binslice)
            bkgh = output_bkg['chan-4mu'].sum('cat').integrate('categ', binslice)
            print('sig/bkg:', sigh.values()[()]/bkgh.values()[()])

        fig.savefig(join(outdir, f'ljpairDphi-4mu_region{rg}.png'))
        fig.savefig(join(outdir, f'ljpairDphi-4mu_region{rg}.pdf'))
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
