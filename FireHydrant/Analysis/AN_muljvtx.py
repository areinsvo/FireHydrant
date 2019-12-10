#!/usr/bin/env python
"""For AN
muon-type leptonjet vertexing
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

np.seterr(divide='ignore', invalid='ignore', over='ignore')
plt.rcParams["savefig.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"


parser = argparse.ArgumentParser(description="[AN] leptonjet pair delta phi")
parser.add_argument("--sync", action='store_true', help="issue rsync command to sync plots folder to lxplus web server")
args = parser.parse_args()

sdml = SigDatasetMapLoader()
sigDS, sigSCALE = sdml.fetch('all')


class MuLJVtxProcessor(processor.ProcessorABC):
    def __init__(self):
        dataset_axis = hist.Cat('dataset', 'dataset')
        reco_axis = hist.Cat('reco', 'reco type')
        vxy_axis = hist.Bin('vxy', 'vxy [cm]', [0, 5, 70, 700])
        lxy_axis = hist.Bin('lxy', 'lxy [cm]', 100, 0, 700)
        bool_axis = hist.Bin('boolean', 'true/false', 2, 0, 2)
        reso_axis = hist.Bin('reso', '(vxy(reco)-vxy(gen)) [cm]', 100, -25, 25)
        error_axis = hist.Bin('error', '$\sigma_{lxy}$', 100, 0, 100)
        cos_axis = hist.Bin('cos', r'$|cos(\theta)|$', 50, -1, 1)
        sig_axis = hist.Bin('sig', 'lxy/$\sigma_{lxy}$', 50, 0, 50)
        self._accumulator = processor.dict_accumulator({
            'vertexgood': hist.Hist('Frequency', dataset_axis, vxy_axis, reco_axis),
            'vxyreso': hist.Hist('Norm. Frequency', dataset_axis, reso_axis, vxy_axis),
            'lxy': hist.Hist('Norm. Frequency', dataset_axis, lxy_axis, vxy_axis),
            'lxyerr': hist.Hist('Norm. Frequency', dataset_axis, error_axis, vxy_axis),
            'lxysig': hist.Hist('Norm. Frequency', dataset_axis, sig_axis, vxy_axis),
            'costheta': hist.Hist('Norm. Frequency', dataset_axis, cos_axis, vxy_axis),
        })


    @property
    def accumulator(self):
        return self._accumulator

    def process(self, df):
        output = self.accumulator.identity()
        dataset = df['dataset']

        genparticles = JaggedCandidateArray.candidatesfromcounts(
            df['gen_p4'],
            px=df['gen_p4.fCoordinates.fX'].content,
            py=df['gen_p4.fCoordinates.fY'].content,
            pz=df['gen_p4.fCoordinates.fZ'].content,
            energy=df['gen_p4.fCoordinates.fT'].content,
            pid=df['gen_pid'].content,
            daupid=df['gen_daupid'].content,
            dauvx=df['gen_dauvtx.fCoordinates.fX'].content,
            dauvy=df['gen_dauvtx.fCoordinates.fY'].content,
            dauvz=df['gen_dauvtx.fCoordinates.fZ'].content,
        )
        genparticles.add_attributes(daurho=np.hypot(genparticles.dauvx, genparticles.dauvy))
        is_dpToMu = (genparticles.pid==32)&(genparticles.daupid==13)
        dpMu = genparticles[is_dpToMu&(genparticles.daurho<700)&(genparticles.pt>20)&(np.abs(genparticles.eta)<2.4)]

        # at least 1 good dpMu
        nDpMuGe1 = dpMu.counts>=1
        dpMu = dpMu[nDpMuGe1]

        leptonjets = JaggedCandidateArray.candidatesfromcounts(
            df['pfjet_p4'],
            px=df['pfjet_p4.fCoordinates.fX'].content,
            py=df['pfjet_p4.fCoordinates.fY'].content,
            pz=df['pfjet_p4.fCoordinates.fZ'].content,
            energy=df['pfjet_p4.fCoordinates.fT'].content,
            vx=df['pfjet_klmvtx.fCoordinates.fX'].content,
            vy=df['pfjet_klmvtx.fCoordinates.fY'].content,
            vz=df['pfjet_klmvtx.fCoordinates.fZ'].content,
            lxy=df['pfjet_klmvtx_lxy'].content,
            lxysig=df['pfjet_klmvtx_lxySig'].content,
            costheta=df['pfjet_klmvtx_cosThetaXy'].content
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
        leptonjets = leptonjets[leptonjets.isneutral]

        leptonjets = leptonjets[nDpMuGe1]

        ljmu = leptonjets[leptonjets.ismutype]
        matchidx = dpMu.argmatch(ljmu, deltaRCut=0.4)
        dpMu_ = dpMu[ljmu.counts!=0]
        matchmask = matchidx[ljmu.counts!=0]!=-1
        output['vertexgood'].fill(dataset=dataset, vxy=dpMu_[matchmask].daurho.flatten(), reco='inclusive')

        ljmu = leptonjets[leptonjets.ismutype&(~np.isnan(leptonjets.vxy))] # vertexed good
        matchidx = dpMu.argmatch(ljmu, deltaRCut=0.4)
        dpMu_ = dpMu[ljmu.counts!=0]
        matchmask = matchidx[ljmu.counts!=0]!=-1
        output['vertexgood'].fill(dataset=dataset, vxy=dpMu_[matchmask].daurho.flatten(), reco='vertexed')

        genval = dpMu_[matchmask].daurho.flatten()
        recoObj = ljmu[matchidx][ljmu.counts!=0][matchmask]
        recoval = recoObj.vxy.flatten()
        output['vxyreso'].fill(dataset=dataset, reso=(recoval-genval), vxy=genval)
        output['lxy'].fill(dataset=dataset, lxy=np.abs(recoObj.lxy).flatten(), vxy=genval)
        output['lxysig'].fill(dataset=dataset, sig=recoObj.lxysig.flatten(), vxy=genval)
        output['lxyerr'].fill(dataset=dataset, error=(recoObj.lxy/recoObj.lxysig).flatten(), vxy=genval)
        output['costheta'].fill(dataset=dataset, cos=-recoObj.costheta.flatten(), vxy=genval)

        return output

    def postprocess(self, accumulator):
        return accumulator


if __name__ == "__main__":
    import os
    from os.path import join, isdir, splitext

    reldir = splitext(__file__)[0].replace('_', '/')
    outdir = join(os.getenv('FH_BASE'), "Imgs", reldir)
    if not isdir(outdir): os.makedirs(outdir)

    import re
    longdecay = re.compile('^.*_lxy-300$')

    output = processor.run_uproot_job(sigDS,
                                    treename='ffNtuplizer/ffNtuple',
                                    processor_instance=MuLJVtxProcessor(),
                                    executor=processor.futures_executor,
                                    executor_args=dict(workers=12, flatten=False),
                                    chunksize=500000,
                                    )

    ## vertex efficiency
    fig, ax = plt.subplots(figsize=(8,6))
    hist.plotratio(num=output['vertexgood'][longdecay].sum('dataset').integrate('reco', 'vertexed'),
                   denom=output['vertexgood'][longdecay].sum('dataset').integrate('reco', 'inclusive'),
                   error_opts={'marker': 'o', 'xerr': [5/2, 65/2, 630/2]},
                   ax=ax,)
    ax.set_xscale('symlog')
    ax.autoscale(axis='both', tight=True)
    ax.set_ylim([0.9, 1.02])
    ax.set_yticks(np.arange(0.9, 1.02, 0.02))
    ax.set_xticks([0,1,2,3,4,5,10,20,30,50,70,100,300,500,700])
    ax.set_xticklabels(['0', '$10^0$', '2', '3', '4','5', '$10^1$', '20', '30', '50', '70', '$10^2$', '300', '500', '700'])
    ax.grid(axis='y', ls='--')
    ax.set_title('[signalMC|lxy300cm] muon-type leptonjet\nvertexing efficiency vs. darkphoton(${}$) lxy'.format(r'\rightarrow\mu^+\mu^-'), x=0.0, ha="left")
    ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
    ax.set_ylabel('Efficiency', y=1.0, ha="right")

    fig.savefig(join(outdir, 'mulj-vtxeffi-vs-lxy.png'))
    fig.savefig(join(outdir, 'mulj-vtxeffi-vs-lxy.pdf'))
    plt.close(fig)

    ## vertex resolution
    fig, ax = plt.subplots(figsize=(8,6))
    hist.plot1d(output['vxyreso'][longdecay].sum('dataset'), overlay='vxy', ax=ax, density=True, overflow='all')
    ax.set_title(r'[signalMC|lxy300cm] muon-type leptonjet vertexing resolution', x=0.0, ha="left")
    ax.set_xticks(np.arange(-24, 24.1, 2))
    ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
    ax.set_ylabel(ax.get_ylabel(), y=1.0, ha="right")

    fig.savefig(join(outdir, 'mulj-vtxreso.png'))
    fig.savefig(join(outdir, 'mulj-vtxreso.pdf'))
    plt.close(fig)

    ## vertex significance
    fig, ax = plt.subplots(figsize=(8,6))
    hist.plot1d(output['lxysig'][longdecay].sum('dataset'), overlay='vxy', ax=ax, density=True, overflow='over')
    ax.set_title(r'[signalMC|lxy300cm] muon-type leptonjet vertexing significance', x=0.0, ha="left")
    ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
    ax.set_ylabel(ax.get_ylabel(), y=1.0, ha="right")

    fig.savefig(join(outdir, 'mulj-vtxsignificance.png'))
    fig.savefig(join(outdir, 'mulj-vtxsignificance.pdf'))
    plt.close(fig)

    ## vertex error
    fig, ax = plt.subplots(figsize=(8,6))
    hist.plot1d(output['lxyerr'][longdecay].sum('dataset'), overlay='vxy', ax=ax, density=True, overflow='over')
    ax.set_title(r'[signalMC|lxy300cm] muon-type leptonjet vertexing error', x=0.0, ha="left")
    ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
    ax.set_ylabel(ax.get_ylabel(), y=1.0, ha="right")

    fig.savefig(join(outdir, 'mulj-vtxerror.png'))
    fig.savefig(join(outdir, 'mulj-vtxerror.pdf'))
    plt.close(fig)

    ## vertex lxy
    fig, ax = plt.subplots(figsize=(8,6))
    hist.plot1d(output['lxy'][longdecay].sum('dataset'), overlay='vxy', ax=ax, density=True, overflow='over')
    ax.set_title(r'[signalMC|lxy300cm] muon-type leptonjet lxy', x=0.0, ha="left")
    ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
    ax.set_ylabel(ax.get_ylabel(), y=1.0, ha="right")

    fig.savefig(join(outdir, 'mulj-vtxlxy.png'))
    fig.savefig(join(outdir, 'mulj-vtxlxy.pdf'))
    plt.close(fig)

    ## cosTheta
    fig, ax = plt.subplots(figsize=(8,6))
    hist.plot1d(output['costheta'][longdecay].sum('dataset'), overlay='vxy', ax=ax, density=True, overflow='all')
    ax.set_title(r'[signalMC|lxy300cm] muon-type leptonjet |cos($\theta$)|', x=0.0, ha="left")
    ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
    ax.set_ylabel(ax.get_ylabel(), y=1.0, ha="right")

    fig.savefig(join(outdir, 'mulj-costheta.png'))
    fig.savefig(join(outdir, 'mulj-costheta.pdf'))
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
        ## copy to AN image folder
        an_dir = '/uscms_data/d3/wsi/lpcdm/AN-18-125/image'
        if isdir(an_dir):
            cmd = f'cp {outdir}/*.pdf {an_dir}'
            print(f'--> copy to AN folder: {an_dir}')
            os.system(cmd)
