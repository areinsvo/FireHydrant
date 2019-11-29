#!/usr/bin/env python
"""For AN
leptonjets efficiencies and resolutions
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


parser = argparse.ArgumentParser(description="[AN] leptonjet efficiencies/resolutions")
parser.add_argument("--sync", action='store_true', help="issue rsync command to sync plots folder to lxplus web server")
args = parser.parse_args()

# dml = DatasetMapLoader()
# bkgDS, bkgMAP, bkgSCALE = dml.fetch('bkg')
# dataDS, dataMAP = dml.fetch('data')

sdml = SigDatasetMapLoader()
sigDS_2mu2e, sigSCALE_2mu2e = sdml.fetch('2mu2e')
sigDS_4mu, sigSCALE_4mu = sdml.fetch('4mu')


class MuEffiResoProcessor(processor.ProcessorABC):
    def __init__(self):
        dataset_axis = hist.Cat('dataset', 'dataset')
        lxy_axis = hist.Bin('lxy', 'lxy [cm]', 100, 0, 700)
        reso_axis = hist.Bin('reso', '($p_T$(reco)-$p_T$(gen))/$p_T$(gen)', 100, -1, 2)
        reco_axis = hist.Cat('reco', 'reco type')
        self._accumulator = processor.dict_accumulator({
            'lxy': hist.Hist('Counts', dataset_axis, lxy_axis, reco_axis),
            'lxy-pf': hist.Hist('Counts', dataset_axis, lxy_axis, reco_axis),
            'lxy-dsa': hist.Hist('Counts', dataset_axis, lxy_axis, reco_axis),
            'reso': hist.Hist('Norm. Frequency/0.03', dataset_axis, reso_axis, reco_axis),
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
            vx=df['gen_vtx.fCoordinates.fX'].content,
            vy=df['gen_vtx.fCoordinates.fY'].content,
            vz=df['gen_vtx.fCoordinates.fZ'].content,
            charge=df['gen_charge'].content,
        )
        genparticles.add_attributes(rho=np.hypot(genparticles.vx, genparticles.vy))
        genmuons = genparticles[(np.abs(genparticles.pid)==13)&(genparticles.pt>10)&(np.abs(genparticles.eta)<2.4)&(genparticles.rho<700)]

        ## at least 2 good gen muons
        nmuGe2 = genmuons.counts>=2
        genmuons = genmuons[nmuGe2]

        ljsources = JaggedCandidateArray.candidatesfromcounts(
            df['ljsource_p4'],
            px=df['ljsource_p4.fCoordinates.fX'].content,
            py=df['ljsource_p4.fCoordinates.fY'].content,
            pz=df['ljsource_p4.fCoordinates.fZ'].content,
            energy=df['ljsource_p4.fCoordinates.fT'].content,
            pid=df['ljsource_type'].content,
            charge=df['ljsource_charge'].content,
        )
        muons = ljsources[(ljsources.pid==3)|(ljsources.pid==8)][nmuGe2]
        matchidx = genmuons.argmatch(muons, deltaRCut=0.3)
        genmuons_ = genmuons[muons.counts!=0]
        sameq = (muons[matchidx][muons.counts!=0].charge==genmuons_.charge)&(matchidx[muons.counts!=0]!=-1)

        output['lxy'].fill(dataset=dataset, lxy=genmuons_[sameq].rho.flatten(), reco='true')
        output['lxy'].fill(dataset=dataset, lxy=genmuons.rho.flatten(), reco='inclusive')

        genpt = genmuons_[sameq].pt.flatten()
        recopt = muons[matchidx][muons.counts!=0][sameq].pt.flatten()
        output['reso'].fill(dataset=dataset, reso=(recopt-genpt)/genpt, reco='(PFMu+DSAMu)')

        muons = ljsources[(ljsources.pid==3)][nmuGe2]
        matchidx = genmuons.argmatch(muons, deltaRCut=0.3)
        genmuons_ = genmuons[muons.counts!=0]
        sameq = (muons[matchidx][muons.counts!=0].charge==genmuons_.charge)&(matchidx[muons.counts!=0]!=-1)

        output['lxy-pf'].fill(dataset=dataset, lxy=genmuons_[sameq].rho.flatten(), reco='true')
        output['lxy-pf'].fill(dataset=dataset, lxy=genmuons.rho.flatten(), reco='inclusive')

        genpt = genmuons_[sameq].pt.flatten()
        recopt = muons[matchidx][muons.counts!=0][sameq].pt.flatten()
        output['reso'].fill(dataset=dataset, reso=(recopt-genpt)/genpt, reco='PFMu')

        muons = ljsources[(ljsources.pid==8)][nmuGe2]
        matchidx = genmuons.argmatch(muons, deltaRCut=0.3)
        genmuons_ = genmuons[muons.counts!=0]
        sameq = (muons[matchidx][muons.counts!=0].charge==genmuons_.charge)&(matchidx[muons.counts!=0]!=-1)

        output['lxy-dsa'].fill(dataset=dataset, lxy=genmuons_[sameq].rho.flatten(), reco='true')
        output['lxy-dsa'].fill(dataset=dataset, lxy=genmuons.rho.flatten(), reco='inclusive')

        genpt = genmuons_[sameq].pt.flatten()
        recopt = muons[matchidx][muons.counts!=0][sameq].pt.flatten()
        output['reso'].fill(dataset=dataset, reso=(recopt-genpt)/genpt, reco='DSAMu')


        return output

    def postprocess(self, accumulator):
        return accumulator




class MuLJEffiResoProcessor(processor.ProcessorABC):
    def __init__(self):
        dataset_axis = hist.Cat('dataset', 'dataset')
        lxy_axis = hist.Bin('lxy', 'lxy [cm]', 100, 0, 700)
        reso_axis = hist.Bin('reso', '($p_T$(reco)-$p_T$(gen))/$p_T$(gen)', 100, -1, 2)
        reco_axis = hist.Cat('reco', 'reco type')
        self._accumulator = processor.dict_accumulator({
            'lxy': hist.Hist('Counts', dataset_axis, lxy_axis, reco_axis),
            'lxy-pf': hist.Hist('Counts', dataset_axis, lxy_axis, reco_axis),
            'lxy-dsa': hist.Hist('Counts', dataset_axis, lxy_axis, reco_axis),
            'reso': hist.Hist('Norm. Frequency/0.03', dataset_axis, reso_axis, reco_axis),
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
        ljdaucharge = awkward.fromiter(df['pfjet_pfcand_charge']).sum()
        leptonjets.add_attributes(qsum=ljdaucharge)
        leptonjets.add_attributes(isneutral=(leptonjets.iseltype | (leptonjets.ismutype&(leptonjets.qsum==0))))
        leptonjets = leptonjets[leptonjets.isneutral]

        leptonjets = leptonjets[nDpMuGe1]

        ljmu = leptonjets[leptonjets.ismutype]
        matchidx = dpMu.argmatch(ljmu, deltaRCut=0.4)
        dpMu_ = dpMu[ljmu.counts!=0]
        matchmask = matchidx[ljmu.counts!=0]!=-1

        output['lxy'].fill(dataset=dataset, lxy=dpMu_[matchmask].daurho.flatten(), reco='true')
        output['lxy'].fill(dataset=dataset, lxy=dpMu.daurho.flatten(), reco='inclusive')

        genpt = dpMu_[matchmask].pt.flatten()
        recopt = ljmu[matchidx][ljmu.counts!=0][matchmask].pt.flatten()
        output['reso'].fill(dataset=dataset, reso=(recopt-genpt)/genpt, reco='(PFMu+DSAMu)-type leptonjet')

        ljmu = leptonjets[leptonjets.label==2]
        matchidx = dpMu.argmatch(ljmu, deltaRCut=0.4)
        dpMu_ = dpMu[ljmu.counts!=0]
        matchmask = matchidx[ljmu.counts!=0]!=-1

        genpt = dpMu_[matchmask].pt.flatten()
        recopt = ljmu[matchidx][ljmu.counts!=0][matchmask].pt.flatten()
        output['reso'].fill(dataset=dataset, reso=(recopt-genpt)/genpt, reco='PFMu-type leptonjet')

        output['lxy-pf'].fill(dataset=dataset, lxy=dpMu_[matchmask].daurho.flatten(), reco='true')
        output['lxy-pf'].fill(dataset=dataset, lxy=dpMu.daurho.flatten(), reco='inclusive')

        ljmu = leptonjets[leptonjets.label==3]
        matchidx = dpMu.argmatch(ljmu, deltaRCut=0.4)
        dpMu_ = dpMu[ljmu.counts!=0]
        matchmask = matchidx[ljmu.counts!=0]!=-1

        genpt = dpMu_[matchmask].pt.flatten()
        recopt = ljmu[matchidx][ljmu.counts!=0][matchmask].pt.flatten()
        output['reso'].fill(dataset=dataset, reso=(recopt-genpt)/genpt, reco='DSAMu-type leptonjet')

        output['lxy-dsa'].fill(dataset=dataset, lxy=dpMu_[matchmask].daurho.flatten(), reco='true')
        output['lxy-dsa'].fill(dataset=dataset, lxy=dpMu.daurho.flatten(), reco='inclusive')

        return output

    def postprocess(self, accumulator):
        return accumulator



class EGMEffiProcessor(processor.ProcessorABC):
    def __init__(self):
        dataset_axis = hist.Cat('dataset', 'dataset')
        lxy_axis = hist.Bin('lxy', 'lxy [cm]', 100, 0, 250)
        reco_axis = hist.Cat('reco', 'reco type')
        self._accumulator = processor.dict_accumulator({
            'lxy': hist.Hist('Counts', dataset_axis, lxy_axis, reco_axis),
            'lxy-el': hist.Hist('Counts', dataset_axis, lxy_axis, reco_axis),
            'lxy-pho': hist.Hist('Counts', dataset_axis, lxy_axis, reco_axis),
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
            vx=df['gen_vtx.fCoordinates.fX'].content,
            vy=df['gen_vtx.fCoordinates.fY'].content,
            vz=df['gen_vtx.fCoordinates.fZ'].content,
            charge=df['gen_charge'].content,
        )
        genparticles.add_attributes(rho=np.hypot(genparticles.vx, genparticles.vy))
        genel = genparticles[(np.abs(genparticles.pid)==11)&(genparticles.pt>10)&(np.abs(genparticles.eta)<2.4)&(genparticles.rho<250)]

        ## at least 2 good gen electrons
        nelGe2 = genel.counts>=2
        genel = genel[nelGe2]

        ljsources = JaggedCandidateArray.candidatesfromcounts(
            df['ljsource_p4'],
            px=df['ljsource_p4.fCoordinates.fX'].content,
            py=df['ljsource_p4.fCoordinates.fY'].content,
            pz=df['ljsource_p4.fCoordinates.fZ'].content,
            energy=df['ljsource_p4.fCoordinates.fT'].content,
            pid=df['ljsource_type'].content,
            charge=df['ljsource_charge'].content,
        )
        egms = ljsources[(ljsources.pid==2)|(ljsources.pid==4)][nelGe2]
        matchidx = genel.argmatch(egms, deltaRCut=0.3)
        genel_ = genel[egms.counts!=0]
        matchmask = matchidx[egms.counts!=0]!=-1

        output['lxy'].fill(dataset=dataset, lxy=genel_[matchmask].rho.flatten(), reco='true')
        output['lxy'].fill(dataset=dataset, lxy=genel.rho.flatten(), reco='inclusive')

        egms = ljsources[(ljsources.pid==2)][nelGe2]
        matchidx = genel.argmatch(egms, deltaRCut=0.3)
        genel_ = genel[egms.counts!=0]
        matchmask = matchidx[egms.counts!=0]!=-1

        output['lxy-el'].fill(dataset=dataset, lxy=genel_[matchmask].rho.flatten(), reco='true')
        output['lxy-el'].fill(dataset=dataset, lxy=genel.rho.flatten(), reco='inclusive')

        egms = ljsources[(ljsources.pid==4)][nelGe2]
        matchidx = genel.argmatch(egms, deltaRCut=0.3)
        genel_ = genel[egms.counts!=0]
        matchmask = matchidx[egms.counts!=0]!=-1

        output['lxy-pho'].fill(dataset=dataset, lxy=genel_[matchmask].rho.flatten(), reco='true')
        output['lxy-pho'].fill(dataset=dataset, lxy=genel.rho.flatten(), reco='inclusive')

        return output

    def postprocess(self, accumulator):
        return accumulator



class EGMLJEffiResoProcessor(processor.ProcessorABC):
    def __init__(self):
        dataset_axis = hist.Cat('dataset', 'dataset')
        lxy_axis = hist.Bin('lxy', 'lxy [cm]', 100, 0, 250)
        reso_axis = hist.Bin('reso', '($p_T$(reco)-$p_T$(gen))/$p_T$(gen)', 100, -0.5, 0.5)
        reco_axis = hist.Cat('reco', 'reco type')
        self._accumulator = processor.dict_accumulator({
            'lxy': hist.Hist('Counts', dataset_axis, lxy_axis, reco_axis),
            'lxy-el': hist.Hist('Counts', dataset_axis, lxy_axis, reco_axis),
            'lxy-pho': hist.Hist('Counts', dataset_axis, lxy_axis, reco_axis),
            'reso': hist.Hist('Norm. Frequency/0.01', dataset_axis, reso_axis, reco_axis),
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
        is_dpToEl = (genparticles.pid==32)&(genparticles.daupid==11)
        dpEl = genparticles[is_dpToEl&(genparticles.daurho<250)&(genparticles.pt>20)&(np.abs(genparticles.eta)<2.4)]

        # at least 1 good dpEl
        nDpElGe1 = dpEl.counts>=1
        dpEl = dpEl[nDpElGe1]

        leptonjets = JaggedCandidateArray.candidatesfromcounts(
            df['pfjet_p4'],
            px=df['pfjet_p4.fCoordinates.fX'].content,
            py=df['pfjet_p4.fCoordinates.fY'].content,
            pz=df['pfjet_p4.fCoordinates.fZ'].content,
            energy=df['pfjet_p4.fCoordinates.fT'].content,
        )
        ljdautype = awkward.fromiter(df['pfjet_pfcand_type'])
        nel = (ljdautype==2).sum()
        npfmu = (ljdautype==3).sum()
        ndsa = (ljdautype==8).sum()
        isegammajet = (npfmu==0)&(ndsa==0)
        iseljet = (isegammajet)&(nel!=0)
        ispfmujet = (npfmu>=2)&(ndsa==0)
        isdsajet = ndsa>0
        label = isegammajet.astype(int)*1+ispfmujet.astype(int)*2+isdsajet.astype(int)*3+iseljet.astype(int)*4
        leptonjets.add_attributes(label=label)
        nmu = ((ljdautype==3)|(ljdautype==8)).sum()
        leptonjets.add_attributes(ismutype=(nmu>=2), iseltype=(nmu==0))
        ljdaucharge = awkward.fromiter(df['pfjet_pfcand_charge']).sum()
        leptonjets.add_attributes(qsum=ljdaucharge)
        leptonjets.add_attributes(isneutral=(leptonjets.iseltype | (leptonjets.ismutype&(leptonjets.qsum==0))))
        leptonjets = leptonjets[leptonjets.isneutral]

        leptonjets = leptonjets[nDpElGe1]

        ljegm = leptonjets[leptonjets.iseltype]
        matchidx = dpEl.argmatch(ljegm, deltaRCut=0.4)
        dpEl_ = dpEl[ljegm.counts!=0]
        matchmask = matchidx[ljegm.counts!=0]!=-1

        output['lxy'].fill(dataset=dataset, lxy=dpEl_[matchmask].daurho.flatten(), reco='true')
        output['lxy'].fill(dataset=dataset, lxy=dpEl.daurho.flatten(), reco='inclusive')

        genpt = dpEl_[matchmask].pt.flatten()
        recopt = ljegm[matchidx][ljegm.counts!=0][matchmask].pt.flatten()
        output['reso'].fill(dataset=dataset, reso=(recopt-genpt)/genpt, reco='EGM-type leptonjet')

        ljegm = leptonjets[leptonjets.label==5]
        matchidx = dpEl.argmatch(ljegm, deltaRCut=0.4)
        dpEl_ = dpEl[ljegm.counts!=0]
        matchmask = matchidx[ljegm.counts!=0]!=-1

        genpt = dpEl_[matchmask].pt.flatten()
        recopt = ljegm[matchidx][ljegm.counts!=0][matchmask].pt.flatten()
        output['reso'].fill(dataset=dataset, reso=(recopt-genpt)/genpt, reco='Electron-type leptonjet')

        output['lxy-el'].fill(dataset=dataset, lxy=dpEl_[matchmask].daurho.flatten(), reco='true')
        output['lxy-el'].fill(dataset=dataset, lxy=dpEl.daurho.flatten(), reco='inclusive')

        ljegm = leptonjets[leptonjets.label==1]
        matchidx = dpEl.argmatch(ljegm, deltaRCut=0.4)
        dpEl_ = dpEl[ljegm.counts!=0]
        matchmask = matchidx[ljegm.counts!=0]!=-1

        genpt = dpEl_[matchmask].pt.flatten()
        recopt = ljegm[matchidx][ljegm.counts!=0][matchmask].pt.flatten()
        output['reso'].fill(dataset=dataset, reso=(recopt-genpt)/genpt, reco='Photon-type leptonjet')

        output['lxy-pho'].fill(dataset=dataset, lxy=dpEl_[matchmask].daurho.flatten(), reco='true')
        output['lxy-pho'].fill(dataset=dataset, lxy=dpEl.daurho.flatten(), reco='inclusive')

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

    # ----------------------------------------------------------
    ## mu cand efficiency, resolution

    output = processor.run_uproot_job(sigDS_2mu2e,
                                    treename='ffNtuplizer/ffNtuple',
                                    processor_instance=MuEffiResoProcessor(),
                                    executor=processor.futures_executor,
                                    executor_args=dict(workers=12, flatten=False),
                                    chunksize=500000,
                                    )

    fig, ax = plt.subplots(figsize=(8,6))
    hist.plotratio(num=output['lxy'][longdecay].sum('dataset').integrate('reco', 'true'),
                   denom=output['lxy'][longdecay].sum('dataset').integrate('reco', 'inclusive'),
                   overflow='over',
                   error_opts={'marker': 'o',},
                   ax=ax,
                   label='PFMu+DSAMu')
    hist.plotratio(num=output['lxy-pf'][longdecay].sum('dataset').integrate('reco', 'true'),
                   denom=output['lxy-pf'][longdecay].sum('dataset').integrate('reco', 'inclusive'),
                   overflow='over',
                   error_opts={'marker': 'o', 'color': 'tab:red', 'fillstyle': 'none',},
                   ax=ax,
                   clear=False,
                   label='PFMu')
    hist.plotratio(num=output['lxy-dsa'][longdecay].sum('dataset').integrate('reco', 'true'),
                   denom=output['lxy-dsa'][longdecay].sum('dataset').integrate('reco', 'inclusive'),
                   overflow='over',
                   error_opts={'marker': 'o', 'color': 'tab:green', 'fillstyle': 'none',},
                   ax=ax,
                   clear=False,
                   label='DSAMu')
    ax.set_ylim([0, 1.05])
    ax.set_yticks(np.arange(0, 1.05, 0.1))
    ax.set_xticks(np.arange(0, 701, 50))
    ax.grid(axis='y', ls='--')
    ax.legend()
    ax.set_title('[signalMC|2mu2e] leptonjet source - muon candidates (PFMuon+DSAMu) \nreconstruction efficiency vs. gen muon lxy', x=0.0, ha="left")
    ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
    ax.set_ylabel('Efficiency/7', y=1.0, ha="right")
    ax.text( 0.6, 0.6, '$\geqslant$2 gen muons with\n$p_T>10GeV, |\eta|<2.4, vxy<700cm$\n$\Delta R$(gen,reco)<0.3, same charge', transform=ax.transAxes)

    fig.savefig(join(outdir, 'mucand-effi-vs-lxy_2mu2e.png'))
    fig.savefig(join(outdir, 'mucand-effi-vs-lxy_2mu2e.pdf'))
    plt.close(fig)


    fig, ax = plt.subplots(figsize=(8,6))
    hist.plot1d(output['reso'][longdecay].sum('dataset'), overlay='reco', ax=ax, overflow='all', density=True)
    ax.set_title('[signalMC|2mu2e] leptonjet source - muon candidates (PFMuon+DSAMu)\n$p_T$ resolution', x=0, ha='left')
    ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
    ax.set_ylabel(ax.get_ylabel(), y=1.0, ha="right")
    ax.set_xticks(np.arange(-1, 2.01, 0.2))

    fig.savefig(join(outdir, 'mucand-ptreso_2mu2e.png'))
    fig.savefig(join(outdir, 'mucand-ptreso_2mu2e.pdf'))
    plt.close(fig)


    # ----------------------------------------------------------
    ## mu-type leptonjet efficiency, resolution

    output = processor.run_uproot_job(sigDS_2mu2e,
                                    treename='ffNtuplizer/ffNtuple',
                                    processor_instance=MuLJEffiResoProcessor(),
                                    executor=processor.futures_executor,
                                    executor_args=dict(workers=12, flatten=False),
                                    chunksize=500000,
                                    )

    fig, ax = plt.subplots(figsize=(8,6))
    hist.plotratio(num=output['lxy'][longdecay].sum('dataset').integrate('reco', 'true'),
                   denom=output['lxy'][longdecay].sum('dataset').integrate('reco', 'inclusive'),
                   overflow='over',
                   error_opts={'marker': 'o',},
                   ax=ax,
                   label='(PFMu+DSAMu)-type leptonjet')
    hist.plotratio(num=output['lxy-pf'][longdecay].sum('dataset').integrate('reco', 'true'),
                   denom=output['lxy-pf'][longdecay].sum('dataset').integrate('reco', 'inclusive'),
                   overflow='over',
                   error_opts={'marker': 'o', 'color': 'tab:red', 'fillstyle': 'none',},
                   ax=ax,
                   clear=False,
                   label='PFMu-type leptonjet')
    hist.plotratio(num=output['lxy-dsa'][longdecay].sum('dataset').integrate('reco', 'true'),
                   denom=output['lxy-dsa'][longdecay].sum('dataset').integrate('reco', 'inclusive'),
                   overflow='over',
                   error_opts={'marker': 'o', 'color': 'tab:green', 'fillstyle': 'none',},
                   ax=ax,
                   clear=False,
                   label='DSAMu-type leptonjet')
    ax.set_ylim([0, 1.05])
    ax.set_yticks(np.arange(0, 1.05, 0.1))
    ax.set_xticks(np.arange(0, 701, 50))
    ax.grid(axis='y', ls='--')
    ax.legend()
    ax.set_title('[signalMC|2mu2e] mu-type leptonjet\nreconstruction efficiency vs. gen muon lxy', x=0.0, ha="left")
    ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
    ax.set_ylabel('Efficiency/7', y=1.0, ha="right")
    ax.text( 0.6, 0.6, '$\geqslant$1 gen darkphoton(${}$) with\n$p_T>20GeV, |\eta|<2.4, lxy<700cm$\n$\Delta R$(gen,reco)<0.4'.format(r'\rightarrow\mu^+\mu^-'), transform=ax.transAxes)

    fig.savefig(join(outdir, 'mulj-effi-vs-lxy_2mu2e.png'))
    fig.savefig(join(outdir, 'mulj-effi-vs-lxy_2mu2e.pdf'))
    plt.close(fig)


    fig, ax = plt.subplots(figsize=(8,6))
    hist.plot1d(output['reso'][longdecay].sum('dataset'), overlay='reco', ax=ax, overflow='all', density=True)
    ax.set_title('[signalMC|2mu2e] mu-type leptonjet $p_T$ resolution', x=0, ha='left')
    ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
    ax.set_ylabel(ax.get_ylabel(), y=1.0, ha="right")
    ax.set_xticks(np.arange(-1, 2.01, 0.2))

    fig.savefig(join(outdir, 'mulj-ptreso_2mu2e.png'))
    fig.savefig(join(outdir, 'mulj-ptreso_2mu2e.pdf'))
    plt.close(fig)

    # ----------------------------------------------------------
    ## EGM cand efficiency

    output = processor.run_uproot_job(sigDS_2mu2e,
                                    treename='ffNtuplizer/ffNtuple',
                                    processor_instance=EGMEffiProcessor(),
                                    executor=processor.futures_executor,
                                    executor_args=dict(workers=12, flatten=False),
                                    chunksize=500000,
                                    )

    fig, ax = plt.subplots(figsize=(8,6))
    hist.plotratio(num=output['lxy'][longdecay].sum('dataset').integrate('reco', 'true'),
                   denom=output['lxy'][longdecay].sum('dataset').integrate('reco', 'inclusive'),
                   overflow='over',
                   error_opts={'marker': 'o',},
                   ax=ax,
                   label='PFElectron+PFPhoton')
    hist.plotratio(num=output['lxy-el'][longdecay].sum('dataset').integrate('reco', 'true'),
                   denom=output['lxy-el'][longdecay].sum('dataset').integrate('reco', 'inclusive'),
                   overflow='over',
                   error_opts={'marker': 'o', 'color': 'tab:red', 'fillstyle': 'none',},
                   ax=ax,
                   clear=False,
                   label='PFElectron')
    hist.plotratio(num=output['lxy-pho'][longdecay].sum('dataset').integrate('reco', 'true'),
                   denom=output['lxy-pho'][longdecay].sum('dataset').integrate('reco', 'inclusive'),
                   overflow='over',
                   error_opts={'marker': 'o', 'color': 'tab:green', 'fillstyle': 'none',},
                   ax=ax,
                   clear=False,
                   label='PFPhoton')
    ax.set_ylim([0, 1.05])
    ax.set_yticks(np.arange(0, 1.05, 0.1))
    ax.set_xticks(np.arange(0, 251, 25))
    ax.grid(axis='y', ls='--')
    ax.legend()
    ax.set_title('[signalMC|2mu2e] leptonjet source - egamma candidates (PFElectron+PFPhoton)\nreconstruction efficiency vs. gen electron lxy', x=0.0, ha="left")
    ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
    ax.set_ylabel('Efficiency/2.5', y=1.0, ha="right")
    ax.text( 0.6, 0.6, '$\geqslant$2 gen electrons with\n$p_T>10GeV, |\eta|<2.4, vxy<250cm$\n$\Delta R$(gen,reco)<0.3', transform=ax.transAxes)

    fig.savefig(join(outdir, 'egmcand-effi-vs-lxy_2mu2e.png'))
    fig.savefig(join(outdir, 'egmcand-effi-vs-lxy_2mu2e.pdf'))
    plt.close(fig)


    # ----------------------------------------------------------
    ## EGM leptonjet efficiency, resolution

    output = processor.run_uproot_job(sigDS_2mu2e,
                                    treename='ffNtuplizer/ffNtuple',
                                    processor_instance=EGMLJEffiResoProcessor(),
                                    executor=processor.futures_executor,
                                    executor_args=dict(workers=12, flatten=False),
                                    chunksize=500000,
                                    )

    fig, ax = plt.subplots(figsize=(8,6))
    hist.plotratio(num=output['lxy'][longdecay].sum('dataset').integrate('reco', 'true'),
                   denom=output['lxy'][longdecay].sum('dataset').integrate('reco', 'inclusive'),
                   overflow='over',
                   error_opts={'marker': 'o',},
                   ax=ax,
                   label='EGM-type leptonjet')
    hist.plotratio(num=output['lxy-el'][longdecay].sum('dataset').integrate('reco', 'true'),
                   denom=output['lxy-el'][longdecay].sum('dataset').integrate('reco', 'inclusive'),
                   overflow='over',
                   error_opts={'marker': 'o', 'color': 'tab:red', 'fillstyle': 'none',},
                   ax=ax,
                   clear=False,
                   label='Electron-type leptonjet')
    hist.plotratio(num=output['lxy-pho'][longdecay].sum('dataset').integrate('reco', 'true'),
                   denom=output['lxy-pho'][longdecay].sum('dataset').integrate('reco', 'inclusive'),
                   overflow='over',
                   error_opts={'marker': 'o', 'color': 'tab:green', 'fillstyle': 'none',},
                   ax=ax,
                   clear=False,
                   label='Photon-type leptonjet')
    ax.set_ylim([0, 1.05])
    ax.set_yticks(np.arange(0, 1.05, 0.1))
    ax.set_xticks(np.arange(0, 251, 25))
    ax.grid(axis='y', ls='--')
    ax.legend()
    ax.set_title('[signalMC|2mu2e] EGM-type leptonjet\nreconstruction efficiency vs. gen electron lxy', x=0.0, ha="left")
    ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
    ax.set_ylabel('Efficiency/2.5', y=1.0, ha="right")
    ax.text( 0.6, 0.6, '$\geqslant$1 gen darkphoton(${}$) with\n$p_T>20GeV, |\eta|<2.4, lxy<250cm$\n$\Delta R$(gen,reco)<0.4'.format(r'\rightarrow e^+e^-'), transform=ax.transAxes)

    fig.savefig(join(outdir, 'egmlj-effi-vs-lxy_2mu2e.png'))
    fig.savefig(join(outdir, 'egmlj-effi-vs-lxy_2mu2e.pdf'))
    plt.close(fig)


    fig, ax = plt.subplots(figsize=(8,6))
    hist.plot1d(output['reso'][longdecay].sum('dataset'), overlay='reco', ax=ax, overflow='all', density=True)
    ax.set_title('[signalMC|2mu2e] EGM-type leptonjet $p_T$ resolution', x=0, ha='left')
    ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
    ax.set_ylabel(ax.get_ylabel(), y=1.0, ha="right")
    ax.set_xticks(np.arange(-0.5, 0.51, 0.1))

    fig.savefig(join(outdir, 'egmlj-ptreso_2mu2e.png'))
    fig.savefig(join(outdir, 'egmlj-ptreso_2mu2e.pdf'))
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
