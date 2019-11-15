#!/usr/bin/env
"""
find isolation cut for EGM-type leptonjet (2mu2e)
"""
import argparse
from contextlib import contextmanager

import awkward
import coffea.processor as processor
import numpy as np
from coffea import hist
from coffea.analysis_objects import JaggedCandidateArray
from FireHydrant.Analysis.DatasetMapLoader import (DatasetMapLoader,
                                                   SigDatasetMapLoader)
from FireHydrant.Tools.correction import (get_nlo_weight_function,
                                          get_pu_weights_function,
                                          get_ttbar_weight)
from FireHydrant.Tools.metfilter import MetFilters
from FireHydrant.Tools.trigger import Triggers

from FireHydrant.Analysis.StudyIsolationDependency import LeptonjetIsoProcessor

parser = argparse.ArgumentParser(description="find isolation cut for EGM-type leptonjet")
parser.add_argument("--sync", action='store_true', help="issue rsync command to sync plots folder to lxplus web server")
parser.add_argument("--preserve", action='store_true', help="preserve plots in ROOT file")
args = parser.parse_args()

import ROOT
ROOT.gROOT.SetBatch()


sdml = SigDatasetMapLoader()
sigDS, sigSCALE = sdml.fetch('2mu2e')

dml = DatasetMapLoader()
bkgDS, bkgMAP, bkgSCALE = dml.fetch('bkg')
dataDS, dataMAP = dml.fetch('data')

@contextmanager
def _setIgnoreLevel(level):
    originalLevel = ROOT.gErrorIgnoreLevel
    ROOT.gErrorIgnoreLevel = level
    yield
    ROOT.gErrorIgnoreLevel = originalLevel


def root_filling(output, datatype):
    chan = '2mu2e'
    ljtype = 'egm'
    isotype = 'dbeta'
    key = f'{datatype}-{chan}_{ljtype}_{isotype}'

    res = ROOT.TH1F(key, f'{key};isolation;frequency', 100, 0, 0.2)
    for dbeta_, wgt_, ljtype_, chan_ in np.nditer([
        output['dbeta'].value,
        output['wgt'].value,
        output['ljtype'].value,
        output['channel'].value,
    ]):
        if chan_!=1: continue
        if ljtype_!=2: continue
        res.Fill(dbeta_, wgt_)

    return res



if __name__ == "__main__":
    import os
    from os.path import join, isdir, splitext

    reldir = splitext(__file__)[0].replace('_', '/')
    outdir = join(os.getenv('FH_BASE'), "Imgs", reldir)
    if not isdir(outdir): os.makedirs(outdir)

    histos = {}

    print('[signal]')
    outputs = {}
    for k, ds in sigDS.items():
        outputs[k] = processor.run_uproot_job({k: ds},
                                      treename='ffNtuplizer/ffNtuple',
                                      processor_instance=LeptonjetIsoProcessor(dphi_control=False, data_type='sig'),
                                      executor=processor.futures_executor,
                                      executor_args=dict(workers=12, flatten=True),
                                      chunksize=500000,
                                      )
    print("Filling..")
    histos['sig'] = {}
    for k in outputs:
        histos['sig'][k] = root_filling(outputs[k], k)

    print('[background]')
    output = processor.run_uproot_job(bkgDS,
                                      treename='ffNtuplizer/ffNtuple',
                                      processor_instance=LeptonjetIsoProcessor(dphi_control=False, data_type='bkg'),
                                      executor=processor.futures_executor,
                                      executor_args=dict(workers=12, flatten=True),
                                      chunksize=500000,
                                      )
    print("Filling..")
    histos['bkg'] = root_filling(output, 'bkg')

    print('[data]')
    output = processor.run_uproot_job(dataDS,
                                      treename='ffNtuplizer/ffNtuple',
                                      processor_instance=LeptonjetIsoProcessor(dphi_control=True, data_type='data'),
                                      executor=processor.futures_executor,
                                      executor_args=dict(workers=12, flatten=True),
                                      chunksize=500000,
                                      )
    print("Filling..")
    histos['data'] = root_filling(output, 'data')


    print("Saving...")
    sighcums = []
    for sigk, sigh in histos['sig'].items():
        sighcum = sigh.GetCumulative()
        sighcum.Scale(1./sigh.Integral())
        sighcums.append(sighcum)
    sighcums = sorted(sighcums, key=lambda h: h.Integral(), reverse=True)

    bkgh = histos['bkg']
    bkgcum = bkgh.GetCumulative(False)
    bkgcum.Scale(1./bkgh.Integral())

    datah = histos['data']
    datacum = datah.GetCumulative(False)
    datacum.Scale(1./datah.Integral())

    COLORS = [ROOT.kBlue, ROOT.kRed, ROOT.kGreen]
    # pngIdx = [0, 1, 2, 3, 4, len(sighcums)-1] # first 5 and last one
    canvases = []
    for i, sighcum in enumerate(sighcums):
        c = ROOT.TCanvas(f'c_{sighcum.GetName()}', sighcum.GetTitle(), 700, 500)
        toplots = [sighcum, bkgcum, datacum]
        x1, y1, height = None, None, None
        for j, h in enumerate(toplots):
            h.SetLineColor(COLORS[j])
            h.GetYaxis().SetRangeUser(0, 1)
            if j==0: h.Draw()
            else: h.Draw('sames')
            ROOT.gPad.Update() # this is needed, otherwise `FindObject('stats')` would return null ptr.
            statbox = h.FindObject('stats')
            if j==0:
                x1, y1 = statbox.GetX1NDC(), statbox.GetY1NDC()
                height = statbox.GetY2NDC()-statbox.GetY1NDC()
            else:
                statbox.SetY2NDC(y1)
                y1 -= height
                statbox.SetY1NDC(y1)
            statbox.SetTextColor(COLORS[j])
            c.Update()
        # if i in pngIdx:
        print(f'[{i}]\t{sighcum.GetName():60}{sighcum.Integral():.3f}')
        with _setIgnoreLevel(ROOT.kError):
            c.SaveAs(f'{outdir}/{i:02}__{sighcum.GetName()}-pt.png')
            c.SaveAs(f'{outdir}/{i:02}__{sighcum.GetName()}-pt.pdf')
        canvases.append(c)



    if args.preserve:
        outrootfn = f'{outdir}/plots.root'
        print(f"--> preserving")
        outrootf = ROOT.TFile(outrootfn, 'RECREATE')
        outrootf.cd()
        for c in canvases:
            c.Write()
        outrootf.Close()

    for c in canvases: c.Close()

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
