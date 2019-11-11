#!/usr/bin/env python
import concurrent.futures
import json
import shlex
import subprocess
from copy import deepcopy
from datetime import datetime
from os.path import join

import uproot

# skimmed background sample path
EOSPATHS_BKG = dict(
    QCD={
        "QCD_Pt-15to20": ["/store/group/lpcmetx/SIDM/Skim/2018/QCD_Pt-15to20_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v3",],
        "QCD_Pt-20to30": ["/store/group/lpcmetx/SIDM/Skim/2018/QCD_Pt-20to30_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v4",],
        "QCD_Pt-30to50": ["/store/group/lpcmetx/SIDM/Skim/2018/QCD_Pt-30to50_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v3",],
        "QCD_Pt-50to80": ["/store/group/lpcmetx/SIDM/Skim/2018/QCD_Pt-50to80_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v3",],
        "QCD_Pt-80to120": [
            "/store/group/lpcmetx/SIDM/Skim/2018/QCD_Pt-80to120_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1",
            "/store/group/lpcmetx/SIDM/Skim/2018/QCD_Pt-80to120_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_ext1-v2",
        ],
        "QCD_Pt-120to170": [
            "/store/group/lpcmetx/SIDM/Skim/2018/QCD_Pt-120to170_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1",
            "/store/group/lpcmetx/SIDM/Skim/2018/QCD_Pt-120to170_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_ext1-v2",
        ],
        "QCD_Pt-170to300": ["/store/group/lpcmetx/SIDM/Skim/2018/QCD_Pt-170to300_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v3",],
        "QCD_Pt-300to470": [
            "/store/group/lpcmetx/SIDM/Skim/2018/QCD_Pt-300to470_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v3",
            "/store/group/lpcmetx/SIDM/Skim/2018/QCD_Pt-300to470_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_ext3-v1",
        ],
        "QCD_Pt-470to600": [
            "/store/group/lpcmetx/SIDM/Skim/2018/QCD_Pt-470to600_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1",
            "/store/group/lpcmetx/SIDM/Skim/2018/QCD_Pt-470to600_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_ext1-v2",
        ],
        "QCD_Pt-600to800": ["/store/group/lpcmetx/SIDM/Skim/2018/QCD_Pt-600to800_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1",],
        "QCD_Pt-800to1000": ["/store/group/lpcmetx/SIDM/Skim/2018/QCD_Pt-800to1000_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_ext3-v2",],
        "QCD_Pt-1000toInf": ["/store/group/lpcmetx/SIDM/Skim/2018/QCD_Pt-1000toInf_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1",],
    },
    DYJetsToLL={
        "DYJetsToLL-M-10to50": ["/store/group/lpcmetx/SIDM/Skim/2018/DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2"],
        "DYJetsToLL_M-50": ["/store/group/lpcmetx/SIDM/Skim/2018/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1"],
    },
    TTJets={
        "TTJets": ["/store/group/lpcmetx/SIDM/Skim/2018/TTJets_TuneCP5_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1",],
    },
    DiBoson={
        "WW": ["/store/group/lpcmetx/SIDM/Skim/2018/WW_TuneCP5_13TeV-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2",],
        "WZ": ["/store/group/lpcmetx/SIDM/Skim/2018/WZ_TuneCP5_13TeV-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v3",],
        "ZZ": ["/store/group/lpcmetx/SIDM/Skim/2018/ZZ_TuneCP5_13TeV-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2",],
    }
)

# background sample xsec
BKG_XSEC = dict(
    TTJets={
        "TTJets": 491.3,
        "TTJets_SingleLeptFromT": 108.5,
        "TTJets_SingleLeptFromTbar": 109.1,
        "TTJets_DiLept": 54.29,
    },
    ST={
        "Top": 34.91,
        "AntiTop": 34.97,
    },
    WJets={
        "WJets_HT-70To100": 1353,
        "WJets_HT-100To200": 1395,
        "WJets_HT-200To400": 407.9,
        "WJets_HT-400To600": 57.48,
        "WJets_HT-600To800": 12.87,
        "WJets_HT-800To1200": 5.366,
        "WJets_HT-1200To2500": 1.074,
        "WJets_HT-2500ToInf": 0.008001,
    },
    DYJetsToLL={
        "DYJetsToLL-M-10to50": 15820,
        "DYJetsToLL_M-50": 5317,
    },
    DiBoson={
        "WW": 75.91,
        "ZZ": 12.14,
        "WZ": 27.55
    },
    TriBoson={
        "WWW": 0.2154,
        "WWZ": 0.1676,
        "WZZ": 0.05701,
        "ZZZ": 0.01473,
        "WZG": 0.04345,
        "WWG": 0.2316,
        "WGG": 2.001,
    },
    QCD={
        "QCD_Pt-15to20": 279900,
        "QCD_Pt-20to30": 2526000,
        "QCD_Pt-30to50": 1362000,
        "QCD_Pt-50to80": 376600,
        "QCD_Pt-80to120": 88930,
        "QCD_Pt-120to170": 21230,
        "QCD_Pt-170to300": 7055,
        "QCD_Pt-300to470": 619.8,
        "QCD_Pt-470to600": 59.24,
        "QCD_Pt-600to800": 18.19,
        "QCD_Pt-800to1000": 3.271,
        "QCD_Pt-1000toInf": 1.08,
    },
)

XDIRECTOR = 'root://cmseos.fnal.gov/'

def processed_genwgt_sum(ntuplefile):
    """Given a ntuplefile path, return the sum of gen weights."""

    f_ = uproot.open(ntuplefile)
    key_ = f_.allkeys(filtername=lambda k: k.endswith(b"weight"))
    if key_:
        key_ = key_[0]
        return f_[key_]['weight'].array().sum()
    else:
        key_ = f_.allkeys(filtername=lambda k: k.endswith(b"history"))[0]
        return f_[key_].values[3]  # 0: run, 1: lumi, 2: events, 3: genwgtsum


def total_genwgt_sum(filelist):
    """Given a list of ntuple files, return the total sum of gen weights"""

    numsum = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(processed_genwgt_sum, f): f for f in filelist}
        for future in concurrent.futures.as_completed(futures):
            filename = futures[future]
            try:
                numsum += future.result()
            except Exception as e:
                print(f">> Fail to get genwgts for {filename}\n{str(e)}")
    return numsum


def list_ffNtuple_files(eospath):
    """list all files LIKE ffNtuple.root under an eospath"""

    subflist = []
    try:
        timestamps_ = subprocess.check_output(shlex.split('eos {0} ls {1}'.format(XDIRECTOR, eospath))).split()
        timestamps = [ts.decode() for ts in timestamps_ if ts]
        timestamps = sorted(timestamps, key=lambda x: datetime.strptime(x, "%y%m%d_%H%M%S"))
        eospath = join(eospath, timestamps[-1]) # most recent submission
        subflist = subprocess.check_output(shlex.split('eos {0} find -name "*ffNtuple*.root" -f --xurl {1}'.format(XDIRECTOR, eospath))).split()
    except:
        print(f"Error whenn stat eospath: {eospath}\nEmpty list returned!")
    subflist = [f.decode() for f in subflist if f]
    return subflist


def generate_skimmed_background_files():
    """list ffNtuple-like files from a group of paths"""

    bkgFileLists = deepcopy(EOSPATHS_BKG)
    for category in bkgFileLists:
        for tag in bkgFileLists[category]:
            paths = bkgFileLists[category][tag]
            bkgFileLists[category][tag] = [f for p in paths for f in list_ffNtuple_files(p)]
    return bkgFileLists


def generate_background_scale():
    """parse all files to get number of events processed => scale
        scale = xsec/#genwgtsum, scale*lumi-> gen weight
    """

    bkgfilelist = generate_skimmed_background_files()
    generated = dict()
    for i, group in enumerate(bkgfilelist, start=1):
        print(f"[{i}/{len(bkgfilelist)}] {group}")
        generated[group] = {}

        for tag in bkgfilelist[group]:
            xsec = BKG_XSEC[group][tag]
            sumgenwgt = total_genwgt_sum(bkgfilelist[group][tag])
            generated[group][tag] = xsec / sumgenwgt
#             nevents = total_event_number(bkgfilelist[group][tag])
#             generated[group][tag] = xsec / nevents

    fn = f"backgrounds_scale_{datetime.now().strftime('%y%m%d')}.json"
    print(f"=> Saving as {fn}")
    with open(fn, "w") as outf:
        outf.write(json.dumps(generated, indent=4))


if __name__ == "__main__":
    generate_background_scale()
