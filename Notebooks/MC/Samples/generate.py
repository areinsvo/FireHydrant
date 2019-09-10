#!/usr/bin/env python
"""Generate MC sample json file, until proper sample management tool show up.
"""
import concurrent.futures
import json
from datetime import datetime
from os.path import join

import uproot
from FireHydrant.Tools.commonhelpers import eosfindfile, eosls

EOSPATHS_BKG = dict(
    TTJets={
        "TTJets": ["/store/group/lpcmetx/SIDM/ffNtuple/2018/TTJets_TuneCP5_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1",],
        "TTJets_SingleLeptFromT": ["/store/group/lpcmetx/SIDM/ffNtuple/2018/TTJets_SingleLeptFromT_TuneCP5_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1",],
        "TTJets_SingleLeptFromTbar": ["/store/group/lpcmetx/SIDM/ffNtuple/2018/TTJets_SingleLeptFromTbar_TuneCP5_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1",],
        "TTJets_DiLept": ["/store/group/lpcmetx/SIDM/ffNtuple/2018/TTJets_DiLept_TuneCP5_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1",],
    },
    ST={
        "AntiTop": ["/store/group/lpcmetx/SIDM/ffNtuple/2018/ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_ext1-v1",],
        "Top": ["/store/group/lpcmetx/SIDM/ffNtuple/2018/ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_ext1-v1",],
    },
    WJets={
        "WJets_HT-70To100": ["/store/group/lpcmetx/SIDM/ffNtuple/2018/WJetsToLNu_HT-70To100_TuneCP5_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1",],
        "WJets_HT-100To200": ["/store/group/lpcmetx/SIDM/ffNtuple/2018/WJetsToLNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1",],
        "WJets_HT-200To400": ["/store/group/lpcmetx/SIDM/ffNtuple/2018/WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1",],
        "WJets_HT-400To600": ["/store/group/lpcmetx/SIDM/ffNtuple/2018/WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1",],
        "WJets_HT-600To800": ["/store/group/lpcmetx/SIDM/ffNtuple/2018/WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1",],
        "WJets_HT-800To1200": ["/store/group/lpcmetx/SIDM/ffNtuple/2018/WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1",],
        "WJets_HT-1200To2500": ["/store/group/lpcmetx/SIDM/ffNtuple/2018/WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1",],
        "WJets_HT-2500ToInf": ["/store/group/lpcmetx/SIDM/ffNtuple/2018/WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1",],
    },
    DYJetsToLL={
        "DYJetsToLL-M-10to50": ["/store/group/lpcmetx/SIDM/ffNtuple/2018/DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2"],
        "DYJetsToLL_M-50": ["/store/group/lpcmetx/SIDM/ffNtuple/2018/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1"],
    },
    DiBoson={
        "WW": ["/store/group/lpcmetx/SIDM/ffNtuple/2018/WW_TuneCP5_13TeV-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2",],
        "ZZ": ["/store/group/lpcmetx/SIDM/ffNtuple/2018/ZZ_TuneCP5_13TeV-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2",],
        "WZ": ["/store/group/lpcmetx/SIDM/ffNtuple/2018/WZ_TuneCP5_13TeV-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v3",],
    },
    TriBoson={
        "WWW": ["/store/group/lpcmetx/SIDM/ffNtuple/2018/WWW_4F_TuneCP5_13TeV-amcatnlo-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_ext1-v2",],
        "WWZ": ["/store/group/lpcmetx/SIDM/ffNtuple/2018/WWZ_TuneCP5_13TeV-amcatnlo-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_ext1-v2",],
        "WZZ": ["/store/group/lpcmetx/SIDM/ffNtuple/2018/WZZ_TuneCP5_13TeV-amcatnlo-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_ext1-v2",],
        "ZZZ": ["/store/group/lpcmetx/SIDM/ffNtuple/2018/ZZZ_TuneCP5_13TeV-amcatnlo-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_ext1-v2",],
        "WZG": ["/store/group/lpcmetx/SIDM/ffNtuple/2018/WZG_TuneCP5_13TeV-amcatnlo-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1",],
        "WWG": ["/store/group/lpcmetx/SIDM/ffNtuple/2018/WWG_TuneCP5_13TeV-amcatnlo-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_ext1-v2",],
        "WGG": ["/store/group/lpcmetx/SIDM/ffNtuple/2018/WGG_5f_TuneCP5_13TeV-amcatnlo-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1",],
    },
    QCD={
        "QCD_Pt-15to20": ["/store/group/lpcmetx/SIDM/ffNtuple/2018/QCD_Pt-15to20_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v3",],
        "QCD_Pt-20to30": ["/store/group/lpcmetx/SIDM/ffNtuple/2018/QCD_Pt-20to30_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v4",],
        # "QCD_Pt-30to50": ["/store/group/lpcmetx/SIDM/ffNtuple/2018/QCD_Pt-30to50_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v3",],
        "QCD_Pt-50to80": ["/store/group/lpcmetx/SIDM/ffNtuple/2018/QCD_Pt-50to80_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v3",],
        "QCD_Pt-80to120": [
            "/store/group/lpcmetx/SIDM/ffNtuple/2018/QCD_Pt-80to120_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1",
            "/store/group/lpcmetx/SIDM/ffNtuple/2018/QCD_Pt-80to120_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_ext1-v2",
        ],
        "QCD_Pt-120to170": [
            "/store/group/lpcmetx/SIDM/ffNtuple/2018/QCD_Pt-120to170_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1",
            "/store/group/lpcmetx/SIDM/ffNtuple/2018/QCD_Pt-120to170_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_ext1-v2",
        ],
        "QCD_Pt-170to300": ["/store/group/lpcmetx/SIDM/ffNtuple/2018/QCD_Pt-170to300_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v3",],
        "QCD_Pt-300to470": [
            "/store/group/lpcmetx/SIDM/ffNtuple/2018/QCD_Pt-300to470_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v3",
            "/store/group/lpcmetx/SIDM/ffNtuple/2018/QCD_Pt-300to470_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_ext3-v1",
        ],
        "QCD_Pt-470to600": [
            "/store/group/lpcmetx/SIDM/ffNtuple/2018/QCD_Pt-470to600_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1",
            "/store/group/lpcmetx/SIDM/ffNtuple/2018/QCD_Pt-470to600_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_ext1-v2",
        ],
        "QCD_Pt-600to800": ["/store/group/lpcmetx/SIDM/ffNtuple/2018/QCD_Pt-600to800_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1",],
        "QCD_Pt-800to1000": ["/store/group/lpcmetx/SIDM/ffNtuple/2018/QCD_Pt-800to1000_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_ext3-v2",],
        "QCD_Pt-1000toInf": ["/store/group/lpcmetx/SIDM/ffNtuple/2018/QCD_Pt-1000toInf_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1",],
    },
)


# BKG_XSEC = dict(
#     TTJets={
#         "TTJets": 491.3,
#         "TTJets_SingleLeptFromT": 108.5,
#         "TTJets_SingleLeptFromTbar": 109.1,
#         "TTJets_DiLept": 54.29,
#     },
#     ST={
#         "Top": 34.91,
#         "AntiTop": 34.97,
#     },
#     WJets={
#         "WJets_HT-70To100": 1290,
#         "WJets_HT-100To200": 1394,
#         "WJets_HT-200To400": 409.8,
#         "WJets_HT-400To600": 57.77,
#         "WJets_HT-600To800": 12.93,
#         "WJets_HT-800To1200": 5.454,
#         "WJets_HT-1200To2500": 1.085,
#         "WJets_HT-2500ToInf": 0.008062,
#     },
#     DYJetsToLL={
#         "DYJetsToLL-M-10to50": 15820,
#         "DYJetsToLL_M-50": 5317,
#     },
#     DiBoson={
#         "WW": 75.91,
#         "ZZ": 12.14,
#         "WZ": 27.55
#     },
#     TriBoson={
#         "WWW": 0.2154,
#         "WWZ": 0.1676,
#         "WZZ": 0.05701,
#         "ZZZ": 0.01473,
#         "WZG": 0.04345,
#         "WWG": 0.2316,
#         "WGG": 2.001,
#     },
#     QCD={
#         # "QCD_Pt-15to20": 2805000,
#         "QCD_Pt-20to30": 2536000,
#         # "QCD_Pt-30to50": 1375000,
#         "QCD_Pt-50to80": 377900,
#         "QCD_Pt-80to120": 89730,
#         "QCD_Pt-120to170": 21410,
#         "QCD_Pt-170to300": 7022,
#         "QCD_Pt-300to470": 619.8,
#         "QCD_Pt-470to600": 59.32,
#         "QCD_Pt-600to800": 18.19,
#         "QCD_Pt-800to1000": 3.271,
#         "QCD_Pt-1000toInf": 1.08,
#     },
# )

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
        # "QCD_Pt-30to50": 1362000,
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

# private signal MC
EOSPATH_SIG = '/store/group/lpcmetx/SIDM/ffNtuple/2018/CRAB_PrivateMC/'
EOSPATH_SIG2 = {
    "4mu": "/store/group/lpcmetx/SIDM/ffNtuple/2018/SIDM_XXTo2ATo4Mu",
    "2mu2e": "/store/group/lpcmetx/SIDM/ffNtuple/2018/SIDM_XXTo2ATo2Mu2E",
}


# _____________________________________________________________________________

def generate_background_json():

    generated = dict()
    for group in EOSPATHS_BKG:
        generated[group] = {}
        for tag in EOSPATHS_BKG[group]:
            generated[group][tag] = []
            for path in EOSPATHS_BKG[group][tag]:
                timestampdirs = eosls(path)
                timestampdirs = sorted(
                    timestampdirs, key=lambda x: datetime.strptime(x, "%y%m%d_%H%M%S")
                )
                latest = join(path, timestampdirs[-1])
                for filepath in eosfindfile(latest):
                    if "/failed/" in filepath:
                        continue  # filter out those in *failed* folder
                    generated[group][tag].append(filepath)

    with open("backgrounds.json", "w") as outf:
        outf.write(json.dumps(generated, indent=4))


def processed_event_number(ntuplefile):
    """Given a ntuplefile path, return the number of events it ran over."""

    f_ = uproot.open(ntuplefile)
    key_ = f_.allkeys(filtername=lambda k: k.endswith(b"history"))[0]
    return f_[key_].values[2]  # 0: run, 1: lumi, 2: events


def total_event_number(filelist):
    """Given a list of ntuple files, return the total number of events processed"""

    numevents = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(processed_event_number, f): f for f in filelist}
        for future in concurrent.futures.as_completed(futures):
            filename = futures[future]
            try:
                numevents += future.result()
            except Exception as e:
                print(f">> Fail to get numEvents for {filename}\n{str(e)}")
    return numevents


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


def generate_background_scale():
    """parse all files to get number of events processed => scale
        scale = xsec/#genwgtsum, scale*lumi-> gen weight
    """

    bkgfilelist = json.load(open("backgrounds.json"))
    generated = dict()
    for group in BKG_XSEC:
        generated[group] = {}

        for tag in BKG_XSEC[group]:
            xsec = BKG_XSEC[group][tag]
            sumgenwgt = total_genwgt_sum(bkgfilelist[group][tag])
            generated[group][tag] = xsec / sumgenwgt
#             nevents = total_event_number(bkgfilelist[group][tag])
#             generated[group][tag] = xsec / nevents

    with open("backgrounds_scale.json", "w") as outf:
        outf.write(json.dumps(generated, indent=4))


def remove_empty_file(filepath):
    """given a file, if the tree has non-zero number of events, return filepath"""
    f_ = uproot.open(filepath)
    key_ = f_.allkeys(filtername=lambda k: k.endswith(b"ffNtuple"))[0]
    if uproot.open(filepath)[key_].numentries != 0:
        return filepath
    else:
        return None


def remove_empty_files(filelist):
    """given a list of files, return all files with a tree of non-zero number of events"""
    cleanlist = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(remove_empty_file, f): f for f in filelist}
        for future in concurrent.futures.as_completed(futures):
            filename = futures[future]
            try:
                res = future.result()
                if res:
                    cleanlist.append(res)
            except Exception as e:
                print(f">> Fail to get numEvents for {filename}\n{str(e)}")
    return cleanlist


def clean_background_json():
    """parse all background files, remove empty tree files
    """
    bkgfilelist = json.load(open("backgrounds.json"))
    for group in bkgfilelist:
        for tag in bkgfilelist[group]:
            files = bkgfilelist[group][tag]
            bkgfilelist[group][tag] = remove_empty_files(files)
    with open("backgrounds_nonempty.json", "w") as outf:
        outf.write(json.dumps(bkgfilelist, indent=4))


def generate_signal_json():
    """generate private signal file list json"""
    paramsubdirs = eosls(EOSPATH_SIG)
    json_4mu, json_2mu2e = {}, {}
    for subdir in paramsubdirs:
        if '0p8' in subdir or '2p5' in subdir: continue # skipping unrequested lifetime points
        if '4Mu' in subdir:
            key = subdir.replace('SIDM_BsTo2DpTo4Mu_', '').split('_ctau')[0].replace('MBs', 'mXX').replace('MDp', 'mA')
            key += '_lxy-300' # mXX-1000_mA-0p25_lxy-300
            timestampdirs = eosls(join(EOSPATH_SIG, subdir))
            timestampdirs = sorted(timestampdirs, key=lambda x: datetime.strptime(x, "%y%m%d_%H%M%S"))
            latest = join(EOSPATH_SIG, subdir, timestampdirs[-1])
            json_4mu[key] = [f for f in eosfindfile(latest) if '/failed/' not in f]
        if '2Mu2e' in subdir:
            key = subdir.replace('SIDM_BsTo2DpTo2Mu2e_', '').split('_ctau')[0].replace('MBs', 'mXX').replace('MDp', 'mA')
            key += '_lxy-300'
            timestampdirs = eosls(join(EOSPATH_SIG, subdir))
            timestampdirs = sorted(timestampdirs, key=lambda x: datetime.strptime(x, "%y%m%d_%H%M%S"))
            latest = join(EOSPATH_SIG, subdir, timestampdirs[-1])
            json_2mu2e[key] = [f for f in eosfindfile(latest) if '/failed/' not in f]

    ## samples with new naming
    for subdir in eosls(EOSPATH_SIG2['4mu']):
        key = subdir.split('_ctau')[0]  # mXX-100_mA-5_lxy-0p3
        timestampdirs = eosls(join(EOSPATH_SIG2['4mu'], subdir))
        timestampdirs = sorted(timestampdirs, key=lambda x: datetime.strptime(x, "%y%m%d_%H%M%S"))
        latest = join(EOSPATH_SIG2['4mu'], subdir, timestampdirs[-1])
        json_4mu[key] = [f for f in eosfindfile(latest) if '/failed/' not in f]
    for subdir in eosls(EOSPATH_SIG2['2mu2e']):
        key = subdir.split('_ctau')[0]  # mXX-100_mA-5_lxy-0p3
        timestampdirs = eosls(join(EOSPATH_SIG2['2mu2e'], subdir))
        timestampdirs = sorted(timestampdirs, key=lambda x: datetime.strptime(x, "%y%m%d_%H%M%S"))
        latest = join(EOSPATH_SIG2['2mu2e'], subdir, timestampdirs[-1])
        json_2mu2e[key] = [f for f in eosfindfile(latest) if '/failed/' not in f]

    with open('signal_4mu.json', 'w') as outf:
        outf.write(json.dumps(json_4mu, indent=4))
    with open('signal_2mu2e.json', 'w') as outf:
        outf.write(json.dumps(json_2mu2e, indent=4))


def generate_signal_scale():

    from signalnumbers import genfiltereff, genxsec, darkphotonbr

    filelists = {}
    filelists['2mu2e'] = json.load(open("signal_2mu2e.json"))
    filelists['4mu'] = json.load(open("signal_4mu.json"))

    def decompose_paramtag(paramtag):
        # paramtag  mXX-1000_mA-0p8_lxy-300 => (1000, 0.8, 300)
        paramtags = paramtag.split('_')
        assert (len(paramtags) == 3)

        res = []
        for t in paramtags:
            v = t.split('-')[1]
            if 'p' in v:
                res.append(float(v.replace('p', '.')))
            else:
                res.append(int(v))

        return tuple(res)

    scale_2mu2e = {}
    for paramtag in filelists['2mu2e']:
        # paramtag like : mXX-1000_mA-0p8_lxy-300
        parameters = decompose_paramtag(paramtag)
        if parameters not in genfiltereff['2mu2e']:
            # didn't run those samples
            scale_2mu2e[paramtag] = 0
            continue

        filtereff = genfiltereff['2mu2e'][parameters]
        xsec = genxsec[parameters[0]]
        brs = darkphotonbr['ee'][parameters[1]] * darkphotonbr['mumu'][parameters[1]]
        nevents = total_event_number(filelists['2mu2e'][paramtag])

        scale_2mu2e[paramtag] = xsec * brs * filtereff / nevents

    with open("signal_2mu2e_scale.json", "w") as outf:
        outf.write(json.dumps(scale_2mu2e, indent=4))

    scale_4mu = {}
    for paramtag in filelists['4mu']:
        # paramtag like : mXX-1000_mA-0p8_lxy-300
        parameters = decompose_paramtag(paramtag)
        if parameters not in genfiltereff['4mu']:
            # didn't run those samples
            scale_4mu[paramtag] = 0
            continue

        filtereff = genfiltereff['4mu'][parameters]
        xsec = genxsec[parameters[0]]
        brs = darkphotonbr['mumu'][parameters[1]] * darkphotonbr['mumu'][parameters[1]]
        nevents = total_event_number(filelists['4mu'][paramtag])

        scale_4mu[paramtag] = xsec * brs * filtereff / nevents

    with open("signal_4mu_scale.json", "w") as outf:
        outf.write(json.dumps(scale_4mu, indent=4))



if __name__ == "__main__":

    import sys
    ## Here we are only keeping the most recent submission batch
    if sys.argv[1]=='bkg':
        generate_background_json()
        generate_background_scale()
        clean_background_json()

    if sys.argv[1]=='sig':
        generate_signal_json()
        generate_signal_scale()
