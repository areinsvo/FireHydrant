#!/usr/bin/env python
import concurrent.futures
import json
from datetime import datetime
from os.path import join

import uproot
from FireHydrant.Tools.commonhelpers import eosfindfile, eosls

# private signal MC
EOSPATH_SIG = '/store/group/lpcmetx/SIDM/ffNtupleV2/2018/CRAB_PrivateMC/'
EOSPATH_SIG2 = {
    "4mu": "/store/group/lpcmetx/SIDM/ffNtupleV2/2018/SIDM_XXTo2ATo4Mu",
    "2mu2e": "/store/group/lpcmetx/SIDM/ffNtupleV2/2018/SIDM_XXTo2ATo2Mu2E",
}

# background sample path
EOSPATHS_BKG = dict(
    QCD={
        "QCD_Pt-15to20": ["/store/group/lpcmetx/SIDM/ffNtupleV2/2018/QCD_Pt-15to20_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v3",],
        "QCD_Pt-20to30": ["/store/group/lpcmetx/SIDM/ffNtupleV2/2018/QCD_Pt-20to30_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v4",],
        "QCD_Pt-30to50": ["/store/group/lpcmetx/SIDM/ffNtupleV2/2018/QCD_Pt-30to50_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v3",],
        "QCD_Pt-50to80": ["/store/group/lpcmetx/SIDM/ffNtupleV2/2018/QCD_Pt-50to80_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v3",],
        "QCD_Pt-80to120": [
            "/store/group/lpcmetx/SIDM/ffNtupleV2/2018/QCD_Pt-80to120_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1",
            "/store/group/lpcmetx/SIDM/ffNtupleV2/2018/QCD_Pt-80to120_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_ext1-v2",
        ],
        "QCD_Pt-120to170": [
            "/store/group/lpcmetx/SIDM/ffNtupleV2/2018/QCD_Pt-120to170_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1",
            "/store/group/lpcmetx/SIDM/ffNtupleV2/2018/QCD_Pt-120to170_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_ext1-v2",
        ],
        "QCD_Pt-170to300": ["/store/group/lpcmetx/SIDM/ffNtupleV2/2018/QCD_Pt-170to300_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v3",],
        "QCD_Pt-300to470": [
            "/store/group/lpcmetx/SIDM/ffNtupleV2/2018/QCD_Pt-300to470_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v3",
            "/store/group/lpcmetx/SIDM/ffNtupleV2/2018/QCD_Pt-300to470_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_ext3-v1",
        ],
        "QCD_Pt-470to600": [
            "/store/group/lpcmetx/SIDM/ffNtupleV2/2018/QCD_Pt-470to600_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1",
            "/store/group/lpcmetx/SIDM/ffNtupleV2/2018/QCD_Pt-470to600_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_ext1-v2",
        ],
        "QCD_Pt-600to800": ["/store/group/lpcmetx/SIDM/ffNtupleV2/2018/QCD_Pt-600to800_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1",],
        "QCD_Pt-800to1000": ["/store/group/lpcmetx/SIDM/ffNtupleV2/2018/QCD_Pt-800to1000_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15_ext3-v2",],
        "QCD_Pt-1000toInf": ["/store/group/lpcmetx/SIDM/ffNtupleV2/2018/QCD_Pt-1000toInf_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1",],
    },
    DYJetsToLL={
        "DYJetsToLL-M-10to50": ["/store/group/lpcmetx/SIDM/ffNtupleV2/2018/DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v2"],
        "DYJetsToLL_M-50": ["/store/group/lpcmetx/SIDM/ffNtupleV2/2018/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1"],
    },
    TTJets={
        "TTJets": ["/store/group/lpcmetx/SIDM/ffNtupleV2/2018/TTJets_TuneCP5_13TeV-madgraphMLM-pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v1",],
    },
)

# background skim sample path
EOSPATHS_BKGSKIM = dict(
    QCD={
        "QCD_Pt-80to120": ["/store/group/lpcmetx/SIDM/ffNtupleV2/2018/SkimmedBkgMC/QCD_Pt-80to120_MuEnrichedPt5_TuneCP5_13TeV_pythia8",],
        "QCD_Pt-120to170": ["/store/group/lpcmetx/SIDM/ffNtupleV2/2018/SkimmedBkgMC/QCD_Pt-120to170_MuEnrichedPt5_TuneCP5_13TeV_pythia8",],
        "QCD_Pt-170to300": ["/store/group/lpcmetx/SIDM/ffNtupleV2/2018/SkimmedBkgMC/QCD_Pt-170to300_MuEnrichedPt5_TuneCP5_13TeV_pythia8",],
        "QCD_Pt-300to470": ["/store/group/lpcmetx/SIDM/ffNtupleV2/2018/SkimmedBkgMC/QCD_Pt-300to470_MuEnrichedPt5_TuneCP5_13TeV_pythia8",],
        "QCD_Pt-470to600": ["/store/group/lpcmetx/SIDM/ffNtupleV2/2018/SkimmedBkgMC/QCD_Pt-470to600_MuEnrichedPt5_TuneCP5_13TeV_pythia8",],
        "QCD_Pt-600to800": ["/store/group/lpcmetx/SIDM/ffNtupleV2/2018/SkimmedBkgMC/QCD_Pt-600to800_MuEnrichedPt5_TuneCP5_13TeV_pythia8",],
        "QCD_Pt-800to1000": ["/store/group/lpcmetx/SIDM/ffNtupleV2/2018/SkimmedBkgMC/QCD_Pt-800to1000_MuEnrichedPt5_TuneCP5_13TeV_pythia8",],
        "QCD_Pt-1000toInf": ["/store/group/lpcmetx/SIDM/ffNtupleV2/2018/SkimmedBkgMC/QCD_Pt-1000toInf_MuEnrichedPt5_TuneCP5_13TeV_pythia8",],
    },
    DYJetsToLL={
        # "DYJetsToLL-M-10to50": ["/store/group/lpcmetx/SIDM/ffNtupleV2/2018/SkimmedBkgMC/DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8",],
        "DYJetsToLL_M-50": ["/store/group/lpcmetx/SIDM/ffNtupleV2/2018/SkimmedBkgMC/DYJetsToLL_M-50_TuneCP5_13TeV-madgraphMLM-pythia8",],
    },
    TTJets={
        "TTJets": ["/store/group/lpcmetx/SIDM/ffNtupleV2/2018/SkimmedBkgMC/TTJets_TuneCP5_13TeV-madgraphMLM-pythia8",],
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


def generate_signal_json(version=-1):
    """generate private signal file list json"""
    paramsubdirs = eosls(EOSPATH_SIG)
    json_4mu, json_2mu2e = {}, {}
    for subdir in paramsubdirs:
        if 'MDp-0p8' in subdir or 'MDp-2p5' in subdir: continue # skipping unrequested darkphoton mass points
        if '4Mu' in subdir:
            key = subdir.replace('SIDM_BsTo2DpTo4Mu_', '').split('_ctau')[0].replace('MBs', 'mXX').replace('MDp', 'mA')
            key += '_lxy-300' # mXX-1000_mA-0p25_lxy-300
            timestampdirs = eosls(join(EOSPATH_SIG, subdir))
            timestampdirs = sorted(timestampdirs, key=lambda x: datetime.strptime(x, "%y%m%d_%H%M%S"))
            latest = join(EOSPATH_SIG, subdir, timestampdirs[version])
            json_4mu[key] = [f for f in eosfindfile(latest) if '/failed/' not in f]
        if '2Mu2e' in subdir:
            key = subdir.replace('SIDM_BsTo2DpTo2Mu2e_', '').split('_ctau')[0].replace('MBs', 'mXX').replace('MDp', 'mA')
            key += '_lxy-300'
            timestampdirs = eosls(join(EOSPATH_SIG, subdir))
            timestampdirs = sorted(timestampdirs, key=lambda x: datetime.strptime(x, "%y%m%d_%H%M%S"))
            latest = join(EOSPATH_SIG, subdir, timestampdirs[version])
            json_2mu2e[key] = [f for f in eosfindfile(latest) if '/failed/' not in f]

    ## samples with new naming
    for subdir in eosls(EOSPATH_SIG2['4mu']):
        key = subdir.split('_ctau')[0]  # mXX-100_mA-5_lxy-0p3
        timestampdirs = eosls(join(EOSPATH_SIG2['4mu'], subdir))
        timestampdirs = sorted(timestampdirs, key=lambda x: datetime.strptime(x, "%y%m%d_%H%M%S"))
        latest = join(EOSPATH_SIG2['4mu'], subdir, timestampdirs[version])
        json_4mu[key] = [f for f in eosfindfile(latest) if '/failed/' not in f]
    for subdir in eosls(EOSPATH_SIG2['2mu2e']):
        key = subdir.split('_ctau')[0]  # mXX-100_mA-5_lxy-0p3
        timestampdirs = eosls(join(EOSPATH_SIG2['2mu2e'], subdir))
        timestampdirs = sorted(timestampdirs, key=lambda x: datetime.strptime(x, "%y%m%d_%H%M%S"))
        latest = join(EOSPATH_SIG2['2mu2e'], subdir, timestampdirs[version])
        json_2mu2e[key] = [f for f in eosfindfile(latest) if '/failed/' not in f]

    with open(f'signal_4mu_v2{version}.json', 'w') as outf:
        outf.write(json.dumps(json_4mu, indent=4))
    with open(f'signal_2mu2e_v2{version}.json', 'w') as outf:
        outf.write(json.dumps(json_2mu2e, indent=4))


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

    with open("backgrounds_v2.json", "w") as outf:
        outf.write(json.dumps(generated, indent=4))


def generate_skimmed_background_json(outputname="skimmedbackgrounds_v2"):

    generated = dict()
    for group in EOSPATHS_BKGSKIM:
        generated[group] = {}
        for tag in EOSPATHS_BKGSKIM[group]:
            generated[group][tag] = []
            for path in EOSPATHS_BKGSKIM[group][tag]:
                timestampdirs = eosls(path)
                timestampdirs = sorted(
                    timestampdirs, key=lambda x: datetime.strptime(x, "%y%m%d_%H%M%S")
                )
                latest = join(path, timestampdirs[-1])
                for filepath in eosfindfile(latest):
                    if "/failed/" in filepath:
                        continue  # filter out those in *failed* folder
                    generated[group][tag].append(filepath)

    with open(f"{outputname}.json", "w") as outf:
        outf.write(json.dumps(generated, indent=4))


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

    bkgfilelist = json.load(open("backgrounds_v2.json"))
    generated = dict()
    for group in bkgfilelist.keys():
        generated[group] = {}

        for tag in bkgfilelist[group]:
            xsec = BKG_XSEC[group][tag]
            sumgenwgt = total_genwgt_sum(bkgfilelist[group][tag])
            generated[group][tag] = xsec / sumgenwgt
#             nevents = total_event_number(bkgfilelist[group][tag])
#             generated[group][tag] = xsec / nevents

    with open("backgrounds_scale_v2.json", "w") as outf:
        outf.write(json.dumps(generated, indent=4))


def remove_empty_file(filepath):
    """given a file, if the tree has non-zero number of events, return filepath"""
    f_ = uproot.open(filepath)
    key_ = f_.allkeys(filtername=lambda k: k.endswith(b"ffNtuple"))
    if key_ and uproot.open(filepath)[key_[0]].numentries != 0:
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
    bkgfilelist = json.load(open("backgrounds_v2.json"))
    for group in bkgfilelist:
        for tag in bkgfilelist[group]:
            files = bkgfilelist[group][tag]
            bkgfilelist[group][tag] = remove_empty_files(files)
    with open("backgrounds_nonempty_v2.json", "w") as outf:
        outf.write(json.dumps(bkgfilelist, indent=4))



if __name__ == "__main__":

    import sys
    ## Here we are only keeping the most recent submission batch
    if sys.argv[1]=='bkg':
        generate_background_json()
        generate_background_scale()
        clean_background_json()

    if sys.argv[1]=='sig':
        version = -1
        if len(sys.argv)==3:
            version = int(sys.argv[2])
        generate_signal_json(version)

    if sys.argv[1]=='bkgskim':
        if len(sys.argv) == 2:
            generate_skimmed_background_json()
        else:
            generate_skimmed_background_json(outputname=sys.argv[2])