#!/usr/bin/env python
"""generate data sample list, until proper sample management tool show up.
"""
import argparse
import json
import os
from datetime import datetime
from os.path import basename, join, relpath

from FireHydrant.Samples.eospaths import *
from FireHydrant.Tools.commonhelpers import eosfindfile, eosls

parser = argparse.ArgumentParser(description="produce data ntuple files")
parser.add_argument("datatype", type=str, nargs=1, choices=["sigmc", "bkgmc", "data"], help="Type of dataset",)
parser.add_argument("--skim", action='store_true', help="make ntuple files from skimmed samples. Only valid when datatype is data/bkgmc")
args = parser.parse_args()


def list_files(dir, pattern=None):
    """
    wrapper of `eos find` subcommand to list files recursively, excluding failed crab job output

    :param str dir: eos directory
    :param str pattern: restrict pattern of file names
    :return: a list of file names
    :rtype: list
    """
    return [f for f in eosfindfile(dir, pattern=pattern) if f and "/failed" not in f]


def latest_files(parentPathOfTimestamps, pattern=None):
    """
    list file names of latest batch job submission

    :param str parentPathOfTimestamps: eos directory up to timestamp level
    :param str pattern: restrict pattern of file names
    :return: a list of file names
    :rtype: list
    """
    try:
        timestampdirs = eosls(parentPathOfTimestamps)
        timestampdirs = sorted(timestampdirs, key=lambda x: datetime.strptime(x, "%y%m%d_%H%M%S"))
        latest = join(parentPathOfTimestamps, timestampdirs[-1])

        return list_files(latest, pattern=pattern)
    except Exception as e:
        print(e)
        print(f"Error when stat eos path: {parentPathOfTimestamps}! Empty list returned")
        return []


def generate_data_files(skim=False):
    print(f"[generate_data_files(skim={skim})]")
    _dirmap = EOSPATHS_DATA_SKIM if skim else EOSPATH_DATA
    return {k: latest_files(v) for k, v in _dirmap.items()}


def generate_background_files(skim=False, forscale=False):
    print(f"[generate_background_files(skim={skim}, forscale={forscale})]")
    _dirmap = EOSPATHS_BKGSKIM if skim else EOSPATHS_BKG
    if forscale:
        _dirmap = EOSPATHS_BKGAOD

    generated = dict()
    for group in _dirmap:
        generated[group] = {}
        for tag in _dirmap[group]:
            generated[group][tag] = []
            for path in _dirmap[group][tag]:
                generated[group][tag].extend(latest_files(path, pattern='*ffNtuple*.root'))

    return generated


def remove_empty_file(filepath):
    """given a file, if the tree has non-zero number of events, return filepath"""
    import uproot
    f_ = uproot.open(filepath)
    key_ = f_.allkeys(filtername=lambda k: k.endswith(b"ffNtuple"))
    if key_ and uproot.open(filepath)[key_[0]].numentries != 0:
        return filepath
    else:
        return None


def remove_empty_files(filelist):
    """given a list of files, return all files with a tree of non-zero number of events"""
    import concurrent.futures
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


def clean_background_files(filedict):
    """parse all background files, remove empty tree files
    """
    print("[clean_background_files]")
    cleaneddict = filedict
    for i, group in enumerate(cleaneddict, start=1):
        print(f"[{i}/{len(cleaneddict)}] {group}")
        for tag in cleaneddict[group]:
            files = cleaneddict[group][tag]
            cleaneddict[group][tag] = remove_empty_files(files)
    return cleaneddict


def processed_genwgt_sum(ntuplefile):
    """Given a ntuplefile path, return the sum of gen weights."""
    import uproot
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
    import concurrent.futures
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


def generate_background_scale(filedict):
    """parse all files to get number of events processed => scale
        scale = xsec/#genwgtsum, scale*lumi-> gen weight
    """
    from FireHydrant.Samples.xsecs import BKG_XSEC

    generated = dict()
    print('[generate_background_scale]')
    for i, group in enumerate(filedict, start=1):
        print(f"[{i}/{len(filedict)}] {group}")
        generated[group] = {}

        for tag in filedict[group]:
            xsec = BKG_XSEC[group][tag]
            sumgenwgt = total_genwgt_sum(filedict[group][tag])
            generated[group][tag] = xsec / sumgenwgt
            # nevents = total_event_number(filedict[group][tag])
            # generated[group][tag] = xsec / nevents

    return generated


def generate_signal_files():
    """generate private signal file list json"""
    print("[generate_signal_files]")
    paramsubdirs = eosls(EOSPATH_SIG)
    json_4mu, json_2mu2e = {}, {}
    for subdir in paramsubdirs:
        if 'MDp-0p8' in subdir or 'MDp-2p5' in subdir:
            continue # skipping unrequested darkphoton mass points

        if '4Mu' in subdir:
            key = subdir.replace('SIDM_BsTo2DpTo4Mu_', '').split('_ctau')[0]\
                        .replace('MBs', 'mXX').replace('MDp', 'mA')
            key += '_lxy-300' # mXX-1000_mA-0p25_lxy-300
            json_4mu[key] = latest_files(join(EOSPATH_SIG, subdir))
        if '2Mu2e' in subdir:
            key = subdir.replace('SIDM_BsTo2DpTo2Mu2e_', '').split('_ctau')[0]\
                        .replace('MBs', 'mXX').replace('MDp', 'mA')
            key += '_lxy-300'
            json_2mu2e[key] = latest_files(join(EOSPATH_SIG, subdir))

    ## samples with new naming
    for subdir in eosls(EOSPATH_SIG2['4mu']):
        key = subdir.split('_ctau')[0]  # mXX-100_mA-5_lxy-0p3
        json_4mu[key] = latest_files(join(EOSPATH_SIG2['4mu'], subdir))
    for subdir in eosls(EOSPATH_SIG2['2mu2e']):
        key = subdir.split('_ctau')[0]  # mXX-100_mA-5_lxy-0p3
        json_2mu2e[key] = latest_files(join(EOSPATH_SIG2['2mu2e'], subdir))

    return json_4mu, json_2mu2e


def processed_event_number(ntuplefile):
    """Given a ntuplefile path, return the number of events it ran over."""
    import uproot
    f_ = uproot.open(ntuplefile)
    key_ = f_.allkeys(filtername=lambda k: k.endswith(b"history"))[0]
    return f_[key_].values[2]  # 0: run, 1: lumi, 2: events


def total_event_number(filelist):
    """Given a list of ntuple files, return the total number of events processed"""
    import concurrent.futures
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


def generate_signal_scale(fl_4mu, fl_2mu2e):

    from FireHydrant.Samples.signalnumbers import genfiltereff, genxsec, darkphotonbr
    print("[generate_signal_scale]")

    filelists = {}
    filelists['2mu2e'] = fl_2mu2e
    filelists['4mu'] = fl_4mu

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
    print("\t> 2mu2e")
    for i, paramtag in enumerate(filelists['2mu2e'], start=1):
        print(f"({i}/{len(filelists['2mu2e'])})", paramtag)
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


    scale_4mu = {}
    print("\t> 4mu")
    for i, paramtag in enumerate(filelists['4mu'], start=1):
        print(f"({i}/{len(filelists['4mu'])})", paramtag)
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

    return scale_4mu, scale_2mu2e


def stage_out_json(outfn, datasets, shortcutdir, shortcutfn):
    with open(outfn, "w") as outf:
        outf.write(json.dumps(datasets, indent=4))
    os.chdir(shortcutdir)
    if os.path.islink(shortcutfn): os.remove(shortcutfn)
    os.symlink(relpath(outfn, shortcutdir), basename(shortcutfn))



if __name__ == "__main__":

    outdir = join(os.getenv('FH_BASE'), 'FireHydrant/Samples/store')
    shortcutdir = join(os.getenv('FH_BASE'), 'FireHydrant/Samples/latest')
    if not os.path.isdir(outdir): os.makedirs(outdir)
    if not os.path.isdir(shortcutdir): os.makedirs(shortcutdir)

    if args.datatype[0] == "data":
        if args.skim:
            datasets = generate_data_files(skim=True)
            outfn = join(outdir, f"skimmed_control_data2018_{datetime.now().strftime('%y%m%d')}.json")
            shortcutfn = join(shortcutdir, 'skimmed_control_data2018.json')
        else:
            datasets = generate_data_files()
            outfn = join(outdir, f"control_data2018_{datetime.now().strftime('%y%m%d')}.json")
            shortcutfn = join(shortcutdir, 'control_data2018.json')

        stage_out_json(outfn, datasets, shortcutdir, shortcutfn)

    if args.datatype[0] == "bkgmc":
        if args.skim:
            datasets = generate_background_files(skim=True)
            outfn = join(outdir, f"skimmed_backgrounds_{datetime.now().strftime('%y%m%d')}.json")
            shortcutfn = join(shortcutdir, 'skimmed_backgrounds.json')
            stage_out_json(outfn, datasets, shortcutdir, shortcutfn)

            scales = generate_background_scale(datasets)
            outfn = join(outdir, f"skimmed_backgrounds_scale_{datetime.now().strftime('%y%m%d')}.json")
            shortcutfn = join(shortcutdir, 'skimmed_backgrounds_scale.json')
            stage_out_json(outfn, datasets, shortcutdir, shortcutfn)
        else:
            datasets_ = generate_data_files()

            scales = generate_background_scale(datasets_)
            outfn = join(outdir, f"backgrounds_scale_{datetime.now().strftime('%y%m%d')}.json")
            shortcutfn = join(shortcutdir, 'backgrounds_scale.json')
            stage_out_json(outfn, datasets, shortcutdir, shortcutfn)

            datasets = clean_background_files(datasets_)
            outfn = join(outdir, f"backgrounds_{datetime.now().strftime('%y%m%d')}.json")
            shortcutfn = join(shortcutdir, 'backgrounds.json')
            stage_out_json(outfn, datasets, shortcutdir, shortcutfn)


    if args.datatype[0] == "sigmc":
        ds_4mu, ds_2mu2e = generate_signal_files()

        outfn = join(outdir, f"signal_4mu_{datetime.now().strftime('%y%m%d')}.json")
        shortcutfn = join(shortcutdir, 'signal_4mu.json')
        stage_out_json(outfn, ds_4mu, shortcutdir, shortcutfn)

        outfn = join(outdir, f"signal_2mu2e_{datetime.now().strftime('%y%m%d')}.json")
        shortcutfn = join(shortcutdir, 'signal_2mu2e.json')
        stage_out_json(outfn, ds_2mu2e, shortcutdir, shortcutfn)

        ## signal scales
        scale_4mu, scale_2mu2e = generate_signal_scale(ds_4mu, ds_2mu2e)

        outfn = join(outdir, f"signal_4mu_scale_{datetime.now().strftime('%y%m%d')}.json")
        shortcutfn = join(shortcutdir, 'signal_4mu_scale.json')
        stage_out_json(outfn, scale_4mu, shortcutdir, shortcutfn)

        outfn = join(outdir, f"signal_2mu2e_scale_{datetime.now().strftime('%y%m%d')}.json")
        shortcutfn = join(shortcutdir, 'signal_2mu2e_scale.json')
        stage_out_json(outfn, scale_2mu2e, shortcutdir, shortcutfn)
