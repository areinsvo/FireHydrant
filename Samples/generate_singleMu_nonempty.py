#!/usr/bin/env python
"""generate data sample list, until proper sample management tool show up.
"""
import json
import concurrent.futures
import uproot
from FireHydrant.Tools.commonhelpers import eosls, eosfindfile





# This is control region events with L1 triggers and no pt cut
'''
EOSPATHS = dict(
    A="store/user/pimeloni/SIDM/ffNtuple4trigger/2018/SingleMuon/Run2018A-17Sep2018-v2/190903_170031",
    B="store/user/pimeloni/SIDM/ffNtuple4trigger/2018/SingleMuon/Run2018B-17Sep2018-v1/190903_170117",
    C="store/user/pimeloni/SIDM/ffNtuple4trigger/2018/SingleMuon/Run2018C-17Sep2018-v1/190903_170204",
    D="store/user/pimeloni/SIDM/ffNtuple4trigger/2018/SingleMuon/Run2018D-22Jan2019-v2/190903_170249",
)
REDIRECTOR = "root://cmseos.fnal.gov/"
'''





# This is control region events.

EOSPATHS = dict(
    A="/store/user/pimeloni/ffNtuple4trigger/2018/SingleMuon/Run2018A-17Sep2018-v2/190903_170031",
    B="/store/user/pimeloni/ffNtuple4trigger/2018/SingleMuon/Run2018B-17Sep2018-v1/190903_170117",
    C="/store/user/pimeloni/ffNtuple4trigger/2018/SingleMuon/Run2018C-17Sep2018-v1/190903_170204",
    D="/store/user/pimeloni/ffNtuple4trigger/2018/SingleMuon/Run2018D-22Jan2019-v2/190903_170249",
)
REDIRECTOR = "root://cmseos.fnal.gov/"


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


       
if __name__ == "__main__":
    
    """parse all background files, remove empty tree files"""
    datasets = {k: eosfindfile(v) for k, v in EOSPATHS.items()}#json.load(open("trigger_data.json"))
    for group in datasets:
        files = datasets[group]
        datasets[group] = remove_empty_files(files)
    with open("trigger_data_nonempty_no_pt_cut.json", "w") as outf:
        outf.write(json.dumps(datasets, indent=4))
   


