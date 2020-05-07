#!/usr/bin/env python
"""generate data sample list, until proper sample management tool show up.
"""
import json
import concurrent.futures
import uproot
from FireHydrant.Tools.commonhelpers import eosls, eosfindfile

# This is control region events.
EOSPATHS = dict(
    A="/store/user/pimeloni/ffNtuple4trigger/2018/SingleMuon/Run2018A-17Sep2018-v2/190903_170031",
    B="/store/user/pimeloni/ffNtuple4trigger/2018/SingleMuon/Run2018B-17Sep2018-v1/190903_170117",
    C="/store/user/pimeloni/ffNtuple4trigger/2018/SingleMuon/Run2018C-17Sep2018-v1/190903_170204",
    D="/store/user/pimeloni/ffNtuple4trigger/2018/SingleMuon/Run2018D-22Jan2019-v2/190903_170249",
)
REDIRECTOR = "root://cmseos.fnal.gov/"

if __name__ == "__main__":
    
    datasets = {k: eosfindfile(v) for k, v in EOSPATHS.items()}#json.load(open("trigger_data.json"))
    with open("trigger_data.json", "w") as outf:
        outf.write(json.dumps(datasets, indent=4))
   


