#!/usr/bin/env python
"""generate data sample list, until proper sample management tool show up.
"""
import json
from FireHydrant.Tools.commonhelpers import eosls, eosfindfile

# This is control region events.
EOSPATHS = dict(
    A="/store/group/lpcmetx/SIDM/ffNtuple/2018/DoubleMuon/Run2018A-17Sep2018-v2/190821_224704",
    B="/store/group/lpcmetx/SIDM/ffNtuple/2018/DoubleMuon/Run2018B-17Sep2018-v1/190821_224721",
    C="/store/group/lpcmetx/SIDM/ffNtuple/2018/DoubleMuon/Run2018C-17Sep2018-v1/190821_224730",
    D="/store/group/lpcmetx/SIDM/ffNtuple/2018/DoubleMuon/Run2018D-PromptReco-v2/190821_224740",
)
REDIRECTOR = "root://cmseos.fnal.gov/"


if __name__ == "__main__":

    datasets = {k: eosfindfile(v) for k, v in EOSPATHS.items()}

    with open("control_data2018.json", "w") as outf:
        outf.write(json.dumps(datasets, indent=4))
