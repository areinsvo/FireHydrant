#!/usr/bin/env python
"""generate data sample list, until proper sample management tool show up.
"""
import json
from FireHydrant.Tools.commonhelpers import eosls, eosfindfile

# This is control region events.
EOSPATHS = dict(
    A="/store/group/lpcmetx/SIDM/ffNtuple/2018/DoubleMuon/Run2018A-17Sep2018-v2/190701_165735",
    B="/store/group/lpcmetx/MCSIDM/ffNtuple/2018/DoubleMuon/Run2018B-17Sep2018-v1/190625_233608",
    C="/store/group/lpcmetx/MCSIDM/ffNtuple/2018/DoubleMuon/Run2018C-17Sep2018-v1/190625_233628",
    D="/store/group/lpcmetx/MCSIDM/ffNtuple/2018/DoubleMuon/Run2018D-PromptReco-v2/190625_233354",
)
REDIRECTOR = "root://cmseos.fnal.gov/"


if __name__ == "__main__":

    datasets = {k: eosfindfile(v) for k, v in EOSPATHS.items()}

    with open("control_data2018.json", "w") as outf:
        outf.write(json.dumps(datasets, indent=4))
