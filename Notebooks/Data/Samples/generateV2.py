#!/usr/bin/env python
"""generate data sample list, until proper sample management tool show up.
"""
import json
from datetime import datetime
from os.path import join

from FireHydrant.Tools.commonhelpers import eosls, eosfindfile

# This is control region events.
EOSPATHS = dict(
    A="/store/group/lpcmetx/SIDM/ffNtupleV2/2018/DoubleMuon/Run2018A-17Sep2018-v2",
    B="/store/group/lpcmetx/SIDM/ffNtupleV2/2018/DoubleMuon/Run2018B-17Sep2018-v1",
    C="/store/group/lpcmetx/SIDM/ffNtupleV2/2018/DoubleMuon/Run2018C-17Sep2018-v1",
    D="/store/group/lpcmetx/SIDM/ffNtupleV2/2018/DoubleMuon/Run2018D-PromptReco-v2",
)

# This is skimmed control region events.
EOSPATHS_SKIM = dict(
    A="/store/group/lpcmetx/SIDM/ffNtupleV2/Skim/2018/DoubleMuon/Run2018A-17Sep2018-v2",
    B="/store/group/lpcmetx/SIDM/ffNtupleV2/Skim/2018/DoubleMuon/Run2018B-17Sep2018-v1",
    C="/store/group/lpcmetx/SIDM/ffNtupleV2/Skim/2018/DoubleMuon/Run2018C-17Sep2018-v1",
    D="/store/group/lpcmetx/SIDM/ffNtupleV2/Skim/2018/DoubleMuon/Run2018D-PromptReco-v2",
)

REDIRECTOR = "root://cmseos.fnal.gov/"


def list_files(dir):
    "remove crab failed files"
    return [f for f in eosfindfile(dir) if "/failed" not in f]

def latest_files(parentPathOfTimestamps):
    timestampdirs = eosls(parentPathOfTimestamps)
    timestampdirs = sorted(timestampdirs, key=lambda x: datetime.strptime(x, "%y%m%d_%H%M%S"))
    latest = join(parentPathOfTimestamps, timestampdirs[-1])

    return list_files(latest)


if __name__ == "__main__":

    import sys

    if len(sys.argv)==1:
        datasets = {k: latest_files(v) for k, v in EOSPATHS.items()}

        with open("control_data2018_v2.json", "w") as outf:
            outf.write(json.dumps(datasets, indent=4))

    if len(sys.argv)==2 and sys.argv[1]=='skim':
        datasets = {k: latest_files(v) for k, v in EOSPATHS_SKIM.items()}

        with open("skimmed_control_data2018_v2.json", "w") as outf:
            outf.write(json.dumps(datasets, indent=4))
