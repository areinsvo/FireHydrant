#!/bin/bash

source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /uscms/home/wsi/nobackup/lpcdm/CMSSW_10_2_14/src
eval `scramv1 runtime -sh`
source /cvmfs/cms.cern.ch/crab3/crab.sh
cd /uscms/home/wsi/nobackup/lpcdm/FireHydrant/Scripts
export X509_USER_PROXY=/uscms/home/wsi/x509up_u50352
python makeBackgroundSummary2.py