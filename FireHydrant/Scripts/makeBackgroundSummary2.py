#!/usr/bin/env python2.7
from __future__ import print_function

import json
import shutil
import socket
import ssl
import sys
import time
import urllib2
from collections import defaultdict
from threading import Thread

import yaml
from dbs.apis.dbsClient import DbsApi

hostname = socket.gethostname()
if 'lxplus' in hostname:
    OUTPUT_FILE = "/eos/user/w/wsi/www/lpcdm/backgroundsummary.txt"
    OUTPUT_URL = "https://wsi.web.cern.ch/wsi/lpcdm/backgroundsummary.txt"
elif 'cmslpc' in hostname:
    OUTPUT_FILE = "/publicweb/w/wsi/public/lpcdm/backgroundsummary.txt"
    OUTPUT_URL = "http://home.fnal.gov/~wsi/public/lpcdm/backgroundsummary.txt"
else:
    sys.exit('Not on lxplus nor cmslpc platforms -> exiting')


def lookup_summary(ds):
    """lookup basic information of a dataset from DBS3 API

    :param ds: dataset name
    :type ds: str
    :return: infodict
    :rtype: dict
    """
    dbs3api = DbsApi("https://cmsweb.cern.ch/dbs/prod/global/DBSReader")
    res = dbs3api.listFileSummaries(dataset=ds)
    # [{'num_file': 1599, 'file_size': 10341982224399, 'num_event': 28159421,
    # 'num_lumi': 198621, 'num_block': 234}]
    return res[0]


def lookup_sites(ds):
    """Given a dataset name, lookup site presence percentage from Phedex

    :param ds: dataset name
    :type ds: str
    :return: sitelist
    :rtype: dict
    """
    filelist=set()
    sitelist=defaultdict(float)
    url='https://cmsweb.cern.ch/phedex/datasvc/json/prod/filereplicas?dataset=' + ds
    jstr = urllib2.urlopen(url, context=ssl._create_unverified_context()).read()
    jstr = jstr.replace("\n", " ")
    result = json.loads(jstr)
    for block in result['phedex']['block']:
        for item in block['file']:
            filelist.add(item['name'])
            for replica in item['replica']:
                site, addr = replica['node'], replica['se']
                if site is None: continue
                if addr is None: addr = ""
                sitelist[(site,addr)] += 1
    nfiles_tot = len(filelist)
    for site,addr in sitelist:
        this_percent = float(sitelist[(site,addr)])/float(nfiles_tot)*100
        sitelist[(site,addr)] = this_percent
    return sitelist


def check_tapesite(site):
    """check if a storage site as tape site, if so, return True

    :param site: site
    :type site: str
    :return: if it's a tape site
    :rtype: bool
    """
    return site.startswith('T0') or (site.startswith('T1') and not site.endswith('Disk'))

# https://stackoverflow.com/questions/1094841/reusable-library-to-get-human-readable-version-of-file-size
def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, "Yi", suffix)


def record(dataset, resultdict):
    """make up a record containing needed info for dataset

    :param dataset: dataset name
    :type dataset: str
    :param resultdict: dictionary holding info
    :type resultdict: dict
    """

    res = {}

    _summary = lookup_summary(dataset)
    res['num_event'] = _summary['num_event']
    res['num_file'] = _summary['num_file']
    res['file_size'] = sizeof_fmt(_summary['file_size'])

    _siteinfo = lookup_sites(dataset)
    diskonly = {}
    for k in _siteinfo:
        if check_tapesite((k[0])): continue
        diskonly[k[0].encode('ascii')] = _siteinfo[k]
    fullyAvailable, transferring = [], []
    for s in diskonly:
        if diskonly[s] == 100: fullyAvailable.append(s)
        else: transferring.append(s)

    if fullyAvailable:
        res['sites'] = ' '.join(fullyAvailable)
    else:
        res['sites'] = str({s: '{:.2f}%'.format(diskonly[s]) for s in transferring})

    resultdict[dataset] = res


def manyrecord(datasets, resultdict):
    """make up records for a list of datasets

    :param datasets: datasets names
    :type datasets: list
    :param resultdict: dictionary holding info
    :type resultdict: dict
    """
    threads = [Thread(target=record, args=(d, resultdict)) for d in datasets]
    for t in threads: t.start()
    for t in threads:
        t.join()


def test():
    ds = '/QCD_Pt-15to20_MuEnrichedPt5_TuneCP5_13TeV_pythia8/RunIIAutumn18DRPremix-102X_upgrade2018_realistic_v15-v3/AODSIM'
    print('dataset:', ds)

    res = {}
    record(ds, res)
    print('Standalone:', res)

    res = {}
    manyrecord([ds,], res)
    print('Threading:', res)


def main():
    DATASET_GRP = yaml.load(open("backgroundlist.yml"), Loader=yaml.SafeLoader)
    datasets_flat = []
    for v in DATASET_GRP.values():
        datasets_flat.extend(v)
    recordstore = {}
    manyrecord(datasets_flat, recordstore)

    fmt = "{:145}{:>10}{:>10}{:>10}  {}"
    outf = open('backgroundsummary.log', 'w')
    print("Generated at {}".format(time.ctime()), file=outf)
    print('=' * 140, end='\n\n', file=outf)
    print(fmt.format("Dataset", "#Events", "#Files", "size", "sites(disk only)"), file=outf)
    for cat in sorted(DATASET_GRP):
        print('+++ {} +++'.format(cat), file=outf)
        print('-' * 160, file=outf)
        for ds in DATASET_GRP[cat]:
            print(fmt.format(ds, recordstore[ds]['num_event'],
                             recordstore[ds]['num_file'],
                             recordstore[ds]['file_size'],
                             recordstore[ds]['sites']),
                  file=outf)
        print('-' * 160, file=outf)
    outf.close()
    shutil.move('backgroundsummary.log', OUTPUT_FILE)


if __name__ == "__main__":
    main()
