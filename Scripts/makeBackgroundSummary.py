#!/usr/bin/env python
"""Text filing dataset info returned by dasgoclient queries
"""
import json
import os
import shlex
import socket
import subprocess
import sys
import time
import os

import yaml

hostname = socket.gethostname()
if 'lxplus' in hostname:
    OUTPUT_FILE = "/eos/user/w/wsi/www/lpcdm/backgroundsummary.txt"
    OUTPUT_URL = "https://wsi.web.cern.ch/wsi/lpcdm/backgroundsummary.txt"
elif 'cmslpc' in hostname:
    OUTPUT_FILE = "/publicweb/w/wsi/public/lpcdm/backgroundsummary.txt"
    OUTPUT_URL = "http://home.fnal.gov/~wsi/public/lpcdm/backgroundsummary.txt"
else:
    sys.exit('Not on lxplus nor cmslpc platforms -> exiting')

DATASET_GRP = yaml.load(open("backgroundlist.yml"), Loader=yaml.FullLoader)


# https://stackoverflow.com/questions/1094841/reusable-library-to-get-human-readable-version-of-file-size
def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, "Yi", suffix)


def main():

    ## check certificate
    if subprocess.call(shlex.split('voms-proxy-info -exists')) != 0:
        os.system("voms-proxy-init -voms cms -valid 192:00")

    with open(OUTPUT_FILE, "w") as outf:
        outf.write("Generated at {}\n".format(time.ctime()))
        outf.write("=" * 100 + "\n\n")
        fmt = "{:145}{:>10}{:>10}{:>10}  {}"
        outf.write(fmt.format("Dataset", "#Events", "#Files", "size", "sites(disk only)") + "\n")

        starttime = time.time()
        starttimec = time.time()

        print('='*60)
        print('**', time.asctime())
        for categ, datasets in DATASET_GRP.items():

            outf.write(f"+++ {categ} +++\n")
            print('{:25}'.format(f"+++ {categ} +++ [{len(datasets)}]"), end="\t")
            outf.write("-" * 160 + "\n")
            for ds in datasets:
                nFiles, nEvents, fileSize = (-1, -1, -1)
                sites = set()
                transfering = {}
                try:
                    dasquery = 'dasgoclient -query="summary dataset={}" -json'.format(ds)
                    dasres = subprocess.check_output(shlex.split((dasquery)))
                    dasres = json.loads(dasres)[0]

                    dasquery_site = 'dasgoclient -query="site dataset={}" -json'.format(ds)
                    dasres_site = subprocess.check_output(shlex.split((dasquery_site)))
                    dasres_site = json.loads(dasres_site.decode())

                    nFiles = dasres["summary"][0]["num_file"]
                    nEvents = dasres["summary"][0]["num_event"]
                    fileSize = dasres["summary"][0]["file_size"]
                    _sites = set()
                    _transfering = {}
                    for siteinfo in dasres_site:
                        if not siteinfo["das"]["services"][0].startswith('combined'): continue
                        if siteinfo["site"][0]["kind"]!="Disk": continue
                        _sitename = siteinfo["site"][0]["name"]

                        if (
                            siteinfo["site"][0]["block_completion"] != "100.00%"
                            or siteinfo["site"][0]["block_fraction"] != "100.00%"
                            or siteinfo["site"][0]["dataset_fraction"] != "100.00%"
                            or siteinfo["site"][0]["replica_fraction"] != "100.00%"
                        ):
                            _transfering[_sitename] = (siteinfo["site"][0]["block_fraction"], siteinfo["site"][0]["dataset_fraction"])
                        else:
                            _sites.add(_sitename)

                    sites.update(_sites)
                    transfering.update(_transfering)

                except Exception:
                    ds += " **"
                    sites.clear()
                    transfering = {}
                # if none available on disk, show transfer fraction if any
                if (not sites) and transfering:
                    sitesinfo = str(list(sites))+str(transfering)
                else:
                    sitesinfo = str(list(sites))
                outf.write(fmt.format(ds, nEvents, nFiles, sizeof_fmt(fileSize), sitesinfo) + "\n")
            outf.write("-" * 160 + "\n")
            print("--> took {:.3f}s".format(time.time() - starttimec))
            starttimec = time.time()

    print('_'*60)
    print("--> took {:.3f}s".format(time.time() - starttime))
    print("Write summary at ", OUTPUT_FILE)
    print("Please visit", OUTPUT_URL)


if __name__ == "__main__":
    main()
