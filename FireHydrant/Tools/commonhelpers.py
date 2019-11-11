#!/usr/bin/env python
"""system utilities
"""
import subprocess
import shlex


def eosls(eospath, xdirector="root://cmseos.fnal.gov/"):
    """list file on EOS with eos command line tool
    """

    if eospath.startswith("root://"):
        cmd = "eos ls {}".format(eospath)
    else:
        cmd = "eos {0} ls {1}".format(xdirector, eospath)

    try:
        return subprocess.check_output(shlex.split(cmd)).decode().split()
    except:
        print(f"ERROR when calling: {cmd}")
        return []


def eosfindfile(eospath, xdirector="root://cmseos.fnal.gov/", pattern=None):
    """find all files under ``eospath`` with eos command line tool
    """

    if eospath.startswith("root://"):
        if pattern:
            cmd = 'eos find -name "{0}" -f --xurl {1}'.format(pattern, eospath)
        else:
            cmd = "eos find -f --xurl {0}".format(eospath)
    else:
        if pattern:
            cmd = 'eos {0} find -name "{1}" -f --xurl {2}'.format(xdirector, pattern, eospath)
        else:
            cmd = "eos {0} find -f --xurl {1}".format(xdirector, eospath)

    return subprocess.check_output(shlex.split(cmd)).decode().split()
