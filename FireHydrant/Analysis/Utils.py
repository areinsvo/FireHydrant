#!/usr/bin/env python
"""Utilities collection for Analysis"""

def sigsort(param):
    # mXX-1000_mA-0p25_lxy-0p3
    params_ = param.split('_')
    mxx = float(params_[0].split('-')[-1])
    ma  = float(params_[1].split('-')[-1].replace('p', '.'))
    lxy = float(params_[2].split('-')[-1].replace('p', '.'))

    return lxy*1e6 + mxx*1e3 + ma
