#!/usr/bin/env python

# background sample xsec
BKG_XSEC = dict(
    TTJets={
        "TTJets": 491.3,
        "TTJets_SingleLeptFromT": 108.5,
        "TTJets_SingleLeptFromTbar": 109.1,
        "TTJets_DiLept": 54.29,
    },
    ST={
        "Top": 34.91,
        "AntiTop": 34.97,
    },
    WJets={
        "WJets_HT-70To100": 1353,
        "WJets_HT-100To200": 1395,
        "WJets_HT-200To400": 407.9,
        "WJets_HT-400To600": 57.48,
        "WJets_HT-600To800": 12.87,
        "WJets_HT-800To1200": 5.366,
        "WJets_HT-1200To2500": 1.074,
        "WJets_HT-2500ToInf": 0.008001,
    },
    DYJetsToLL={
        "DYJetsToLL-M-10to50": 15820,
        "DYJetsToLL_M-50": 5317,
    },
    DiBoson={
        "WW": 75.91,
        "ZZ": 12.14,
        "WZ": 27.55
    },
    TriBoson={
        "WWW": 0.2154,
        "WWZ": 0.1676,
        "WZZ": 0.05701,
        "ZZZ": 0.01473,
        "WZG": 0.04345,
        "WWG": 0.2316,
        "WGG": 2.001,
    },
    QCD={
        "QCD_Pt-15to20": 2799000,
        "QCD_Pt-20to30": 2526000,
        "QCD_Pt-30to50": 1362000,
        "QCD_Pt-50to80": 376600,
        "QCD_Pt-80to120": 88930,
        "QCD_Pt-120to170": 21230,
        "QCD_Pt-170to300": 7055,
        "QCD_Pt-300to470": 619.8,
        "QCD_Pt-470to600": 59.24,
        "QCD_Pt-600to800": 18.19,
        "QCD_Pt-800to1000": 3.271,
        "QCD_Pt-1000toInf": 1.08,
    },
)
