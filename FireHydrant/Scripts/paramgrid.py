#!/usr/bin/env python
import os

PROC = ["XXTo2ATo4Mu", "XXTo2ATo2Mu2E"]
M_XX = [100, 150, 200, 500, 800, 1000]
M_DP = [0.25, 1.2, 5]
LXY = [0.3, 3, 30, 150, 300]  # cm

rate3d_2d = 0.75


def floatpfy(num):

    num_as_str = "{:5.2}".format(num)
    num = float(num_as_str)

    if abs(int(num) - num) > 10e-7:
        return str(num).replace(".", "p")
    else:
        return str(int(num))


def printParamTable():

    headrow = "{:^20} {:^20} {:^54}".format(
        "m_boundstate [GeV]", "m_darkphoton [GeV]", "ctau [mm]"
    )
    print(headrow)
    print("=" * len(headrow))

    for xx in M_XX:
        for dp in M_DP:
            ctaus = [
                "{:5.3e}".format(2 * dp * lxy / xx / rate3d_2d * 10) for lxy in LXY
            ]
            # line = '{:^20} {:^20} {:<54}'.format(xx, dp, ', '.join(ctaus))
            line = "{:^20} {:^20} {:<54}".format(xx, dp, str([float(x) for x in ctaus]))
            print(line)
        print()

    print("-" * len(headrow))


def printPD():

    pdFormat = "SIDM_{}_mXX-{}_mA-{}_ctau-{}_TuneCP5_13TeV-madgraph"

    for proc_ in PROC:
        for xx_ in M_XX:
            for dp_ in M_DP:
                ctaus = [
                    floatpfy(2 * dp_ * lxy_ / xx_ / rate3d_2d * 10) for lxy_ in LXY
                ]
                for ctau_ in ctaus:
                    _pd = pdFormat.format(proc_, xx_, str(dp_).replace(".", "p"), ctau_)
                    print(_pd)


def writeGenFragments():

    fragmentTemplate = """
import FWCore.ParameterSet.Config as cms

from Configuration.Generator.Pythia8CommonSettings_cfi import *
from Configuration.Generator.MCTunes2017.PythiaCP5Settings_cfi import *
from Configuration.Generator.PSweightsPythia.PythiaPSweightsSettings_cfi import *

# Hadronizer configuration
generator = cms.EDFilter("Pythia8HadronizerFilter",
    maxEventsToPrint = cms.untracked.int32(1),
    pythiaPylistVerbosity = cms.untracked.int32(1),
    filterEfficiency = cms.untracked.double(1.0),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    comEnergy = cms.double(13000.),
    PythiaParameters = cms.PSet(
        pythia8CommonSettingsBlock,
        pythia8CP5SettingsBlock,
        pythia8PSweightsSettingsBlock,
        processParameters = cms.vstring(
            'ParticleDecays:tau0Max = 1000.1',
            'LesHouches:setLifetime = 2',
            '32:tau0 = {CTAU}'
        ),
        parameterSets = cms.vstring('pythia8CommonSettings',
                                    'pythia8CP5Settings',
                                    'pythia8PSweightsSettings',
                                    'processParameters',
                                    )
    )
)

genParticlesForFilter = cms.EDProducer(
    "GenParticleProducer",
    saveBarCodes=cms.untracked.bool(True),
    src=cms.InputTag("generator", "unsmeared"),
    abortOnUnknownPDGCode=cms.untracked.bool(False)
)

genfilter = cms.EDFilter(
    "GenParticleSelector",
    src = cms.InputTag("genParticlesForFilter"),
    cut = cms.string(' && '.join([
        '(abs(pdgId)==11 || abs(pdgId)==13)',
        'abs(eta)<2.4',
        '(vertex.rho<740. && abs(vertex.Z)<960.)',
        'pt>5.',
        'isLastCopy()',
        'isPromptFinalState()',
        'fromHardProcessFinalState()',
    ]))
)
gencount = cms.EDFilter(
    "CandViewCountFilter",
    src = cms.InputTag("genfilter"),
    minNumber = cms.uint32(4)
)

ProductionFilterSequence = cms.Sequence(
    generator * (genParticlesForFilter + genfilter + gencount)
)
"""

    pdFormat = "SIDM_{}_mXX-{}_mA-{}_ctau-{}_TuneCP5_13TeV-madgraph"

    fragmentOutputDir = "generatorFragments"
    if not os.path.exists(fragmentOutputDir):
        os.makedirs(fragmentOutputDir)
    for proc_ in PROC:
        for xx_ in M_XX:
            for dp_ in M_DP:
                ctaus = [
                    floatpfy(2 * dp_ * lxy_ / xx_ / rate3d_2d * 10) for lxy_ in LXY
                ]
                for ctau_ in ctaus:
                    _pd = pdFormat.format(proc_, xx_, str(dp_).replace(".", "p"), ctau_)
                    _fn = os.path.join(fragmentOutputDir, "{}_cff.py".format(_pd))
                    with open(_fn, "w") as f:
                        f.write(fragmentTemplate.format(CTAU=ctau_.replace("p", ".")))


def printGridPackLocation():

    pdFormat = "SIDM_{}_mXX-{}_mA-{}_ctau-{}_TuneCP5_13TeV-madgraph"

    fragmentOutputDir = "generatorFragments"
    if not os.path.exists(fragmentOutputDir):
        os.makedirs(fragmentOutputDir)
    for proc_ in PROC:
        for xx_ in M_XX:
            for dp_ in M_DP:
                ctaus = [
                    floatpfy(2 * dp_ * lxy_ / xx_ / rate3d_2d * 10) for lxy_ in LXY
                ]
                for ctau_ in ctaus:
                    _pd = pdFormat.format(proc_, xx_, str(dp_).replace(".", "p"), ctau_)
                    gridpackName = "{}slc6_amd64_gcc481_CMSSW_7_1_30_tarball.tar.xz".format(
                        _pd.split("ctau")[0]
                    )
                    print("https://wsi.web.cern.ch/wsi/mc/gridpacks/" + gridpackName)


def printCardsURL():

    pdFormat = "SIDM_{}_mXX-{}_mA-{}_ctau-{}_TuneCP5_13TeV-madgraph"
    urlFormat = "https://github.com/phylsix/genproductions/tree/master/bin/MadGraph5_aMCatNLO/cards/production/2017/13TeV/SIDM_LO/{0}/SIDM_{0}_mXX-{1}_mA-{2}"

    fragmentOutputDir = "generatorFragments"
    if not os.path.exists(fragmentOutputDir):
        os.makedirs(fragmentOutputDir)
    for proc_ in PROC:
        for xx_ in M_XX:
            for dp_ in M_DP:
                ctaus = [
                    floatpfy(2 * dp_ * lxy_ / xx_ / rate3d_2d * 10) for lxy_ in LXY
                ]
                for ctau_ in ctaus:
                    _pd = pdFormat.format(proc_, xx_, str(dp_).replace(".", "p"), ctau_)

                    _url = urlFormat.format(proc_, xx_, str(dp_).replace(".", "p"))
                    print(_url)


if __name__ == "__main__":
    writeGenFragments()
