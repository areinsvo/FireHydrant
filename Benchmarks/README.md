FireHydrant Benchmarks
=======

Some benchmark tasks completed with ffNtuples with [Coffea](https://github.com/CoffeaTeam/coffea).

1. [x] Plot leptonJets multiplicity of all events.
2. [x] Plot leading and subleading leptonJet pair deltaPhi for *mXX-100_mA-0p25_lxy-300* signals.
3. [x] Matching leptonJets with gen dark photons (pid=32) by `deltaRCut=0.3`, overlay matched and unmatched leptonJets pT distribution for *mXX-100_mA-0p25_lxy-300* signals.
4. [x] Overlay leading and subleading leptonJet pair invariant mass for *all* signals, in [0, 200] GeV and [0, 1200] GeV range.
5. [x] **Trigger efficiency** | Plot the efficiency of logical OR of [DoubleL2Mu triggers](../FireHydrant/Tools/trigger.py) wrt. the pT of subleading displacedStandAloneMuons which satisfied the condition that *|eta|<2.4 && #stations>1 && normalized Chi2<10*.


### Info

- You can import the [dataset](./Samples/signal_4mu.json) with
    ```python
    import json
    datasets = json.load(open('Samples/signal_4mu.json')) # -> dictionary {'tag': [files]}
    ```
- JaggedCandidateArray can be constructed by
    ```python
    from coffea.analysis_objects import JaggedCandidateArray
    leptonjets = JaggedCandidateArray.candidatesfromcounts(
        df['pfjet_p4'],
        px=df['pfjet_p4.fCoordinates.fX'],
        py=df['pfjet_p4.fCoordinates.fY'],
        pz=df['pfjet_p4.fCoordinates.fZ'],
        energy=df['pfjet_p4.fCoordinates.fT']
    )
    ```
- [Branch names](../Docs/ffBranchNames.md) for ffNtuples.
- [Coffea binder notebooks](https://github.com/CoffeaTeam/coffea/tree/master/binder)
