# Samples

generate JSON files of ntuples and scales, versioned by date under _store/_, the latest ones are symlinked under _latest/_.

```bash
python generateV2.py data --skim  # skimmed data files
python generateV2.py bkgmc --skim # skimmed bkg files and scales
python generateV2.py sigmc        # signal files and scales
```