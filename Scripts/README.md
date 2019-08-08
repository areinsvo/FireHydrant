# FireHydrant/Scripts

Standalone scripts

dasgoclient writes cache in home directory, loses permissions in nohup. Thus just do `python makeBackgroundSummary.py` everytime.

---
Update/install dependencies

```bash
conda env update --file environment.yml
```

### run `scheduledChecker.py`

```bash
nohup python scheduledChecker.py&
```
