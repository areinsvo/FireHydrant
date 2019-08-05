FireHydrant
===========

Analysis code for SIDM, making use of [Coffea](https://github.com/CoffeaTeam/coffea).

Environment management: [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

Requirements:

- python3
- [Coffea](https://github.com/CoffeaTeam/coffea)
- [xrootd](https://github.com/xrootd/xrootd-python)
- [jupyterlab](https://github.com/jupyterlab/jupyterlab)

---

# Activation/Deactivation

- `conda activate FireHydrant`
- `conda deactivate`

# Start `jupyterlab`

```bash
jupyter lab --no-browser --port=8888 # replace by your favourite port
```

# Setup (@FNAL LPC)

1. Install Miniconda (if you have not install Miniconda yet)

    - go to https://docs.conda.io/en/latest/miniconda.html,
    - get the bash installer script of Linux, Python3.7 64-bit with `wget` or `curl`,
    - then `bash Miniconda3-latest-Linux-x86_64.sh`
    - (you need to type `ENTER` or `yes` at some point, add conda sourcing to your `.bashrc` or equivalent)
    - a quick [reference](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) on managing environments with `conda`.

2. Edit/create .bash_profile file (for Bash users only)
    
    - open `~/.bash_profile` 
    - add the line: `source ~/.bashrc`
    - save and exit

3. Clone repo

    ```bash
    git clone https://github.com/phylsix/FireHydrant.git
    cd FireHydrant
    ```

4. Install packages

    ```bash
    ./setup.sh
    ```
