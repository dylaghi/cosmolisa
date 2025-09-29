# cosmolisa
Source code for the inference of cosmological parameters with LISA observations.

Based on github.com/wdpozzo/cosmolisa

Download this repository, with

```
git clone git@github.com:dylaghi/cosmolisa.git
```

## How to install the code with Conda

Use the environment file `environment.yml` to create the conda environment for cosmoLISA.
Run the following lines from terminal:

```
yes | conda env create -f environment.yml
conda activate cosmolisa_env
conda install -c conda-forge -c pytorch nessai
export LAL_PREFIX=$HOME/.conda/envs/cosmolisa_env
python setup.py install
```

If the installation s completed successfully, the code is ready for use. 

You should be able to view the help message by entering the following command via CLI:

```
cosmoLISA --help
```

To run any analysis, specify the (relative or absolute) path of the config file with the `--config-file` option.
To run the EMRI example:

```
cosmoLISA --config-file config_EMRI.ini
```
