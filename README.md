# Code to extract sources from EEG-IP

## How to use this code to compute sources for EEG-IP (resting-state only for now)

First download the script:
```
wget https://raw.githubusercontent.com/christian-oreilly/extract_source_eegip/master/extract_source_eegip.py
```

Change its permissions so that it can be execute:
```
chmod u+x extract_source_eegip.py 
```

You can now get and example of configuration file that specify the various source estimation parameters:
```
./extract_source_eegip.py  --get_example_config
```

To have a look at this file...
```
less config.json 
```

Whatever in this data structure that is in "global" describes parameters used for ad hoc functions defined in `extract_source_eegip.py`, for example for specifying the event definitions. Unfortunatly, these are currently not "user-fliendly" parameters and to see how they are used you need to look at the script. But, good news, you should not need to modify these.

All the other parameters are defines the parameters use for source reconstruction and are defined as imbricated dictionaries that points to the function and parameters that they relate to... for example config["mne"]["minimum_norm"]["make_inverse_operator"]["loose"] point to the parameter "loose" of the `mne.minimum_norm.make_inverse_operator`, which [definition is of course specified in MNE documentation](https://mne.tools/stable/generated/mne.minimum_norm.make_inverse_operator.html). Even if a parameter is present in the configuration file, if it is available for `mne.minimum_norm.make_inverse_operator` (as per MNE documentation), such a parameters can be added to the configuration file and it will be properly taken into account... e.g., 

```
    "mne": {
    [...]
        "minimum_norm": {
            "make_inverse_operator": {
                "loose": 0.0,
                "rank": "full"
            },
    [...]
```
will work just fine!

Then, if you are suspicious about my work, you van always validate that the head models used for source reconstruction looks ok by doing
```
./extract_source_eegip.py --validate_head_models
```
This will generate PNG files like this one:

<a href="https://ibb.co/9WgsvLP"><img src="https://i.ibb.co/2dhg6QJ/coregistration-ANTS18-0-Months3-T-2.png" alt="coregistration-ANTS18-0-Months3-T-2" border="0"></a>

These templates are as defined in this [paper](https://www.sciencedirect.com/science/article/pii/S1053811920311678) and the [corresponding code](https://github.com/christian-oreilly/infant_template_paper).

Now, when you want to get into buisness you can actually perform the source reconstruction:
```
./extract_source_eegip.py --bids_root "/home/christian/Globus" --derivatives_name "test_sources"
```

And if you are not sure what parameters this program accept, just check the help:
```
./extract_source_eegip.py --help
```

## How to use this code to compute sources for EEG-IP (resting-state only for now)

The libraries necesary to run this code can be installed with:
```
pip install numpy tqdm xarray
```
Further, this code use a patch that has not been integrated to MNE yet so it require to install MNE from the corresponding pull request for the time being:
```
pip install git+https://github.com/mne-tools/mne-python.git@refs/pull/8869/merge
```


## How the f*** am I supposed to use these .nc files?

These netCDF files which I would suggest manipulating with the great (albeit coming with a learning curve) XArray package:
```python
import xarray as xr
bids_root = "/home/christian/Globus/london/derivatives/lossless/derivatives/test_sources/"
file_path = "sub-s720/ses-m06/eeg/sub-s720_ses-m06_eeg_qcr_source_labels.nc"
sources = xr.open_dataset(bids_root + file_path).to_array()
print(sources)
```
output:
```
<xarray.DataArray (variable: 1, epoch: 19, region: 67, time: 500)>
array([[[[ 1.87818218e-11, -7.86477259e-12, -1.07909828e-11, ...,
          -6.14985586e-11, -6.89699106e-11, -6.47266245e-11],
         [-9.85630578e-12, -7.67703880e-12, -1.35715053e-11, ...,
          -8.47077000e-11, -8.18195180e-11, -1.00695440e-10],
         [-2.48309142e-11, -5.40558718e-13, -9.11517958e-12, ...,
           4.42311377e-11,  2.83430388e-11,  2.96824192e-11],
         ...,
         [ 2.77010531e-11,  2.48137081e-11,  4.36731266e-11, ...,
           4.57430512e-11,  3.84855392e-11,  2.32374021e-11],
         [-2.34795363e-11, -1.25781721e-11, -6.72145480e-12, ...,
           3.17994500e-11,  2.75076511e-11,  1.23424075e-11],
         [-3.45082468e-12, -2.90525636e-11, -3.51056933e-11, ...,
           6.96156913e-11,  5.95077239e-11,  6.04564566e-11]],

        [[-2.98248306e-11, -7.27313059e-12,  2.55709159e-11, ...,
           1.26124778e-11,  3.13372747e-11,  6.50894168e-11],
         [-1.18105581e-10, -1.08038981e-10, -1.16940162e-10, ...,
           2.24796807e-12,  2.73202508e-11, -8.86329428e-12],
         [ 1.00638224e-11,  2.55094725e-11,  4.81736936e-11, ...,
          -6.37122723e-11, -6.56622378e-11, -7.18291302e-11],
...
         [-3.68139004e-11, -4.55119783e-11, -2.80960131e-11, ...,
          -1.47320304e-10, -1.52234728e-10, -1.36493323e-10],
         [ 3.19664181e-11,  3.85898135e-11,  4.35751504e-11, ...,
          -6.00239103e-11, -4.92587488e-11, -4.43055389e-11],
         [-1.86526812e-11, -3.41365347e-11, -2.59046085e-11, ...,
          -2.66130738e-11, -2.37554182e-11, -4.44844324e-11]],

        [[ 8.59252879e-11,  3.88610829e-11,  4.13191325e-11, ...,
          -9.76343372e-11, -5.99005427e-11, -1.92819985e-11],
         [ 7.86528521e-11,  5.61962141e-11,  3.73763012e-11, ...,
          -4.04052776e-11, -9.57115900e-12, -1.94912722e-11],
         [ 1.38090602e-12,  3.58649194e-12,  2.61644197e-11, ...,
           3.87649661e-12,  5.13089499e-12, -2.89363088e-11],
         ...,
         [-1.21563320e-10, -1.36773237e-10, -1.58955005e-10, ...,
           3.20956187e-11,  1.48786846e-11,  4.10208661e-11],
         [-4.62144332e-11, -3.07841337e-11, -4.83128034e-11, ...,
           7.22809474e-11,  6.84347577e-11,  5.40160803e-11],
         [-5.76975810e-11, -3.92953583e-11, -3.25637409e-11, ...,
           4.66442790e-11,  2.05134879e-11,  3.43952058e-11]]]])
Coordinates:
  * epoch     (epoch) int64 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18
  * region    (region) object 'bankssts-lh' ... 'transversetemporal-rh'
  * time      (time) float64 -0.2 -0.198 -0.196 -0.194 ... 0.794 0.796 0.798
  * variable  (variable) <U29 '__xarray_dataarray_variable__'
```

```python
print(sources.region)
```
ouput:
```
<xarray.DataArray 'region' (region: 67)>
array(['bankssts-lh', 'bankssts-rh', 'caudalanteriorcingulate-lh',
       'caudalanteriorcingulate-rh', 'caudalmiddlefrontal-lh',
       'caudalmiddlefrontal-rh', 'cuneus-lh', 'cuneus-rh', 'entorhinal-lh',
       'frontalpole-lh', 'frontalpole-rh', 'fusiform-lh', 'fusiform-rh',
       'inferiorparietal-lh', 'inferiorparietal-rh', 'inferiortemporal-lh',
       'inferiortemporal-rh', 'insula-lh', 'insula-rh', 'isthmuscingulate-lh',
       'isthmuscingulate-rh', 'lateraloccipital-lh', 'lateraloccipital-rh',
       'lateralorbitofrontal-lh', 'lateralorbitofrontal-rh', 'lingual-lh',
       'lingual-rh', 'medialorbitofrontal-lh', 'medialorbitofrontal-rh',
       'middletemporal-lh', 'middletemporal-rh', 'paracentral-lh',
       'paracentral-rh', 'parahippocampal-lh', 'parahippocampal-rh',
       'parsopercularis-lh', 'parsopercularis-rh', 'parsorbitalis-lh',
       'parsorbitalis-rh', 'parstriangularis-lh', 'parstriangularis-rh',
       'pericalcarine-lh', 'pericalcarine-rh', 'postcentral-lh',
       'postcentral-rh', 'posteriorcingulate-lh', 'posteriorcingulate-rh',
       'precentral-lh', 'precentral-rh', 'precuneus-lh', 'precuneus-rh',
       'rostralanteriorcingulate-lh', 'rostralanteriorcingulate-rh',
       'rostralmiddlefrontal-lh', 'rostralmiddlefrontal-rh',
       'superiorfrontal-lh', 'superiorfrontal-rh', 'superiorparietal-lh',
       'superiorparietal-rh', 'superiortemporal-lh', 'superiortemporal-rh',
       'supramarginal-lh', 'supramarginal-rh', 'temporalpole-lh',
       'temporalpole-rh', 'transversetemporal-lh', 'transversetemporal-rh'],
      dtype=object)
Coordinates:
  * region   (region) object 'bankssts-lh' ... 'transversetemporal-rh'
```

```python
import matplotlib.pyplot as plt
plt.plot(sources.time, sources.sel(region="bankssts-rh").mean("epoch").squeeze())
```
output:
<a href="https://imgbb.com/"><img src="https://i.ibb.co/nq20f0v/index.png" alt="index" border="0"></a>


And, if you insist to suffer, these file should also be manageable in [Matlab](https://www.mathworks.com/help/matlab/ref/ncread.html).

## Using this on Beluga

```bash
wget https://raw.githubusercontent.com/christian-oreilly/extract_source_eegip/master/extract_source_eegip.py

# Load python...
module load python/3.7 qt/5.12.3 vtk/8.1.1

# You should work in a virtual environement when working with python...
# Here I used my "env_mayada" virtual environement but you'll have to
# create one (google python virtual environement if not familiar) if
# you don't have one
source env_mayada/bin/activate

# Install the MNE PR
pip install git+https://github.com/mne-tools/mne-python.git@refs/pull/8869/merge

# Get a config file... I'll use it as-is because its default parameters are reasonable 
./extract_source_eegip.py  --get_example_config

# Get the head models... The source extraction routine would download
# them automatically if they have not already been downloaded, but since
# the source extraction will be performed on a compute node which have 
# no internet access, it needs to be explicitly downloaded before
# getting on the compute note. 
./extract_source_eegip.py --get_head_models

# Create an interactive job. Could be done through 
# srun or sbatch but I want to monitor this task as it runs...
salloc --time=12:0:0  --mem-per-cpu=10G --account=def-emayada

# Source the virtual environment again since we are now on a computing node...
source env_mayada/bin/activate

./extract_source_eegip.py  --derivatives_name "sources"

# I don't need the downloaded json config file anymore...
rm config.json

# I we don't want to keep the head models...
rm -r fs_models
```


## Using precomputed sources

Precomputed sources are available on Beluga as the following derivatives:
```
/project/def-emayada/eegip/[site]/derivatives/lossless/derivatives/sources/
```
