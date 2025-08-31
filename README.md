# Project Dialogue-Systems: DAIC-WOZ ADR
This repository deals with automatic depression recognition on the DAIC-WOZ data set. It is a project at *Ulm University*, part of the *Dialogue Systems* Course at the *Institute of Communications Engineering*.

The DAIC-WOZ Data Set was collected by the University of Southern California and can be requested for scientific use [on the university website](https://dcapswoz.ict.usc.edu).

# Goal
The Goal of this project is to use audio data from the DAIC-WOZ data set to construct machine learning models, able to predict depression (Major Depressive Disorder) in human individuals.

# Recreating results
The repository consists of standalone python scripts that are meant to be executed one after another.
In order to flawlessly recreate results, the structure of the input data as well as the order in which to execute the python scripts needs to be followed closely. 
For obvious reasons, the interview data is NOT part of the repository. Only the data processing / machine learning code is.

## Virtual environment
Before running any code in this repository, make sure to create a virtual environment, activate it and install all requirements. 
### 1. Create and Activate Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

## Data structure 
The unprocessed data needs to be placed at the directory root as follows:

```text
data/
├── daic_woz/
│   ├── 300_P/
│   │   └── diarisation_participant/
│   │       ├── part_0_62.328_63.178.wav
│   │       ├── part_1_68.978_70.288.wav
│   │       ├── part_2_75.028_78.128.wav
│   │       └── ...
│   ├── 301_P/
│   │   └── diarisation_participant/
│   │       └── ...
```

## Audio concatenation
Once the data is provided in the structure outlined above, run the following script to generate a long audio file per patient. Make sure the venv is activated.
```bash
python src/preprocessing/audio_concatenator.py
```
The results will be saved to `data/created_data/concatenated_diarisation/300_P/300_P_concatenated.wav` an so on. The `data/created_data` directory will be the target directory for all further created data.
