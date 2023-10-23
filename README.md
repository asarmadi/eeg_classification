# EEG Classification
In this project, we use collected EEG data from 9 subjects to solve 2 binary classification problems.

## First vs Third thumb
The goal is to see whether EEG data could be used to detect the first thumb and the third thumb.

## Flex vs Extend
This is a binary classification problem, for detection flex vs extention. 

## Installation
You can generate a python virtual environment by running the following command
```
python -m venv .venv
``` 

To activate the environment
Linux:
```
source ./.venv/bin/activate
```
Windows:
```
.\.venv\Scripts\activate
```

The required packages are listed in requirements.txt file and could be installed in the virtual environment by
```
pip install -r requirements.txt
```

## Data Prepration
The data is stored in *.mat files under "data" folder. To convert this data into labeled set of samples with desired input and output shape, you can run 

```
python mat2h5Conversion.py
```
