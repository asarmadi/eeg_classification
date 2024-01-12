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

There is another script for the case when the trials from the test subject is included in the training set. In other words, the training has all the users, but the test set has only 1 subject. The difference between these two sets is the set of trials which have no intersection. Run the following
```
python mat2h5ConversionFull.py
```

## Credits
We thank authors of the following repos for their contributions to our codebase:

* The CapsuleNet implementation is derived from [Pytorch-CapsuleNet](https://github.com/jindongwang/Pytorch-CapsuleNet.git).

* Stockwell implementation is derived from [Stockwell](https://github.com/claudiodsf/stockwell.git)