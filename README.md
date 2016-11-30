# Snape

Snape is a convenient artificial dataset generator that wraps sklearn's make_classification and make_regression
and then adds in 'realism' features such as complex formating, varying scales, categorical variables,
and missing values.

## Motivation

Snape was primarily created for academic and educational settings.  It has used to create datasets that are unique per
student, per assignment for various homework assignments.  It has also been used to create class wide assessments in
conjunction with 'Kaggle In the Classroom.'

Other users have suggested non-academic uses cases as well, including 'interview screening problems,' model comparison,
etc.

## Installation


### Via Github
```bash
git clone https://github.com/mbernico/snape.git
cd snape
python setup.py install
```
### Via pip
*Coming Soon...*

## Quick Start

Snape can run either as a python module or as a command line application.

### Command Line Usage

#### Creating a Dataset

From the main directory in the git repo:
```bash

python snape/make_dataset.py -c example/config_classification.json
```
Will use the configuration file example/config_classification.json to create an artificial dataset called 'my_dataset'
(which is specified in the json config, more on this later...).

The dataset will consist of three files:
*  my_dataset_train.csv   (80% of the artificial dataset with all dependent and independent variables)
*  my_dataset_test.csv    (20% of the artificial dataset with only the dependent variables present)
*  my_dataset_testkey.csv (the same 20% as _test, including the dependent variables)

The train and test files can be given to a student.  The student can respond with a file of predictions, which can be
scored against the testkey as follows:

#### Scoring a Dataset

```bash
snape/score_dataset.py  -p example/student_predictions.csv  -k example/student_testkey.csv
```
Snape's score_dataset.py will attempt to detect the problem type and then score it, printing some metrics


```
Problem Type Detection: binary
---Binary Classification Score---
             precision    recall  f1-score   support

          0       0.81      0.99      0.89      1601
          1       0.50      0.06      0.11       399

avg / total       0.75      0.80      0.73      2000
```


### Python Module Usage


#### Creating a Dataset
```python
todo
```

#### Scoring a Dataset

```python
todo
````


## Usage Documentation
[Dataset Creation](doc/make_dataset.md)
[Dataset Scoring](doc/score_dataset.md)
[Classification JSON](doc/config_classification.md)
[Regression JSON](doc/config_regression.md)


## Why Snape?
Snape is primarily used for creating complex datasets that *challenge* students and teach defense against the dark
arts of machine learning.  :)


