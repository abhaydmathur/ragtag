# ragtag

## Usage

Our repository contains submodules, linking to the datasets made available for the hackathon. To reproduce our results
it is thus required to clone this repository with the recursion flag, as below,

```bash
git clone --recurse-submodules git@github.com:abhaydmathur/ragtag.git
cd qallm & pip install .
```

```bash
usage: qallm [-h] [-c file] [-i question] [-o file] [-v]

Adversarial machine learning robustness evaluations for ML based N-IDS.

options:
  -h, --help            show this help message and exit
  -c file, --csv file   relative path to the csv file configuration the questions to be answered. If not defined, it is required to define an input
                        question as a string.
  -i question, --input question
                        String input question to be answered. If not defined, it is required to define an input csv file.
  -o file, --output file
                        relative path to the file in which the output shall be written to. If not defined, the prediction results will be displayed
                        on the standard output.
  -v, --verbose         Set logging level to INFO to obtain information about the internal program execution. Without the verbose flag, only Errors
                        are diplayed in stdout.
```
