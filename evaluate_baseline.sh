#!/bin/bash

pipenv shell 

python -m thesis task=evaluate_baseline benchmark=refact
python -m thesis task=evaluate_baseline benchmark=truthfulqa
python -m thesis task=evaluate_baseline benchmark=haluleval