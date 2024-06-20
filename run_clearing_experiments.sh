#!/bin/bash

# rm -rf figures
# mkdir -p figures

python3 main.py --name synthetic -B 50 --num_iters 50 --method fractional -L 50 
python3 main.py --name venmo -B 100 --num_iters 50 --method fractional -L 100 

python3 main.py --name tlc -B 100 -L 10 --num_iters 1 --method discrete 
python3 main.py --name safegraph -B 500000 --method discrete

python3 main.py --name synthetic -B 50 --num_iters 50 --method fractional -L 50 --gini 0.5
python3 main.py --name venmo -B 100 --num_iters 50 --method fractional -L 100 --gini 0.5

python3 main.py --name synthetic -B 50 --num_iters 50 --method discrete -L 50
python3 main.py --name venmo -B 100 --num_iters 50 --method discrete -L 100 

python3 main.py --name tlc -B 100 -L 10 --num_iters 50 --method fractional 
python3 main.py --name safegraph -B 500000 --num_iters 1 --method fractional

python3 main.py --name tlc -B 0 -L 0 --num_iters 1 --method fractional
python3 main.py --name tlc -B 100 -L 100 --num_iters 1 --method fractional
python3 main.py --name tlc -B 500 -L 500 --num_iters 1 --method fractional
