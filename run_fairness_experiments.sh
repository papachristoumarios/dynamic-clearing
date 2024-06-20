#!/bin/bash

python3 main.py --name synthetic --num_iters 50 --method fractional --gini 0.5 --gini_type sgc -L -1 -B 50 &>synthetic_fairness_sgc &
python3 main.py --name tlc --num_iters 1 --method fractional --gini 0.5 --gini_type sgc -L -1 -B 100 &>tlc_fairness_sgc &
python3 main.py --name venmo --num_iters 10 --method fractional --gini 0.5 --gini_type sgc -L -1 -B 100 --solver SCS &>venmo_fairness_sgc & 
python3 main.py --name safegraph --num_iters 1 --method fractional --gini 0.5 --gini_type sgc -L -1 -B 500000 --solver SCS &>safegraph_fairness_sgc & 

python3 main.py --name synthetic --num_iters 50 --method fractional --gini 0.5 --gini_type standard -L -1 -B 50 &>synthetic_fairness_gc &
python3 main.py --name tlc --num_iters 1 --method fractional --gini 0.5 --gini_type standard -L -1 -B 100 &>tlc_fairness_gc &
python3 main.py --name venmo --num_iters 10 --method fractional --gini 0.5 --gini_type standard -L -1 -B 100 --solver SCS &>venmo_fairness_gc & 
python3 main.py --name safegraph --num_iters 1 --method fractional --gini 0.5 --gini_type standard -L -1 -B 500000 --solver SCS &>safegraph_fairness_gc & 
