#!/bin/bash

# Train LO/MSO from scratch
#python3 main.py linear --random-init --results-filename "new" --dataset-directory "datasets/scz_decomp"
#python3 main.py BENDR --random-init --results-filename "new" --dataset-directory "datasets/scz_decomp"

# Train LO/MSO from checkpoint
#python3 main.py linear --results-filename "new" --dataset-directory "datasets/scz_decomp"
python3 main.py BENDR --multi-gpu --results-filename "new" --dataset-directory "datasets/scz_decomp2"

# Train LO/MSO from checkpoint with frozen encoder
#python3 main.py linear --freeze-encoder --results-filename "new" --dataset-directory "datasets/scz_decomp"
#python3 main.py BENDR --freeze-encoder --results-filename "new" --dataset-directory "datasets/scz_decomp"

#0.00028420302659993245
#-0.000300692818895227
