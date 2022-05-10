#!/bin/bash

echo "Started"

python train.py -hl 15 -l 0.00001
echo "task1 done"

python train.py -hl 15 -l 0.01
echo "task2 done"

python train.py -hl 15 -l 0.001
echo "task3 done"

python train.py -hl 15 -l 0.0001
echo "task4 done"

python train.py -hl 1 -l 0.01
echo "task5 done"

python train.py -hl 3 -l 0.01
echo "task6 done"

python train.py -hl 5 -l 0.01
echo "task7 done"


