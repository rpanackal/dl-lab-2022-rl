#!/bin/bash

echo "Started"

python test.py -hl 1 -l 0.001
echo "task1 done"

python test.py -hl 3 -l 0.001
echo "task2 done"

python test.py -hl 5 -l 0.001
echo "task3 done"

python test.py -hl 1 -l 0.0001
echo "task4 done"

python test.py -hl 3 -l 0.0001
echo "task5 done"

python test.py -hl 5 -l 0.0001
echo "task6 done"

python test.py -hl 1 -l 0.00001
echo "task7 done"

python test.py -hl 3 -l 0.00001
echo "task8 done"

python test.py -hl 5 -l 0.00001
echo "task9 done"


