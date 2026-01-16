#!/bin/bash

###  Java file scorer, using cpplint
###  Gauthier HEISS, 24/11/2023
###  gauthier.heiss@esiea.fr
###
###  Usage : ./score.sh [file]
###  If there is no file argument, the scorer will find recurcively all .c/.h files

# Stop on error
#set -e

# Clear "output.txt" files
echo "" > output.txt

# If there is no launching argument, find all files
if [[ $# -eq 0 ]]; then
    # Run checkstyle on all .c/.h recursively, and append results to output.txt
    cpplint --filter=-whitespace/tab,-legal --linelength=120 --recursive . >> output.txt 2>&1
    # Count all line of codes in all .c/.h recurcively, ignoring comments and empty lines
    count=$(find . -type f \( -name "*.c" -o -name "*.h" \) -exec grep -vE '^\s*(//|\*)' {} \; | wc -l)
else
    if [[ -f $1 ]]; then
        # Run checkstyle only on asked file
        cpplint --filter=-whitespace/tab,-legal --linelength=120 --recursive "$1" >> output.txt 2>&1
        # Count lines in the asked file, ignoring comments and empty lines
        count=$(grep -c -vE '^\s*(//|\*)' "$1")
    else
        echo -e "\e[31mNo such file or directory \"$1\"\e[0m"
        echo "---- Final score: ----"
        echo "0.00"
        exit
    fi
fi

# Displaying output.txt
cat output.txt

# Line break
echo -e "\n"

# If there is no code, give 0 as score
if [[ ${count} == 0 ]]; then
    echo -e "\e[31mNo code to analyze. Do you have .c/.h files in this directory or its subdirectories?\e[0m"
    echo "---- Final score: ----"
    echo "0.00"
    exit
fi

# Parse error count
errors=$(grep -oP 'Total errors found: \K[0-9]+' output.txt)

# Displaying errors count
echo "---- Results: ----"
echo "Lines: ${count}"
echo "Errors: ${errors}"

# Compute score
score=$(echo "scale=5; 10 - (${errors} / ${count} * 10)" | bc | awk '{printf "%.2f\n", $0}')

# If score is negative, set to 0
if [[ ${score} == *-* ]]; then
    score="0.00"
fi

# Show score
echo -e "\n---- Final score: ----"
echo "${score}"