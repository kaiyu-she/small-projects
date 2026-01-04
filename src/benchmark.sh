#!/usr/bin/env bash
set -e
FILENAME="pi-benchmark.csv"

if [ -e $FILENAME ]; then
    echo "$FILENAME already exists"
    exit 1
fi

echo "prec,time" > $FILENAME
for prec in {0..1000000..100}; do
    [ $((prec % 10000)) -eq 0 ] && echo "prec $prec"
    echo "$prec,$(./pi -tq $prec)" >> $FILENAME
done
