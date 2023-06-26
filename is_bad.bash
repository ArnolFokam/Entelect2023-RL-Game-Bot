#!/bin/bash

USED=`nvidia-smi -d MEMORY -q -i 0 | grep Used | head -n 1 | grep -Eo '[0-9]{1,5}'`
# [[ `nvidia-smi | grep -q Error` ]] ||

if `nvidia-smi | grep -q Error`; then 
    echo -n $2, >> $1/bad_ones_private;
else
    if `[[ ! $USED == 1   &&  ! $3 == alloc ]]`; then
        echo -n $2, >> $1/bad_ones_private;
    fi
fi