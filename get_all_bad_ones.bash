#!/bin/bash

touch bad_ones_private
truncate -s 0 bad_ones_private
DIR=`pwd`
USER=`whoami`

# 42 89 for bigbatch

for i in `seq 42 89`; do
    NODE="mscluster$i"
    ALLOCATED=`sinfo -N | grep ${NODE} | grep alloc -o `
    echo "sshing into $NODE"
    # ssh-keygen -f $HOME/.ssh/known_hosts -R $NODE
    CODE="bash ${DIR}/is_bad.bash ${DIR} ${NODE} ${ALLOCATED};"
    timeout 30 ssh -oStrictHostKeyChecking=no $NODE $CODE
    exit_status=$?
    if [[ $exit_status -eq 124 ]]; then
        echo -n $NODE, >> $DIR/bad_ones_private;
    fi

done

truncate -s -1 $DIR/bad_ones_private