# usage: ./run_responsibly  max runs

mkdir -p scheduled_private
alias python=python3

bash get_all_bad_ones.bash >  /dev/null
python update_excluded_nodes.py

for i in `ls waiting_private/`; do
    USER=`whoami`
    NUM_JOBS=`squeue | grep $USER | wc -l`

    if  ! ls -a scheduled_private | grep -q $i; then

        # echo "Not found $i. Possibly Running"
        if [ $NUM_JOBS -lt  $1 ]; then
            # echo "$NUM_JOBS JOBS LESS $1, will definitely run $i"
            sbatch waiting_private/$i
            mv waiting_private/$i scheduled_private/$i
            # echo "Ran $i"
        else
            # echo "$NUM_JOBS bigger $1. Will not run $i"
            break
        fi

    fi
done