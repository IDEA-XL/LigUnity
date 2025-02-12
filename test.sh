batch_size=256

TASK=${1}
arch=${2}
weight_path=${3}
results_path=${4}
echo "writing to ${results_path}"

mkdir -p $results_path
python ./unimol/test.py "./test_datasets" --user-dir ./unimol --valid-subset test \
       --results-path $results_path \
       --num-workers 8 --ddp-backend=c10d --batch-size $batch_size \
       --task test_task --loss rank_softmax --arch $arch \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256  --seed 1 \
       --path $weight_path \
       --log-interval 100 --log-format simple \
       --max-pocket-atoms 511 \
       --test-task $TASK
