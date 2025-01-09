batch_size=256

lig_file=${1}
prot_file=${2}
uniprot=${3}
arch=${4}
weight_path=${5}
results_path=${6}
echo "writing to ${results_path}"

mkdir -p $results_path
python ./unimol/test.py "./vocab" --user-dir ./unimol --valid-subset test \
       --results-path $results_path \
       --num-workers 8 --ddp-backend=c10d --batch-size $batch_size \
       --task test_task --loss rank_softmax --arch $arch \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256  --seed 1 \
       --path $weight_path \
       --log-interval 100 --log-format simple \
       --max-pocket-atoms 511 --demo-lig-file $lig_file --demo-prot-file $prot_file --demo-uniprot $uniprot \
       --test-task DEMO
