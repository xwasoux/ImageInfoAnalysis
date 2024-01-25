# Super Resolution with CNN

## Dataset

```bash
cd dataset
python3 dataset_stl10.py --data_shuffle
```

## Train

```bash
cd code
python3 run.py \
    --do_train \
    --do_eval \
    --epochs 10 \
    --batch_size 32 \
    --in_channels 3 \
    --out_channels 3 \
    --train_data_file ../data/train.pkl \
    --eval_data_file ../data/eval.pkl \
    --test_data_file ../data/test.pkl
```

## Test
```bash
cd code
python3 run.py \
    --do_test \
    --batch_size 32 \
    --in_channels 3 \
    --out_channels 3 \
    --train_data_file ../data/train.pkl \
    --eval_data_file ../data/eval.pkl \
    --test_data_file ../data/test.pkl
```