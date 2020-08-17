# A Fast Implementation of C-PCFGs

## Model
This implementation reaches an average sentence-level F1 **56%**, slightly higher than Yoon's **55.2%**. 
Notably, it takes only **25** minutes per epoch on a GeForce GTX 1080 Ti. I will release a report in a couple of days.

## Data
The same data processing as in Yoon's [code](https://github.com/harvardnlp/compound-pcfg#data).

## Train
Replace `DROOT` by your data directory before running.
```shell
python train_fast.py \
  --train_file $DROOT"/ptb-train.pkl" \
  --val_file $DROOT"/ptb-val.pkl" \
  --save_path "./model"
```

## Test
Specify `MODEL` and `DATA` before running.
```shell
python eval_best.py \
    --model_file $MODEL \
    --data_file  $DATA \
    --out_file "./test.pred" \
    --gold_out_file "./test.gold"
```

## Dependencies
It requires a tailored [Torch-Struct](https://github.com/zhaoyanpeng/pytorch-struct).

## Acknowledgements
This repo is developed based on [C-PCFGs](https://github.com/harvardnlp/compound-pcfg) and [Torch-Struct](https://github.com/harvardnlp/pytorch-struct).

## License
MIT
