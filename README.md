# A Fast Implementation of C-PCFGs

## Model
This implementation reaches an average sentence-level F1 **56%**, slightly higher than Yoon's **55.2%**. 
Notably, it takes only **25** minutes per epoch on a GeForce GTX 1080 Ti. ~~I will release a report in a couple of
days.~~ Here is the [report](https://zhaoyanpeng.github.io/files/An%20Empirical%20Study%20of%20Compound%20PCFGs.pdf).

## Data
I am using the same data processing as in Yoon's [code](https://github.com/harvardnlp/compound-pcfg#data). If you are looking for a **unified data pipeline** for [WSJ](https://catalog.ldc.upenn.edu/LDC99T42), [CTB](https://catalog.ldc.upenn.edu/LDC2005T01), and [SPMRL](https://dokufarm.phil.hhu.de/spmrl2014/), I suggest you have a look at [XCFG](https://github.com/zhaoyanpeng/xcfg). It makes data creation easier. If you still find it annoying processing all data from scratch, contact me and I can give you access to all processed data, but you must have been granted licenses for these treebanks.

## Learning 
Replace `DROOT` by your data directory before running.
```shell
python train_fast.py \
  --train_file $DROOT"/ptb-train.pkl" \
  --val_file $DROOT"/ptb-val.pkl" \
  --save_path "./model"
```

## Parsing 
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
```shell
git clone --branch messup https://github.com/zhaoyanpeng/fast-cpcfg.git
cd fast-cpcfg
virtualenv -p python3.7 ./pyenv/oops
source ./pyenv/oops/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
git clone --branch infer_pos_tag https://github.com/zhaoyanpeng/pytorch-struct.git
cd pytorch-struct
pip install -e .
```

## Citation
If you use the fast implementation of C-PCFGs in your research or wish to refer to the results in the [report](https://zhaoyanpeng.github.io/files/An%20Empirical%20Study%20of%20Compound%20PCFGs.pdf), please use the following BibTeX entry.
```
@article{zhao2020xcfg,
  author = {Yanpeng Zhao},
  title  = {An Empirical Study of Compound PCFGs},
  journal= {https://github.com/zhaoyanpeng/cpcfg},
  url    = {https://github.com/zhaoyanpeng/cpcfg},
  year   = {2020}
}
```

## Acknowledgements
This repo is developed based on [C-PCFGs](https://github.com/harvardnlp/compound-pcfg) and [Torch-Struct](https://github.com/harvardnlp/pytorch-struct).

## License
MIT
