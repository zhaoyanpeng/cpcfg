# A Fast Implementation of C-PCFGs

**Update (09/30/2021):** I have been working on a better implementation of XCFG models. Checkout the [beta](https://github.com/zhaoyanpeng/cpcfg/tree/beta) branch and give a try!

## Model
This implementation reaches an average sentence-level F1 **56%**, slightly higher than Yoon's **55.2%**. 
Notably, it takes only **25** minutes per epoch on a GeForce GTX 1080 Ti. ~~I will release a report in a couple of
days.~~ Here is the report: [An Empirical Study of Compound PCFGs](https://arxiv.org/abs/2103.02298).

## Data
I am using the same data processing as in Yoon's [code](https://github.com/harvardnlp/compound-pcfg#data). If you are looking for a **unified data pipeline** for [WSJ](https://catalog.ldc.upenn.edu/LDC99T42), [CTB](https://catalog.ldc.upenn.edu/LDC2005T01), and [SPMRL](https://dokufarm.phil.hhu.de/spmrl2014/), I suggest you have a look at [XCFG](https://github.com/zhaoyanpeng/xcfg). It makes data creation easier. If you still find it annoying processing all data from scratch, contact me and I can give you access to all processed data, but you must have been granted licenses for these treebanks.

## Mean sentence-level F1 numbers
Here is an overview of model performance on WSJ, CTB, and SPMRL. Find more details in the [report](https://arxiv.org/abs/2103.02298).

<details><summary>On WSJ and CTB</summary></details>

| Model | WSJ | CTB |
|:-:|:-|:-|
| Yoon's | 55.2 | 36.0 |
| This Repo | 55.7<sub>±1.3<sub> | 35.1<sub>±6.1<sub> |

<details><summary>On SPMRL</summary><p>

| Model | Basque | German | French | Hebrew | Hungarian | Korean | Polish | Swedish |
|:-:|:-|:-|:-|:-|:-|:-|:-|:-|
| N-PCFG | **30.2**<sub>±0.9<sub> | **37.8**<sub>±1.7<sub> | **42.2**<sub>±1.4<sub> | **41.0**<sub>±0.6<sub> | 37.9<sub>±0.8<sub> | 25.7<sub>±2.8<sub> | 31.7<sub>±1.8<sub> | 14.5<sub>±12.7<sub> |
| C-PCFG | 27.9<sub>±2.0<sub> | 37.3<sub>±1.8<sub> | 40.5<sub>±0.8<sub> | 39.2<sub>±1.2<sub> | **38.3**<sub>±0.7<sub> | **27.7**<sub>±2.8<sub> | **32.4**<sub>±1.1<sub> | **23.7**<sub>±14.3<sub> |

</p></details>

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
If you use the fast implementation of C-PCFGs in your research or wish to refer to the results in the [report](https://arxiv.org/abs/2103.02298), please use the following BibTeX entries.
```
@inproceedings{zhao-titov-2021-empirical,
    title = "An Empirical Study of Compound {PCFG}s",
    author = "Zhao, Yanpeng and Titov, Ivan",
    booktitle = "Proceedings of the Second Workshop on Domain Adaptation for NLP",
    month = apr,
    year = "2021",
    address = "Kyiv, Ukraine",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.adaptnlp-1.17",
    pages = "166--171",
}
```
```
@inproceedings{kim2019compound,
  title    = {Compound Probabilistic Context-Free Grammars for Grammar Induction},
  author   = {Kim, Yoon and Dyer, Chris and Rush, Alexander},
  booktitle= {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
  address  = {Florence, Italy},
  publisher= {Association for Computational Linguistics},
  url      = {https://www.aclweb.org/anthology/P19-1228},
  doi      = {10.18653/v1/P19-1228},
  pages    = {2369--2385},
  month    = {jul},
  year     = {2019},
}
```
```
@inproceedings{rush2020torch,
  title    = {Torch-Struct: Deep Structured Prediction Library},
  author   = {Rush, Alexander},
  booktitle= {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics: System Demonstrations},
  address  = {Online},
  publisher= {Association for Computational Linguistics},
  url      = {https://www.aclweb.org/anthology/2020.acl-demos.38},
  doi      = {10.18653/v1/2020.acl-demos.38},
  pages    = {335--342},
  month    = {jul},
  year     = {2020},
}
```

## Acknowledgements
This repo is developed based on [C-PCFGs](https://github.com/harvardnlp/compound-pcfg) and [Torch-Struct](https://github.com/harvardnlp/pytorch-struct).

## License
MIT
