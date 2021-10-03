# A Fast and Modularized Implementation of C-PCFGs

**Update (03/10/2021):** This is a modularized implementation of C-PCFG. Check out this [branch](https://github.com/zhaoyanpeng/cpcfg/tree/messup) if you are looking for a lightweight implementation.

## Model
This implementation reaches an average sentence-level F1 **56%**, slightly higher than Yoon's **55.2%**. 
Notably, it takes only **25** minutes per epoch on a GeForce GTX 1080 Ti. Here is the report: [An Empirical Study of Compound PCFGs](https://arxiv.org/abs/2103.02298).

## Data
I am using the same data processing as [Yoon](https://github.com/harvardnlp/compound-pcfg#data). If you are looking for a **unified data pipeline** for [WSJ](https://catalog.ldc.upenn.edu/LDC99T42), [CTB](https://catalog.ldc.upenn.edu/LDC2005T01), and [SPMRL](https://dokufarm.phil.hhu.de/spmrl2014/), I suggest you take a look at [XCFG](https://github.com/zhaoyanpeng/xcfg). It makes data creation easier. If you still find it annoying processing all the data from scratch, contact me and I can give you access to all the processed data (please make sure you have acquired licenses for these treebanks).

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
Specify the model saving path `M_ROOT` and the data path `D_ROOT` before running. Check out [XCFG](https://github.com/zhaoyanpeng/xcfg) if you are unsure about how to prepare data.
```shell
python train.py num_gpus=1 eval=False alias_root=$M_ROOT data.data_root=$D_ROOT \
    running.peep_rate=500 running.save_rate=1e9 running.save_epoch=True data.eval_samples=50000 \
    +model/pcfg=default \
    +optimizer=default \
    +data=default \
    +running=default
```

## Parsing 
Inference needs two more runninng parameters than learning: (1) `M_NAME` is the name of an experiment you have run and want to test and (2) `M_FILE` is the name of a model weight file from the experiment you have run.
```shell
python train.py num_gpus=1 eval=True alias_root=$M_ROOT data.data_root=$D_ROOT \
    model_name=$M_NAME model_file=$M_FILE data.eval_samples=50000 data.eval_name=english-test.json \
    +model/pcfg=default \
    +optimizer=default \
    +data=default \
    +running=default
```

## Dependencies
It requires a tailored [Torch-Struct](https://github.com/zhaoyanpeng/pytorch-struct).
```shell
git clone --branch beta https://github.com/zhaoyanpeng/cpcfg.git
cd cpcfg
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
