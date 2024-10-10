This program is an edited version of the [origin Pytorch implementation of Informer](https://github.com/zhouhaoyi/Informer2020) in the following paper: 
[Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://arxiv.org/abs/2012.07436). 

## Data

The ETT dataset used in the paper can be downloaded in the repo [ETDataset](https://github.com/zhouhaoyi/ETDataset).

## Usage

Commands for training and testing the model with score attention on Dataset ETTh1, ETTh2 and ETTm1 respectively:

```bash
# ETTh1
python -u main.py --model simplifiedinformer --data ETTh1  --freq h

# ETTh2
python -u main.py --model scoreinformer --data ETTh2 --attn prob --freq h --output_attention

# ETTm1
python -u main.py --model scoreinformer --data ETTm1 --attn prob --freq t --output_attention
```

More parameter information please refer to `main.py` and the [origin repository](https://github.com/zhouhaoyi/Informer2020).
