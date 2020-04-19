# ALBERT-BiLSTM-CRF-NER

Tensorflow 1.x solution of chinese NER task Using ALBERT-BiLSTM-CRF model with Google ALBERT Fine-tuning

**Disclaimer, this project is for techinical learning and has not been used in commercial scenarios.**

You can get the chinese model from [here](https://github.com/google-research/albert).

This project run in python3 and tensorflow 1.x.

## HOW TO USE

You can upload [how-to-use.ipynb](https://github.com/grallage/ALBERT-BiLSTM-CRF-NER/example/how-to-use.ipynb) to [google colab](https://colab.research.google.com/notebooks/welcome.ipynb), or see my blog here.

## UPDATE:
- 2020.4.19 create project 
    
## RAW DATA FORMAT

data/train.txt, and also dev.txt、test.txt dataset is like this:

```
海 O
钓 O
比 O
赛 O
地 O
点 O
在 O
厦 B-LOC
门 I-LOC
与 O
金 B-LOC
门 I-LOC
之 O
间 O
的 O
海 O
域 O
。 O
```

Each line contain a token and a token's label, each sentences divide by blank line(not show here, please check train.txt).

## REFERENCE: 

* [https://github.com/google-research/albert](https://github.com/google-research/albert)

* [https://github.com/macanv/BERT-BiLSTM-CRF-NER](https://github.com/macanv/BERT-BiLSTM-CRF-NER)

* [https://github.com/ProHiryu/albert-chinese-ner](https://github.com/ProHiryu/albert-chinese-ner)
