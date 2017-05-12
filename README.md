# Course project of DS-GA 1003 

# Machine Learning and Statistical Computation

This project is mainly about predicting the helpfulness of a product review.

### Data

Helpfulness evaluation is similar to [sentiment analysis](deeplearning.net/tutorial/lstm.html). These two tasks can use the same model. Due to the privacy of the data, we choose not to mention much about the dataset we use. However, IMDB dataset is open sourced [here](http://www.iro.umontreal.ca/~lisa/deep/data/imdb.pkl). For convenience, we have formatted IMDB data to well fit into our program. The data has been separated into three sets, and uploaded to [Dropbox](https://www.dropbox.com/s/7a2zjwp0iq1m629/aclImdb.tar.gz?dl=0). 

As a result, the IMDB data, for the binary classification task, can reach 0.86 accuracy. More details are in *report.pdf*.

### Files

In *src* folder, there are many python files, most of which are data processing and some are deprecated. We will skip those. And we will talk about RNN model files.

* process.py: split and process data, generate vocabulary and convert word indices.

* rnn.py: build three versions of RNN model, use different tricks.

* main.py: training and evaluation of RNN model.

### Run the model

The model is hard coded (Sorry for that). You may need to change some configurations If you like. But be aware that some parameter settings may lead to bad result.

If you use IMDB data we provide, note change the following code

```python
vocab_size = len(process.get_vocab()) + 1
```

to 

```python
vocab_size = 20000 + 1
```

because we set vocabulary size as 20000 as we generate and reformat the data.

Finally, run in the terminal

```python
python main.py
```

