# rank-text-cnn

# Intro
This code is an independent implementation of Learning to Rank Short Text Pairs with Convolutional Deep Neural Networks SIGIR'15 in Keras.

# Results
1. Evaluation done using TREC-Eval and without word overlap features.
2. Trained the model for 15 epochs, setting more or less the same parameters as mentioned in the paper.
3. Used only a single dropout layer, after the Dense layer.
4. Used `sigmoid` as the activation function, instead of `softmax` (mentioned in the paper).

For the provided data `jacana-qa-naacl2013-..`


|  Metric |  Training Set | Score  |
|---|---|---|
|map|TRAIN-ALL|~~0.6675~~ 0.6760|
|recip_rank(mrr)|TRAIN-ALL|~~0.7178~~ 0.7387|

Reported in paper for `qg-emnlp07-data` 

|  Metric |  Test Set | Score  |
|---|---|---|
|map|TRAIN-ALL|0.6709|
|recip_rank(mrr)|TRAIN-ALL|0.7280|

# Implemented
1. Neural architecture as given in the paper, complete with regularization and adadelta, with optimal hyperparameters.
2. Used `parse.py` from the implementations in Other Implementations section.
3. Implemented `map_score()` using `sklearn`.

# PS
1. Accuracy printed while training does not have any relevance.

# Issues / TODO
1. `sklearn.metrics.average_precision_score()` gives a division by zero error and is thus unable to compute the scores, return `nan`.
1. Add early stopping and store parameters with best MAP score on dev set (as per the paper). This will increase the scores a fair bit (The last iteration had a high loss, as compared to some of the previous ones.).
1. Add dropout at a few other places.
1. Use word overlap features.

# Other Implementations
1. https://github.com/shashankg7/Keras-CNN-QA
2. https://github.com/aseveryn/deep-qa
