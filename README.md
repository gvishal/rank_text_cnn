# rank-text-cnn

# Intro
This code is an independent implementation of Learning to Rank Short Text Pairs with Convolutional Deep Neural Networks SIGIR'15 in Keras.

# Results
For the provided data `jacana-qa-naacl2013-..`
Without extra features
|  Metric |  Test Set | Score  |
|---|---|---|
|map|TRAIN-ALL|0.6675|
|recip_rank(mrr)|TRAIN-ALL|0.7178|

Reported in paper for `qg-emnlp07-data`
Without extra features
|  Metric |  Test Set | Score  |
|---|---|---|
|map|TRAIN-ALL|.6709|
|recip_rank(mrr)|TRAIN-ALL|0.7280|

# Other Implementations
1. https://github.com/shashankg7/Keras-CNN-QA
2. https://github.com/aseveryn/deep-qa