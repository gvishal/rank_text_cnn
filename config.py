
# Config file for datasets / parameters
import os
# datasets contains differently named data.
DATASETS = os.environ['VG_DATASETS']
DATA_PATH = ''
OUTPUT_PATH = 'output/'
EMBEDDING_PATH = DATASETS + 'aquaint+wiki.txt.gz.ndim=50.bin'
VOCAB_PATH = OUTPUT_PATH + 'vocab.json'

EPOCHS = 2
RANDOM_STATE = 42
BATCH_SIZE = 50