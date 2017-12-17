'''Main file to run the setup.'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys

from keras.callbacks import TensorBoard
import numpy as np
import pandas as pd
import sklearn
import subprocess
import tensorflow as tf
# import tqdm

sys.path.insert(0, '../')
from rank_text_cnn import config
from rank_text_cnn.code.batch_generator import batch_gen
from rank_text_cnn.code import model, utils


def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()


def train_model(mode):
    '''Train the model.
    1. Read numpy arrays for input data
    2. Batch train the model
    3. Calculate map scores using our method.
    4. Dump predicted values in csv format for evaluation using Trec-eval
    '''
    if mode not in ['TRAIN-ALL', 'TRAIN']:
        print 'Invalid mode'
        return

    print 'Training on %s set' %(mode)
    data_dir = os.path.join(config.OUTPUT_PATH, mode)

    # Load train set.
    q_train = np.load(
        os.path.join(data_dir, '%s.questions.npy' %(mode.lower())))
    a_train = np.load(os.path.join(data_dir, '%s.answers.npy' %(mode.lower())))
    y_train = np.load(os.path.join(data_dir, '%s.labels.npy' %(mode.lower())))
    qids_train = np.load(os.path.join(data_dir, '%s.qids.npy' %(mode.lower())))
    addn_feat_train = np.zeros(y_train.shape)
    print '''q_train.shape, q_train.shape, y_train.shape, qids_train.shape,
             addn_feat_train.shape: ''',
    print(q_train.shape, q_train.shape, y_train.shape, qids_train.shape,
          addn_feat_train.shape)

    # Load dev and test sets.
    q_dev = np.load(os.path.join(data_dir, 'dev.questions.npy'))
    a_dev = np.load(os.path.join(data_dir, 'dev.answers.npy'))
    y_dev = np.load(os.path.join(data_dir, 'dev.labels.npy'))
    qids_dev = np.load(os.path.join(data_dir, 'dev.qids.npy'))
    addn_feat_dev = np.zeros(y_dev.shape)

    q_test = np.load(os.path.join(data_dir, 'test.questions.npy'))
    a_test = np.load(os.path.join(data_dir, 'test.answers.npy'))
    y_test = np.load(os.path.join(data_dir, 'test.labels.npy'))
    qids_test = np.load(os.path.join(data_dir, 'test.qids.npy'))
    addn_feat_test = np.zeros(y_test.shape)

    vocab = utils.load_json(config.VOCAB_PATH)
    max_ques_len = q_train.shape[1]
    max_ans_len = a_train.shape[1]
    embedding, embed_dim, _ = model.load_embeddings(
        config.EMBEDDING_PATH, vocab)
    # Only zeros
    addit_feat_len = 1
    if addn_feat_train.ndim > 1:
        addit_feat_len = addn_feat_train.shape[1]

    # Get model
    cnn_model = model.cnn_model(embed_dim, max_ques_len, max_ans_len,
                                len(vocab), embedding,
                                addit_feat_len=addit_feat_len)

    def testing_things():
        ques_conv = cnn_model.get_layer('ques_conv')
        print ques_conv
        print(ques_conv.input, ques_conv.output, ques_conv.input_shape,
              ques_conv.output_shape)
        ques_pool = cnn_model.get_layer('ques_pool')
        print ques_pool
        print ques_pool.input_shape, ques_pool.output_shape
    # testing_things()
    
    bs = config.BATCH_SIZE
    np.set_printoptions(threshold=np.nan)
    # np.seterr(divide='ignore', invalid='ignore')
    # Train manually, epoch by epoch
    # TODO: Add tqdm
    log_path = './logs'
    callback = TensorBoard(log_path)
    callback.set_model(cnn_model)
    train_names = ['train_loss', 'train_acc']
    dev_names = ['dev_loss', 'dev_acc']

    for epoch in range(config.EPOCHS):
        # Obtain a shuffled batch of the samples
        q_train, a_train, y_train, addn_feat_train = sklearn.utils.shuffle(
            q_train, a_train, y_train, addn_feat_train,
            random_state=config.RANDOM_STATE)
        # print y_train.shape, np.where(y_train == 1)[0].shape
        # break
        # history = cnn_model.fit([q_train, a_train], y_train, batch_size=bs,
                                  # epochs=5, shuffle=False)
        # print 'Loss, Acc: %f %f' %(loss, acc)
        batch_no = 0
        for b_q_train, b_a_train, b_y_train, b_addn_feat_train in zip(
                batch_gen(q_train, bs), batch_gen(a_train, bs),
                batch_gen(y_train, bs), batch_gen(addn_feat_train, bs)):
            logs = cnn_model.train_on_batch(
                [b_q_train, b_a_train, b_addn_feat_train], b_y_train)
            if batch_no%10 == 0:
                write_log(callback, train_names, logs, batch_no*(epoch+1))
            batch_no += 1
            # sys.stdout.write('\r Loss, Acc: %f %f' %(loss, acc))
            # sys.stdout.flush()


        # Predict result on dev set
        y_pred = cnn_model.predict([q_dev, a_dev, addn_feat_dev])
        # print y_pred
        # print y_pred.shape, np.where(y_pred == 0)[0].shape
        dev_acc = sklearn.metrics.roc_auc_score(y_dev, y_pred)
        # TODO: add checking if best accuracy has been reached
        dev_map = utils.map_score(qids_dev, y_test, y_pred)
        # write_log(callback, dev_names, logs, epoch)
        print ' Dev AUC: %f, MAP: %f' %(dev_acc, dev_map)

    y_pred = cnn_model.predict([q_test, a_test, addn_feat_test])
    test_acc = sklearn.metrics.roc_auc_score(y_test, y_pred)
    test_map = utils.map_score(qids_test, y_test, y_pred)
    print 'Test AUC: %f, MAP: %f' %(test_acc, test_map)

    # Dump data for trec eval
    N = len(y_pred)
    nnet_outdir = 'output/'

    df_submission = pd.DataFrame(index=np.arange(N), columns=['qid', 'iter', 'docno', 'rank', 'sim', 'run_id'])
    df_submission['qid'] = qids_test
    df_submission['iter'] = 0
    df_submission['docno'] = np.arange(N)
    df_submission['rank'] = 0
    df_submission['sim'] = y_pred
    df_submission['run_id'] = 'nnet'
    df_submission.to_csv(os.path.join(nnet_outdir, 'submission.txt'), header=False, index=False, sep=' ')

    df_gold = pd.DataFrame(index=np.arange(N), columns=['qid', 'iter', 'docno', 'rel'])
    df_gold['qid'] = qids_test
    df_gold['iter'] = 0
    df_gold['docno'] = np.arange(N)
    df_gold['rel'] = y_test
    df_gold.to_csv(os.path.join(nnet_outdir, 'gold.txt'), header=False, index=False, sep=' ')

    subprocess.call("/bin/sh eval/run_eval.sh '{}'".format(nnet_outdir), shell=True)


def main():
    train_model(mode='TRAIN')


if __name__ == '__main__':
    main()
