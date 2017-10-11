# coding: utf-8
'''Utils'''
from collections import defaultdict
import json
import numpy as np
from sklearn import metrics


def ap_score(cands):
    '''cands: (predicted_scores, actual_labels)
    Using: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
    It uses roc-auc and then computes avg-prec.
    '''
    y_true, y_pred = map(list, zip(*cands))
    # print y_true, y_pred
    count = 0
    score = 0
    for i, (y_true, y_pred) in enumerate(cands):
        if y_true > 0:
            count += 1.0
            score += count / (i + 1.0)
    return score / (count + 1e-6)
    # return metrics.average_precision_score(y_true, y_pred)


def map_score(qids, labels, preds):
    '''Method that computes Mean Average Precision for the given input.

    Authors use their custom method to train. Actual benchmark is done using TREC eval.

    Original Code:
    https://github.com/aseveryn/deep-qa/blob/master/run_nnet.py#L403
    Read more about it:
    https://github.com/scikit-learn/scikit-learn/blob/ef5cb84a/sklearn/metrics/ranking.py#L107
    https://makarandtapaswi.wordpress.com/2012/07/02/intuition-behind-average-precision-and-map/
    http://fastml.com/what-you-wanted-to-know-about-mean-average-precision/
    '''

    qid_2_cand = defaultdict(list)
    for qid, label, pred in zip(qids, labels, preds):
        assert pred >= 0 and pred <= 1
        qid_2_cand[qid].append((label, pred))

    avg_precs = []
    for qid, cands in qid_2_cand.iteritems():
        # get average prec score for all cands of qid
        avg_prec = ap_score(sorted(cands, reverse=True, key=lambda x: x[1]))
        avg_precs.append(avg_prec)

    return sum(avg_precs) / len(avg_precs)


def load_json(file_path):
    '''Huh?'''
    return json.load(open(file_path, 'r'))
