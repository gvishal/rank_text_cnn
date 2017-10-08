# coding: utf-8
'''Utils'''
from collections import defaultdict

def map(qids, labels, preds):
    '''Method that computes Mean Average Precision for the given input.
    All inputs are numpy arrays.
    preds: Values between 0 and 1, due to softmax. Round off values to 0 or 1.

    We count a prediction as correct if its predicted score is same as its 
    label. 
    eg. score = 0.6 ~ 1, label = 1: correct
        score = 0.2 ~ 0, label = 0: correct
    See tests.utils_test.py for more

    Read more about it:
    https://makarandtapaswi.wordpress.com/2012/07/02/intuition-behind-average-precision-and-map/
    http://fastml.com/what-you-wanted-to-know-about-mean-average-precision/
    '''

    qid_2_cand = defaultdict(list)
    for qid, label, pred in zip(qids, labels, preds):
        # We are rounding off predicted values which are in range [0, 1]
        assert pred >= 0 and pred <= 1
        pred = 1 if pred >= 0.5 else 0
        qid_2_cand[qid].append((pred, label))

    avg_precs = []
    for qid, cands in qid_2_cand.iteritems():
        avg_prec = 0
        correct_cnt = 0

        for i, (score, label) in enumerate(sorted(cands, reverse=True), 1):
            if score == label:
                correct_cnt += 1
                avg_prec += float(correct_cnt) / i
        # adding small value to prevent div by zero
        avg_precs.append(avg_prec / (correct_cnt + 1e-6))

    map_score = sum(avg_precs) / len(avg_precs)
    return map_score
