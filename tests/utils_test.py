# coding: utf-8
'''Test utils methods'''
import numpy as np

import sys
sys.path.insert(0, '../')
from rank_text_cnn.code import utils

def test_map():
    qids = np.array([1, 2, 3])
    labels = np.array([1, 0, 1])
    preds = np.array([0.6, 0.5, 0.7])

    # Answer should be (1/1 + 0 + 1/3)/2 = 0.667
    expected_answer = round((1/1 + 0 + 1.0/3)/2, 3)
    map_score = round(utils.map(qids, labels, preds), 3)
    print expected_answer, map_score
    assert expected_answer == map_score


def main():
    test_map()


if __name__ == '__main__':
    main()