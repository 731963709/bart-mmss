import codecs
import logging
import os
import subprocess
import sys

from rouge import Rouge


def rouge(reference, candidate, print_log):
    """
    compute the rouge score
    :param reference: reference
    :param candidate: candidate
    :param log_path: path to log
    :param print_log: function to print log
    :param config: configuration
    :return: rouge-2 score
    """

    # check if of equal amount.
    assert len(reference) == len(candidate)

    # use rouge(Rouge155) see https://github.com/pltrdy/rouge ( else: https://hub.fastgit.org/pltrdy/rouge)
    files_rouge = Rouge()
    try:
        scores = files_rouge.get_scores(candidate, reference, avg=True)
        # recall
        recall = [round(scores["rouge-1"]['r'] * 100, 2),
                    round(scores["rouge-2"]['r'] * 100, 2),
                    round(scores["rouge-l"]['r'] * 100, 2)]
        # precision
        precision = [round(scores["rouge-1"]['p'] * 100, 2),
                        round(scores["rouge-2"]['p'] * 100, 2),
                        round(scores["rouge-l"]['p'] * 100, 2)]
        # f score
        f_score = [round(scores["rouge-1"]['f'] * 100, 2),
                    round(scores["rouge-2"]['f'] * 100, 2),
                    round(scores["rouge-l"]['f'] * 100, 2)]
        # print
        print_log("F_measure: %s Recall: %s Precision: %s\n"
                    % (str(f_score), str(recall), str(precision)))

        return f_score[1]
    except Exception as e:
        print(f"error: {e}")
        return 0

def srouge(reference, candidate):
    """
    compute the rouge score
    :param reference: reference
    :param candidate: candidate
    :param log_path: path to log
    :param print_log: function to print log
    :param config: configuration
    :return: rouge-2 score
    """

    # check if of equal amount.
    assert len(reference) == len(candidate)

    # use rouge(Rouge155) see https://github.com/pltrdy/rouge ( else: https://hub.fastgit.org/pltrdy/rouge)
    files_rouge = Rouge()
    try:
        scores = files_rouge.get_scores(candidate, reference, avg=False)
        f_scores = []
        for score in scores:
            f_scores.append({
                "RG1": score["rouge-1"]['f'],
                "RG2": score["rouge-2"]['f'],
                "RGL": score["rouge-l"]['f'],
            })
        return f_scores
    except Exception as e:
        print(f"error: {e}")
        return 0