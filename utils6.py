# -*- coding: utf-8 -*-
# @Time    : 2020/8/21 3:56 AM
# @Author  : Yinghao Qin
# @Email   : y.qin@hss18.qmul.ac.uk
# @File    : utils6.py
# @Software: PyCharm


#######################################################################################
# 'utils6' is used to calculate the Levenshtein distance between the predicted results#
# and ground truth.                                                                   #
#######################################################################################


def LevenshteinDistance(a, b):
    """
    The Levenshtein distance is a metric for measuring the difference between two sequences.
    Calculates the Levenshtein distance between a and b.
    :param a: the first sequence, such as [1, 2, 3, 4]
    :param b: the second sequence
    :return: Levenshtein distance
    """
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a, b = b, a
        n, m = m, n

    current = range(n + 1)
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete = previous[j] + 1, current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change = change + 1
            current[j] = min(add, delete, change)
    if current[n]<0:
        return 0
    else:
        return current[n]


# x = [0, 23, 4, 5]
# y = [0, 24, 4, 5, 7]
# print(LevenshteinDistance(x, y))
