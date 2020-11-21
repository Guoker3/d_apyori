"""
origin : https://github.com/mattzheng/python-Apriori/blob/master/apriori2.py
与apriori唯一不同的是:runApriori()函数，新增了提升度的计算
同时生成的表格之中，有每两对内容的：支持度、置信度、提升度
"""

import preCal
import loadData
import numpy as np

import sys

from itertools import chain, combinations
from collections import defaultdict
from optparse import OptionParser
from tqdm import tqdm
import pandas as pd


def subsets(arr):
    """ Returns non empty subsets of arr"""
    return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])

def returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet):
    """calculates the support for items in the itemSet and returns a subset
    of the itemSet each of whose elements satisfies the minimum support"""
    _itemSet = set()
    localSet = defaultdict(int)

    for item in itemSet:
        # time cost
        for transaction in transactionList:
            if item.issubset(transaction):
                freqSet[item] += 1
                localSet[item] += 1

    for item, count in list(localSet.items()):
        support = float(count) / len(transactionList)

        if support >= minSupport:
            _itemSet.add(item)

    return _itemSet

def returnItemsWithMinSupport_1itemLoop(itemSet, minSupport,freqSet,pre):
    """calculates the support for items in the itemSet and returns a subset
   of the itemSet each of whose elements satisfies the minimum support"""
    _itemSet = set()
    if np.array([x !='atom' for x in pre.mode]).all():
        for item in itemSet:
            support = sum(pre.pre_1item[str(int(list(item)[0]/3))]) / pre.data_inf['row_number']
            if support >= minSupport:
                _itemSet.add(item)
    else:
        pd=dict()
        for di in pre.pre_1item:
            ##TODO(robust) detect the same atom-key from different attribute and dicard the mistake
            pd.update(di)
        for item in itemSet:
            support = sum(pd[list(item)[0]]) / pre.data_inf['row_number']
            if support >= minSupport:
                freqSet[item] = support
                _itemSet.add(item)

    return _itemSet


def joinSet(itemSet, length):
    """Join a set with itself and returns the n-element itemsets"""
    return set([i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length])


def getItemSetTransactionList(data_iterator):
    transactionList = list()
    itemSet = set()
    print('Generate 1-itemSets ... ')
    for record in data_iterator:
        transaction = frozenset(record)
        transactionList.append(transaction)
        for item in transaction:
            itemSet.add(frozenset([item]))  # Generate 1-itemSets
    return itemSet, transactionList


def runApriori(data_iter,preClass, minSupport, minConfidence, minLift=0):
    """
    run the apriori algorithm. data_iter is a record iterator
    Return both:
     - items (tuple, support)
     - rules ((pretuple, posttuple), confidence)
    """
    itemSet, transactionList = getItemSetTransactionList(data_iter)

    freqSet = defaultdict(int)
    largeSet = dict()
    # Global dictionary which stores (key=n-itemSets,value=support)
    # which satisfy minSupport

    assocRules = dict()
    # Dictionary which stores Association Rules

    oneCSet = returnItemsWithMinSupport_1itemLoop(itemSet,minSupport,freqSet,preClass)

    currentLSet = oneCSet
    k = 2
    while (currentLSet != set([])):
        largeSet[k - 1] = currentLSet
        currentLSet = joinSet(currentLSet, k)
        currentCSet = returnItemsWithMinSupport(currentLSet,transactionList,minSupport,freqSet)
        currentLSet = currentCSet
        k = k + 1

    def getSupport(item):
        """local function which Returns the support of an item"""
        return float(freqSet[item]) / len(transactionList)

    toRetRules = []
    print('Calculation the pretuple words and confidence ... ')
    for key, value in list(largeSet.items())[1:]:
        for item in value:
            _subsets = map(frozenset, [x for x in subsets(item)])
            for element in _subsets:
                remain = item.difference(element)
                if len(remain) > 0:
                    confidence = getSupport(item) / getSupport(element)
                    # lift = getSupport(item)/( getSupport(element) * getSupport(remain))
                    lift = confidence / getSupport(remain)
                    self_support = getSupport(item)
                    if self_support >= minSupport:
                        if confidence >= minConfidence:
                            if confidence >= minLift:
                                toRetRules.append(((tuple(element), tuple(remain)), tuple(item),
                                                   self_support, confidence, lift))
    return toRetRules

if __name__ == "__main__":
    t = loadData.d_apyori_cookDataSet()
    dataset = None
    dataset = 'small'
    if dataset == 'small':
        smallDataSet = [[0, 0, 'a',0], [1, 1, 'a',0], [0, 0, 'b',0], [0, 0, 'b',0], [0.1, 0.2, 'a',0], [0.87, 0, 'c',1]]
        t.loadDataSet(smallDataSet, haveHeader=False)
        func_tid1 = t.create_rTOx_DistanceFunc(raw_func='l_sigmoid', section_pick=[-100, 100])
        func_tid2 = t.create_rTOx_DistanceFunc(raw_func='l_atom')
        distFunc = [t.distanceFuncList[func_tid1], t.distanceFuncList[func_tid1], t.distanceFuncList[func_tid2],t.distanceFuncList[func_tid1]]
    # dataset='luntai'
    if dataset == 'luntai':
        t.loadDataSet('test9_11.csv', haveHeader=True, data_set_cut=[0, 100])
        func_tid = t.create_rTOx_DistanceFunc(raw_func='l_sigmoid', section_pick=[-100, 100])
        distFunc = [t.distanceFuncList[func_tid], ] * len(t.header)

    t.normalization()
    t.division()
    p = preCal.d_apyori_preCal(t.d_data, t.header, distFunc, t.data_type)
    p.preCal_1item()

    minSupport = 0
    minConfidence = 0
    rules = runApriori(t.d_data, p, minSupport, minConfidence)
    #     - items (tuple, support)
    #     - rules ((pretuple, posttuple),support, confidence, lift)


    # ------------ print函数 ------------
    #for i in range(len(items)):
    #print(items)
    for i in rules:
        print(i)

