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

def returnItemsWithMinSupport(itemSet, pClass, minSupport, freqSet):
    """
    calculates the support for items in the itemSet and returns a subset
    of the itemSet each of whose elements satisfies the minimum support
    """
    """for i in range(1):
        transactionList=pClass.data.values.tolist()
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
    """
    _itemSet = set()
    if np.array([x !='atom' for x in pClass.mode]).all():
        for items in itemSet:
            itemSet_iter=iter(items)
            item=itemSet_iter.__next__()
            D=pClass.pre_1item[int(item/3)][item]
            for item in itemSet_iter:
                D = D * pClass.pre_1item[int(item/3)][item]
            support = sum(D) /pClass.data_inf['row_number']
            if support >=minSupport:
                _itemSet.add(items)
                freqSet[items] = support
    else:
        for items in itemSet:
            itemSet_iter = iter(items)
            item = itemSet_iter.__next__()
            D = pClass.pre_1item_uni_dict[item]
            for item in itemSet_iter:
                D = D * pClass.pre_1item_uni_dict[item]
            support = sum(D) / pClass.data_inf['row_number']
            if support >= minSupport:
                _itemSet.add(items)
                freqSet[items] = support
    return _itemSet

def returnItemsWithMinSupport_1itemLoop(itemSet, minSupport,freqSet,pre):
    """calculates the support for items in the itemSet and returns a subset
   of the itemSet each of whose elements satisfies the minimum support"""
    _itemSet = set()
    if np.array([x !='atom' for x in pre.mode]).all():
        for item in itemSet:
            f_item=list(item)[0]
            support = sum(pre.pre_1item[int(f_item/3)][f_item]) / pre.data_inf['row_number']
            if support >= minSupport:
                _itemSet.add(item)
                freqSet[item] = support
    else:
        for item in itemSet:
            support = sum(pre.pre_1item_uni_dict[list(item)[0]]) / pre.data_inf['row_number']
            if support >= minSupport:
                freqSet[item] = support
                _itemSet.add(item)
    return _itemSet,freqSet

def joinSet(itemSet, length,pre):
    """Join a set with itself and returns the n-element itemsets"""
    s_ret=set()
    s = set([i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length])
    if np.array([x !='atom' for x in pre.mode]).all():
        for f in s:
            flag=1
            ff=list(f)
            ff.sort()
            t=-10
            for i in ff:
                if i-t < 1.5:
                    flag = 0
                t=i
            if flag:
                s_ret.add(f)
    else:
        for f in s:
            flag=1
            ff=list(f)
            for i in range(len(ff)):
                if isinstance(ff[i],str):
                    ff[i] = int(ff[i].split(' ')[0])
            ff.sort()
            t=-10 ##TODO(check)related to divide function
            for i in ff:
                d=i-t
                if d < 1.5:
                    flag = 0
            if flag:
                s_ret.add(f)
    return s_ret
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
    itemSet = [frozenset([x,]) for y in data_iter.values.tolist() for x in y]
    #freqSet = defaultdict(int)
    freqSet = dict()
    largeSet = dict()

    oneCSet,freqSet = returnItemsWithMinSupport_1itemLoop(itemSet,minSupport,freqSet,preClass)
    currentLSet = oneCSet

    k = 2
    while (currentLSet != set([])) and k < preClass.data_inf['column_number']:
        print('itemSet number : ',k)
        largeSet[k - 1] = currentLSet
        currentLSet = joinSet(currentLSet, k,preClass)
        print('itemset length ',k,' has',len(currentLSet),' items')
        currentCSet = returnItemsWithMinSupport(currentLSet, preClass, minSupport, freqSet)
        currentLSet = currentCSet
        k = k + 1

    print('calculate completed screening......')

    def getSupport(item):
        """local function which Returns the support of an item"""
        return freqSet[item] / preClass.data_inf['row_number']

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
                                toRetRules.append(((tuple(element), tuple(remain)),
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
    #dataset='luntai'
    if dataset == 'luntai':
        t.loadDataSet('test9_11.csv', haveHeader=True, data_set_cut=[0, 2])
        func_tid = t.create_rTOx_DistanceFunc(raw_func='l_sigmoid', section_pick=[-100, 100])
        distFunc = [t.distanceFuncList[func_tid], ] * len(t.header)

    t.normalization()
    t.division()
    p = preCal.d_apyori_preCal(t.d_data, t.header, distFunc, t.data_type)
    p.preCal_1item()
    ##TODO(important speed) make a list of minSupport to deal with items-boommm
    ##TODO(importtant development) make minsupport-threhold be a relative thing.
    minSupport = 0
    minConfidence = 0
    rules = runApriori(p.data, p, minSupport, minConfidence)
    #     - items (tuple, support)
    #     - rules ((pretuple, posttuple),support, confidence, lift)


    # ------------ print函数 ------------
    for i in rules:
        print(i)

