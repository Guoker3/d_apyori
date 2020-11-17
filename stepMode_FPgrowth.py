import os
import time
from copy import deepcopy
import pyfpgrowth

class Node:
    def __init__(self, node_name, count, parentNode):
        self.name = node_name
        self.count = count
        self.nodeLink = None  # 根据nideLink可以找到整棵树中所有nodename一样的节点
        self.parent = parentNode  # 父亲节点
        self.children = {}  # 子节点{节点名字:节点地址}


class Fp_growth():
    def __init__(self, Mode):
        self.mode = Mode
        if mode == 'brute':
            self.generate_R = self.generate_R_brute
            self.generate_L = self.generate_L_brute
            self.create_cond_fptree = self.create_cond_fptree_brute
            self.create_fptree = self.create_fptree_brute
            self.update_fptree = self.update_fptree_brute
        if mode == 'weight':
            self.generate_R = self.generate_R_weight
            self.generate_L = self.generate_L_weight
            self.create_cond_fptree = self.create_cond_fptree_weight
            self.create_fptree = self.create_fptree_weight
            self.update_fptree = self.update_fptree_weight

    def update_header(self, node, targetNode):  # 更新headertable中的node节点形成的链表
        while node.nodeLink != None:
            node = node.nodeLink
        node.nodeLink = targetNode

    def find_path(self, node, nodepath):
        '''
        递归将node的父节点添加到路径
        '''
        if node.parent != None:
            nodepath.append(node.parent.name)
            self.find_path(node.parent, nodepath)
    def supportCalculator(self, rule_list, supportData):
        """:argument
            item in rule_list [pre_set,sub_set,conf,support]
        """
        rl = [list(l) for l in rule_list]
        for r in range(len(rl)):
            rl[r].append(supportData[rl[r][0]])
        return rl

    def find_cond_pattern_base(self, node_name, headerTable):
        '''
        根据节点名字，找出所有条件模式基
        '''
        treeNode = headerTable[node_name][1]
        cond_pat_base = {}  # 保存所有条件模式基
        while treeNode != None:
            nodepath = []
            self.find_path(treeNode, nodepath)
            if len(nodepath) > 1:
                cond_pat_base[frozenset(nodepath[:-1])] = treeNode.count
            treeNode = treeNode.nodeLink
        return cond_pat_base

#************************************part of BRUTE**********************************************************
    def update_fptree_brute(self, items, node, headerTable):  # 用于更新fptree
        if items[0] in node.children:
            # 判断items的第一个结点是否已作为子结点
            node.children[items[0]].count += 1
        else:
            # 创建新的分支
            node.children[items[0]] = Node(items[0], 1, node)
            # 更新相应频繁项集的链表，往后添加
            if headerTable[items[0]][1] == None:
                headerTable[items[0]][1] = node.children[items[0]]
            else:
                self.update_header(headerTable[items[0]][1], node.children[items[0]])
        # 递归
        if len(items) > 1:
            self.update_fptree(items[1:], node.children[items[0]], headerTable)

    def create_fptree_brute(self, data_set, min_support, flag=False):  # 建树主函数
        '''
        根据data_set创建fp树
        header_table结构为
        {"nodename":[num,node],..} 根据node.nodelink可以找到整个树中的所有nodename
        '''
        item_count = {}  # 统计各项出现次数
        for t in data_set:  # 第一次遍历，得到频繁一项集
            for item in t:
                if item not in item_count:
                    item_count[item] = 1
                else:
                    item_count[item] += 1
        headerTable = {}
        for k in item_count:  # 剔除不满足最小支持度的项
            if item_count[k] >= min_support:
                headerTable[k] = item_count[k]

        freqItemSet = set(headerTable.keys())  # 满足最小支持度的频繁项集
        if len(freqItemSet) == 0:
            return None, None
        for k in headerTable:
            headerTable[k] = [headerTable[k], None]  # element: [count, node]
        tree_header = Node('head node', 1, None)
        ite = data_set
        for t in ite:  # 第二次遍历，建树
            localD = {}
            for item in t:
                if item in freqItemSet:  # 过滤，只取该样本中满足最小支持度的频繁项
                    localD[item] = headerTable[item][0]  # element : count
            if len(localD) > 0:
                # 根据全局频数从大到小对单样本排序
                order_item = [v[0] for v in sorted(localD.items(), key=lambda x: x[1], reverse=True)]
                # 用过滤且排序后的样本更新树
                self.update_fptree(order_item, tree_header, headerTable)
        return tree_header, headerTable

    def create_cond_fptree_brute(self, headerTable, min_support, temp, freq_items, support_data):
        # 最开始的频繁项集是headerTable中的各元素
        freqs = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1][0])]  # 根据频繁项的总频次排序
        for freq in freqs:  # 对每个频繁项
            freq_set = temp.copy()
            freq_set.add(freq)
            freq_items.add(frozenset(freq_set))
            if frozenset(freq_set) not in support_data:  # 检查该频繁项是否在support_data中
                support_data[frozenset(freq_set)] = headerTable[freq][0]
            else:
                support_data[frozenset(freq_set)] += headerTable[freq][0]

            cond_pat_base = self.find_cond_pattern_base(freq, headerTable)  # 寻找到所有条件模式基
            cond_pat_dataset = []  # 将条件模式基字典转化为数组
            for item in cond_pat_base:
                item_temp = list(item)
                item_temp.sort()
                for i in range(cond_pat_base[item]):
                    cond_pat_dataset.append(item_temp)
            # 创建条件模式树
            cond_tree, cur_headtable = self.create_fptree(cond_pat_dataset, min_support)
            if cur_headtable != None:
                self.create_cond_fptree_brute(cur_headtable, min_support, freq_set, freq_items, support_data)  # 递归挖掘条件FP树

    def generate_L_brute(self, data_set, min_support):
        freqItemSet = set()
        support_data = {}
        tree_header, headerTable = self.create_fptree(data_set, min_support, flag=True)  # 创建数据集的fptree
        # 创建各频繁一项的fptree，并挖掘频繁项并保存支持度计数
        self.create_cond_fptree_brute(headerTable, min_support, set(), freqItemSet, support_data)

        max_l = 0
        for i in freqItemSet:  # 将频繁项根据大小保存到指定的容器L中
            if len(i) > max_l: max_l = len(i)
        L = [set() for _ in range(max_l)]
        for i in freqItemSet:
            L[len(i) - 1].add(i)
        for i in range(len(L)):
            print("frequent item {}:{}".format(i + 1, len(L[i])))
        #print('L:',L)
        #print('S:',support_data)
        return L, support_data

    def generate_R_brute(self, data_set, min_support, min_conf):
        L, support_data = self.generate_L(data_set, min_support)
        rule_list = []
        sub_set_list = []
        for i in range(0, len(L)):
            for freq_set in L[i]:
                for sub_set in sub_set_list:
                    if sub_set.issubset(
                            freq_set) and freq_set - sub_set in support_data:  # and freq_set-sub_set in support_data
                        conf = support_data[freq_set] / support_data[freq_set - sub_set]
                        big_rule = (freq_set - sub_set, sub_set, conf)
                        if conf >= min_conf and big_rule not in rule_list:
                            # print freq_set-sub_set, " => ", sub_set, "conf: ", conf
                            rule_list.append(big_rule)
                sub_set_list.append(freq_set)
        rule_list = self.supportCalculator(rule_list,support_data)
        rule_list = sorted(rule_list, key=lambda x: (x[2]), reverse=True)
        return rule_list

#***************************************part of WEIGHT************************************************

    def update_fptree_weight(self, items, node, headerTable,weight):  # 用于更新fptree


        if items[0] in node.children:
            # 判断items的第一个结点是否已作为子结点
            node.children[items[0]].count = node.children[items[0]].count+weight
        else:
            # 创建新的分支
            node.children[items[0]] = Node(items[0], weight, node)
            # 更新相应频繁项集的链表，往后添加
            tmp=items[0]
            if headerTable[tmp][1] == None:
                headerTable[tmp][1] = node.children[items[0]]
            else:
                self.update_header(headerTable[tmp][1], node.children[items[0]])
        # 递归
        if len(items) > 1:
            self.update_fptree(items[1:], node.children[items[0]], headerTable,weight)

    def create_fptree_weight(self, data_set, min_support, flag=False):  # 建树主函数
        '''
        根据data_set创建fp树
        header_table结构为
        {"nodename":[num,node],..} 根据node.nodelink可以找到整个树中的所有nodename
        '''
        item_count = {}  # 统计各项出现次数
        #for i in data_set:
            #if type(i[0]) != type([1,2]):
        #    print(i)
        for t in data_set:  # 第一次遍历，得到频繁一项集
            #print('t',t)
            for item in t[0]:
                #print(item)
                ##TODO(Jump Point 1) change count in frequent items mining
                if item not in item_count:
                    item_count[item] = t[1]
                else:
                    item_count[item] = item_count[item] + t[1]
        headerTable = {}
        for k in item_count:  # 剔除不满足最小支持度的项
            #print(type(item_count[k]))

            if item_count[k] >= min_support:
                headerTable[k] = item_count[k]
        #    print('k:',type(k))
        #input()
        freqItemSet = set(headerTable.keys())  # 满足最小支持度的频繁项集
        if len(freqItemSet) == 0:
            return None, None
        for k in headerTable:
            headerTable[k] = [headerTable[k], None]  # element: [count, node]

        tree_header = Node('head node', 1, None)

        ite = data_set
        for t in ite:  # 第二次遍历，建树
            localD = {}
            for item in t[0]:
                if item in freqItemSet:  # 过滤，只取该样本中满足最小支持度的频繁项
                    localD[item] = headerTable[item][0]  # element : count
            if len(localD) > 0:
                # 根据全局频数从大到小对单样本排序
                order_item = [v[0] for v in sorted(localD.items(), key=lambda x: x[1], reverse=True)]
                #print(order_item)
                #for i in order_item:
                #   print(i)
                #input()
                # 用过滤且排序后的样本更新树
                self.update_fptree_weight(order_item, tree_header, headerTable,t[1])
        return tree_header, headerTable

    def create_cond_fptree_weight(self, headerTable, min_support, temp, freq_items, support_data):
        # 最开始的频繁项集是headerTable中的各元素
        freqs = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1][0])]  # 根据频繁项的总频次排序
        for freq in freqs:  # 对每个频繁项
            freq_set = temp.copy()
            freq_set.add(freq)
            freq_items.add(frozenset(freq_set))
            if frozenset(freq_set) not in support_data:  # 检查该频繁项是否在support_data中
                support_data[frozenset(freq_set)] = headerTable[freq][0]
            else:
                support_data[frozenset(freq_set)] = support_data[frozenset(freq_set)] + headerTable[freq][0]

            cond_pat_base = self.find_cond_pattern_base(freq, headerTable)  # 寻找到所有条件模式基
            cond_pat_dataset = []  # 将条件模式基字典转化为数组
            for item,weight in cond_pat_base.items():
                item_temp = list(item)
                item_temp.sort()
                for i in range(cond_pat_base[item]):
                    ##TODO(Jump Point 3 when go through the )
                    cond_pat_dataset.append([item_temp,1])
            # 创建条件模式树
            cond_tree, cur_headtable = self.create_fptree(cond_pat_dataset, min_support)
            if cur_headtable != None:
                self.create_cond_fptree_weight(cur_headtable, min_support, freq_set, freq_items,
                                              support_data)  # 递归挖掘条件FP树

    def generate_L_weight(self, data_set, min_support):
        freqItemSet = set()
        support_data = {}
        tree_header, headerTable = self.create_fptree(data_set, min_support, flag=True)  # 创建数据集的fptree
        #print('T:',tree_header)
        #print('H:',headerTable)
        # 创建各频繁一项的fptree，并挖掘频繁项并保存支持度计数
        self.create_cond_fptree(headerTable, min_support, set(), freqItemSet, support_data)
        max_l = 0
        for i in freqItemSet:  # 将频繁项根据大小保存到指定的容器L中
            if len(i) > max_l: max_l = len(i)
        L = [set() for _ in range(max_l)]
        for i in freqItemSet:
            L[len(i) - 1].add(i)
        for i in range(len(L)):
            print("frequent item {}:{}".format(i + 1, len(L[i])))
        return L, support_data

    def generate_R_weight(self, data_set, min_support, min_conf):
        """:param:
                L : list of frequent items' set
                support_data : dictionary about {frequent set : support counts}
        """
        L, support_data = self.generate_L(data_set, min_support)

        #print('L:',L)
        #for i in support_data.items():
        #    print('S:',i)

        rule_list = []
        sub_set_list = []
        for i in range(0, len(L)):
            for freq_set in L[i]:
                for sub_set in sub_set_list:
                    if sub_set.issubset(
                            freq_set) and freq_set - sub_set in support_data:  # and freq_set-sub_set in support_data
                        conf = support_data[freq_set] / support_data[freq_set - sub_set]
                        big_rule = (freq_set - sub_set, sub_set, conf)
                        if conf >= min_conf and big_rule not in rule_list:
                            # print freq_set-sub_set, " => ", sub_set, "conf: ", conf
                            rule_list.append(big_rule)
                sub_set_list.append(freq_set)
        rule_list = self.supportCalculator(rule_list,support_data)
        rule_list = sorted(rule_list, key=lambda x: (x[2]), reverse=True)
        return rule_list


if __name__ == "__main__":
    min_support_param = 25  # 最小支持度
    min_conf = 0.7  # 最小置信度

    from step_mode import *
    from loadData import *

    dataSet = d_apyori_cookDataSet()

    dataChoice='luntai'
    #dataChoice='small'

    if dataChoice=='luntai':
        dataSet.quickStart(fileName='test9_11.csv', haveHeader=True)
        dif_list = [60,90, 100, 90,60]
        dif_step=[[-0.2,-0.1, 0, 0.1,0.2], ] * len(dataSet.n_data[0])
        new_min_support = int(min_support_param * sum(dif_list))
        print('support selected in luntai : ', new_min_support)

    if dataChoice=='small':
        #easyDataSet=[[0,0,0],[0,0,1]]
        easyDataSet=[[0,0,0],[1,1,1],[0,0,0.22],[0,0,0.7],[0.1,0.2,0.3],[0.87,0,0.22]]
        dataSet.quickStart(fileName=easyDataSet,haveHeader=False)

        #dif_list=[1000,30000,1000]
        dif_list=[1000,2000,3000,4000,5000,6000,7000,6000,5000,4000,3000,2000,1000]
        dif_step = [[-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6], ] * len(dataSet.n_data[0])
        #dif_list = [1,]
        #dif_step=[[0,], ] * len(dataSet.n_data[0])
        min_conf=0.5
        new_min_support = 2
    mode = None
    #mode = 'brute'
    time_b_s=time.time()
    print('****************\n******************* brute ******************\n***************************')
    if mode == 'brute':
        dataSet_b = stepDiffusion(dataSet.d_data, dif_list, dif_step, mode='brute')
        dataSet_b = [[str(round(x, 2)) for x in row] for row in dataSet_b]
        fp = Fp_growth(Mode='brute')
        rule_list = fp.generate_R(dataSet_b, new_min_support, min_conf)
        for i in rule_list:
            print(i)
        print('number of results',len(rule_list))
    time_b=time.time()-time_b_s
    time_w_s=time.time()
    print('****************\n******************* weight ******************\n***************************')
    mode ='weight'
    if mode == 'weight':
        dataSet_w = stepDiffusion(dataSet.d_data, dif_list, dif_step, mode='weight')
        for r in range(len(dataSet_w)):
            dataSet_w[r][0] = [str(round(x,2)) for x in dataSet_w[r][0]]
        fp = Fp_growth(Mode='weight')
        rule_list = fp.generate_R(dataSet_w, new_min_support, min_conf)
        for i in rule_list:
            print('rule',i)
        print('number of results:',len(rule_list))
        print('total diffuse number : ',sum(dif_list) )
    time_w=time.time()-time_w_s
    print('brute time : ',time_b)
    print('weight time : ',time_w)
"""
    print('****************\n***************** pyfpgrowth ********************\n***************************')
    mode = 'pyfpgrowth'
    if mode == 'pyfpgrowth':
        patterns = pyfpgrowth.find_frequent_patterns(dataSet_b, new_min_support)
        rules = pyfpgrowth.generate_association_rules(patterns, min_conf)
        for r in rules:
            print(r)
        print('number of results:', len(rules))
"""