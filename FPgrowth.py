import os
import time



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
            self.find_cond_pattern_base = self.find_cond_pattern_base_brute
            self.create_fptree=self.create_fptree_brute
            self.update_fptree=self.update_fptree_brute
        if mode == 'weight':
            self.generate_R = self.generate_R_weight
            self.generate_L = self.generate_L_weight
            self.find_cond_pattern_base = self.find_cond_pattern_base_weight
            self.create_fptree=self.create_fptree_weight
            self.update_fptree=self.update_fptree_weight

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

    def find_cond_pattern_base_brute(self, node_name, headerTable):
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
                        big_rule = (freq_set - sub_set, sub_set, conf, 0)
                        if conf >= min_conf and big_rule not in rule_list:
                            # print freq_set-sub_set, " => ", sub_set, "conf: ", conf
                            rule_list.append(big_rule)
                sub_set_list.append(freq_set)
        rule_list = self.supportCalculator(rule_list,support_data)
        rule_list = sorted(rule_list, key=lambda x: (x[2]), reverse=True)
        return rule_list

#***************************************part of WEIGHT************************************************

    def update_fptree_weight(self, items, node, headerTable):  # 用于更新fptree

        #for i in headerTable:
        #    print(i)
        if type(items[0][0]) == type(str()):
            c_tmp=items[0][1]
        else:
            c_tmp=items[0][0]
        #print(items)
        #input()
        f_items=frozenset(items[0])

        if f_items in node.children:
            # 判断items的第一个结点是否已作为子结点
            node.children[f_items].count = node.children[f_items].count+c_tmp
        else:
            # 创建新的分支
            node.children[f_items] = Node(f_items, c_tmp, node)
            # 更新相应频繁项集的链表，往后添加
            tmp=items[0][0]
            if headerTable[tmp][1] == None:
                headerTable[tmp][1] = node.children[f_items]
            else:
                self.update_header(headerTable[tmp][1], node.children[f_items])
        # 递归
        if len(items) > 1:
            self.update_fptree(items[1:], node.children[f_items], headerTable)

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
                order_item = [v for v in sorted(localD.items(), key=lambda x: x[1], reverse=True)]
                #print(order_item)
                #for i in order_item:
                #   print(i)
                #input()
                # 用过滤且排序后的样本更新树
                self.update_fptree_weight(order_item, tree_header, headerTable)
        return tree_header, headerTable

    def find_cond_pattern_base_weight(self, node_name, headerTable):
        '''
        根据节点名字，找出所有条件模式基
        '''
        if type(list(node_name)[0])==type(str()):
            tmp=list(node_name)[0]
        else:
            tmp=list(node_name)[1]
        treeNode = headerTable[tmp][1]
        cond_pat_base = {}  # 保存所有条件模式基
        while treeNode != None:
            nodepath = []
            self.find_path(treeNode, nodepath)
            if len(nodepath) > 1:
                nodepath_tmp=[]
                for i in nodepath[:-1]:
                    ii=list(i)
                    if type(ii[0])==type(str()):
                        nodepath_tmp.append(ii[0])
                    else:
                        nodepath_tmp.append(ii[1])

                cond_pat_base[frozenset(nodepath_tmp)] = treeNode.count
            treeNode = treeNode.nodeLink
        #print('cpd:',cond_pat_base)
        return cond_pat_base

    def create_cond_fptree_weight(self, headerTable, min_support, temp, freq_items, support_data):
        """:param
                headerTable:dictionary of the "('39.6', [18, <__main__.Node object at 0x000001B355482DC8>])"
        """
        # 最开始的频繁项集是headerTable中的各元素
        #for i in headerTable.items():
        #    print('h',i)
        freqs = [v for v in sorted(headerTable.items(), key=lambda p: p[1][0])]  # 根据频繁项的总频次排序
        for i in range(len(freqs)):
            #print('i-1:',freqs[i])
            freqs[i]=frozenset([freqs[i][0],freqs[i][1][0]])
            #print('i:',freqs[i])

        #for a in support_data.items():
        #for a in freqs:
        #    print('s:',a)
        #input()
        for freq in freqs:  # 对每个频繁项
            freq_set = temp.copy()
            freq_set.add(freq)

            #if len(freq_set)==2:
            #    print(freq_set)
            #    input()

            freq_items.add(frozenset(freq_set))
            ##TODO(main) rewrite the frozenset to make it ordered and repeat-able OR make list to str for hash

            if type(list(freq)[0]) == type(str()):
                tmp = list(freq)[0]
            else:
                tmp = list(freq)[1]

            if frozenset(freq_set) not in support_data:  # 检查该频繁项是否在support_data中
                #print(freq_set)
                support_data[frozenset(freq_set)] = headerTable[tmp][0]
                #print(support_data[frozenset(freq_set)])
            else:
                #print(freq_set)
                support_data[frozenset(freq_set)] += headerTable[tmp][0]
                #print(support_data[frozenset(freq_set)])

            cond_pat_base = self.find_cond_pattern_base(freq, headerTable)  # 寻找到所有条件模式基 ##TODO(done) make support return together
            #for i in cond_pat_base:
            #    print(i)
            cond_pat_dataset = []  # 将条件模式基字典转化为数组
            for item in cond_pat_base.items():
                item_temp = list(item[0])
                #print('item_temp:',item_temp)
                item_temp.sort()
                #print(item)
                cond_pat_dataset.append([item_temp,item[1]])
            #for i in cond_pat_dataset:
                #print(':::',i)
            # 创建条件模式树
            cond_tree, cur_headtable = self.create_fptree_weight(cond_pat_dataset, min_support)
            if cur_headtable != None:
                #print(cur_headtable)
                #input()
                self.create_cond_fptree_weight(cur_headtable, min_support, freq_set, freq_items, support_data)  # 递归挖掘条件FP树

    def generate_L_weight(self, data_set, min_support):
        freqItemSet = set()
        support_data = {}
        tree_header, headerTable = self.create_fptree(data_set, min_support, flag=True)  # 创建数据集的fptree
        #print('T:',tree_header)
        #print('H:',headerTable)
        # 创建各频繁一项的fptree，并挖掘频繁项并保存支持度计数
        self.create_cond_fptree_weight(headerTable, min_support, set(), freqItemSet, support_data)
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
                ##TODO(consider) judge whether "sub_set_list.append(freq_set)" should be here
                for sub_set in sub_set_list:
                    if sub_set.issubset(
                            freq_set) and freq_set - sub_set in support_data:  # and freq_set-sub_set in support_data
                        conf = support_data[freq_set] / support_data[freq_set - sub_set]
                        big_rule = (freq_set - sub_set, sub_set, conf)
                        if conf >= min_conf and big_rule not in rule_list:
                            # print freq_set-sub_set, " => ", sub_set, "conf: ", conf
                            rule_list.append(big_rule)
                    #else:
                        #print('sub_set:',sub_set)
                        #print('freq_set:',freq_set)
                        #print(freq_set-sub_set)
                        #print('1:',sub_set.issubset(freq_set))
                        #print('2:',freq_set-sub_set in support_data)
                        #if sub_set.issubset(freq_set):
                        #    print(freq_set in support_data)
                        #    print('sub_set:',sub_set)
                        #    print('freq_set:',freq_set)
                        #    print(freq_set-sub_set)
                        #    print(support_data)
                sub_set_list.append(freq_set)
        rule_list = self.supportCalculator(rule_list,support_data)
        rule_list = sorted(rule_list, key=lambda x: (x[2]), reverse=True)
        return rule_list


if __name__ == "__main__":
    ##TODO check why result differ between the brute and weight
    min_support = 3  # 最小支持度
    min_conf = 0  # 最小置信度
    ##TODO(fyt) estimate the min_conf relationship

    from step_mode import *
    from loadData import *

    dataSet = d_apyori_cookDataSet()

    #dataChoice='luntai'
    dataChoice='small'

    if dataChoice=='luntai':
        dataSet.quickStart(fileName='test9_11.csv', haveHeader=True)
        dif_list = [1, 3, 1]
        new_min_support = int(min_support * sum(dif_list))

    if dataChoice=='small':
        easyDataSet=[[0,0,0],[1.5,1.5,1.5]]
        dataSet.quickStart(fileName=easyDataSet,haveHeader=False)
        dif_list=[100,2,0]
        new_min_support = 1

    mode = 'weight'
    #mode = 'brute'

    if mode == 'brute':
        dataSet = stepDiffusion(dataSet.d_data, dif_list, [[-0.1, 0, 0.1], ] * len(dataSet.n_data[0]), mode='brute')
        dataSet = [[str(round(x, 2)) for x in row] for row in dataSet]
        fp = Fp_growth(Mode='brute')
        rule_list = fp.generate_R(dataSet[0:500], new_min_support, min_conf)
        #for i in rule_list:
        #    print(i)
        print(len(rule_list))

    if mode == 'weight':
        dataSet = stepDiffusion(dataSet.d_data, dif_list, [[-0.1, 0, 0.1], ] * len(dataSet.n_data[0]), mode='weight')
        for r in range(len(dataSet)):
            dataSet[r][0] = [str(round(x,2)) for x in dataSet[r][0]]
        fp = Fp_growth(Mode='weight')
        rule_list = fp.generate_R(dataSet[0:100], new_min_support, min_conf)
        for i in rule_list:
            print('rule',i)
        print(len(rule_list))