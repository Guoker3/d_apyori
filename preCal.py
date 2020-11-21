import loadData
import numpy as np
import pandas as pd
from copy import deepcopy
import threading
import time

class d_apyori_preCal:
    def __init__(self, dividedData, dataHeader, fault_distanceFunction =None ,modeList=None):
        """
        :param
            dividedData: format-dataset(d_data in d_apyori_cookDataSet)
            distanceFunction: list of distance for all attributes
            mode:list of column data distance attribute,['insolate'] can be 'insolate' or 'linked' or 'atom'
        """
        if modeList is None:
            self.mode = [modeList, ] * len(dataHeader)
        elif isinstance(modeList,list):
            self.mode = modeList
        else:
            raise ValueError('modeList wrong')
        self.fault_distanceFunction = fault_distanceFunction
        if isinstance(dividedData,list) and np.array([isinstance(x, list) for x in dividedData]).all():
            self.data=pd.DataFrame(dividedData,columns=dataHeader)
            self.data_inf = dict()
            #infomation about dataset of other informations can be easily save in this dict
            self.data_inf['column_number'] = len(dataHeader)
            self.data_inf['row_number'] = len(dividedData)

            self.pre_1item = list()
            for i in range(self.data_inf['column_number']):
                self.pre_1item.append(dict())
        else:
            raise ValueError('list no in format')

    def __preCal_1item_columnLoop(self,tid,distFuncIn,se_t,mode):
        time_t=time.time()
        if mode == 'insolate' or mode == 'atom':
            se_t_2 = deepcopy(se_t)
            for i in se_t:
                dl = list()
                for r in se_t_2:
                    dl.append(distFuncIn(r, i))
                self.pre_1item[tid][i] = dl
        elif mode == 'linked':
            for i in se_t:
                dl = list()
                data=deepcopy(self.data)
                for R in data:
                    dl.append(distFuncIn(R, i))
                self.pre_1item[tid][i] = dl

        print('1item precal (tid) complete : ', tid, '/', self.data_inf['column_number']-1, 'cost time : ', time.time()-time_t)

    def preCal_1item(self,distFuncList=None):
        if distFuncList == None :
            distFuncList=self.fault_distanceFunction

        data=deepcopy(self.data)
        n=0
        threadingPool = list()
        for column in data.columns:
            distFunc = distFuncList[n]
            se_t = data[column]
            threadingPool.append(threading.Thread(target=self.__preCal_1item_columnLoop(n,distFunc,se_t,self.mode[n])))
            n = n + 1
        th_it=iter(threadingPool)
        maxThreading=5
        count=0
        while count<maxThreading:
            try:
                th=next(th_it)
            except Exception:
                break
            th.start()
            count=count+1
            time.sleep(3)


if __name__ == '__main__':
    t = loadData.d_apyori_cookDataSet()
    dataset = None
    dataset='small'
    if dataset == 'small':
        smallDataSet=[[0,0,'a'],[1,1,'a'],[0,0,'b'],[0,0,'b'],[0.1,0.2,'a'],[0.87,0,'c']]
        t.loadDataSet(smallDataSet,haveHeader=False)
        func_tid1 = t.create_rTOx_DistanceFunc(raw_func='l_sigmoid', section_pick=[-100, 100])
        func_tid2 = t.create_rTOx_DistanceFunc(raw_func='l_atom')
        distFunc=[t.distanceFuncList[func_tid1],t.distanceFuncList[func_tid1],t.distanceFuncList[func_tid2]]
    #dataset='luntai'
    if dataset == 'luntai':
        t.loadDataSet('test9_11.csv', haveHeader=True,data_set_cut=[0,100])
        func_tid = t.create_rTOx_DistanceFunc(raw_func='l_sigmoid',section_pick=[-100,100])
        distFunc=[t.distanceFuncList[func_tid],] * len(t.header)

    t.normalization()
    t.division()

    p=d_apyori_preCal(t.d_data,t.header,distFunc, t.data_type)
    p.preCal_1item()
    for i in p.pre_1item:
        print(str(i)[0:100])