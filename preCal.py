import loadData
import numpy as np
import pandas as pd
from copy import deepcopy
import threading
import time

class d_apyori_preCal:
    def __init__(self, dividedData, dataHeader, fault_distanceFunction =None , mode = 'insolate'):
        """
        :param
            dividedData: format-dataset(d_data in d_apyori_cookDataSet)
            distanceFunction: list of distance for all attributes
            mode: can be 'insolate' or 'linked'
        """
        if mode == 'insolate':
            self.mode='insolate'
        elif mode == 'linked':
            self.mode = 'linked'
        else:
            raise ValueError(' mode should be insolate or linked ')

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

    def __preCal_1item_insolate_columnLoop(self,tid,distFuncIn,se_t):
        time_t=time.time()
        se_t_2 = deepcopy(se_t)
        for i in se_t:
            dl = list()
            for r in se_t_2:
                dl.append(distFuncIn(r, i))
            self.pre_1item[tid][i] = dl
        print('1item precal (tid) complete : ', tid, '/', self.data_inf['column_number']-1, 'cost time : ', time.time()-time_t)

    def preCal_1item(self,distFuncList=None):
        if distFuncList == None :
            distFuncList=self.fault_distanceFunction

        data=deepcopy(self.data)
        if self.mode == 'insolate':
            n=0
            threadingPool = list()
            for column in data.columns:
                distFunc = distFuncList[n]
                se_t = data[column]
                threadingPool.append(threading.Thread(target=self.__preCal_1item_insolate_columnLoop(n,distFunc,se_t)))
                n = n + 1
            th_it=iter(threadingPool)
            maxThreading=1
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
    t.loadDataSet('test9_11.csv', haveHeader=True,data_set_cut=[0,2000])
    t.normalization()
    t.division()

    func_tid = t.create_rTOx_DistanceFunc(raw_func='l_sigmoid',section_pick=[-100,100])
    distFunc=[t.distanceFuncList[func_tid],] * len(t.header)
    p=d_apyori_preCal(t.d_data,t.header,distFunc, mode = 'insolate')
    p.preCal_1item()
    for i in p.pre_1item:
        print(str(i)[0:100])