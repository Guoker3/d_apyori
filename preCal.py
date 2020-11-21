import loadData
import numpy as np
import pandas as pd
from copy import deepcopy
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

    def preCal_1item(self,distFuncList=None):
        if distFuncList == None :
            distFuncList=self.fault_distanceFunction

        data=deepcopy(self.data)
        if self.mode == 'insolate':
            n=0
            for column in data.columns:
                print('1item precal_loop : ', n,'/',self.data_inf['column_number'])
                distFunc = distFuncList[n]
                se_t = data[column]
                se_t_2=deepcopy(se_t)
                for i in se_t:
                    dl=list()
                    for r in se_t_2:
                        dl.append(distFunc(r,i))
                    self.pre_1item[n][i] = dl
                n = n + 1

if __name__ == '__main__':
    t = loadData.d_apyori_cookDataSet()
    t.loadDataSet('test9_11.csv', haveHeader=True,data_set_cut=[0,100])
    t.normalization()
    t.division()

    func_tid = t.create_rTOx_DistanceFunc(raw_func='l_sigmoid',section_pick=[-100,100])
    distFunc=[t.distanceFuncList[func_tid],] * len(t.header)
    p=d_apyori_preCal(t.d_data,t.header,distFunc, mode = 'insolate')
    p.preCal_1item()
    for i in p.pre_1item:
        print(i)