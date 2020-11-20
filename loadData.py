from csv import reader as csvReader
import numpy as np
import pandas as pd
from copy import deepcopy


class d_apyori_cookDataSet:
    def __init__(self):
        """
        :param
            header: header list of the dataset if there is header in.
            r_data: data set in form of list<float>

            n_data: normalized data set
            n_data_inf: information about the normalization

            d_data: divided data set to make float data from different attribute not mixed in the algorithm
                    [0.1,0.4,0.2] -> [0.1, 3.4, 6.2] when divideStep == 3
        """
        self.fileName = None
        self.header = None
        self.r_data = None

        self.n_data = None
        self.n_data_inf = dict()

        self.d_data = None

        self.distanceFuncList=list()

    def __loadCSV(self, csvName, haveHeader, myEncoding='utf-8'):
        data = []
        with open(csvName, encoding=myEncoding) as csvFile:
            csv_reader = csvReader(csvFile)  # 使用csv.reader读取csvFile中的文件
            if haveHeader:
                self.header = next(csv_reader)
                # print(csvName, " header : ", header)
            for row in csv_reader:
                data.append(row)
        self.r_data = [[float(x) for x in row] for row in data]
        print("csv loaded")

    def loadDataSet(self, fileName, haveHeader, myEncoding='utf-8'):
        if type(fileName) == type(list()):
            self.r_data = fileName
            return

        self.fileName = fileName
        if fileName[-4:] == '.csv':
            self.__loadCSV(fileName, haveHeader=haveHeader, myEncoding=myEncoding)
        else:
            raise TypeError('unknown kind of files')

    def flattenBEFOREnormalization(self):
        pass  ##TODO(extra precision optimazation) complete this function

    def normalization(self):
        """:argument
            normalize the dataset and save the information of the normalize-process in (n_inf_min and n_inf_range)\
            raw = norm * range + min
        """
        if self.r_data == None:
            raise Exception('raw data have not loaded')
        if self.header == None:
            self.header = [x for x in range(len(self.r_data[0]))]
        pd_r_data = pd.DataFrame(self.r_data, columns=self.header)
        n_inf_min = []
        n_inf_range = []
        for i in range(len(self.r_data[0])):
            minData = np.min(pd_r_data[self.header[i]].tolist())
            n_inf_min.append(minData)
            t_range = np.max(pd_r_data[self.header[i]].to_list()) - minData
            n_inf_range.append(t_range)

            def normalization(number):
                return (number - minData) / t_range

            pd_r_data[self.header[i]] = pd.Series([normalization(x) for x in pd_r_data[self.header[i]].tolist()])
        self.n_data_inf['min'] = n_inf_min
        self.n_data_inf['range'] = n_inf_range
        self.n_data = pd_r_data.values.tolist()
        print('data normalized')

    def division(self):
        """:argument
                make data of different attribute can be recognized by the algorithm (by add a tid-interval)
        """
        if self.n_data == None:
            raise Exception('have no data set normalized')
        if self.header == None:
            self.header = [x for x in range(len(self.r_data[0]))]
        divideStep = 3
        pd_n_data = pd.DataFrame(self.r_data, columns=self.header)
        for i in range(len(self.r_data[0])):
            pd_n_data[self.header[i]] = pd.Series([x + divideStep * i for x in pd_n_data[self.header[i]].tolist()])
        self.d_data = pd_n_data.values.tolist()
        print('data divided')

    def quickStart_stepmode(self, fileName='test9_11.csv', haveHeader=True):
        self.loadDataSet(fileName=fileName, haveHeader=haveHeader)
        self.normalization()
        self.division()
        print('quick started')

    def baseDistanceFunc(self):
        ##TODO(extra toolkit) add func choice
        funcChoiceName=[]
        funcChoice=dict()

        return funcChoiceName,funcChoice


    def create_rTOx_DistanceFunc_insolate(self, raw_func=None, section_pick=None):
        """
        :paramIn
            :section_pick in types [a,b] , func(a,b) -> func(-1, 1), func(others) = 0
        :argument
            :range(distanceFunc) in [0,1]
        """
        funcChoiceName, funcChoice = self.baseDistanceFunc()
        if raw_func == None:
            print('you can choose some builtin funcs or write one in')
            print('builtin funcs:')
            print('\n'.join(funcChoiceName))
        elif type(raw_func) == type('str'):
            _func = funcChoice['raw_func']
        else:
            _func = raw_func

        # reflect func to section if select
        if section_pick == None:
            pass
        #r to x
        elif isinstance(section_pick,list) and len(section_pick) == 2:
            a, b = section_pick
            _func2=deepcopy(_func)
            def s_func(r, x):
                value_t = (r - x) * (b / 2 - a / 2) + a / 2 + b / 2
                d = _func2(value_t,0)
                return d

            _func = s_func
        #R to x
        elif isinstance(section_pick,list) and np.array([isinstance(s,list) and len(s)==2 for s in section_pick]).all():
            pass
        else:
            raise TypeError('section_pick should in form [a,b] OR in form[[tid,a,b], [tid,a,b], ...]')

        _len_dis=len(self.distanceFuncList)
        self.distanceFuncList.append(_func)
        print("new func saved in self.distanceFuncList ,position : ", _len_dis)

    def create_RTOx_DistanceFunc_connect(self, raw_func=None, t_section_pick=None):
        pass##TODO(mode funcs) complete this func like rTOx

    def preCal_1itemDistance(self, distance_func_list=None, attr_insolate=True):
        """
            :paramIn:
        """
        if self.d_data == None:
            raise Exception('have no data set normalized')


if __name__ == '__main__':
    t = d_apyori_cookDataSet()
    t.loadDataSet('test9_11.csv', haveHeader=True)
    t.normalization()
    t.division()

    def myfunc(a,b):
        return (a-b)**2

    t.create_rTOx_DistanceFunc_insolate(raw_func=myfunc,section_pick=[-1,1])
    print(t.distanceFuncList[0](1,3))
