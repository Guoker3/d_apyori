from csv import reader as csvReader
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt


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

        self.data_type =None
        #make str-like type work. for stepmode, just transfer the data. and for distancemode, it's need to transfer this attribute

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
        return data

    def loadDataSet(self, fileName, haveHeader, myEncoding='utf-8',data_set_cut=None):
        """
        :param
            data_set_cut:[a,b] pick part of the data_set_cut
        :argument
            it's better to use data all of float,but you can use str-like type(be signed as atom in this algorithm
            ,and pay attention to make a distance function for atom attribute (reference to l_atom in baseDistanceFunc))
        """

        if type(fileName) == type(list()):
            self.data_type=list()
            for i in fileName[0]:
                try:
                    a = float(i)
                    self.data_type.append('insolate')
                except:
                    self.data_type.append('atom')
            self.r_data = fileName
            return

        self.fileName = fileName
        if fileName[-4:] == '.csv':
            data=self.__loadCSV(fileName, haveHeader=haveHeader, myEncoding=myEncoding)
        else:
            raise TypeError('unknown kind of files')

        self.data_type=list()
        for i in data[0]:
            try:
                a=float(i)
                self.data_type.append('insolate')#default to insolate, can be change to linked if need
            except:
                self.data_type.append('atom')

        pd_data=pd.DataFrame(data)
        n=0
        for column in pd_data.columns:
            if self.data_type[n] == 'insolate':
                se=pd.Series([float(x) for x in pd_data[column]])
            else:
                se=pd_data[column]
            pd_data[column]=se
            n=n+1
        self.r_data=pd_data.values.tolist()
        #self.r_data = [[float(x) for x in row] for row in data]

        if data_set_cut != None:
            self.r_data=self.r_data[data_set_cut[0]: data_set_cut[1]]

    def flattenBEFOREnormalization_stepmode(self):
        pass  ##TODO(extra flatten stepMode) complete this function

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
            if self.data_type[i] != 'atom':
                l=pd_r_data[self.header[i]].tolist()
                minData = np.min(l)
                n_inf_min.append(minData)
                t_range = np.max(l) - minData
                n_inf_range.append(t_range)
                
                pd_r_data[self.header[i]] = pd.Series([(x-minData)/t_range for x in l])
            else:
                n_inf_min.append(None)
                n_inf_range.append((None))
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
        pd_n_data = pd.DataFrame(self.n_data, columns=self.header)
        for i in range(len(self.r_data[0])):
            if self.data_type[i] != 'atom':
                pd_n_data[self.header[i]] = pd.Series([x + divideStep * i for x in pd_n_data[self.header[i]].tolist()])
            else:
                pd_n_data[self.header[i]] = pd.Series([str(divideStep * i)+' ' + str(x) for x in pd_n_data[self.header[i]].tolist()])
        self.d_data = pd_n_data.values.tolist()
        print('data divided')

    def quickStart_stepmode(self, fileName='test9_11.csv', haveHeader=True,data_set_cut_in=[0,100]):
        self.loadDataSet(fileName=fileName, haveHeader=haveHeader,data_set_cut=data_set_cut_in)
        self.normalization()
        self.division()
        print('quick started')

    def baseDistanceFunc(self):
        ##TODO(extra toolkit) add func choice and put more math on it
        funcChoice=dict()

        def l_sigmoid(r,x):
            return 2 - (2 / (1 + np.exp(-abs(r-x))))
        funcChoice['l_sigmoid']=l_sigmoid

        def l_tanh(r,x):
            d=abs(r-x)
            a=np.exp(d)
            b=1/a
            return 1 - (a - b) / (a + b)
        funcChoice['l_tanh'] = l_tanh

        def l_atom(r,x):
            if r==x:
                return 1
            else:
                return 0
        funcChoice['l_atom'] = l_atom

        return funcChoice


    def create_rTOx_DistanceFunc(self, raw_func=None, section_pick=None):
        """
        :paramIn
            :section_pick in types [a,b] , func(-1,1) -> func(a, b) from norm-data to output_func
                help easy scale the func ,if None,calculate directly
        :argument
            :range(distanceFunc) in [0,1]
        """
        funcChoice = self.baseDistanceFunc()
        if raw_func == None and not isinstance(section_pick, dict):
            print('you can choose some builtin funcs or write one in')
            print('builtin funcs:')
            print('\n'.join(funcChoice.keys()))
        elif raw_func == None and isinstance(section_pick, dict):
            _func = None
        elif type(raw_func) == type('str'):
            _func = funcChoice[raw_func]
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
                #check if tid is same
                if int(r/3) != int(x/3):
                    return 0
                value_t = (r - x) * (b / 2 - a / 2) + a / 2 + b / 2
                d = _func2(value_t,0)
                return d

            _func = s_func

        #R to x, float only
        #assume that x separately associate with the attribute in R
        ##TODO(advanced) with topology or something, make all attributes in a pluralistic function,and make it more forward-useful
        elif isinstance(section_pick,dict):
            tid_func=deepcopy(section_pick)

            def s_func(R,x):
                D = 0
                num = 0
                for tid in tid_func.keys():
                    D = tid_func[tid](R[tid] % 3, x % 3)+D
                    num = num+1
                return D/num

            _func=s_func

        else:
            raise TypeError('section_pick should in form [a,b] for rTOx OR in form{int_tid1:func1, int_tid2:func2, ...} for RTOx')

        _len_dis=len(self.distanceFuncList)
        self.distanceFuncList.append(_func)
        print("new func saved in self.distanceFuncList ,position : ", _len_dis)
        return _len_dis

    def preCal_1itemDistance(self, distance_func_list=None, attr_insolate=True):
        """
            :paramIn:
        """
        if self.d_data == None:
            raise Exception('have no data set normalized')

def plotFunc(func,scale='single'):
    if scale == 'single':
        plt.title(func.__name__)
        x1=np.arange(-3,-1,0.01)
        y1=[func(xx,0) for xx in x1]
        x2=np.arange(1,3,0.01)
        y2=[func(xx,0) for xx in x2]
        x3=np.arange(-1,1,0.01)
        y3=[func(xx,0) for xx in x3]
        plt.plot(x1,y1,'b',x2,y2,'b',x3,y3,'y')
        plt.show()

if __name__ == '__main__':
    t = d_apyori_cookDataSet()
    t.loadDataSet('test9_11.csv', haveHeader=True)
    t.normalization()
    t.division()

    def myfunc(a,b):
        return (a-b)**2

    t.create_rTOx_DistanceFunc(raw_func=myfunc,section_pick=[-1,1])
    print(t.distanceFuncList[0](0.2,0.6))

    t.create_rTOx_DistanceFunc(raw_func=None,section_pick={0:t.distanceFuncList[0],2:t.distanceFuncList[0],3:t.distanceFuncList[0]})
    print(t.distanceFuncList[1]([0.1,3.7,6.2,9.9],9.4))

    t.create_rTOx_DistanceFunc(raw_func='l_sigmoid',section_pick=[-10,10])
    plotFunc(t.distanceFuncList[2])
    print(t.distanceFuncList[2](0.0,0.2))
