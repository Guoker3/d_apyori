from csv import reader as csvReader
import numpy as np
import pandas as pd


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

    def quickStart(self, fileName='test9_11.csv', haveHeader=True):
        self.loadDataSet(fileName=fileName, haveHeader=haveHeader)
        self.normalization()
        self.division()
        print('quick started')


if __name__ == '__main__':
    t = d_apyori_cookDataSet()
    t.loadDataSet('test9_11.csv', haveHeader=True)
    t.normalization()
    t.division()
