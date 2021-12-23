import loadData as ld
import pandas as pd
from distanceMode_apriori import *
from matplotlib import pyplot as plt
import copy
def createDataset(name):
    data=ld.d_apyori_cookDataSet()
    data.loadDataSet(name,haveHeader=True)
    data.normalization()
    data.division()
    return data

def  plotData(data,ignoreBorder=False):
    lineNumber=0
    featureNumber=len(data.header)
    cl=[0,]*100
    ac=list()
    for i in range(featureNumber):
        ac.append(copy.deepcopy(cl))
    for line in data.n_data:
        lineNumber+=1
        for n in range(featureNumber):
            pos=int(line[n]*100)
            if pos==100:
                pos=99
            ac[n][pos]+=1
    x=list()
    for i in range(100):
        x.append(i/100)
    i=0
    for y in ac:
        plt.plot(x,y)
        plt.title(data.header[i])
        i+=1
        plt.show()

def plotFunc(funcs,title):
    i=0
    for fun in funcs:
        ld.plotFunc(fun,title[i])
        i+=1

if __name__=="__main__":
    controlFlag=list()
    #controlFlag.append("plotData")
    controlFlag.append("plotFunc")

    #controlFlag.append("chooseRow")
    #chosedRow=[0,1,2,3,4]

    #controlFlag.append("calculate")


    #used to deal with "***Raw2***.csv"
    datasetName="../dataset/GuanWangRaw2_firstTry.csv"
    data=createDataset(datasetName) #data.d_data is the data for next use

    #plot the data for hand analyse
    if "plotData" in controlFlag:
        plotData(data)

    #generate and plot the distant func

    func_tid = data.create_rTOx_DistanceFunc(raw_func='l_sigmoid', section_pick=[-100, 100])
    distFunc = [data.distanceFuncList[func_tid], ] * len(data.header)

    if "plotFunc" in controlFlag:
        plotFunc(distFunc,data.header)

    if "chooseRow" in controlFlag:
        dataIn=list()
        dataHeaderIn=list()
        dataTypeIn=list()
        for line in data.d_data:
            l=list()
            for i in chosedRow:
                l.append(line[i])
            dataIn.append(l)
        for i in chosedRow:
            dataHeaderIn.append(data.header[i])
            dataTypeIn.append(data.data_type[i])
    else:
        dataIn=data.d_data
        dataHeaderIn=data.header

    #calculate
    if "calculate" in controlFlag:
        p = preCal.d_apyori_preCal(dataIn[0:10], dataHeaderIn, distFunc, dataTypeIn)
        p.preCal_1item()
        #argument flowing are similiar with ones in apriori by not the same,espetially in the value
        minSupport = 0.01
        minConfidence = 0
        NumberLimit=2
        rules = runApriori(p.data, p, minSupport, minConfidence,itemNumberLimit=NumberLimit)
        #     - rules ((pretuple, posttuple),support, confidence, lift)

        if rules==list():
            print("rules not found")
    ##TODO filter the useful rules
        for i in rules:
            print(i)