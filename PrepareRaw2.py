import loadData as ld
import pandas as pd
from distanceMode_apriori import *
from matplotlib import pyplot as plt
import copy
import aprioriRule as ar
import time

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
2
def plotFunc(funcs,title):
    i=0
    for fun in funcs:
        ld.plotFunc(fun,title[i])
        i+=1

if __name__=="__main__":
    startTime=time.time()
    controlFlag=list()
    #controlFlag.append("plotData")
    #controlFlag.append("plotFunc")

    controlFlag.append("chooseRow")
    chosedRow=[0,1,2,5,6,7,8,9]   #increase

    controlFlag.append("calculate")


    #used to deal with "***Raw2***.csv"
    datasetName="../dataset/GuanWangRaw2_firstTry.csv"
    data=createDataset(datasetName) #data.d_data is the data for next use

    #plot the data for hand analyse
    if "plotData" in controlFlag:
        plotData(data)

    #generate and plot the distant func

    func_tid = data.create_rTOx_DistanceFunc(raw_func='l_sigmoid', section_pick=[-30, 30])
    distFuncAll = [data.distanceFuncList[func_tid], ] * len(data.header)

    if "plotFunc" in controlFlag:
        plotFunc(distFuncAll,data.header)

    if "chooseRow" in controlFlag:
        dataIn,dataHeaderIn,dataTypeIn=data.chooseRow(data,chosedRow=chosedRow)
        dataIn = dataIn[0:300]  # change here if need to control the line number of dataset
        distFuncIn=list()
        for x in chosedRow:
            distFuncIn.append(distFuncAll[x])
    else:
        dataIn = data.d_data
        #dataIn=dataIn[0:20]    #change here if need to control the line number of dataset
        dataHeaderIn = data.header
        dataTypeIn=data.data_type
        distFuncIn=distFuncAll
    #calculate
    if "calculate" in controlFlag:
        p = preCal.d_apyori_preCal(dataIn, dataHeaderIn, distFuncIn, dataTypeIn)
        p.preCal_1item()
        #argument flowing are similiar with ones in apriori by not the same,espetially in the value
        minSupport = iter([0.3,0.25,0.25,0.0001])#number of minsupport equals to (ItemNumberLimit+1),[  ,   ,   ,filterMinSuppport]
        #minSupport = iter([0.00001,0.00001,0.00001,0.00001,0.00001,0.00001])
        minConfidence = 0.00001
        ItemNumberLimit=3
        rules = runApriori(p.data, p, minSupport, minConfidence,itemNumberLimit=ItemNumberLimit)
        #     - rules ((pretuple, posttuple),support, confidence, lift)

        if rules==list():
            print("rules not found")
    ##TODO filter the useful rules
        rc=ar.rules(data.header)                #total header to generate class
        rc.addRules(rules,dataHeaderIn)         #header of chosen feature
        rc.savePickle(rc.rules,dataHeaderIn)
        for rule in rc.rules:
            rc.showFeatureName(rule,rc.totalHeader)
        print("number of rules: "+str(len(rc.rules)))
        print("cost time:",int(time.time()-startTime))