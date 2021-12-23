import loadData as ld
from distanceMode_apriori import *
def createDataset(name):
    data=ld.d_apyori_cookDataSet()
    data.loadDataSet(name,haveHeader=True)
    data.normalization()
    data.division()
    return data

def  plotData(n_data):
    pass

def plotFunc(func):
    pass

def useD_apriori():
    pass

if __name__=="__main__":
    plotFlag=list()

    #used to deal with "***Raw2***.csv"
    datasetName="../dataset/GuanWangRaw2_firstTry.csv"
    data=createDataset(datasetName) #data.d_data is the data for next use

    #plot the data for hand analyse
    if "data" in plotFlag:
        plotData()

    #generate and plot the distant func

    if "func" in plotFlag:
        plotFlag()
    func_tid = data.create_rTOx_DistanceFunc(raw_func='l_sigmoid', section_pick=[-100, 100])
    distFunc = [data.distanceFuncList[func_tid], ] * len(data.header)

    #calculate

    p = preCal.d_apyori_preCal(data.d_data[0:10], data.header, distFunc, data.data_type)
    p.preCal_1item()
    ##TODO(important speed) make a list of minSupport to deal with items-boommm
    ##TODO(importtant development) make minsupport-threhold be a relative thing.
    minSupport = 0
    minConfidence = 0
    rules = runApriori(p.data, p, minSupport, minConfidence,itemNumberLimit=2)
    #     - items (tuple, support)
    #     - rules ((pretuple, posttuple),support, confidence, lift)

    if rules==list():
        print("rules not found")

    # ------------ print函数 ------------
    for i in rules:
        print(i)