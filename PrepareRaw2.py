import loadData as ld
def loadDataset(name):
    data=ld.d_apyori_cookDataSet()
    data.loadDataSet(name,haveHeader=True)
    data.normalization()
    data.division()
    return data

if __name__=="__main__":
    datasetName="../dataset/GuanWangRaw2_firstTry.csv"
    data=loadDataset(datasetName)
    print(data)