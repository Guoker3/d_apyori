import pickle
class rules:
    def __init__(self,allHeader):
        # rule: ((pretuple, posttuple),support, confidence, lift)
        self.rules=list()
        self.totalHeader=allHeader

    def __chosedHeader_TO_totalHeader(self,chosedHeader):
        offset=dict()
        of=0
        th=iter(self.totalHeader)
        for i in range(len(chosedHeader)):
            while(chosedHeader[i]!=next(th)):
                of+=3
            offset[chosedHeader[i]]=of
        return offset

    def chosedRule_TO_totalRule(self,ruleIn,chosedHeader):
        offset=self.__chosedHeader_TO_totalHeader(chosedHeader=chosedHeader)
        rule = [[[x+offset[chosedHeader[int(x/3)]] for x in list(ruleIn[0][0])], [x+offset[chosedHeader[int(x/3)]] for x in list(ruleIn[0][1])]], ruleIn[1], ruleIn[2], ruleIn[3]]
        return rule

    def addRule(self,ruleIn,header):
        ruleIn=self.chosedRule_TO_totalRule(ruleIn,header)
        rule = [[[x for x in list(ruleIn[0][0])],[x for x in list(ruleIn[0][1])]], ruleIn[1], ruleIn[2], ruleIn[3]]
        self.rules.append(rule)

    def addRules(self,rulesIn,header):
        for ruleIn in rulesIn:
            self.addRule(ruleIn,header)

    def filterRules(self,filterDict):
        retRules=list()
        for rule in self.rules:
            flag=True
            if "support" in filterDict:
                if rule[1]<filterDict["support"]:
                    flag=False
            if "confidence" in filterDict:
                if rule[2]<filterDict["confidence"]:
                    flag=False
            if "lift" in filterDict:
                if rule[3]<filterDict["lift"]:
                    flag=False

            if flag:
                retRules.append(rule)
        return retRules

    def sortRules(self,rules,amount=-1,key="confidence"):
        if key =="confidence":
            sRules=sorted(rules, key=lambda rule: rule[2],reverse=True)
        elif key =="support":
            sRules=sorted(rules, key=lambda rule: rule[1],reverse=True)
        elif key =="lift":
            sRules=sorted(rules, key=lambda rule: rule[3],reverse=True)
        else:
            return None
        if amount>len(rules):
            amount=len(rules)
        elif amount==-1:
            return sRules
        return sRules[0:amount]

    def searchFeatureLeft(self,featuresLeft):
        retRules=list()
        for rule in self.rules:
            flag = False
            for i in list(rule[0][0]):
                if self.totalHeader[int(i / 3)] in featuresLeft:
                    flag=True
            if flag:
                retRules.append(rule)
        return retRules
    def showFeatureName(self,rule,header):
        ii = [[header[int(x / 3)] + ": " + str(x) for x in list(rule[0][0])],[header[int(x / 3)] + ": " + str(x) for x in list(rule[0][1])], rule[1], rule[2], rule[3]]
        print(ii)

    def savePickle(self,rules,header,name=None):
        if name==None:
            with open("../dataset/rules"+str(len(self.rules))+".pickle", 'wb') as f:
                pickle.dump([rules,header], f)
                print("save in: "+ "../dataset/rules"+str(len(self.rules))+".pickle")
        else:
            with open("../dataset/"+name +".pickle", 'wb') as f:
                pickle.dump([rules,header], f)
                print("save in: "+"../dataset/"+name +".pickle")

    def addRulesFromPickle(self,name):
        with open("../dataset/"+name+".pickle", 'rb') as f:
            pk=pickle.load(f)
            rules=pk[0]
            header=pk[1]
            for rule in rules:
                self.addRule(rule,self.totalHeader)

if __name__=="__main__":
    allHeaders=["embeddedDepth","lineNumber","imgWidth","imgHeight","widthHeightRatio","red","green","blue","colorVariety","contrast","levelDistanceLowRatio","levelDistanceHighRatio","levelSimiliarDistanceLowRatio","levelSimiliarDistanceHighRatio","verticalZeroRatio","verticalMinusRatio","verticalPositiveRatio","verticalSimiliarZeroRatio","verticalSimiliarMinusRatio","verticalSimiliarPositiveRatio","horizonDistanceCloserRatio","horizonDistanceFatherRatio","horizonDistanceInFoundLevelCloserRatio","horizonDistanceInFoundLevelFatherRatio","childNumber","childTagNumber","siblingNumber","siblingTagNumber","uncleNumber","uncleTagNumber"]
    rl=rules(allHeaders)
    rl.addRulesFromPickle("rules_LnCv")
    rl.addRulesFromPickle("rules_EdCt")
    #for rule in rl.rules:
    fr=rl.searchFeatureLeft("colorVariety")
    for rule in rl.sortRules(fr,10):
        print(rl.showFeatureName(rule,allHeaders))
