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

    def searchRules(self,filterDict):
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

    def sortRules(self,sortDict):
        pass

    def searchFeature(self):
        pass

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
        with open("../dataset"+name, 'wb') as f:
            pk=pickle.load(f)
            rules=pk[0]
            header=pk[1]
            for rule in rules:
                self.addRule(rule,header)

if __name__=="__main__":
    rl=rules()
    rl.addRulesFromPickle("rule5352")