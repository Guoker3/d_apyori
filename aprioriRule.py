import pickle
class rules:
    def __init__(self,header):
        # rule: ((pretuple, posttuple),support, confidence, lift)
        self.rules=list()
        self.header=header

    def addRule(self,ruleIn):
        rule = [[[x for x in list(ruleIn[0][0])],[x for x in list(ruleIn[0][1])]], ruleIn[1], ruleIn[2], ruleIn[3]]
        self.rules.append(rule)

    def addRules(self,rulesIn):
        for ruleIn in rulesIn:
            self.addRule(ruleIn)

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

            if "maxConfidence" in filterDict:
                #bigger in retRules
                pass
        return retRules

    def showFeatureName(self,rule):
        ii = [[self.header[int(x / 3)] + ": " + str(x) for x in list(rule[0][0])],[self.header[int(x / 3)] + ": " + str(x) for x in list(rule[0][1])], rule[1], rule[2], rule[3]]
        print(ii)

    def savePickle(self,rules,name=None):
        if name==None:
            with open("../dataset/rules"+str(len(self.rules))+".pickle", 'wb') as f:
                pickle.dump(rules, f)
        else:
            with open("../dataset/"+name +".pickle", 'wb') as f:
                pickle.dump(rules, f)

    def addFromPickle(self,name):
        with open("../dataset"+name, 'wb') as f:
            rules=pickle.load(f)
            for rule in rules:
                self.addRule(rule)