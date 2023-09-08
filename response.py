from models.run import Run
from classifier.network import NETWORK
from utils.spider import spider
class Chatbot():
    def __init__(self):
        self.Ans_classifier = NETWORK()
        self.Ans_machinelearning = Run()
        self.Ans_Spider = spider()
    def response(self,input):
        res = self.Ans_Spider.Getcontent(input)
        if res == None:
            res = self.Ans_classifier.predict(input)
            if res == None:
                res = self.Ans_machinelearning.predict(input)
        return res

