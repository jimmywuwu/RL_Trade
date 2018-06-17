
import numpy as np
import time
import pandas as pd
import sys


class Market(object):
    def __init__(self,begin_date="2016/1/1",end_date="2017/5/5"):
        self.action_space = [-1,0,1]
        #action record price,time,action
        self.lstm_length=10
        self.get_price=self.price_data(begin_date,end_date)
        self.action_record=pd.DataFrame(columns=['Date', 'time','price','Vol','action'])
        self.ot=list(self.get_price.__next__())
        self.count_episode=0
        self.n_actions=3
        
    
    def price_data(self,begin_date,end_date):
        series=pd.read_csv('parse_over_data.csv')
        for x in range(self.lstm_length):
            series["price.lag"+str(x+1)]=series.price.shift(x+1)
            series["Vol.lag"+str(x+1)]=series.Vol.shift(x+1)
        series=series[(series.Date>begin_date)&(series.Date<end_date)]
        series=series.dropna()
        self.n_features=series.shape[1]-3
        for index,row in series.iterrows():
            yield row
    

    def step(self, action):
        
        # record St,Ot,At

        self.action_record.loc[self.count_episode]=self.ot[1:5]+[action]
        
        # reward function
        self.ot=list(self.get_price.__next__())
        s_ = self.ot[3:]
        reward=sum((s_[0]-self.action_record[self.action_record.action != 0].price)*self.action_record[self.action_record.action != 0].action)
        self.count_episode+=1
        return s_,reward

    

