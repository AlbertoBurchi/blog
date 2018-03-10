# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 00:59:07 2018

@author: MILA
"""

import random
from math import sqrt, exp
import numpy as np
import matplotlib.pyplot as plt


# create class of Option
class Option(object):
    def __init__(self, strike, S_beg, risk_free, maturity, st_dev,
                 number_trials):
        self.s = strike
        self.rf = risk_free
        self.cur = S_beg
        self.m = maturity
        self.std = st_dev
        self.ntr = number_trials
        
    # define brownian motion
    def random_step(self):

        step = random.gauss(0, 1)
        return step

    # define stock behavior
    def trace(self):
        t=1
        d_t = 1/360
        
        path = np.array([self.cur])
        S_0 = self.cur
        while t< self.m:
            S_0 = S_0*exp((self.rf - self.std**2/2)*d_t + self.std*self.random_step()*sqrt(d_t))
            t=t+1
            path = np.append(path, [S_0])
        return path
        
    # price of call option    
    def call(self, price = True):
        
        price_disc_c = [max(self.trace()[-1] - self.s,0)*exp(-self.m/360*self.rf) for i in range(self.ntr)]
                
        # proportion of exericsable call options
        self.p_call = sum(1 for x in price_disc_c if x>0)/self.ntr        
        if price == True:
            return np.mean(price_disc_c)
        else: 
            return price_disc_c
    
    # price of put option
    def put(self, price = True):
        
        price_disc_p = [max(self.s -self.trace()[-1],0)*exp(-self.m/360*self.rf) for i in range(self.ntr)]
                
        # proportion of exericsable call options
        self.p_put = sum(1 for y in price_disc_p if y>0)/self.ntr
        
        if price == True:
            return np.mean(price_disc_p)
        else:
            return price_disc_p
    
    # graph distribution of potential prices of an option
    def graph_dist(self, type_option = 'call'):
        
        if type_option == 'call':
            plt.hist(self.call(price = False), bins = min(100, int(self.ntr/10)), normed = True)
            plt.xlabel('call option price')
            plt.ylabel('density')
            plt.show()
        if type_option == 'put':
            plt.hist(self.put(price = False), bins = min(100, int(self.ntr/10)), normed = True)
            plt.xlabel('put option price')
            plt.ylabel('density')
            plt.show()
        else:
            print('no such option')
    
v_1 = Option(40, 42, 0.1, 180, 0.2, 1000)