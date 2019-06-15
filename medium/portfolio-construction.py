import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns

etfs = ['mtum.csv','reet.csv', 'pdbc.csv', 'iwd.csv', 'fm.csv', 'igov.csv']
db_list = []
for etf in etfs:
    db = pd.read_csv(etf)
    db_list.append(db)

etf_0 = db_list[0][['date', 'close']]
for etf_1 in db_list[1:]:
    etf_0 = etf_0.merge(etf_1[['date', 'close']], 'left', on = 'date')
etf_0.columns = ['date', 'mtum', 'reet', 'pdbc', 'iwd', 'fm', 'igov']
etf_0.dropna(axis = 0, inplace = True)

etf_return = etf_0.drop('date', axis = 1).pct_change(1).dropna()


def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = mean_returns.shape[0]
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = minimize(neg_sharpe_ratio, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_var, p_ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_var

def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns*weights ) *252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns

def equal_risk_contributon(weights, cov_matrix):
    marginal_risk = weights.T*np.dot(cov_matrix, weights)
    marginal_risk = np.abs(np.sum(marginal_risk)/4-marginal_risk)
    
    return np.sum(marginal_risk)*1000

def min_risk_diff(mean_returns, cov_matrix):
    num_assets = mean_returns.shape[0]
    args = (cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = minimize(equal_risk_contributon, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

#modified
def neg_sharpe_ratio_m(weights, mean_returns, cov_matrix, risk_free_rate, alpha):
    p_var, p_ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
    m_risk = equal_risk_contributon(weights, cov_matrix)
    return -(p_ret - risk_free_rate)/p_var*alpha + (1-alpha)*m_risk

def max_sharpe_ratio_m(mean_returns, cov_matrix, risk_free_rate, alpha):
    num_assets = mean_returns.shape[0]
    args = (mean_returns, cov_matrix, risk_free_rate, alpha)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = minimize(neg_sharpe_ratio_m, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

i= 0
length = etf_return.shape[0]//66
risk_free_rate =0.0
weights = np.empty([0,6])
weights_parity = np.empty([0,6])
while i<length:
    distance = min(66,etf_return.shape[0]-i*66)
    cov_matrix = np.cov(etf_return.iloc[i*66: (i+1)*min(distance, 66) , :].values, 
                                        rowvar = False)    
    rt = np.mean(etf_return.iloc[i*66: (i+1)*min(distance, 66) , :].values, axis=0)
    result = np.around(max_sharpe_ratio(rt, cov_matrix, risk_free_rate), 3)
    result_parity = np.around(min_risk_diff(rt, cov_matrix), 3)
    weights = np.vstack((weights, result))
    weights_parity = np.vstack((weights_parity, result_parity))
    i+=1

def portfolio_performance(weights):
    portfolio = np.empty([0,6])
    for i, row in enumerate(etf_return.iloc[66:].values):
        index = i//66
        #print(row)
        portfolio = np.vstack((portfolio,row*weights[index]))
    return pd.Series(np.sum(portfolio, 1)+1).cumprod()
weights_equal = np.ones([17,6])/6
efficient = portfolio_performance(weights)
risk_parity = portfolio_performance(weights_parity)
equal = portfolio_performance(weights_equal)

dates = etf_0.iloc[1:].iloc[66::66,0].values

'''performance graph'''
plt.clf()
fig, ax = plt.subplots(1,1, figsize=(12,9))

_ = plt.gcf().subplots_adjust(bottom=0.2)

_ = plt.plot(efficient-1, label = 'efficient', color = 'blue')
_ = plt.plot(risk_parity-1, label = 'risk_parity', color = 'green')
_ = plt.plot(equal-1, label = 'equal', color='red')
vals = ax.get_yticks()
_ = ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])

xposition = [x*66 for x in range(17)]
for xc in xposition:
    _ = plt.axvline(x=xc, color='k', linestyle='--', linewidth = 0.5 )
_ = plt.legend()
_ = plt.xticks(np.arange(0, 1089, step=66), dates, rotation = 90, fontsize = 9)
_ = plt.yticks(fontsize = 9)
_ = plt.xlabel('Reweighting intervals', fontsize = 9)
_ = plt.ylabel('Return', fontsize = 9)
_ = plt.title('Comparison of Strategies performance' ,fontsize = 15)
plt.plot()

'''weights graph '''
def plot_weight(weights, name):
    plt.clf()
    colors = sns.color_palette(["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"])
    fig, ax = plt.subplots(1,1, figsize=(12,9))
    _ = plt.gcf().subplots_adjust(bottom=0.2)
    for i, etf in enumerate(etf_return.columns.values):
        _ = plt.bar(np.array(range(17)), weights[:,i], bottom = np.sum(weights[:,:i],1), 
                    label = etf, color = colors[i], width = 0.99, edgecolor = 'white')
    _ = ax.legend(bbox_to_anchor=(-0.05, .99))
    vals = ax.get_yticks()
    _ = ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    _ = plt.xticks(np.arange(0, 17, 1), dates, rotation = 90, fontsize = 9)
    _ = plt.xlabel('Time', fontsize = 9)
    _ = plt.ylabel('Portfolio weights', fontsize = 9 )
    _ = plt.title('Changes in portfolio weights ({})'.format(name))
    plt.show() 

plot_weight(weights_parity, 'equal risk')

'''cross-asset graph'''
def weights_variance(weight):
    cross_asset = np.std(weight, 1)
    cross_time = np.std(weight,0)
    wvr = (1+np.mean(cross_asset))/(1+np.mean(cross_time))
    return cross_time, cross_asset, wvr

ca_e, ct_e, _ = weights_variance(weights)
ca_rp, ct_rp, _ = weights_variance(weights_parity)

from scipy.ndimage.filters import gaussian_filter1d
plt.clf()
colors = sns.color_palette(["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"])
fig, ax = plt.subplots(1,1, figsize=(12,9))
_ = plt.gcf().subplots_adjust(bottom=0.2)
_ =  plt.plot(np.array(range(17)), gaussian_filter1d(ct_e, sigma=2), label = 'efficient', color = colors[0])
_ =  plt.plot(np.array(range(17)), gaussian_filter1d(ct_rp,sigma=2), label = 'risk-parity', color = colors[5])
_ = plt.xticks(np.arange(0, 17, 1), dates, rotation = 90, fontsize = 9)
_ = plt.legend()
_ = plt.xlabel('Time', fontsize = 9)
_ = plt.ylabel('Volatility (std)')
_ = plt.title("Portfolios' cross-asset volatility")
plt.show()

'''cross-asset cross-time graph'''
plt.clf()
fig, ax = plt.subplots(1,1, figsize=(12,9))
_ = plt.gcf().subplots_adjust(bottom=0.2)
x = [np.mean(ca_e), np.mean(ca_rp)]
y = [np.mean(ct_e), np.mean(ct_rp)]
txt = ['efficient', 'risk-parity']
marker = ['*', 'o']
for i, txt in enumerate(txt):
    _ = plt.scatter(x[i], y[i], color = colors[i], marker = marker[i], 
                s = 40)
    _ = plt.annotate('{} ({:.2f}, {:.2f})'.format(txt, x[i], y[i]), (x[i]+0.01, y[i]+0.01))
_ = plt.xlim(0,0.5)
_ = plt.ylim(0,0.5)
_ = plt.xlabel("Cross-asset weights' volatility (std)", fontsize = 9)
_ = plt.ylabel("Cross-time weights' volatility (std)", fontsize = 9)
_ = plt.title("Mapping portfolios' weights volatility")
plt.show()

'''blended portfolio'''
weights_alpha =[]
blended_list = []

for alpha in 1/np.exp2(np.arange(0,10,1)):
    i= 0
    risk_free_rate =0.0
    weights = np.empty([0,6])
    while i<length:
        distance = min(66,etf_return.shape[0]-i*66)
        cov_matrix = np.cov(etf_return.iloc[i*66: (i+1)*min(distance, 66) , :].values, 
                                            rowvar = False)    
        rt = np.mean(etf_return.iloc[i*66: (i+1)*min(distance, 66) , :].values, axis=0)
        result = np.around(max_sharpe_ratio_m(rt, cov_matrix, risk_free_rate, alpha), 3)       
        weights = np.vstack((weights, result))
        i+=1
    weights_alpha.append(weights)
    blended = portfolio_performance(weights)
    blended_list.append(blended)

'''blended grapgh'''
plt.clf()
fig, ax = plt.subplots(1,1, figsize=(12,9))
_ = plt.gcf().subplots_adjust(bottom=0.2)
colors = sns.color_palette("RdBu_r", 10)
alpha = 1/np.exp2(np.arange(0,10,1))
for idx, blended in enumerate(blended_list):
    _ = plt.plot(blended, label = 'alpha={:.3f}'.format(alpha[idx]), color = colors[idx])
vals = ax.get_yticks()
_ = ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
xposition = [x*66 for x in range(17)]
for xc in xposition:
    _ = plt.axvline(x=xc, color='k', linestyle='--', linewidth = 0.5 )
_ = _ = ax.legend(bbox_to_anchor=(0.95, .99))
_ = plt.xticks(np.arange(0, 1089, step=66), dates, rotation = 90, fontsize = 9)
_ = plt.xlabel("Time", fontsize = 9)
_ = plt.ylabel("Performance, %", fontsize = 9)
_ = plt.title("Blended portfolios' performance", fontsize = 12)
plt.show()
