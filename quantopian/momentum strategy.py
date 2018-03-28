from quantopian.pipeline import Pipeline
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import ExponentialWeightedMovingAverage
from quantopian.pipeline.filters.morningstar import Q1500US
from quantopian.pipeline.classifiers.fundamentals import Sector  
import pandas as pd
import numpy as np

def initialize(context):
    # Schedule our rebalance function to run at the start of each month.
    schedule_function(my_rebalance, date_rules.month_start(), time_rules.market_open(hours=1))

    # Record variables at the end of each day.
    schedule_function(my_record_vars, date_rules.every_day(), time_rules.market_close())

    # Create our pipeline and attach it to our algorithm.
    my_pipe = make_pipeline()
    attach_pipeline(my_pipe, 'my_pipeline')
    
    set_commission(commission.PerShare(cost=0.0075, min_trade_cost=1.00))
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1))
    
     
def make_pipeline():
    """
    Create our pipeline.
    """

    # Base universe set to the Q1500US.
    base_universe = Q1500US()

    # 10-day close price average.
    mean_30 = ExponentialWeightedMovingAverage(inputs=[USEquityPricing.close], window_length=30, mask=base_universe, decay_rate = 0.9)

    # 30-day close price average.
    mean_120 = ExponentialWeightedMovingAverage(inputs=[USEquityPricing.close], window_length=120, mask=base_universe, decay_rate = 0.9)

    percent_difference = (mean_30 - mean_120) / mean_120

    # Filter to select securities to long.
    longs = percent_difference.top(50)
    long_value = percent_difference
    
    # sector attribution
    sector = Sector()
    
    #pipeline construction
    return Pipeline(
        columns={
			'long value': long_value,
            'sector': sector
        },
        screen=longs,
    )


def before_trading_start(context, data):
    # Gets our pipeline output every day.
    context.output = pipeline_output('my_pipeline')

    # Go long in securities for which the 'longs' value is True.
    context.longs = context.output.index.tolist()
    #print(context.output.head(5))

def choice_within_sector(context, data):
    
    sectors = [101, 102, 103, 104, 205, 206, 207, 308, 309, 310, 311]
    l = []  
    for sector in sectors:
        a = context.output[context.output['sector'] == sector]['long value'].max()
        l.append(a)
    l = [x for x in l if isinstance(x, np.float64) == True]
    
    # reverse function of funding index by value
    l_reverse = []
    for val in l:
        b = context.output[context.output['long value'] == val].index[0]
    	l_reverse.append(b)
    print(l_reverse)
    return l_reverse, l
    
def my_rebalance(context, data):
    """
    Rebalance monthly.
    """
    
    best_performers_in_sector, value_of_best_performers = choice_within_sector(context, data)
    
        
    for security in context.portfolio.positions:
        if security not in best_performers_in_sector and data.can_trade(security):
            order_target(security, int(context.portfolio.positions[security].amount/2))

    for i, security in enumerate(best_performers_in_sector):
        if data.can_trade(security):
            order_target_percent(security, value_of_best_performers[i]/sum(value_of_best_performers)*0.50)



def my_record_vars(context, data):
    """
    Record variables at the end of each day.
    """
    longs = 0
    for position in context.portfolio.positions.itervalues():
        if position.amount > 0:
            longs += 1


    # Record our variables.
    record(leverage=context.account.leverage, long_count=longs)
