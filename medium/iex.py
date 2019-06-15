import requests
import pandas as pd
from sklearn.externals import joblib

token = 'YOUR TOKEN HERE'

def get_quotes(symbol):
    link = 'https://cloud.iexapis.com/stable/stock/'+symbol+'/chart/5y?token='+token
    r = requests.get(link)
    return pd.DataFrame().from_dict(r.json())
