from os import register_at_fork
from numpy.core.numeric import outer
import pandas as pd
import datetime as dt

# -------------- How To ---------------
# To use this program change the paths, and and the hardcoded column names in line 12 and 13

# Reading CSV files in, and creating new DF's to hold the data this is hardcoded.
DXData = pd.read_csv("DX-Y.NYB.csv")
OKData = pd.read_csv("OK_EU.csv")
UpdatedOK = pd.DataFrame(columns=["Date","RWTC","RBRTE"])
UpdatedDX = pd.DataFrame(columns=["Date","Open","High","Low","Close","Adj_Close","Volume"])

# Looping over the to files that has been read in, and using pandas inbuild date converter
i = 0    
for row in DXData.itertuples():
    UpdatedDX.loc[i] = pd.to_datetime(row.Date), row.Open, row.High, row.Low, row.Close, row.Adj_Close, row.Volume
    i += 1
    print(pd.to_datetime(row.Date))
 
i = 0  
for row in OKData.itertuples():
    UpdatedOK.loc[i] = pd.to_datetime(row.Date), row.RWTC, row.RBRTE
    i += 1
    print(pd.to_datetime(row.Date))
    
# Printing the updated date and data to files
UpdatedDX.to_csv(r"DX-Y.NYB.csv", index=False, header=True)
UpdatedOK.to_csv(r"OK_EU.csv", index=False, header=True)
    
