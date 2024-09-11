import pandas as pd
import datetime as dt

# -------------- To Do ----------------
# - Merge DateFormater.py into this file
# - Check what file to merge into to ensure no incorrect data at the end of the shorter file
# -

# ------------- How To ----------------
# This program is made to help users combine 2 CSV files, to do so just change
# the file paths below. The progam will guide you through the use in the terminal after.
# If order is of importance uncomment line 60 and change it to the column order you want-


# These can be changed to the 2 CSV files you want to merge
data1 = pd.read_csv("DX-Y.NYB.csv")
data2 = pd.read_csv("OK_EU.csv") 

fuelDataHeader = list(data1.columns)
dollarRateHeader = list(data2.columns)

# Helper fucntion for dropping columns, takes the union of 2 strings
def union(arr1, arr2):
    outArr = []
    for val1 in arr1:
        for val2 in arr2:
            if (val1 == val2):
                outArr.append(val1) 
    return outArr

# Helper fucntion to change dateformat, but is currently moved to another file called DateFormater.py. 
def DateFormater (unConvertedDate) :
    #if "-" not in unConvertedDate :
    #    convertedDate = dt.datetime.utcfromtimestamp(unConvertedDate).strftime("%Y-%m-%d %H:%M")
    #    return convertedDate
    #else :
        return pd.to_datetime(unConvertedDate)

# Printing the files headers so the user can drop columns
print("Fuel Data Header: ")
print(fuelDataHeader)
print()
print("Dollar Rate Header: ")
print(dollarRateHeader)
print()

# Getting user input
dropList =  [item for item in input("Enter the collums to drop : ").split()]

# Creating droplist and dropping columns
data1Drop = union(dropList, fuelDataHeader)
data2Drop = union(dropList, dollarRateHeader)

data1.drop(data1Drop, axis=1, inplace=True)
data2.drop(data2Drop, axis=1, inplace=True)

# Merging the DF's and ensureing the correct order we wanted, and printing to flile
mergedData = data1.merge(data2, on="Date", how="outer").fillna(method="ffill")
print(mergedData.columns)
#mergedData = mergedData[["Date","RWTC","Open"]] # This line is hardcoded can can be uncommented if the order is of importance

mergedData.to_csv(r"mergedData.csv", index=False, header=True)   