import numpy as np
import pandas as pd
import cProfile

df = pd.read_excel("clinics.xls", sheet_name = "Results")
print(df.head())

def haversine(lat1, lon1, lat2, lon2):
    MILES = 3959
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1 
    dlon = lon2 - lon1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    total_miles = MILES * c
    return total_miles

def haversine_looping(df):
    distance_list = []
    for i in range(0, len(df)):
        d = haversine(40.671, -73.985, df.iloc[i]['locLat'], df.iloc[i]['locLong'])
        distance_list.append(d)
    return distance_list
cProfile.run("df['distance'] = haversine_looping(df)")

#%%
cProfile.run("df['distance'] = haversine(40.671, -73.985, df['locLat'], df['locLong'])")

#%%
cProfile.run("df['distance'] = haversine(40.671, -73.985, df['locLat'].values, df['locLong'].values)")