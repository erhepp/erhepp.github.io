
## Dataset Characteristics and Initial Observations


#### Dataset
In the above steps data were converted to the appropriate type.  



All data was obtained from the USGS Common Catalog API, and was supplied in text format.   Prior to this, in the Get-USGS-ComCat-Data.ipynb notebook,  earthquake records for a rectangular area encompassing the continental United States were pulled using the USGS Common Catalog API.  Records were restricted to those quakes occuring in 1970 or later having a magnitude greater than 2.5.   API size limitations required this to be done in smaller subsets, which were partially cleaned and then then reassembled into a complete csv file.   This file is the starting point for this notebook, focused on additional cleanup and EDA.
  
  - **lat, long, depth, cdi, dmin, gap, mag, mmi, rms, felt, nst, tz**  have been converted from string to float
  - **sig, tsunami** have been converted to int
  - **magType, net, sources, status** will be dummied when they are needed in later analysis
  - **id, code detail, ids, place, title, type, types, url**
 are left as string values.  They are informational and reference only, not needed for model learning or prediction
 
The dataset is incomplete for cdi, mmi, felt.  This is expected, as not all information is recorded for every quake.  Analysis of mmi cdi/felt relationship will be done using samples where this data exists. Likewise, dmin, gap, nst, rms, and tz have missing values. No correct or imputation is anticipated, as there ino planned analysis or modeling using these features in this project.

The Modified Mercali Intensity (mmi) is only computed for quakes that cause damage, and cdi is computed from citizen reports beginning around 2005.  The full dataset will be used to examine any changes in earthquake frequency over time, and the smaller dataset of mmi/cdi will be used to calibrate the cdi intensity data.  With that done, the slighty larger cdi dataset can be used to explore the relationship between magnitude and intensity, and to validate the current formulas, and to develop new, more regionally focused, formulas.  

A data dictionary describing this data is availabe at https://earthquake.usgs.gov/data/comcat/data-eventterms.php   

Of most importance to this project will be **time, lat, long, depth, magnitude, mmi** and **cdi.**  



```python
# Read in the dateset from csv

import numpy as np
import pandas as pd

df = pd.read_csv('./datasets/US-quake-raw.csv')
df.shape
```




    (114503, 30)




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 114503 entries, 0 to 114502
    Data columns (total 30 columns):
    id         114503 non-null object
    lat        114503 non-null float64
    long       114503 non-null float64
    depth      114498 non-null float64
    alert      693 non-null object
    cdi        14130 non-null float64
    code       114503 non-null object
    detail     114503 non-null object
    dmin       68118 non-null float64
    felt       14130 non-null float64
    gap        105961 non-null float64
    ids        114503 non-null object
    mag        114503 non-null float64
    magType    114417 non-null object
    mmi        2955 non-null float64
    net        114503 non-null object
    nst        98355 non-null float64
    place      114503 non-null object
    rms        108641 non-null float64
    sig        114503 non-null int64
    sources    114503 non-null object
    status     114503 non-null object
    time       114503 non-null int64
    title      114503 non-null object
    tsunami    114503 non-null int64
    type       114503 non-null object
    types      114503 non-null object
    tz         17198 non-null float64
    updated    114503 non-null int64
    url        114503 non-null object
    dtypes: float64(12), int64(4), object(14)
    memory usage: 26.2+ MB



```python
# Correctly type the dataset

# convert tsunami to boolian - either a tsunami alert was issued or it wasn't
df['tsunami'] = df['tsunami'].astype(bool)

# convert times to a true datetime format
import datetime

df['time'] = df['time'].astype(int)
df['updated'] = df['updated'].astype(int)

df['time'] = df['time'].map(lambda x: datetime.datetime.fromtimestamp(int(x)/1000.0) )
df['updated'] = df['updated'].map(lambda x: datetime.datetime.fromtimestamp(int(x)/1000.0) )


# Convert alert to ordinal 
#
def ordinize(strval, ordered_list, start_idx, idx_skip):
    i = 0
    for val in ordered_list:
        if strval == val:
            return i*idx_skip + start_idx
        i += 1

df['alert'] = df['alert'].apply(lambda x: ordinize(x, ['green', 'yellow', 'red'], 1, 1))
df['alert'] = df['alert'].fillna(value=0)

```

#### Missing values
  - Five quakes missing depth data were deleted.  
  - For 86 quakes missing the type of calculation used to determine magnitude, the type was set to 'Unknown." 


```python
# Handle missing values

# Depth: 5 missing - delete these rows
df = df.dropna(subset=['depth'])

# magType: 86 missing,  fill with 'Unknown'  (there is one existing with Unknown)
df['magType'] = df['magType'].fillna(value='Unknown')


# cdi, mmi, felt:  Leave as NaN, analysis of mmi cdi/felt relationship will be done for existing values only
# dmin, gap, nst, rms, tz: Leave as NaN, no planned analysis or modeling uses these features, but they can be
# converted later should the need arise.

df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 114498 entries, 0 to 114502
    Data columns (total 30 columns):
    id         114498 non-null object
    lat        114498 non-null float64
    long       114498 non-null float64
    depth      114498 non-null float64
    alert      114498 non-null float64
    cdi        14130 non-null float64
    code       114498 non-null object
    detail     114498 non-null object
    dmin       68118 non-null float64
    felt       14130 non-null float64
    gap        105961 non-null float64
    ids        114498 non-null object
    mag        114498 non-null float64
    magType    114498 non-null object
    mmi        2955 non-null float64
    net        114498 non-null object
    nst        98355 non-null float64
    place      114498 non-null object
    rms        108641 non-null float64
    sig        114498 non-null int64
    sources    114498 non-null object
    status     114498 non-null object
    time       114498 non-null datetime64[ns]
    title      114498 non-null object
    tsunami    114498 non-null bool
    type       114498 non-null object
    types      114498 non-null object
    tz         17198 non-null float64
    updated    114498 non-null datetime64[ns]
    url        114498 non-null object
    dtypes: bool(1), datetime64[ns](2), float64(13), int64(1), object(13)
    memory usage: 26.3+ MB



```python
df.head(2)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>long</th>
      <th>lat</th>
      <th>depth</th>
      <th>alert</th>
      <th>cdi</th>
      <th>code</th>
      <th>detail</th>
      <th>dmin</th>
      <th>felt</th>
      <th>...</th>
      <th>sources</th>
      <th>status</th>
      <th>time</th>
      <th>title</th>
      <th>tsunami</th>
      <th>type</th>
      <th>types</th>
      <th>tz</th>
      <th>updated</th>
      <th>url</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>nc1022389</td>
      <td>-121.8735</td>
      <td>36.593</td>
      <td>4.946</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1022389</td>
      <td>https://earthquake.usgs.gov/fdsnws/event/1/que...</td>
      <td>0.03694</td>
      <td>NaN</td>
      <td>...</td>
      <td>,nc,</td>
      <td>reviewed</td>
      <td>1974-12-30 13:28:16.830</td>
      <td>M 3.4 - Central California</td>
      <td>False</td>
      <td>earthquake</td>
      <td>focal-mechanism,nearby-cities,origin,phase-data</td>
      <td>NaN</td>
      <td>2016-12-14 18:02:44.940</td>
      <td>https://earthquake.usgs.gov/earthquakes/eventp...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>nc1022388</td>
      <td>-121.4645</td>
      <td>36.929</td>
      <td>3.946</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1022388</td>
      <td>https://earthquake.usgs.gov/fdsnws/event/1/que...</td>
      <td>0.04144</td>
      <td>NaN</td>
      <td>...</td>
      <td>,nc,</td>
      <td>reviewed</td>
      <td>1974-12-30 09:46:54.820</td>
      <td>M 3.0 - Central California</td>
      <td>False</td>
      <td>earthquake</td>
      <td>nearby-cities,origin,phase-data</td>
      <td>NaN</td>
      <td>2016-12-14 18:02:33.974</td>
      <td>https://earthquake.usgs.gov/earthquakes/eventp...</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 30 columns</p>
</div>



#### Latitude and Longituded were miss-labeled on retrieval.   

This is corrected here


```python
# Oops, the latitude and longitude are swapped, need to fix the column headers to match data.

df.rename(columns={'lat': 'newlong', \
                        'long': 'newlat'}, inplace=True)
```


```python
df.rename(columns={'newlong': 'long', \
                        'newlat': 'lat'}, inplace=True)
```


```python
df.head(2)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>long</th>
      <th>lat</th>
      <th>depth</th>
      <th>alert</th>
      <th>cdi</th>
      <th>code</th>
      <th>detail</th>
      <th>dmin</th>
      <th>felt</th>
      <th>...</th>
      <th>sources</th>
      <th>status</th>
      <th>time</th>
      <th>title</th>
      <th>tsunami</th>
      <th>type</th>
      <th>types</th>
      <th>tz</th>
      <th>updated</th>
      <th>url</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>nc1022389</td>
      <td>-121.8735</td>
      <td>36.593</td>
      <td>4.946</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1022389</td>
      <td>https://earthquake.usgs.gov/fdsnws/event/1/que...</td>
      <td>0.03694</td>
      <td>NaN</td>
      <td>...</td>
      <td>,nc,</td>
      <td>reviewed</td>
      <td>1974-12-30 13:28:16.830</td>
      <td>M 3.4 - Central California</td>
      <td>False</td>
      <td>earthquake</td>
      <td>focal-mechanism,nearby-cities,origin,phase-data</td>
      <td>NaN</td>
      <td>2016-12-14 18:02:44.940</td>
      <td>https://earthquake.usgs.gov/earthquakes/eventp...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>nc1022388</td>
      <td>-121.4645</td>
      <td>36.929</td>
      <td>3.946</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1022388</td>
      <td>https://earthquake.usgs.gov/fdsnws/event/1/que...</td>
      <td>0.04144</td>
      <td>NaN</td>
      <td>...</td>
      <td>,nc,</td>
      <td>reviewed</td>
      <td>1974-12-30 09:46:54.820</td>
      <td>M 3.0 - Central California</td>
      <td>False</td>
      <td>earthquake</td>
      <td>nearby-cities,origin,phase-data</td>
      <td>NaN</td>
      <td>2016-12-14 18:02:33.974</td>
      <td>https://earthquake.usgs.gov/earthquakes/eventp...</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 30 columns</p>
</div>



#### Future plotting and display of data Discussion

Future plotting and display of data will sometimes be more convenient if the magnitudes can be grouped by decace, i.e. all earthquakes with magnitude between 3.0 and 3.999 will be grouped in decade 2.  
  
It will also be convenient to group by year of occurance.  


```python
# Create a 'binned' magnitude column for later gropuing and plotting
# Need to make this an integer number for color coding points, so ....
# This will be called the magDecade, and will be set to the lowest integer in the decace
# Thus,  7 will mean quakes with magnitude from 7 - 7.999
#        6 will mean quakes with magnitude from 6 - 6.999   etc.


def magDecade(x):
    if x >= 8.0:
        mbin = 8
    elif x >= 7.0 and x < 8.0:
        mbin = 7
    elif x >= 6.0 and x < 7.0:
        mbin = 6
    elif x >= 5.0 and x < 6.0:  
        mbin = 5
    elif x >= 4.0 and x < 5.0:
        mbin = 4
    elif x >= 3.0 and x < 4.0:
        mbin = 3
    else:
        mbin = 2   # Dataset was restricted to quakes with magnitude greater than 2.5

    return mbin

df['magDecade'] = df['mag'].apply(lambda x: magDecade(x))

df['magDecade'].value_counts(dropna=False)

```




    2    71425
    3    36662
    4     5677
    5      656
    6       69
    7        9
    Name: magDecade, dtype: int64




```python
# Create a year column for later grouping and plotting 

df['year'] = df['time'].map(lambda x: getattr(x, 'year'))

```


```python
df[['id','long','lat','mag','time','magDecade','year']].head(3)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>long</th>
      <th>lat</th>
      <th>mag</th>
      <th>time</th>
      <th>magDecade</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>nc1022389</td>
      <td>-121.873500</td>
      <td>36.593000</td>
      <td>3.39</td>
      <td>1974-12-30 13:28:16.830</td>
      <td>3</td>
      <td>1974</td>
    </tr>
    <tr>
      <th>1</th>
      <td>nc1022388</td>
      <td>-121.464500</td>
      <td>36.929000</td>
      <td>2.99</td>
      <td>1974-12-30 09:46:54.820</td>
      <td>2</td>
      <td>1974</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ci3319041</td>
      <td>-116.128833</td>
      <td>29.907667</td>
      <td>4.58</td>
      <td>1974-12-30 08:12:47.870</td>
      <td>4</td>
      <td>1974</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (114498, 32)



#### Save the cleaned up version


```python
# Save this cleaned up dataset for use in this and later notebooks

RUN_BLOCK = False  # Prevent execution unless specifically intended

if RUN_BLOCK:
    df.to_csv("./datasets/clean_quakes_v2.csv", index=False)

```


```python
# Re-read and begin EDA
df = pd.read_csv("./datasets/clean_quakes_v2.csv")
```


```python
print df.shape
df.head(3)
```

    (114498, 32)





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>long</th>
      <th>lat</th>
      <th>depth</th>
      <th>alert</th>
      <th>cdi</th>
      <th>code</th>
      <th>detail</th>
      <th>dmin</th>
      <th>felt</th>
      <th>...</th>
      <th>time</th>
      <th>title</th>
      <th>tsunami</th>
      <th>type</th>
      <th>types</th>
      <th>tz</th>
      <th>updated</th>
      <th>url</th>
      <th>magDecade</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>nc1022389</td>
      <td>-121.873500</td>
      <td>36.593000</td>
      <td>4.946</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1022389</td>
      <td>https://earthquake.usgs.gov/fdsnws/event/1/que...</td>
      <td>0.03694</td>
      <td>NaN</td>
      <td>...</td>
      <td>1974-12-30 13:28:16.830</td>
      <td>M 3.4 - Central California</td>
      <td>False</td>
      <td>earthquake</td>
      <td>focal-mechanism,nearby-cities,origin,phase-data</td>
      <td>NaN</td>
      <td>2016-12-14 18:02:44.940</td>
      <td>https://earthquake.usgs.gov/earthquakes/eventp...</td>
      <td>3</td>
      <td>1974</td>
    </tr>
    <tr>
      <th>1</th>
      <td>nc1022388</td>
      <td>-121.464500</td>
      <td>36.929000</td>
      <td>3.946</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1022388</td>
      <td>https://earthquake.usgs.gov/fdsnws/event/1/que...</td>
      <td>0.04144</td>
      <td>NaN</td>
      <td>...</td>
      <td>1974-12-30 09:46:54.820</td>
      <td>M 3.0 - Central California</td>
      <td>False</td>
      <td>earthquake</td>
      <td>nearby-cities,origin,phase-data</td>
      <td>NaN</td>
      <td>2016-12-14 18:02:33.974</td>
      <td>https://earthquake.usgs.gov/earthquakes/eventp...</td>
      <td>2</td>
      <td>1974</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ci3319041</td>
      <td>-116.128833</td>
      <td>29.907667</td>
      <td>6.000</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>3319041</td>
      <td>https://earthquake.usgs.gov/fdsnws/event/1/que...</td>
      <td>2.73400</td>
      <td>NaN</td>
      <td>...</td>
      <td>1974-12-30 08:12:47.870</td>
      <td>M 4.6 - 206km SSE of Maneadero, B.C., MX</td>
      <td>False</td>
      <td>earthquake</td>
      <td>origin,phase-data</td>
      <td>NaN</td>
      <td>2016-01-28 20:48:03.640</td>
      <td>https://earthquake.usgs.gov/earthquakes/eventp...</td>
      <td>4</td>
      <td>1974</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 32 columns</p>
</div>




```python
print list(df.columns)
```

    ['id', 'long', 'lat', 'depth', 'alert', 'cdi', 'code', 'detail', 'dmin', 'felt', 'gap', 'ids', 'mag', 'magType', 'mmi', 'net', 'nst', 'place', 'rms', 'sig', 'sources', 'status', 'time', 'title', 'tsunami', 'type', 'types', 'tz', 'updated', 'url', 'magDecade', 'year']



```python
# Restrict to columns of interest
print list(df[['depth', 'alert', 'mmi', 'cdi', 'felt', 'mag', 'time']].columns)
```

    ['depth', 'alert', 'mmi', 'cdi', 'felt', 'mag', 'time']


#### Map of missing data
Black indicates data exists, white indicates missing values  
X-axis labels are on the left edge of the column they represent  
  
This is not a surprise, as Modified Mercalli Intensity (MMI) require expert, on site visits too assess, and are generally only done for significant, damage causing quakes.  CDI and Felt data comes from crowd sourced citizen reported data, and this program was only begun in the late '90s.  Also, there are few reports for low magnitude quakes that are not felt.  Comparison of MMI and CDI can be done only for quakes that have values for both.   


```python
# A map of missing data.  Clearly, it will be necessary to subset the quakes to use only those that have 
# available intensity data

# Seaborn set(font_scale=__) is just a quick way to adjust font scale 
# for matplotlib graphs drawn after this is called
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.5)

fig, ax = plt.subplots(figsize=(6,24))

ax.pcolor(df[['depth', 'mmi', 'cdi', 'felt', 'mag', 'time']].isnull(), cmap='gist_gray')
ax.set_ylabel("Row number")
ax.set_xlabel("Feature")
ax.set_xticklabels(df[['depth', 'mmi', 'cdi', 'felt', 'mag', 'time']].columns, rotation=90 )
plt.show()
```


![png](/images/Capstone-EDA_files/Capstone-EDA_25_0.png)


## Reverse Geocode the latitude and longitude

I wanted to get the country, state, county, and zip code for all quakes, and use that to filter by country to get just US quakes, or to examine frequency differences by state to see if there are any regions other than Oklahoma were quake frequency is increasing.   
  
The code below will do this, but Google limitations on the number of requests prevented this from being done for 114498 earthquaks.  A future project is to determine the method and cost to process this dataset through geocode to obtain the location data. 


```python
# Create columns for geographic data

RUN_BLOCK = False  # Prevent execution unless specifically intended

if RUN_BLOCK:
    df['q_country'] = np.nan
    df['q_state'] = np.nan
    df['q_county'] = np.nan
    df['q_zip_code'] = np.nan
```


```python
# Use geocode to add columns for country, state, county and postal_code

from pygeocoder import Geocoder

def is_offshore(lat, long):
    # Filter quakes off west coast before to lower the number that will be geocoded.
    if (qlat > 39) and (qlong < -124):
        return True
    elif qlong < (-114 + (qlat-26)*(-10/13.)):
        return True
    else:
        return False


RUN_BLOCK = False  # Prevent execution unless specifically intended

if RUN_BLOCK:  
    quakes = []
    idxs = []
    for idx in range(114000, 114498):
        if idx % 100 == 0:
            print "Retrieving geopolitical info for records", idx, "to", idx+100
            quake_id = df.loc[idx,'id']
            quakes.append(quake_id)
            idxs.append(idx)

        if is_offshore(df.iloc[idx,2], df.iloc[idx,1]):
            country.append('offshore')
            state.append('offshore')
            county.append('offshore')
            zip_code.append('offshore')
            df.loc[idx, 'country'] = 'offshore'
            df.loc[idx, 'state'] = 'offshore'
            df.loc[idx, 'county'] = 'offshore'
            df.loc[idx, 'zip_code'] = 'offshore'

        else:     
            try:
                results = Geocoder.reverse_geocode(df.iloc[idx,2], df.iloc[idx,1])
                country.append(results.country)
                state.append(results.state)
                county.append(results.county)
                zip_code.append(results.postal_code)
                df.loc[idx, 'country'] = results.country
                df.loc[idx, 'state'] = results.state
                df.loc[idx, 'county'] = results.county
                df.loc[idx, 'zip_code'] = results.postal_code

            except:
                country.append('offshore')
                state.append('offshore')
                county.append('offshore')
                zip_code.append('offshore')
                df.loc[idx, 'country'] = 'offshore'
                df.loc[idx, 'state'] = 'offshore'
                df.loc[idx, 'county'] = 'offshore'
                df.loc[idx, 'zip_code'] = 'offshore'

    print "Done retrieving reverse codes" 

```


```python
## Geo columns do not contain uniform data, so pickle to save dataframe as object rather than csv.

# import pickle

# pickle.dump( df, open( "clean_quakes_geocode.p", "wb" ) )

# df = pickle.load( open( "save_losers_df.p", "rb" ) )
```


```python
# Check for duplicate quakes  - there seem to be none with same ID

df[df.duplicated(subset='id')]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>long</th>
      <th>lat</th>
      <th>depth</th>
      <th>alert</th>
      <th>cdi</th>
      <th>code</th>
      <th>detail</th>
      <th>dmin</th>
      <th>felt</th>
      <th>...</th>
      <th>time</th>
      <th>title</th>
      <th>tsunami</th>
      <th>type</th>
      <th>types</th>
      <th>tz</th>
      <th>updated</th>
      <th>url</th>
      <th>magDecade</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
<p>0 rows × 32 columns</p>
</div>



## Exploratory Data Analysis

Now we get to the more interesting part.  The cells below generate
  - table of earthquake frequency by year for quakes with magnitude > 2.5 occuring in a rectangular box encompassing the continental United States.
  - stacked bar charts showing frequency over time


```python
# Earthquake Frequency

magSummary = df.groupby(['year', 'magDecade']).size().unstack()
magSummary = magSummary.fillna(0).astype(int)
magSummary.loc[1970:1984].T

```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>year</th>
      <th>1970</th>
      <th>1971</th>
      <th>1972</th>
      <th>1973</th>
      <th>1974</th>
      <th>1975</th>
      <th>1976</th>
      <th>1977</th>
      <th>1978</th>
      <th>1979</th>
      <th>1980</th>
      <th>1981</th>
      <th>1982</th>
      <th>1983</th>
      <th>1984</th>
    </tr>
    <tr>
      <th>magDecade</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>220</td>
      <td>277</td>
      <td>214</td>
      <td>354</td>
      <td>938</td>
      <td>1478</td>
      <td>1123</td>
      <td>869</td>
      <td>936</td>
      <td>1266</td>
      <td>1655</td>
      <td>1156</td>
      <td>1467</td>
      <td>2075</td>
      <td>1461</td>
    </tr>
    <tr>
      <th>3</th>
      <td>175</td>
      <td>474</td>
      <td>171</td>
      <td>227</td>
      <td>677</td>
      <td>967</td>
      <td>615</td>
      <td>491</td>
      <td>528</td>
      <td>786</td>
      <td>1773</td>
      <td>725</td>
      <td>674</td>
      <td>1088</td>
      <td>915</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11</td>
      <td>57</td>
      <td>14</td>
      <td>68</td>
      <td>138</td>
      <td>163</td>
      <td>126</td>
      <td>72</td>
      <td>112</td>
      <td>108</td>
      <td>537</td>
      <td>98</td>
      <td>104</td>
      <td>149</td>
      <td>127</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2</td>
      <td>6</td>
      <td>3</td>
      <td>17</td>
      <td>12</td>
      <td>15</td>
      <td>22</td>
      <td>11</td>
      <td>9</td>
      <td>16</td>
      <td>42</td>
      <td>6</td>
      <td>12</td>
      <td>19</td>
      <td>14</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>5</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
magSummary.loc[1985:1999].T
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>year</th>
      <th>1985</th>
      <th>1986</th>
      <th>1987</th>
      <th>1988</th>
      <th>1989</th>
      <th>1990</th>
      <th>1991</th>
      <th>1992</th>
      <th>1993</th>
      <th>1994</th>
      <th>1995</th>
      <th>1996</th>
      <th>1997</th>
      <th>1998</th>
      <th>1999</th>
    </tr>
    <tr>
      <th>magDecade</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>1541</td>
      <td>2692</td>
      <td>2022</td>
      <td>1532</td>
      <td>1516</td>
      <td>1153</td>
      <td>1039</td>
      <td>5418</td>
      <td>1399</td>
      <td>1997</td>
      <td>1127</td>
      <td>995</td>
      <td>1252</td>
      <td>1315</td>
      <td>2306</td>
    </tr>
    <tr>
      <th>3</th>
      <td>680</td>
      <td>1064</td>
      <td>860</td>
      <td>761</td>
      <td>781</td>
      <td>609</td>
      <td>606</td>
      <td>2045</td>
      <td>701</td>
      <td>1158</td>
      <td>611</td>
      <td>499</td>
      <td>599</td>
      <td>654</td>
      <td>1196</td>
    </tr>
    <tr>
      <th>4</th>
      <td>98</td>
      <td>162</td>
      <td>96</td>
      <td>93</td>
      <td>145</td>
      <td>105</td>
      <td>73</td>
      <td>281</td>
      <td>66</td>
      <td>154</td>
      <td>92</td>
      <td>68</td>
      <td>77</td>
      <td>80</td>
      <td>150</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>20</td>
      <td>9</td>
      <td>15</td>
      <td>9</td>
      <td>16</td>
      <td>10</td>
      <td>31</td>
      <td>9</td>
      <td>18</td>
      <td>7</td>
      <td>13</td>
      <td>12</td>
      <td>6</td>
      <td>10</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>5</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
magSummary.loc[2000:2016].T

```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>year</th>
      <th>2000</th>
      <th>2001</th>
      <th>2002</th>
      <th>2003</th>
      <th>2004</th>
      <th>2005</th>
      <th>2006</th>
      <th>2007</th>
      <th>2008</th>
      <th>2009</th>
      <th>2010</th>
      <th>2011</th>
      <th>2012</th>
      <th>2013</th>
      <th>2014</th>
      <th>2015</th>
      <th>2016</th>
    </tr>
    <tr>
      <th>magDecade</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>1116</td>
      <td>1108</td>
      <td>953</td>
      <td>1122</td>
      <td>1684</td>
      <td>1270</td>
      <td>960</td>
      <td>937</td>
      <td>1370</td>
      <td>1109</td>
      <td>3666</td>
      <td>1448</td>
      <td>1138</td>
      <td>1324</td>
      <td>2840</td>
      <td>3270</td>
      <td>2431</td>
    </tr>
    <tr>
      <th>3</th>
      <td>560</td>
      <td>569</td>
      <td>490</td>
      <td>628</td>
      <td>886</td>
      <td>655</td>
      <td>502</td>
      <td>399</td>
      <td>695</td>
      <td>487</td>
      <td>1846</td>
      <td>669</td>
      <td>479</td>
      <td>579</td>
      <td>1241</td>
      <td>1428</td>
      <td>1067</td>
    </tr>
    <tr>
      <th>4</th>
      <td>95</td>
      <td>117</td>
      <td>82</td>
      <td>123</td>
      <td>103</td>
      <td>124</td>
      <td>94</td>
      <td>76</td>
      <td>140</td>
      <td>94</td>
      <td>233</td>
      <td>160</td>
      <td>126</td>
      <td>125</td>
      <td>131</td>
      <td>111</td>
      <td>98</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>19</td>
      <td>11</td>
      <td>11</td>
      <td>11</td>
      <td>13</td>
      <td>15</td>
      <td>23</td>
      <td>36</td>
      <td>16</td>
      <td>19</td>
      <td>22</td>
      <td>13</td>
      <td>9</td>
      <td>9</td>
      <td>9</td>
      <td>18</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### Charting Earthquake Frequency
This bar chart shows the number of earthquakes yearly, separate by magnitude. There are many more quakes of lower magnitude than higher, as we know from experience, but the year to year trends are quite variable.   There were 2 large earthquakes in California in 1992, and the spike is likely to do the number of smaller aftershocks these quakes produced.  The spikes in 2010 and after are influcenced by the increased number of eartquakes in Oklahoma, thought to be caused by re-injection of waste water from oil wells.   

This dataset represents earthquakes entered in the USGS common catalog.  These are sourced from various state and academic agencies, and the number of contributors has increased over time.  The ramp up in the early 1970s is probably due to additional quakes being reported to USGS, rather than an actual sudden increase in seismic activity.


```python
import seaborn as sns
import matplotlib.pyplot as plt

# plt.style.use('fivethirtyeight')
%matplotlib inline

q_colors = ['#007FDD', '#008E7F', '#80E100', '#FFFF00', '#FFA000', '#FF0000']
ax = magSummary.loc[1970:2016,2:7].plot(kind='bar',stacked=True, colors=q_colors, \
                                               figsize=(20, 10), legend=True, fontsize=14)


ax.set_title("US Earthquake Counts", fontsize=18 )
ax.set_xlabel("Year", fontsize=16)
ax.set_ylabel("Earthquake Count", fontsize=16)
ax.legend(fontsize=16)
plt.show()


```


![png](/images/Capstone-EDA_files/Capstone-EDA_36_0.png)


### Mapping Earthquakes 

Here are two plots of earthquake locations.  The circle size and color represents magnitude; larger and more toward read are higher magnitude.  Notice the difference in number of quakes in Oklahoma between 2000, 2001 and 2015 to present.  This is believed to be caused by re-injection of wastewater from oil wells back into the ground, where it then lubricates existing faults. 


```python
import plotly.plotly as py
import pandas as pd

import plotly 
plotly.tools.set_credentials_file(username='erhepp', api_key='F6sL7oLP3PpzBpBDwz4U')
# erhepp   PLg3heim
```


```python
df_magplot = df[(df['year'] >= 2000) & (df['year'] < 2002)]\
  [['id','mag','lat','long', 'magDecade']].sort_values('mag', axis=0, ascending=False)
```


```python
df_magplot['text'] = df_magplot['id'] + '<br>Magnitude ' + (df_magplot['mag']).astype(str)
magDecades=[8, 7, 6, 5, 4, 3, 2]
colors = ["rgb(255,0,0)", "rgb(255,90,0)", "rgb(255,160,0)", "rgb(255,255,0)",\
          "rgb(128,255,0)", "rgb(0,142,128)", "rgb(0,128,255)", "rgb(0,0,255)"]
quakes = []
scale = 4
i = 0

for mbin in magDecades:
    df_sub = df_magplot[df_magplot['magDecade'] == mbin]
    quake = dict(
        type = 'scattergeo',
        locationmode = 'USA-states',
        lon = df_sub['long'],
        lat = df_sub['lat'],
        text = df_sub['text'],
        marker = dict(
            size = (df_sub['mag']**4)/scale,
            color = colors[i],
            line = dict(width=0.5, color='rgb(40,40,40)'),
            sizemode = 'area'
        ),
        name = '{0}'.format(mbin) )
    quakes.append(quake)
    i += 1

layout = dict(
        title = 'US Earthquakes 2000, 2001 to present<br>(Click legend to toggle traces)',
        showlegend = True,
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showland = True,
            landcolor = 'rgb(217, 217, 217)',
            subunitwidth=1,
            countrywidth=1,
            subunitcolor="rgb(255, 255, 255)",
            countrycolor="rgb(255, 255, 255)"
        ),
    )

fig = dict( data=quakes, layout=layout )
py.iplot( fig, validate=False, filename='d3-bubble-map-2000-2001-magnitude' )
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~erhepp/4.embed" height="525px" width="100%"></iframe>




```python
df_magplot = df[df['year'] >= 2015]\
  [['id','mag','lat','long', 'magDecade']].sort_values('mag', axis=0, ascending=False)
```


```python
df_magplot['text'] = df_magplot['id'] + '<br>Magnitude ' + (df_magplot['mag']).astype(str)
magDecades=[8, 7, 6, 5, 4, 3, 2]
colors = ["rgb(255,0,0)", "rgb(255,90,0)", "rgb(255,160,0)", "rgb(255,255,0)",\
          "rgb(128,255,0)", "rgb(0,142,128)", "rgb(0,128,255)", "rgb(0,0,255)"]
quakes = []
scale = 4
i = 0

for mbin in magDecades:
    df_sub = df_magplot[df_magplot['magDecade'] == mbin]
    quake = dict(
        type = 'scattergeo',
        locationmode = 'USA-states',
        lon = df_sub['long'],
        lat = df_sub['lat'],
        text = df_sub['text'],
        marker = dict(
            size = (df_sub['mag']**4)/scale,
            color = colors[i],
            line = dict(width=0.5, color='rgb(40,40,40)'),
            sizemode = 'area'
        ),
        name = '{0}'.format(mbin) )
    quakes.append(quake)
    i += 1

layout = dict(
        title = 'US Earthquakes 2015 to present<br>(Click legend to toggle traces)',
        showlegend = True,
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showland = True,
            landcolor = 'rgb(217, 217, 217)',
            subunitwidth=1,
            countrywidth=1,
            subunitcolor="rgb(255, 255, 255)",
            countrycolor="rgb(255, 255, 255)"
        ),
    )

fig = dict( data=quakes, layout=layout )
py.iplot( fig, validate=False, filename='d3-bubble-map-2015-present-magnitude' )
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~erhepp/6.embed" height="525px" width="100%"></iframe>



### Earthquakes with crowd sourced Intensity data


The first modeling step of this project will be to compare the expert determined Modified Mercalli Intensity (MMI) with the Community Decimal Intensity (CDI).  CDI is calculated by the USGS using "Did You Feel It" (DYFI) reports submitted on-line by those who experienced the quake and chose to contribute.  This is a far more limited dataset, as Modified Mercalli Intensity (MMI) requires expert, on site visits too assess, and are generally only done for significant, damage causing quakes. CDI data collection began in the late '90s.  There are few reports for low magnitude quakes that are not felt. Comparison of MMI and CDI can be done only for quakes that have values for both.  Furthermore, the quality of CDI is improved by multiple responses.    

Intensity decreases with distance from the quake location, and DYFI reports reflect this.  The USGS aggregates and averages reports by zip code, and provides this averaged value at latitude and longitude correpsonding to the center of the zip code region.   The quality of CDI is improved by multiple responses, and so for this initial look at MMI / CDI correlation, I will use only quakes with 5 or more CDI reports, which further reduces the dataset.  The maximum observed MMI and CDI for each quake will be used.  This plot shows that subset of quakes, with both MMI and CDI values, with CDI computed from 5 or more DYFI reports.  



```python
df_magplot = df[(df['felt'] >= 5) & df['mmi'].notnull()]\
   [['id','mag','lat','long', 'magDecade']].sort_values('mag', axis=0, ascending=False)
```


```python
df_magplot['text'] = df_magplot['id'] + '<br>Magnitude ' + (df_magplot['mag']).astype(str)
magDecades=[8, 7, 6, 5, 4, 3, 2]
colors = ["rgb(255,0,0)", "rgb(255,90,0)", "rgb(255,160,0)", "rgb(255,255,0)",\
          "rgb(128,255,0)", "rgb(0,142,128)", "rgb(0,128,255)", "rgb(0,0,255)"]
quakes = []
scale = 4
i = 0

for mbin in magDecades:
    
    df_sub = df_magplot[df_magplot['magDecade'] == mbin]
    
    quake = dict(
        type = 'scattergeo',
        locationmode = 'USA-states',
        lat = df_sub['lat'],
        lon = df_sub['long'],
        text = df_sub['text'],
        marker = dict(
            size = (df_sub['mag']**4)/scale,
            color = colors[i],
            line = dict(width=0.5, color='rgb(40,40,40)'),
            sizemode = 'area'
        ),
        name = '{0}'.format(mbin) )
    quakes.append(quake)
    i += 1

layout = dict(
        title = 'Earthquakes with MMI and CDI (from > 5 DYFI reports) data<br>(Click legend to toggle traces)',
        showlegend = True,
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showland = True,
            landcolor = 'rgb(217, 217, 217)',
            subunitwidth=1,
            countrywidth=1,
            subunitcolor="rgb(255, 255, 255)",
            countrycolor="rgb(255, 255, 255)"
        ),
    )

fig = dict( data=quakes, layout=layout )
py.iplot( fig, validate=False, filename='d3-bubble-map-mmi-cdi-magnitude' )
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~erhepp/8.embed" height="525px" width="100%"></iframe>



### Distribution of Magnitude and Intensity data
Let's look at the distribution of data in the final dataset, from 1970 through June of 2017, with both MMI and CDI values, and with CDI computed from 5 or more DYFI reports.   

We see the dataset is limited to quakes with magnitude >= 2.5, and that MMI (expert deried) seems to assess quakes at slightly higher intensity that does CDI (crowd sourced). 


```python
dfm =  df[(df['felt'] >= 5) & df['mmi'].notnull()]
dfm.shape
```




    (1690, 32)




```python
sns.set(font_scale=1.5)
fig, axs = plt.subplots(ncols=3, figsize=(18,6))
fig.suptitle("Distributions for Earthquakes with both CDI and MMI data", fontsize=22)

dfm['mag'].plot(kind='hist', bins=14, range=(0,7), ax=axs[0])
dfm['mmi'].plot(kind='hist', bins=18, range=(0,9), ax=axs[1])
dfm['cdi'].plot(kind='hist', bins=18, range=(0,9), ax=axs[2])
axs[0].set_xlabel('Magnitude', fontsize=18)
axs[1].set_xlabel('Maximum Modified Mercalli Intensity', fontsize=18)
axs[2].set_xlabel('Maxumum Community Decimal Intensity', fontsize=18)
axs[0].locator_params(numticks=9)

```


![png](/images/Capstone-EDA_files/Capstone-EDA_48_0.png)


What did we lose by restricting to quakes with both MMI and CDI values, and with CDI computed from 5 or more DYFI reports?  Here are similar distribution plots, but using all quakes in the dataset from 1970 onward.  Observe carefuly the y-axis in these plots, as there are many quakes with neither MMI or CDI intensity measurement.  

This makes is logical, remember that intensity quantifies the oberved effect of the quake, and for small quakes there may be no observed effect.  MMI is never computed for these, and CDI is only reported if an individual choses to do so, and they have little motivation or interest to do so for quakes that were not felt.

While we dropped a lot of small quakes (see the magnitude distribution plot),  we also dropped a lot of suspect CDI values, producing a more normally distributed dataset.  There is little change in the MMI plot, because of the understandable lack of MMI assessment for small quakes.  


```python
fig, axs = plt.subplots(ncols=3, figsize=(18,6))
fig.suptitle("Distributions for all Earthquakes", fontsize=22)

df['mag'].plot(kind='hist', bins=14, range=(0,7), ax=axs[0])
df['mmi'].plot(kind='hist', bins=18, range=(0,9), ax=axs[1])
df['cdi'].plot(kind='hist', bins=18, range=(0,9), ax=axs[2])
axs[0].set_xlabel('Magnitude', fontsize=18)
axs[1].set_xlabel('Maximum Modified Mercalli Intensity', fontsize=18)
axs[2].set_xlabel('Maxumum Community Decimal Intensity', fontsize=18)
axs[0].locator_params(numticks=9)
```


![png](/images/Capstone-EDA_files/Capstone-EDA_50_0.png)



```python

```
