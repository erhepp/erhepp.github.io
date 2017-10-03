
### Process for retrieving data using USGS ComCat API

## Work with a sample JSON for quake data in WA and OR 

Learn how to work with the USGS ComCat formats


```python

import json
import pandas as pd
from pprint import pprint
```


```python
with open("/Users/erhepp/dsi-nyc-6/capstone/project-capstone/RetrieveData/WA-OR-catalog.json") as json_file:
    json_data = json.load(json_file)
```


```python
type(json_data)
```




    dict




```python
# print the json, but only the first earthquake as example to shorten output

for key, value in json_data.items():
    print key
    
print    
for key, value in json_data.items():
    if key == 'features':
        print "first feature"
        print value[0]
    else:    
        print key, value
    print
```

    type
    features
    bbox
    metadata
    
    type FeatureCollection
    
    first feature
    {u'geometry': {u'type': u'Point', u'coordinates': [-122.6071667, 48.7741667, 16.27]}, u'type': u'Feature', u'properties': {u'rms': 0.21, u'code': u'61297016', u'cdi': 3.8, u'sources': u',uw,us,', u'nst': 31, u'tz': -480, u'title': u'M 3.1 - 4km S of Marietta, Washington', u'magType': u'ml', u'detail': u'https://earthquake.usgs.gov/fdsnws/event/1/query?eventid=uw61297016&format=geojson', u'sig': 257, u'net': u'uw', u'type': u'earthquake', u'status': u'reviewed', u'updated': 1500447547040, u'felt': 286, u'alert': None, u'dmin': 0.1771, u'mag': 3.1, u'gap': 99, u'types': u',dyfi,geoserve,origin,phase-data,shakemap,', u'url': u'https://earthquake.usgs.gov/earthquakes/eventpage/uw61297016', u'ids': u',uw61297016,us1000955h,', u'tsunami': 0, u'place': u'4km S of Marietta, Washington', u'time': 1498725603370, u'mmi': 2.9}, u'id': u'uw61297016'}
    
    bbox [-126.8669, 42.3306, 3.64, -118.5953333, 48.7741667, 56.45]
    
    metadata {u'status': 200, u'count': 30, u'title': u'USGS Earthquakes', u'url': u'https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=2017-01-01&endtime=2017-07-01&minlatitude=42&maxlatitude=49&minlongitude=-127&maxlongitude=-116&eventtype=earthquake&producttype=dyfi&minmagnitude=2.5', u'generated': 1501363653000, u'api': u'1.5.8'}
    



```python
json_data['bbox']
```




    [-126.8669, 42.3306, 3.64, -118.5953333, 48.7741667, 56.45]




```python
json_data['metadata']['count']
```




    30




```python
event_list = []
for i in range(json_data['metadata']['count']):
    event_line = []
    event_line.append(json_data['features'][i]['id'])
    event_line += json_data['features'][i]['geometry']['coordinates']
    event_line.append(json_data['features'][i]['properties']['alert']),
    event_line.append(json_data['features'][i]['properties']['cdi']),
    event_line.append(json_data['features'][i]['properties']['code']),
    event_line.append(json_data['features'][i]['properties']['felt']),
    event_line.append(json_data['features'][i]['properties']['gap']),
    event_line.append(json_data['features'][i]['properties']['ids'][1:-1]),
    event_line.append(json_data['features'][i]['properties']['mag']),
    event_line.append(json_data['features'][i]['properties']['magType']),
    event_line.append(json_data['features'][i]['properties']['mmi']),
    event_line.append(json_data['features'][i]['properties']['net']),
    event_line.append(json_data['features'][i]['properties']['nst']),
    event_line.append(json_data['features'][i]['properties']['place']),
    event_line.append(json_data['features'][i]['properties']['rms']),
    event_line.append(json_data['features'][i]['properties']['sig']),
    event_line.append(json_data['features'][i]['properties']['sources']),
    event_line.append(json_data['features'][i]['properties']['status']),
    event_line.append(json_data['features'][i]['properties']['time']),
    event_line.append(json_data['features'][i]['properties']['title']),
    event_line.append(json_data['features'][i]['properties']['tsunami']),
    event_line.append(json_data['features'][i]['properties']['type']),
    event_line.append(json_data['features'][i]['properties']['types'][1:-1]),
    event_line.append(json_data['features'][i]['properties']['tz']),
    event_line.append(json_data['features'][i]['properties']['updated']),
    event_line.append(json_data['features'][i]['properties']['url']),

    event_list.append(event_line)

# Show only two event as example    
print event_list[0]
print
print event_list[1]
```

    [u'uw61297016', -122.6071667, 48.7741667, 16.27, None, 3.8, u'61297016', 286, 99, u'uw61297016,us1000955h', 3.1, u'ml', 2.9, u'uw', 31, u'4km S of Marietta, Washington', 0.21, 257, u',uw,us,', u'reviewed', 1498725603370, u'M 3.1 - 4km S of Marietta, Washington', 0, u'earthquake', u'dyfi,geoserve,origin,phase-data,shakemap', -480, 1500447547040, u'https://earthquake.usgs.gov/earthquakes/eventpage/uw61297016']
    
    [u'uw61276967', -121.6675, 48.2565, 8.47, None, 3.8, u'61276967', 31, 74, u'uw61276967,us20009pki', 2.92, u'ml', None, u'uw', 24, u'4km W of Darrington, Washington', 0.32, 143, u',uw,us,', u'reviewed', 1498361114610, u'M 2.9 - 4km W of Darrington, Washington', 0, u'earthquake', u'dyfi,geoserve,origin,phase-data', -480, 1501131675040, u'https://earthquake.usgs.gov/earthquakes/eventpage/uw61276967']



```python
cols = ['id', 'lat', 'long', 'depth', 'alert', 'cdi', 'code', 'felt', 'gap', 'ids', 'mag', 'magType', \
        'mmi', 'net', 'nst', 'place', 'rms', 'sig', 'sources', 'status', 'time', 'title', 'tsunami', \
        'type', 'types', 'tz', 'updated', 'url']
quakes = pd.DataFrame(event_list, columns = cols)
```


```python
quakes.head(2)
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
      <th>lat</th>
      <th>long</th>
      <th>depth</th>
      <th>alert</th>
      <th>cdi</th>
      <th>code</th>
      <th>felt</th>
      <th>gap</th>
      <th>ids</th>
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
      <td>uw61297016</td>
      <td>-122.607167</td>
      <td>48.774167</td>
      <td>16.27</td>
      <td>None</td>
      <td>3.8</td>
      <td>61297016</td>
      <td>286</td>
      <td>99</td>
      <td>uw61297016,us1000955h</td>
      <td>...</td>
      <td>,uw,us,</td>
      <td>reviewed</td>
      <td>1498725603370</td>
      <td>M 3.1 - 4km S of Marietta, Washington</td>
      <td>0</td>
      <td>earthquake</td>
      <td>dyfi,geoserve,origin,phase-data,shakemap</td>
      <td>-480</td>
      <td>1500447547040</td>
      <td>https://earthquake.usgs.gov/earthquakes/eventp...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>uw61276967</td>
      <td>-121.667500</td>
      <td>48.256500</td>
      <td>8.47</td>
      <td>None</td>
      <td>3.8</td>
      <td>61276967</td>
      <td>31</td>
      <td>74</td>
      <td>uw61276967,us20009pki</td>
      <td>...</td>
      <td>,uw,us,</td>
      <td>reviewed</td>
      <td>1498361114610</td>
      <td>M 2.9 - 4km W of Darrington, Washington</td>
      <td>0</td>
      <td>earthquake</td>
      <td>dyfi,geoserve,origin,phase-data</td>
      <td>-480</td>
      <td>1501131675040</td>
      <td>https://earthquake.usgs.gov/earthquakes/eventp...</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 28 columns</p>
</div>




```python
quakes.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 30 entries, 0 to 29
    Data columns (total 28 columns):
    id         30 non-null object
    lat        30 non-null float64
    long       30 non-null float64
    depth      30 non-null float64
    alert      1 non-null object
    cdi        30 non-null float64
    code       30 non-null object
    felt       30 non-null int64
    gap        30 non-null int64
    ids        30 non-null object
    mag        30 non-null float64
    magType    30 non-null object
    mmi        17 non-null float64
    net        30 non-null object
    nst        25 non-null float64
    place      30 non-null object
    rms        30 non-null float64
    sig        30 non-null int64
    sources    30 non-null object
    status     30 non-null object
    time       30 non-null int64
    title      30 non-null object
    tsunami    30 non-null int64
    type       30 non-null object
    types      30 non-null object
    tz         30 non-null int64
    updated    30 non-null int64
    url        30 non-null object
    dtypes: float64(8), int64(7), object(13)
    memory usage: 6.6+ KB



```python
quakes.types.unique()

```




    array([u'dyfi,geoserve,origin,phase-data,shakemap',
           u'dyfi,geoserve,origin,phase-data',
           u'dyfi,geoserve,moment-tensor,origin,phase-data,shakemap',
           u'dyfi,geoserve,impact-link,losspager,moment-tensor,origin,phase-data,shakemap'], dtype=object)




```python

```

## Pull actual data for continental US using USGS API.  

This was done in sections, by editing and re-running the code below


```python
import requests
```


```python
# This is the ComCat reqeust to return earthquakes in WA and OR by lat long box, with magnitude > 2.5 
# I did not limit it to dyfi (did you feel it) events, because I want to also examine any changes in 
# quake frenqency.  I pulled all data from 1970 to end of June 2017.

```


```python
wc_url = "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=1970-01-01&endtime=1974-12-31 \
&minlatitude=25&maxlatitude=49&minlongitude=-130&maxlongitude=-100\
&eventtype=earthquake&minmagnitude=2.5"
```


```python
ec_url = "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=2017-06-01&endtime=2017-06-30\
&minlatitude=25&maxlatitude=49&minlongitude=-99.9999&maxlongitude=-67\
&eventtype=earthquake&minmagnitude=2.5"
```


```python
ec_url
```




    'https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=2017-06-01&endtime=2017-06-30&minlatitude=25&maxlatitude=49&minlongitude=-99.9999&maxlongitude=-67&eventtype=earthquake&minmagnitude=2.5'




```python
# Please do not re-run this section, it pulls a lot of data from the USGS comcat server.   This is restricted
# to one month of eastern US data as an example.

# Package the request, send the request and catch the response: r
r = requests.get(ec_url)

# Decode the JSON data into a dictionary: json_data
json_data = r.json()
```


```python
r
```




    <Response [200]>




```python
# print the json, but only the first earthquake as example to shorten output

for key, value in json_data.items():
    print key
    
print    
for key, value in json_data.items():
    if key == 'features':
        print "first feature"
        print value[0]
    else:    
        print key, value
    print
```

    type
    features
    bbox
    metadata
    
    type FeatureCollection
    
    first feature
    {u'geometry': {u'type': u'Point', u'coordinates': [-98.8858, 36.5142, 5]}, u'type': u'Feature', u'properties': {u'rms': 0.59, u'code': u'100095bz', u'cdi': None, u'sources': u',us,', u'nst': None, u'tz': -360, u'title': u'M 2.8 - 29km ENE of Mooreland, Oklahoma', u'magType': u'mb_lg', u'detail': u'https://earthquake.usgs.gov/fdsnws/event/1/query?eventid=us100095bz&format=geojson', u'sig': 121, u'net': u'us', u'type': u'earthquake', u'status': u'reviewed', u'updated': 1506477798040, u'felt': None, u'alert': None, u'dmin': 0.239, u'mag': 2.8, u'gap': 109, u'types': u',geoserve,origin,phase-data,', u'url': u'https://earthquake.usgs.gov/earthquakes/eventpage/us100095bz', u'ids': u',us100095bz,', u'tsunami': 0, u'place': u'29km ENE of Mooreland, Oklahoma', u'time': 1498760916660, u'mmi': None}, u'id': u'us100095bz'}
    
    bbox [-99.6323, 32.925, 1.37, -74.9348, 46.1156, 24.51]
    
    metadata {u'status': 200, u'count': 98, u'title': u'USGS Earthquakes', u'url': u'https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime=2017-06-01&endtime=2017-06-30&minlatitude=25&maxlatitude=49&minlongitude=-99.9999&maxlongitude=-67&eventtype=earthquake&minmagnitude=2.5', u'generated': 1507041904000, u'api': u'1.5.8'}
    



```python
event_list = []
for i in range(json_data['metadata']['count']):
    event_line = []
    event_line.append(json_data['features'][i]['id'])
    event_line += json_data['features'][i]['geometry']['coordinates']
    event_line.append(json_data['features'][i]['properties']['alert']),
    event_line.append(json_data['features'][i]['properties']['cdi']),
    event_line.append(json_data['features'][i]['properties']['code']),
    event_line.append(json_data['features'][i]['properties']['detail']),
    event_line.append(json_data['features'][i]['properties']['dmin']),
    event_line.append(json_data['features'][i]['properties']['felt']),
    event_line.append(json_data['features'][i]['properties']['gap']),
    event_line.append(json_data['features'][i]['properties']['ids'][1:-1]),
    event_line.append(json_data['features'][i]['properties']['mag']),
    event_line.append(json_data['features'][i]['properties']['magType']),
    event_line.append(json_data['features'][i]['properties']['mmi']),
    event_line.append(json_data['features'][i]['properties']['net']),
    event_line.append(json_data['features'][i]['properties']['nst']),
    event_line.append(json_data['features'][i]['properties']['place']),
    event_line.append(json_data['features'][i]['properties']['rms']),
    event_line.append(json_data['features'][i]['properties']['sig']),
    event_line.append(json_data['features'][i]['properties']['sources']),
    event_line.append(json_data['features'][i]['properties']['status']),
    event_line.append(json_data['features'][i]['properties']['time']),
    event_line.append(json_data['features'][i]['properties']['title']),
    event_line.append(json_data['features'][i]['properties']['tsunami']),
    event_line.append(json_data['features'][i]['properties']['type']),
    event_line.append(json_data['features'][i]['properties']['types'][1:-1]),
    event_line.append(json_data['features'][i]['properties']['tz']),
    event_line.append(json_data['features'][i]['properties']['updated']),
    event_line.append(json_data['features'][i]['properties']['url']),

    event_list.append(event_line)
   
```


```python
cols = ['id', 'lat', 'long', 'depth', 'alert', 'cdi', 'code', 'detail', 'dmin', 'felt', 'gap', 'ids', \
        'mag', 'magType', 'mmi', 'net', 'nst', 'place', 'rms', 'sig', 'sources', 'status', 'time', \
        'title', 'tsunami', 'type', 'types', 'tz', 'updated', 'url']
quakes = pd.DataFrame(event_list, columns = cols)
```


```python
quakes.shape
```




    (98, 30)




```python
quakes.to_csv("./example_pull.csv", index=False),
```




    (None,)




```python
quakes.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 98 entries, 0 to 97
    Data columns (total 30 columns):
    id         98 non-null object
    lat        98 non-null float64
    long       98 non-null float64
    depth      98 non-null float64
    alert      0 non-null object
    cdi        64 non-null float64
    code       98 non-null object
    detail     98 non-null object
    dmin       54 non-null float64
    felt       64 non-null float64
    gap        98 non-null int64
    ids        98 non-null object
    mag        98 non-null float64
    magType    98 non-null object
    mmi        12 non-null float64
    net        98 non-null object
    nst        11 non-null float64
    place      98 non-null object
    rms        98 non-null float64
    sig        98 non-null int64
    sources    98 non-null object
    status     98 non-null object
    time       98 non-null int64
    title      98 non-null object
    tsunami    98 non-null int64
    type       98 non-null object
    types      98 non-null object
    tz         98 non-null int64
    updated    98 non-null int64
    url        98 non-null object
    dtypes: float64(10), int64(6), object(14)
    memory usage: 23.0+ KB


## Download complete - now to examine and clean it.

After downloading the separate files for east and west coast, and for smaller year spans,  I recombined everything into one US-quake.csv file, 45MB in size.  This can be loaded and used as the dataset for this capstone project


```python
import pandas as pd

df = pd.read_csv('US-quake.csv')
df.shape
```




    (114503, 30)




```python
df.head(3)
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
      <th>lat</th>
      <th>long</th>
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
      <td>-121.873500</td>
      <td>36.593000</td>
      <td>4.946</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1022389</td>
      <td>https://earthquake.usgs.gov/fdsnws/event/1/que...</td>
      <td>0.03694</td>
      <td>NaN</td>
      <td>...</td>
      <td>,nc,</td>
      <td>reviewed</td>
      <td>1.576600e+11</td>
      <td>M 3.4 - Central California</td>
      <td>0</td>
      <td>earthquake</td>
      <td>focal-mechanism,nearby-cities,origin,phase-data</td>
      <td>NaN</td>
      <td>1.481760e+12</td>
      <td>https://earthquake.usgs.gov/earthquakes/eventp...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>nc1022388</td>
      <td>-121.464500</td>
      <td>36.929000</td>
      <td>3.946</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1022388</td>
      <td>https://earthquake.usgs.gov/fdsnws/event/1/que...</td>
      <td>0.04144</td>
      <td>NaN</td>
      <td>...</td>
      <td>,nc,</td>
      <td>reviewed</td>
      <td>1.576470e+11</td>
      <td>M 3.0 - Central California</td>
      <td>0</td>
      <td>earthquake</td>
      <td>nearby-cities,origin,phase-data</td>
      <td>NaN</td>
      <td>1.481760e+12</td>
      <td>https://earthquake.usgs.gov/earthquakes/eventp...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ci3319041</td>
      <td>-116.128833</td>
      <td>29.907667</td>
      <td>6.000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3319041</td>
      <td>https://earthquake.usgs.gov/fdsnws/event/1/que...</td>
      <td>2.73400</td>
      <td>NaN</td>
      <td>...</td>
      <td>,ci,</td>
      <td>reviewed</td>
      <td>1.576410e+11</td>
      <td>M 4.6 - 206km SSE of Maneadero, B.C., MX</td>
      <td>0</td>
      <td>earthquake</td>
      <td>origin,phase-data</td>
      <td>NaN</td>
      <td>1.454030e+12</td>
      <td>https://earthquake.usgs.gov/earthquakes/eventp...</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 30 columns</p>
</div>




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
    time       114503 non-null float64
    title      114503 non-null object
    tsunami    114503 non-null int64
    type       114503 non-null object
    types      114503 non-null object
    tz         17198 non-null float64
    updated    114503 non-null float64
    url        114503 non-null object
    dtypes: float64(14), int64(2), object(14)
    memory usage: 26.2+ MB


## Discussion

Everything is currently text, and there is missing data.   This is expected, as not all information is recorded for every quake.  

A data dictionary describing this data is availabe at https://earthquake.usgs.gov/data/comcat/data-eventterms.php   

Of most importance to this project will be **time, lat, long, depth, magnitude, mmi** and **cdi.**  

time       114514 non-null object  
id         114514 non-null object  
lat        114514 non-null object  
long       114514 non-null object  
depth      114509 non-null object  
mag        114514 non-null object  
mmi        2966 non-null object  
cdi        14141 non-null object  

All but **mmi** and **cdi** are compete.   

The Modified Mercali Intensity (mmi) is only computed for quakes that cause damage, and cdi is computed from citizen reports beginning around 2005.  The full dataset will be used to examine any changes in earthquake frequency over time, and the smaller dataset of mmi/cdi will be used to calibrate the cdi intensity data.  With that done, the slighty larger cdi dataset can be used to explore the relationship between magnitude and intensity, and to validate the current formulas, and to develop new, more regionally focused, formulas.  
  
My evolving list of questions to attempt to answer with this data include ...

- have there been any increase or decrease in # of quakes in specific regions?
   - by depth, by magnitude, by intensity  
   - can we group by state, by zip code, by geologic unit, by fault zone …  
     - need to geocode, and somehow classify geologic characteristics  
   - can it be correlated to human activity, especially fracking or injection (these are separate)   
   
- Are there more discrete regional differences in magnitude/intensity relationship   
  
- Is citizen reported DYFI data, converted to CDI, accurate.  Does it correspond with MMI 



```python

```


```python
# Continuous Data
#    Convert to float
#       lat, long, depth, cdi, dmin, gap, mag, mmi, rms
df[['lat','long', 'depth', 'cdi', 'dmin', 'gap', 'mag', 'mmi', 'rms']] = \
df[['lat','long', 'depth', 'cdi', 'dmin', 'gap', 'mag', 'mmi', 'rms']].apply(pd.to_numeric, errors='coerce')

#    Convert to int
#       felt, nst, sig, tz(need to learn what this is)
df[['felt', 'nst', 'sig', 'tz']] = df[['felt', 'nst', 'sig', 'tz']].apply(pd.to_numeric, errors='coerce')
# Need to deal with the NaNs before casting as integer 
# df['felt'] = df['felt'].astype(int)
# df['nst'] = df['nst'].astype(int)
# df['sig'] = df['sig'].astype(int)
# df['tz'] = df['tz'].astype(int)

# Convert to boolian 
#   tsunami
df['tsunami'] = df['tsunami'].apply(pd.to_numeric, errors='coerce')
df['tsunami'] = df['tsunami'].astype(bool)

# Convert to date/time  
#           see: https://stackoverflow.com/questions/21787496/converting-epoch-time-with-milliseconds-to-datetime
#           example code given in cell below
#   time, updated

import datetime

df['time'] = df['time'].apply(pd.to_numeric)
df['updated'] = df['updated'].apply(pd.to_numeric)

df['time'] = df['time'].astype(int)
df['updated'] = df['updated'].astype(int)

df['time'] = df['time'].apply(lambda x: datetime.datetime.fromtimestamp(int(x)/1000.0) )
df['updated'] = df['updated'].apply(lambda x: datetime.datetime.fromtimestamp(int(x)/1000.0) )


# Convert to ordinal 
#   alert
#
# This code will convert, but how to handle blanks, which mean no alert issued? 
# Is it OK to make them 0, or does that give some unintended importance?
def ordinize(strval, ordered_list, start_idx, idx_skip):
    i = 0
    for val in ordered_list:
        if strval == val:
            return i*idx_skip + start_idx
        i += 1

df['alert'] = df['alert'].apply(lambda x: ordinize(x, ['green', 'yellow', 'red'], 1, 1))
df['alert'] = df['alert'].fillna(value=0)

# dummy (if used - not sure they will be needed for any of the project goals)
#   magType, net, sources, status

# leave as string (informational and reference only, not needed for model learning or prediction)
#   id, code detail, ids, place, title, type, types, url

df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 114503 entries, 0 to 114502
    Data columns (total 30 columns):
    id         114503 non-null object
    lat        114503 non-null float64
    long       114503 non-null float64
    depth      114498 non-null float64
    alert      114503 non-null float64
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
    time       114503 non-null datetime64[ns]
    title      114503 non-null object
    tsunami    114503 non-null bool
    type       114503 non-null object
    types      114503 non-null object
    tz         17198 non-null float64
    updated    114503 non-null datetime64[ns]
    url        114503 non-null object
    dtypes: bool(1), datetime64[ns](2), float64(13), int64(1), object(13)
    memory usage: 25.4+ MB



```python
# Deal with missing values

# Depth: 5 missing - delete these rows
df = df.dropna(subset=['depth'])

# magType: 86 missing,  fill with 'Unknown'  (there is one existing with Unknown)
df['magType'] = df['magType'].fillna(value='Unknown')


# cdi, mmi, felt:  Leave as NaN, analysis of mmi cdi/felt relationship will be done for existing values only
# dmin, gap, nst, rms, tz: Leave as NaN, no planned analysis or modeling uses these features.
```

    /Users/erhepp/anaconda/lib/python2.7/site-packages/ipykernel_launcher.py:7: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      import sys



```python
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
df.head()
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
      <th>lat</th>
      <th>long</th>
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
      <td>-121.873500</td>
      <td>36.593000</td>
      <td>4.946</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1022389</td>
      <td>https://earthquake.usgs.gov/fdsnws/event/1/que...</td>
      <td>0.03694</td>
      <td>NaN</td>
      <td>...</td>
      <td>,nc,</td>
      <td>reviewed</td>
      <td>157660096830</td>
      <td>M 3.4 - Central California</td>
      <td>False</td>
      <td>earthquake</td>
      <td>focal-mechanism,nearby-cities,origin,phase-data</td>
      <td>NaN</td>
      <td>1481756564940</td>
      <td>https://earthquake.usgs.gov/earthquakes/eventp...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>nc1022388</td>
      <td>-121.464500</td>
      <td>36.929000</td>
      <td>3.946</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1022388</td>
      <td>https://earthquake.usgs.gov/fdsnws/event/1/que...</td>
      <td>0.04144</td>
      <td>NaN</td>
      <td>...</td>
      <td>,nc,</td>
      <td>reviewed</td>
      <td>157646814820</td>
      <td>M 3.0 - Central California</td>
      <td>False</td>
      <td>earthquake</td>
      <td>nearby-cities,origin,phase-data</td>
      <td>NaN</td>
      <td>1481756553974</td>
      <td>https://earthquake.usgs.gov/earthquakes/eventp...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ci3319041</td>
      <td>-116.128833</td>
      <td>29.907667</td>
      <td>6.000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3319041</td>
      <td>https://earthquake.usgs.gov/fdsnws/event/1/que...</td>
      <td>2.73400</td>
      <td>NaN</td>
      <td>...</td>
      <td>,ci,</td>
      <td>reviewed</td>
      <td>157641167870</td>
      <td>M 4.6 - 206km SSE of Maneadero, B.C., MX</td>
      <td>False</td>
      <td>earthquake</td>
      <td>origin,phase-data</td>
      <td>NaN</td>
      <td>1454032083640</td>
      <td>https://earthquake.usgs.gov/earthquakes/eventp...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>usp00009ad</td>
      <td>-116.402000</td>
      <td>30.424000</td>
      <td>33.000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>p00009ad</td>
      <td>https://earthquake.usgs.gov/fdsnws/event/1/que...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>,us,</td>
      <td>reviewed</td>
      <td>157641135300</td>
      <td>M 4.0 - offshore Baja California, Mexico</td>
      <td>False</td>
      <td>earthquake</td>
      <td>origin</td>
      <td>NaN</td>
      <td>1415316088071</td>
      <td>https://earthquake.usgs.gov/earthquakes/eventp...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>usp000099y</td>
      <td>-116.185000</td>
      <td>30.757000</td>
      <td>33.000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>p000099y</td>
      <td>https://earthquake.usgs.gov/fdsnws/event/1/que...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>,ci,us,</td>
      <td>reviewed</td>
      <td>157550821500</td>
      <td>M 4.2 - offshore Baja California, Mexico</td>
      <td>False</td>
      <td>earthquake</td>
      <td>origin,phase-data</td>
      <td>NaN</td>
      <td>1454030304990</td>
      <td>https://earthquake.usgs.gov/earthquakes/eventp...</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
</div>



## Prototype methods for handling the time data


```python
# Here are the methods for dealing with the two time in milliseconds since the epoch (1970).  I'm leaving times
# In the dataframe, I'm leaving the times as 11 digit integors, and for the moment plan to deal with the 
# conversions when the times are needed for something.  The milliseconds are important in seismology for
# accurate compuation of location and magnitude, it's not clear to me yet whether that will need to be maintained
# for display

import time

ttime = 1481756553974   # example time value for test
s, ms = divmod(ttime, 1000) 
print '{}.{:03d}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(s)), ms)
```

    2016-12-14 23:02:33.974



```python
import datetime
import time

ttime = 1481756553974/1000   # example time value for test

print 'ttime:', ttime
print 'fromtimestamp(ttime):', datetime.date.fromtimestamp(ttime)


```

     ttime: 1481756553
    fromtimestamp(ttime): 2016-12-14



```python
print 'Now    :', datetime.datetime.now()
print 'Today  :', datetime.datetime.today()
print 'UTC Now:', datetime.datetime.utcnow()

d = datetime.datetime.now()
for attr in [ 'year', 'month', 'day', 'hour', 'minute', 'second', 'microsecond']:
    print attr, ':', getattr(d, attr)

```

     Now    : 2017-08-20 13:04:44.009927
    Today  : 2017-08-20 13:04:44.010349
    UTC Now: 2017-08-20 17:04:44.010535
    year : 2017
    month : 8
    day : 20
    hour : 13
    minute : 4
    second : 44
    microsecond : 10849



```python
import datetime
ttime = 1481756553974/float(1000)   # example time value for test
print ttime

dt = datetime.datetime.fromtimestamp(ttime)

for attr in [ 'year', 'month', 'day', 'hour', 'minute', 'second', 'microsecond']:
    print attr, ':', getattr(dt, attr)

```

    1481756553.97
    1975-12-14 18:02:33.974000
    year : 2016
    month : 12
    day : 14
    hour : 18
    minute : 2
    second : 33
    microsecond : 974000

