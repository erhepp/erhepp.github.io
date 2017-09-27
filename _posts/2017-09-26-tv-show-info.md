
# What makes a successful TV show?

This analysis and model attemts to determine the factors that influence high or low IMDb ratings for TV shows.  All generes are examined, and while most originate in the United States, there are a few from the UK and elsewhere included.   

Two separate models are developed.  In both, the top and bottom rated shows are classified as winners and losers, respectively, and an array of 12 classisfiers are applied using cross validation to identify the best performing model. 25% of the data was reserved as a test set, and cross-validation scores and test set scores are both shown in the tables below. Baseline score is 0.52.  

The first utilizes natural language processing (NLP) on the IMDb summary descriptions of each show.  Term Frequency - Inverse Document Frequency and Count Vectorization were used on n-grams of size 2-4 were used.  With both vectorization techniqes, Random Forrest and Naive Bayesian classifiers were most successful, with the highest score of 0.642 was achieved using TF-IDF vectorization and a Multinomial Naive Bayes classifier.   The n-grams with highest cumulative score are identified as the most significant factors in the model, giving a clue as to the words in a summary description that foretell a show's likelihood of being a winner or loser.  

A second model, using factors such as genre, lenght, schedule times, network and format was also built, and the same set of 12 classifiers was applied.  The ADA Boost classifier achieved the top score of 0.92 using these factors.  

_Data collection and cleanup was tedious, and involved multiple runs of webscraping IMDb pages on show ratings, then using the TVmaze API to return show detals.  Unless interested in these details, the reader is encouraged to skip to the section titled "Modeling Section" a bit more than halfway through this notebook._

**Results Summary:**

From the NLP models, it seems shows featuring adult characters in crime and drama series set in times before or after the present in New York will fare better than reality or animated series featuring children or teens and highlighting pop culture.  

The model on the factors other than the summary showed similar tendencies.  Realty formats were the strongest negative factor in predicting success, while the scripted format was the strongest positive predictor.  Game and Talk shows were negative, while crime, science fiction, comedy, drama and documentaries were positive predictors.   Shows aired by HBO and BBC predicted success, while the lower rated shows were found more predominantly on MTV, E!, Comedy Central and Lifetime.  

Though interesting associations have been found, it must be said that nothing in the techniques used here can be interpreted as causality. For example, it cannot be said that reality shows featuring teenagers will always flop.  This report is based on initial efforts to determine factors that may influence a show’s success, and have shown a path for future more detailed modeling.  Suggested future paths include textual analysis of critics’ reviews, analysis based on cast or producers, analysis of differences in rating based on audience demographics, and a more detailed look at the connection between genre and the type/style of show.  


```python
# Import libraries needed for scraping and saving results.  
# Additional libraries needed for modeling, analysis and display will be imported when needed.

import requests
import pandas as pd
from bs4 import BeautifulSoup
import pickle
```

## Data Acquisition:  List of top rated TV Shows 


```python
# Retrieve current top 250 TV shows webpage

url = "http://www.imdb.com/chart/toptv/"
r = requests.get(url)
html = r.text
html[0:200]
```




    u'\n\n\n\n<!DOCTYPE html>\n<html\nxmlns:og="http://ogp.me/ns#"\nxmlns:fb="http://www.facebook.com/2008/fbml">\n    <head>\n        <meta charset="utf-8">\n        <meta http-equiv="X-UA-Compatible" content="IE=ed'




```python
# Use Beautiful soup to extract the imdb numbers from the webpage
soup = BeautifulSoup(html, "lxml")
```


```python
# Scrape the IMDb numbers for the 250 top rated shows

show_list = []
for tbody in soup.findAll('tbody', class_='lister-list'):
    for title in tbody.findAll('td', class_='titleColumn'):
        show_list.append(str(title.findAll('a')).split("/")[2])

show_list

```




    ['tt5491994',
     'tt0185906',
     'tt0795176',
     'tt0944947',
     'tt0903747',
     'tt0306414',
     'tt2861424',
     'tt2395695',
     'tt0081846',
     'tt0071075',
     'tt0141842',
     'tt1475582',
     'tt1533395',
     'tt0417299',
     'tt0098769',
     'tt1806234',
     'tt0303461',
     'tt0092337',
     'tt0052520',
     'tt3530232',
     'tt2356777',
     'tt1355642',
     'tt2802850',
     'tt0103359',
     'tt0296310',
     'tt0877057',
     'tt4508902',
     'tt0475784',
     'tt2092588',
     'tt0213338',
     'tt1856010',
     'tt0063929',
     'tt0112130',
     'tt2571774',
     'tt0081834',
     'tt0367279',
     'tt4742876',
     'tt4574334',
     'tt2085059',
     'tt0108778',
     'tt0098904',
     'tt3718778',
     'tt0081912',
     'tt0098936',
     'tt1518542',
     'tt0074006',
     'tt2707408',
     'tt0193676',
     'tt1865718',
     'tt0096548',
     'tt0072500',
     'tt0384766',
     'tt0118421',
     'tt0096697',
     'tt0090509',
     'tt0121955',
     'tt0386676',
     'tt4299972',
     'tt2560140',
     'tt0472954',
     'tt0412142',
     'tt0214341',
     'tt5555260',
     'tt2442560',
     'tt5712554',
     'tt0200276',
     'tt0353049',
     'tt1910272',
     'tt0086661',
     'tt0248654',
     'tt5189670',
     'tt0121220',
     'tt1486217',
     'tt0096639',
     'tt0120570',
     'tt4786824',
     'tt1628033',
     'tt0348914',
     'tt0403778',
     'tt5288312',
     'tt0459159',
     'tt3032476',
     'tt0407362',
     'tt4093826',
     'tt0773262',
     'tt0417349',
     'tt3322312',
     'tt0264235',
     'tt0106179',
     'tt0286486',
     'tt2297757',
     'tt0088484',
     'tt2098220',
     'tt5425186',
     'tt0318871',
     'tt0094517',
     'tt0436992',
     'tt1586680',
     'tt0092324',
     'tt0994314',
     'tt0203082',
     'tt1606375',
     'tt0380136',
     'tt0187664',
     'tt1513168',
     'tt0118273',
     'tt0421357',
     'tt1641384',
     'tt0314979',
     'tt5834204',
     'tt0092455',
     'tt0115147',
     'tt4295140',
     'tt0080306',
     'tt1266020',
     'tt1831164',
     'tt3920596',
     'tt0804503',
     'tt1492966',
     'tt0053488',
     'tt0086831',
     'tt0758745',
     'tt0995832',
     'tt0434706',
     'tt2401256',
     'tt0423731',
     'tt0111958',
     'tt0863046',
     'tt1733785',
     'tt2049116',
     'tt0275137',
     'tt1305826',
     'tt0472027',
     'tt2100976',
     'tt1489428',
     'tt0112159',
     'tt4158110',
     'tt1227926',
     'tt1870479',
     'tt0979432',
     'tt0106028',
     'tt0387764',
     'tt0237123',
     'tt0047708',
     'tt0088509',
     'tt0290978',
     'tt1984119',
     'tt0098825',
     'tt2306299',
     'tt0280249',
     'tt3647998',
     'tt0094525',
     'tt0163507',
     'tt0118266',
     'tt0182629',
     'tt0080297',
     'tt0061287',
     'tt1758429',
     'tt3671754',
     'tt0487831',
     'tt0388629',
     'tt2575988',
     'tt4189022',
     'tt0458254',
     'tt2788432',
     'tt0096657',
     'tt0346314',
     'tt1474684',
     'tt4288182',
     'tt0417373',
     'tt1298820',
     'tt0262150',
     'tt1695360',
     'tt1230180',
     'tt2243973',
     'tt0129690',
     'tt1632701',
     'tt2433738',
     'tt0149460',
     'tt1124373',
     'tt0075520',
     'tt1795096',
     'tt1442449',
     'tt5249462',
     'tt2937900',
     'tt1439629',
     'tt5071412',
     'tt0397150',
     'tt0083466',
     'tt2701582',
     'tt5114356',
     'tt4156586',
     'tt0319969',
     'tt0103584',
     'tt0302199',
     'tt0070644',
     'tt1883092',
     'tt2311418',
     'tt3428912',
     'tt1442437',
     'tt0362192',
     'tt0278238',
     'tt0387199',
     'tt2384811',
     'tt0098833',
     'tt0074028',
     'tt2303687',
     'tt0807832',
     'tt0056751',
     'tt0173528',
     'tt3358020',
     'tt0103466',
     'tt1526318',
     'tt0185133',
     'tt0075572',
     'tt0112084',
     'tt1837492',
     'tt2919910',
     'tt1299368',
     'tt0094535',
     'tt1520211',
     'tt0108906',
     'tt0988824',
     'tt5421602',
     'tt5853176',
     'tt0934320',
     'tt0337898',
     'tt0495212',
     'tt0460681',
     'tt2407574',
     'tt0290988',
     'tt1598754',
     'tt1119644',
     'tt1220617',
     'tt3398228',
     'tt0411008',
     'tt0163503',
     'tt2249364',
     'tt1409055',
     'tt4270492',
     'tt0060028',
     'tt0118480',
     'tt0925266',
     'tt3012698',
     'tt0402711',
     'tt0068098',
     'tt0442632',
     'tt1839578',
     'tt0043208',
     'tt5673782']




```python
# This code has been executed, and the results pickled and stored locally, so no need to run these requests
# to the API again. The api address with key to look up show with imdb number is
# http://api.tvmaze.com/lookup/shows?imdb=<show imdb identifier>

DO_NOT_RUN = True     # Do not run when notebook is loaded to avoid unnecessary calls to the API

if not DO_NOT_RUN:
    shows = pd.DataFrame()
    for show_id in show_list:
            try:
                print show_id
                # Get the tv show info from the api
                url = "http://api.tvmaze.com/lookup/shows?imdb=" + show_id
                r = requests.get(url)

                # convert the return data to a dictionary
                json_data = r.json()

                # load a temp datafram with the dictionary, then append to the composite dataframe
                temp_df = pd.DataFrame.from_dict(json_data, orient='index', dtype=None)
                ttemp_df = temp_df.T     # Was not able to load json in column orientation, so must transpose
                shows = shows.append(ttemp_df, ignore_index=True)
            except: 
                print show_id, " could not be retrieved from api"

    shows.head()    


```


```python
# write the contents of an object to a file for later retrieval

DO_NOT_RUN = True   # Be sure to check the file name to write before enabling execution on this block

if not DO_NOT_RUN:
    pickle.dump( shows, open( "save_shows_df.p", "wb" ) )
```

## Get list of bottom rated TV Series


```python
# This code block was changed multiple times to pull html with different sets of low rated shows
# ultimately about 1200 imdb ids were scraped, and about 1/3 of those could be pulled from the TV Maze API.

url ="http://www.imdb.com/search/title?count=600&languages=en&title_type=tv_series&user_rating=3.4,5.0&sort=user_rating,asc"
r = requests.get(url)
html = r.text
html[0:200]
```




    u'\n\n\n\n<!DOCTYPE html>\n<html\nxmlns:og="http://ogp.me/ns#"\nxmlns:fb="http://www.facebook.com/2008/fbml">\n    <head>\n        <meta charset="utf-8">\n        <meta http-equiv="X-UA-Compatible" content="IE=ed'




```python
# Use Beautiful soup to extract the imdb numbers from the webpage
soup = BeautifulSoup(html, "lxml")

loser_list = []
for div in soup.findAll('div', class_='lister-list'):
    for h3 in div.findAll('h3', class_='lister-item-header'):
        loser_list.append(str(h3.findAll('a')).split("/")[2])

loser_list

```




    ['tt0773264',
     'tt1798695',
     'tt1307083',
     'tt4845734',
     'tt0046641',
     'tt1519575',
     'tt0853078',
     'tt0118423',
     'tt0284767',
     'tt4052124',
     'tt0878801',
     'tt3703500',
     'tt1105170',
     'tt4363582',
     'tt3155428',
     'tt0362350',
     'tt0287196',
     'tt2766052',
     'tt0405545',
     'tt0262975',
     'tt0367278',
     'tt7134262',
     'tt1695352',
     'tt0421470',
     'tt2466890',
     'tt0343305',
     'tt1002739',
     'tt1615697',
     'tt0274262',
     'tt0465320',
     'tt1388381',
     'tt0358889',
     'tt1085789',
     'tt1011591',
     'tt0364804',
     'tt1489335',
     'tt3612584',
     'tt0363377',
     'tt0111930',
     'tt0401913',
     'tt0808086',
     'tt0309212',
     'tt5464192',
     'tt0080250',
     'tt4533338',
     'tt4741696',
     'tt1922810',
     'tt1793868',
     'tt4789316',
     'tt0185054',
     'tt1079622',
     'tt1786048',
     'tt0790508',
     'tt1716372',
     'tt0295098',
     'tt3409706',
     'tt0222574',
     'tt2171325',
     'tt0442643',
     'tt2142117',
     'tt0371433',
     'tt0138244',
     'tt1002010',
     'tt0495557',
     'tt1811817',
     'tt5529996',
     'tt1352053',
     'tt0439346',
     'tt0940147',
     'tt3075138',
     'tt1974439',
     'tt2693842',
     'tt0092325',
     'tt6772826',
     'tt1563069',
     'tt0489598',
     'tt0142055',
     'tt1566154',
     'tt0338592',
     'tt0167515',
     'tt2330327',
     'tt1576464',
     'tt2389845',
     'tt0186747',
     'tt0355096',
     'tt1821877',
     'tt0112033',
     'tt1792654',
     'tt0472243',
     'tt6453018',
     'tt3648886',
     'tt1599374',
     'tt2946482',
     'tt4672020',
     'tt1016283',
     'tt2649480',
     'tt1229945',
     'tt2390606',
     'tt1876612',
     'tt0140732',
     'tt1176156',
     'tt0158522',
     'tt4922726',
     'tt0068104',
     'tt2798842',
     'tt1150627',
     'tt1545453',
     'tt3685566',
     'tt0287223',
     'tt4185510',
     'tt0329912',
     'tt0289808',
     'tt0358849',
     'tt2320439',
     'tt0906840',
     'tt0800281',
     'tt1103082',
     'tt2416362',
     'tt3493906',
     'tt0381827',
     'tt0817553',
     'tt0252172',
     'tt0799872',
     'tt0816224',
     'tt1077162',
     'tt1918005',
     'tt1240983',
     'tt1415000',
     'tt5039916',
     'tt0451467',
     'tt0296438',
     'tt1159990',
     'tt0144701',
     'tt4718304',
     'tt1095213',
     'tt1453090',
     'tt0168372',
     'tt0425725',
     'tt3300126',
     'tt1415098',
     'tt5459976',
     'tt4041694',
     'tt2322264',
     'tt1441005',
     'tt1117549',
     'tt0365991',
     'tt0364807',
     'tt1591375',
     'tt3562462',
     'tt6118186',
     'tt3587176',
     'tt1372127',
     'tt0445865',
     'tt2088493',
     'tt4658248',
     'tt0103444',
     'tt4956964',
     'tt1326185',
     'tt0406422',
     'tt1973659',
     'tt1578933',
     'tt0446621',
     'tt1850624',
     'tt0159177',
     'tt0490539',
     'tt0306398',
     'tt0288922',
     'tt0465336',
     'tt0176397',
     'tt1641939',
     'tt0498879',
     'tt0306296',
     'tt1394277',
     'tt0398416',
     'tt2849552',
     'tt1433566',
     'tt0806893',
     'tt3252890',
     'tt3774098',
     'tt0791275',
     'tt5690224',
     'tt0361181',
     'tt0486953',
     'tt1514319',
     'tt3697290',
     'tt1342752',
     'tt0478936',
     'tt0094448',
     'tt0795101',
     'tt1340759',
     'tt0840061',
     'tt1151434',
     'tt0281429',
     'tt0845745',
     'tt2993514',
     'tt0783634',
     'tt1650352',
     'tt1249256',
     'tt2135766',
     'tt3231114',
     'tt1702421',
     'tt2940494',
     'tt6664486',
     'tt0081857',
     'tt1319598',
     'tt0247094',
     'tt6392176',
     'tt0320969',
     'tt2720144',
     'tt0360266',
     'tt2287380',
     'tt1715368',
     'tt0282291',
     'tt2248736',
     'tt2010634',
     'tt1489432',
     'tt4855578',
     'tt1721484',
     'tt0380850',
     'tt3084090',
     'tt2392683',
     'tt1381004',
     'tt1628058',
     'tt2935638',
     'tt1837169',
     'tt2404111',
     'tt2364381',
     'tt0888095',
     'tt2352123',
     'tt1013862',
     'tt4295320',
     'tt1249227',
     'tt1879603',
     'tt0167566',
     'tt0924528',
     'tt0361144',
     'tt0133300',
     'tt5888698',
     'tt1468817',
     'tt4006060',
     'tt0106096',
     'tt0287243',
     'tt1287376',
     'tt0060032',
     'tt1535270',
     'tt4831262',
     'tt0416397',
     'tt1546138',
     'tt2203971',
     'tt0214353',
     'tt0368518',
     'tt0382506',
     'tt5317980',
     'tt2313839',
     'tt1202295',
     'tt4146118',
     'tt1226448',
     'tt0403748',
     'tt0415448',
     'tt4665932',
     'tt3016956',
     'tt1412249',
     'tt1829773',
     'tt0872053',
     'tt0481443',
     'tt0493098',
     'tt0039120',
     'tt1411598',
     'tt0106123',
     'tt1740718',
     'tt0362153',
     'tt1637756',
     'tt0120974',
     'tt2328067',
     'tt0057741',
     'tt1261356',
     'tt2559390',
     'tt0083433',
     'tt0380934',
     'tt4388486',
     'tt0108821',
     'tt0115338',
     'tt0167735',
     'tt0460630',
     'tt2330453',
     'tt0398429',
     'tt0294140',
     'tt0804423',
     'tt2191952',
     'tt1118131',
     'tt4016700',
     'tt5786580',
     'tt0950199',
     'tt1760165',
     'tt4896654',
     'tt0414719',
     'tt1675974',
     'tt0465343',
     'tt1477137',
     'tt0115171',
     'tt3565412',
     'tt0382458',
     'tt0945153',
     'tt0199278',
     'tt1353293',
     'tt1426343',
     'tt2180165',
     'tt5117094',
     'tt1191039',
     'tt0497857',
     'tt0780409',
     'tt2670950',
     'tt1385183',
     'tt3396736',
     'tt2563482',
     'tt4094138',
     'tt0295065',
     'tt1696268',
     'tt0891053',
     'tt0914267',
     'tt1786018',
     'tt1988479',
     'tt1707814',
     'tt1595853',
     'tt2310444',
     'tt5434894',
     'tt0267216',
     'tt0855313',
     'tt1832828',
     'tt0426685',
     'tt2309561',
     'tt2486556',
     'tt0284786',
     'tt3136814',
     'tt1989818',
     'tt1179310',
     'tt0424748',
     'tt1126298',
     'tt0944946',
     'tt1882639',
     'tt0439904',
     'tt0875887',
     'tt1624991',
     'tt2747670',
     'tt2324247',
     'tt0403810',
     'tt1724452',
     'tt2366252',
     'tt3752894',
     'tt0198211',
     'tt1491318',
     'tt1666205',
     'tt2460474',
     'tt0303435',
     'tt0453329',
     'tt0220938',
     'tt0299264',
     'tt0783341',
     'tt0850175',
     'tt1191056',
     'tt0235917',
     'tt0111892',
     'tt0166442',
     'tt2643770',
     'tt5633924',
     'tt0075485',
     'tt0423657',
     'tt5327970',
     'tt3326032',
     'tt5785658',
     'tt2190731',
     'tt0101041',
     'tt3317020',
     'tt4732076',
     'tt2305717',
     'tt3828162',
     'tt0890935',
     'tt0449460',
     'tt0126175',
     'tt3601886',
     'tt5062878',
     'tt1579911',
     'tt0407354',
     'tt6723012',
     'tt5819414',
     'tt4180738',
     'tt0300802',
     'tt2649738',
     'tt3181412',
     'tt0382400',
     'tt3189040',
     'tt0324919',
     'tt2168240',
     'tt2560966',
     'tt0168373',
     'tt0403824',
     'tt0375440',
     'tt3746054',
     'tt2488150',
     'tt4081326',
     'tt5011838',
     'tt2644204',
     'tt1210781',
     'tt0246359',
     'tt0048898',
     'tt3398108',
     'tt5701572',
     'tt0426827',
     'tt0425714',
     'tt1252620',
     'tt0800289',
     'tt0111991',
     'tt0479847',
     'tt2429392',
     'tt2901828',
     'tt4147072',
     'tt1442411',
     'tt2093677',
     'tt0498421',
     'tt3006666',
     'tt3017190',
     'tt0193680',
     'tt5952954',
     'tt0381759',
     'tt2539740',
     'tt0369176',
     'tt3016990',
     'tt0328787',
     'tt2197994',
     'tt0478753',
     'tt4530152',
     'tt0372643',
     'tt5693024',
     'tt0855669',
     'tt1263594',
     'tt5935350',
     'tt1589855',
     'tt0367444',
     'tt3384116',
     'tt3790338',
     'tt2007260',
     'tt0343300',
     'tt0813904',
     'tt0883849',
     'tt0433296',
     'tt1342705',
     'tt0444988',
     'tt1333495',
     'tt0969661',
     'tt0272967',
     'tt0283184',
     'tt0444577',
     'tt3064496',
     'tt0436996',
     'tt1796788',
     'tt1879997',
     'tt4800624',
     'tt0497079',
     'tt1755893',
     'tt0329824',
     'tt2245937',
     'tt2147632',
     'tt3218114',
     'tt1583417',
     'tt0367403',
     'tt1963853',
     'tt4854900',
     'tt6415490',
     'tt1520150',
     'tt0236907',
     'tt6672370',
     'tt1055136',
     'tt5865052',
     'tt1231448',
     'tt6315022',
     'tt4351710',
     'tt4346344',
     'tt6043450',
     'tt0096605',
     'tt1181712',
     'tt0182623',
     'tt0307719',
     'tt1056344',
     'tt0328795',
     'tt0098916',
     'tt1584617',
     'tt2354136',
     'tt4287478',
     'tt0426347',
     'tt1874006',
     'tt2006560',
     'tt1694893',
     'tt2338766',
     'tt0843808',
     'tt0115155',
     'tt4354068',
     'tt1134663',
     'tt0495787',
     'tt0088539',
     'tt5426274',
     'tt1797127',
     'tt5763656',
     'tt0360301',
     'tt4245504',
     'tt0318214',
     'tt0080254',
     'tt1430135',
     'tt0892562',
     'tt2603010',
     'tt1038918',
     'tt0390746',
     'tt3773682',
     'tt0969372',
     'tt1470839',
     'tt1477822',
     'tt1056446',
     'tt0340474',
     'tt5104198',
     'tt2815184',
     'tt0468998',
     'tt0772146',
     'tt3920816',
     'tt3654000',
     'tt1753229',
     'tt0865687',
     'tt0459631',
     'tt1314665',
     'tt4660152',
     'tt0086685',
     'tt0150323',
     'tt0338576',
     'tt2118185',
     'tt0198086',
     'tt0412184',
     'tt4420148',
     'tt0497853',
     'tt1240534',
     'tt2479832',
     'tt0174195',
     'tt1999642',
     'tt1155579',
     'tt1640376',
     'tt1227586',
     'tt3784176',
     'tt1958848',
     'tt2778982',
     'tt1273636',
     'tt0357357',
     'tt1287301',
     'tt0852784',
     'tt0482432',
     'tt1651941',
     'tt0043235',
     'tt2110603',
     'tt1178184',
     'tt0846757',
     'tt0170959',
     'tt0413617',
     'tt1726890',
     'tt0220874',
     'tt0859872',
     'tt4219276',
     'tt0327268',
     'tt0843319',
     'tt3131346',
     'tt0795072',
     'tt5650560',
     'tt0827847',
     'tt1525767',
     'tt1043913',
     'tt0266179',
     'tt0413558',
     'tt0307714',
     'tt4693416',
     'tt0409619',
     'tt5684430',
     'tt0134269',
     'tt5486088',
     'tt1252370',
     'tt6370626',
     'tt3824018',
     'tt2555880',
     'tt3310544',
     'tt2125758',
     'tt1973047',
     'tt6748366',
     'tt0106113',
     'tt0934701',
     'tt2059031',
     'tt0088598',
     'tt1056536',
     'tt1618950',
     'tt6987940',
     'tt5915978',
     'tt0106008',
     'tt0115206',
     'tt0120992',
     'tt4575056',
     'tt2889104',
     'tt0428169']




```python
len(loser_list)
```




    600




```python
# first_loser_list = loser_list
```


```python
# This code has been executed, and the results pickled and stored locally, so no need to run these requests
# to the API again

DO_NOT_RUN = True

if not DO_NOT_RUN:
    losers = pd.DataFrame()
    for loser_id in loser_list:
            try:
                print loser_id
                # Get the tv show info from the api
                url = "http://api.tvmaze.com/lookup/shows?imdb=" + loser_id
                r = requests.get(url)

                # convert the return data to a dictionary
                json_data = r.json()

                # load a temp datafram with the dictionary, then append to the composite dataframe
                temp_df = pd.DataFrame.from_dict(json_data, orient='index', dtype=None)
                ttemp_df = temp_df.T     # Was not able to load json in column orientation, so must transpose
                losers = losers.append(ttemp_df, ignore_index=True)
            except: 
                print loser_id, " could not be retrieved from api"

    losers.head()    


```

    tt0465347
    tt0465347  could not be retrieved from api
    tt4427122
    tt4427122  could not be retrieved from api
    tt1015682
    tt1015682  could not be retrieved from api
    tt2505738
    tt2505738  could not be retrieved from api
    tt2402465
    tt2402465  could not be retrieved from api
    tt0278236
    tt0278236  could not be retrieved from api
    tt0268066
    tt0268066  could not be retrieved from api
    tt4813760
    tt4813760  could not be retrieved from api
    tt1526001
    tt1526001  could not be retrieved from api
    tt1243976
    tt1243976  could not be retrieved from api
    tt2058498
    tt3897284
    tt3897284  could not be retrieved from api
    tt3665690
    tt3665690  could not be retrieved from api
    tt4132180
    tt4132180  could not be retrieved from api
    tt0824229
    tt0824229  could not be retrieved from api
    tt0314990
    tt0314990  could not be retrieved from api
    tt5423750
    tt5423750  could not be retrieved from api
    tt5423664
    tt5423664  could not be retrieved from api
    tt2175125
    tt2175125  could not be retrieved from api
    tt0404593
    tt0404593  could not be retrieved from api
    tt4160422
    tt4160422  could not be retrieved from api
    tt4552562
    tt4552562  could not be retrieved from api
    tt5804854
    tt5804854  could not be retrieved from api
    tt0886666
    tt0886666  could not be retrieved from api
    tt5423824
    tt5423824  could not be retrieved from api
    tt3500210
    tt3500210  could not be retrieved from api
    tt0285357
    tt0285357  could not be retrieved from api
    tt0280234
    tt0280234  could not be retrieved from api
    tt1863530
    tt1863530  could not be retrieved from api
    tt0280349
    tt0280349  could not be retrieved from api
    tt2660922
    tt2660922  could not be retrieved from api
    tt0292776
    tt0292776  could not be retrieved from api
    tt4566242
    tt0264230
    tt0264230  could not be retrieved from api
    tt1102523
    tt1102523  could not be retrieved from api
    tt3333790
    tt3333790  could not be retrieved from api
    tt0320863
    tt0320863  could not be retrieved from api
    tt0830848
    tt0830848  could not be retrieved from api
    tt0939270
    tt0939270  could not be retrieved from api
    tt1459294
    tt1459294  could not be retrieved from api
    tt6026132
    tt6026132  could not be retrieved from api
    tt1443593
    tt1443593  could not be retrieved from api
    tt0354267
    tt0354267  could not be retrieved from api
    tt0147749
    tt0147749  could not be retrieved from api
    tt0161180
    tt0161180  could not be retrieved from api
    tt4733812
    tt4733812  could not be retrieved from api
    tt0367362
    tt0367362  could not be retrieved from api
    tt5626868
    tt5626868  could not be retrieved from api
    tt7268752
    tt7268752  could not be retrieved from api
    tt1364951
    tt2341819
    tt0464767
    tt0464767  could not be retrieved from api
    tt3550770
    tt3550770  could not be retrieved from api
    tt6422012
    tt6422012  could not be retrieved from api
    tt3154248
    tt3154248  could not be retrieved from api
    tt5016274
    tt5016274  could not be retrieved from api
    tt1715229
    tt1715229  could not be retrieved from api
    tt0489426
    tt0489426  could not be retrieved from api
    tt5798754
    tt5798754  could not be retrieved from api
    tt2022182
    tt2022182  could not be retrieved from api
    tt0303564
    tt0303564  could not be retrieved from api
    tt3462252
    tt3462252  could not be retrieved from api
    tt0329849
    tt0329849  could not be retrieved from api
    tt5074180
    tt5074180  could not be retrieved from api
    tt3900878
    tt3900878  could not be retrieved from api
    tt3887402
    tt3887402  could not be retrieved from api
    tt1893088
    tt0445890
    tt0149408
    tt0149408  could not be retrieved from api
    tt1360544
    tt1360544  could not be retrieved from api
    tt1718355
    tt1718355  could not be retrieved from api
    tt2364950
    tt2364950  could not be retrieved from api
    tt2279571
    tt0285374
    tt0285374  could not be retrieved from api
    tt5267590
    tt5267590  could not be retrieved from api
    tt0314993
    tt0314993  could not be retrieved from api
    tt0300870
    tt0300870  could not be retrieved from api
    tt7036530
    tt7036530  could not be retrieved from api
    tt5657014
    tt5657014  could not be retrieved from api
    tt0149488
    tt0149488  could not be retrieved from api
    tt1204865
    tt1204865  could not be retrieved from api
    tt1182860
    tt1182860  could not be retrieved from api
    tt0423626
    tt0423626  could not be retrieved from api
    tt4223864
    tt4223864  could not be retrieved from api
    tt1773440
    tt1773440  could not be retrieved from api
    tt0872067
    tt0872067  could not be retrieved from api
    tt0428172
    tt0428172  could not be retrieved from api
    tt0817379
    tt0817379  could not be retrieved from api
    tt1210720
    tt1210720  could not be retrieved from api
    tt3855028
    tt3855028  could not be retrieved from api
    tt1611594
    tt1611594  could not be retrieved from api
    tt5822004
    tt5822004  could not be retrieved from api
    tt6524930
    tt6524930  could not be retrieved from api
    tt1733734
    tt1902032
    tt1902032  could not be retrieved from api
    tt0466201
    tt0466201  could not be retrieved from api
    tt1757293
    tt1757293  could not be retrieved from api
    tt1807575
    tt1807575  could not be retrieved from api
    tt0332896
    tt0332896  could not be retrieved from api
    tt3140278
    tt3140278  could not be retrieved from api
    tt1176297
    tt1176297  could not be retrieved from api
    tt0285406
    tt0285406  could not be retrieved from api
    tt6680212
    tt6680212  could not be retrieved from api
    tt0200336
    tt0200336  could not be retrieved from api
    tt0385483
    tt0385483  could not be retrieved from api
    tt3534894
    tt3534894  could not be retrieved from api
    tt1108281
    tt1108281  could not be retrieved from api
    tt3855016
    tt3855016  could not be retrieved from api
    tt0787948
    tt0787948  could not be retrieved from api
    tt1372153
    tt1292967
    tt1292967  could not be retrieved from api
    tt1466565
    tt1466565  could not be retrieved from api
    tt0435565
    tt0435565  could not be retrieved from api
    tt1817054
    tt2879822
    tt1229266
    tt1229266  could not be retrieved from api
    tt0364837
    tt0364837  could not be retrieved from api
    tt0477409
    tt0477409  could not be retrieved from api
    tt0875097
    tt0875097  could not be retrieved from api
    tt1227542
    tt1227542  could not be retrieved from api
    tt1131289
    tt1131289  could not be retrieved from api
    tt0355135
    tt0355135  could not be retrieved from api
    tt1418598
    tt0290970
    tt0290970  could not be retrieved from api
    tt0184124
    tt0184124  could not be retrieved from api
    tt0490736
    tt0490736  could not be retrieved from api
    tt0439354
    tt0439354  could not be retrieved from api
    tt1157935
    tt1157935  could not be retrieved from api
    tt1425641
    tt1425641  could not be retrieved from api
    tt2830404
    tt2830404  could not be retrieved from api
    tt0835397
    tt0835397  could not be retrieved from api
    tt0880581
    tt0880581  could not be retrieved from api
    tt1078463
    tt1078463  could not be retrieved from api
    tt0190177
    tt1234506
    tt1234506  could not be retrieved from api
    tt0323463
    tt0323463  could not be retrieved from api
    tt5047510
    tt5338860
    tt5168468
    tt5168468  could not be retrieved from api
    tt0296322
    tt0296322  could not be retrieved from api
    tt3911254
    tt3911254  could not be retrieved from api
    tt3827516
    tt3827516  could not be retrieved from api
    tt0364899
    tt0364899  could not be retrieved from api
    tt4204032
    tt4204032  could not be retrieved from api
    tt0259768
    tt0259768  could not be retrieved from api
    tt0287880
    tt0287880  could not be retrieved from api
    tt0270763
    tt0270763  could not be retrieved from api
    tt0846349
    tt0846349  could not be retrieved from api
    tt2699648
    tt2699648  could not be retrieved from api
    tt3616368
    tt3616368  could not be retrieved from api
    tt2672920
    tt2672920  could not be retrieved from api
    tt1848281
    tt0813074
    tt0813074  could not be retrieved from api
    tt1694422
    tt1694422  could not be retrieved from api
    tt0472241
    tt0472241  could not be retrieved from api
    tt0202186
    tt0202186  could not be retrieved from api
    tt1297366
    tt1297366  could not be retrieved from api
    tt3919918
    tt3919918  could not be retrieved from api
    tt1564985
    tt1564985  could not be retrieved from api
    tt3336800
    tt3336800  could not be retrieved from api
    tt6839504
    tt2114184
    tt2254454
    tt2254454  could not be retrieved from api
    tt1674023
    tt0824737
    tt0824737  could not be retrieved from api
    tt1288431
    tt1288431  could not be retrieved from api
    tt1705811
    tt1705811  could not be retrieved from api
    tt0968726
    tt0968726  could not be retrieved from api
    tt2058840
    tt2058840  could not be retrieved from api
    tt1971860
    tt3857708
    tt3857708  could not be retrieved from api
    tt0315030
    tt0315030  could not be retrieved from api
    tt2337185
    tt2337185  could not be retrieved from api
    tt0775356
    tt0775356  could not be retrieved from api
    tt0244356
    tt0244356  could not be retrieved from api
    tt2338400
    tt2338400  could not be retrieved from api
    tt0220047
    tt0220047  could not be retrieved from api
    tt0341789
    tt0341789  could not be retrieved from api
    tt0197151
    tt0197151  could not be retrieved from api
    tt0222529
    tt0222529  could not be retrieved from api
    tt6086050
    tt6086050  could not be retrieved from api
    tt3100634
    tt1625263
    tt1625263  could not be retrieved from api
    tt2289244
    tt2289244  could not be retrieved from api
    tt1936732
    tt0278229
    tt0278229  could not be retrieved from api
    tt0429438
    tt0429438  could not be retrieved from api
    tt1410490
    tt1410490  could not be retrieved from api
    tt5588910
    tt5588910  could not be retrieved from api
    tt3670858
    tt3670858  could not be retrieved from api
    tt1197582
    tt0397182
    tt0397182  could not be retrieved from api
    tt1911975
    tt1911975  could not be retrieved from api
    tt0420366
    tt0420366  could not be retrieved from api
    tt3079034
    tt3079034  could not be retrieved from api
    tt0859270
    tt0859270  could not be retrieved from api
    tt0050070
    tt0050070  could not be retrieved from api
    tt0300798
    tt0300798  could not be retrieved from api
    tt5915502
    tt5915502  could not be retrieved from api
    tt6697244
    tt6697244  could not be retrieved from api
    tt1776388
    tt1776388  could not be retrieved from api
    tt0424639
    tt0424639  could not be retrieved from api
    tt1119204
    tt1119204  could not be retrieved from api
    tt1744868
    tt1744868  could not be retrieved from api
    tt1588824
    tt1588824  could not be retrieved from api
    tt1485389
    tt3696798
    tt3696798  could not be retrieved from api
    tt0301123
    tt0301123  could not be retrieved from api
    tt1018436
    tt1018436  could not be retrieved from api
    tt0815776
    tt0815776  could not be retrieved from api
    tt0407462
    tt0407462  could not be retrieved from api
    tt0198147
    tt0198147  could not be retrieved from api
    tt0997412
    tt0997412  could not be retrieved from api
    tt2288050
    tt1612920
    tt0402701
    tt5047494
    tt5047494  could not be retrieved from api
    tt5368216
    tt5368216  could not be retrieved from api
    tt3356610
    tt3356610  could not be retrieved from api
    tt0491735
    tt1454750
    tt1454750  could not be retrieved from api
    tt5891726
    tt5891726  could not be retrieved from api
    tt2369946
    tt4286824
    tt4286824  could not be retrieved from api
    tt0476926
    tt0476926  could not be retrieved from api
    tt5167034
    tt5167034  could not be retrieved from api
    tt0056759
    tt0056759  could not be retrieved from api
    tt3622818
    tt3622818  could not be retrieved from api
    tt0887788
    tt0887788  could not be retrieved from api
    tt4588620
    tt4588620  could not be retrieved from api
    tt0258341
    tt0258341  could not be retrieved from api
    tt0489430
    tt0489430  could not be retrieved from api
    tt2567210
    tt2567210  could not be retrieved from api
    tt0990403
    tt4674178
    tt4674178  could not be retrieved from api
    tt0125638
    tt0125638  could not be retrieved from api
    tt5146640
    tt5146640  could not be retrieved from api
    tt0196284
    tt0196284  could not be retrieved from api
    tt3075154
    tt3075154  could not be retrieved from api
    tt0436003
    tt0436003  could not be retrieved from api
    tt1538090
    tt1538090  could not be retrieved from api
    tt1728226
    tt1728226  could not be retrieved from api
    tt3796070
    tt3796070  could not be retrieved from api
    tt1381395
    tt1381395  could not be retrieved from api
    tt0190199
    tt0190199  could not be retrieved from api
    tt0855213
    tt0855213  could not be retrieved from api
    tt0358890
    tt0358890  could not be retrieved from api
    tt3484986
    tt3484986  could not be retrieved from api
    tt2208507
    tt2208507  could not be retrieved from api
    tt4896052
    tt4896052  could not be retrieved from api
    tt6148376
    tt0217211
    tt0217211  could not be retrieved from api
    tt0430836
    tt0430836  could not be retrieved from api
    tt1429551
    tt1291098
    tt1291098  could not be retrieved from api
    tt0399968
    tt0399968  could not be retrieved from api
    tt2909920
    tt2909920  could not be retrieved from api
    tt3164276
    tt3164276  could not be retrieved from api
    tt1586637
    tt4873032
    tt0926012
    tt0926012  could not be retrieved from api
    tt1305560
    tt1305560  could not be retrieved from api
    tt1291488
    tt1291488  could not be retrieved from api
    tt0428088
    tt0428088  could not be retrieved from api
    tt1057469
    tt1057469  could not be retrieved from api
    tt3807326
    tt3807326  could not be retrieved from api
    tt3293566
    tt0410964
    tt1579186
    tt0271931
    tt6519752
    tt1417358
    tt4568130
    tt1705611
    tt2235190
    tt0244328
    tt0244328  could not be retrieved from api
    tt0459155
    tt0459155  could not be retrieved from api
    tt1890984
    tt1890984  could not be retrieved from api
    tt0460381
    tt0460381  could not be retrieved from api
    tt0439069
    tt0439069  could not be retrieved from api
    tt0329817
    tt0329817  could not be retrieved from api
    tt1805082
    tt1805082  could not be retrieved from api
    tt0468985
    tt0468985  could not be retrieved from api
    tt1071166
    tt1071166  could not be retrieved from api
    tt1634699
    tt1634699  could not be retrieved from api
    tt1086761
    tt4214468
    tt0170930
    tt0170930  could not be retrieved from api
    tt5937940
    tt0305056
    tt1024887
    tt1024887  could not be retrieved from api
    tt1833558
    tt7062438
    tt7062438  could not be retrieved from api
    tt4411548
    tt4411548  could not be retrieved from api
    tt0105970
    tt0105970  could not be retrieved from api
    tt0348949
    tt0348949  could not be retrieved from api
    tt2309197
    tt2309197  could not be retrieved from api
    tt0327271
    tt0327271  could not be retrieved from api
    tt1729597
    tt1729597  could not be retrieved from api
    tt0428108
    tt0428108  could not be retrieved from api
    tt3144026
    tt3144026  could not be retrieved from api
    tt0292770
    tt0077041
    tt1489024
    tt0458269
    tt1020924
    tt0444578
    tt0787980
    tt0249275
    tt1280868
    tt0462121
    tt3136086
    tt1908157
    tt0055714
    tt0781991
    tt0224517
    tt0426804
    tt0484508
    tt0186742
    tt0460081
    tt0320809
    tt0798631
    tt3119834
    tt3804586
    tt0479614
    tt0479614  could not be retrieved from api
    tt0780447
    tt0780447  could not be retrieved from api
    tt0123366
    tt3481544
    tt3975956
    tt3975956  could not be retrieved from api
    tt5335110
    tt0471990
    tt0471990  could not be retrieved from api
    tt1332074
    tt6846846
    tt6846846  could not be retrieved from api
    tt1259798
    tt0381741
    tt0381741  could not be retrieved from api
    tt2953706
    tt1244881
    tt6208480
    tt6208480  could not be retrieved from api
    tt1232190
    tt0829040
    tt0829040  could not be retrieved from api
    tt3859844
    tt1761662
    tt1761662  could not be retrieved from api
    tt2262354
    tt0103411
    tt0103411  could not be retrieved from api
    tt0356281
    tt0356281  could not be retrieved from api
    tt4628798
    tt4628798  could not be retrieved from api
    tt0283714
    tt1147702
    tt1147702  could not be retrieved from api
    tt0780444
    tt0780444  could not be retrieved from api
    tt1981147
    tt0756524
    tt0312095
    tt0260645
    tt1728958
    tt4688354
    tt1296242
    tt1062211
    tt1500453
    tt0358320
    tt1118205
    tt0480781
    tt0303490
    tt0278256
    tt0812148
    tt0892683
    tt1562042
    tt0218767
    tt2265901
    tt1456074
    tt1978967
    tt0313038
    tt5437800
    tt5437800  could not be retrieved from api
    tt2453016
    tt5209238
    tt5209238  could not be retrieved from api
    tt7165310
    tt7165310  could not be retrieved from api
    tt1277979
    tt0362379
    tt0362379  could not be retrieved from api
    tt0348512
    tt0348512  could not be retrieved from api
    tt1024814
    tt0065343
    tt0065343  could not be retrieved from api
    tt3976016
    tt3976016  could not be retrieved from api
    tt1459376
    tt1459376  could not be retrieved from api
    tt4629950
    tt4629950  could not be retrieved from api
    tt0443361
    tt0443361  could not be retrieved from api
    tt1320317
    tt1320317  could not be retrieved from api
    tt1770959
    tt6212410
    tt6212410  could not be retrieved from api
    tt3731648
    tt5872774
    tt5872774  could not be retrieved from api
    tt4410468
    tt0196232
    tt0196232  could not be retrieved from api
    tt3693866
    tt3693866  could not be retrieved from api
    tt6295148
    tt6295148  could not be retrieved from api
    tt0804424
    tt0804424  could not be retrieved from api
    tt0458252
    tt0458252  could not be retrieved from api
    tt2933730
    tt2933730  could not be retrieved from api
    tt5690306
    tt5690306  could not be retrieved from api
    tt3038492
    tt0854912
    tt0426740
    tt0364787
    tt1033281
    tt0473416
    tt5423592
    tt2064427
    tt1208634
    tt0402660
    tt1566044
    tt0292845
    tt2633208
    tt1685317
    tt0421158
    tt1176154
    tt3099832
    tt0396337
    tt0337790
    tt0287847
    tt0421343
    tt0408364
    tt0346300
    tt0346300  could not be retrieved from api
    tt2908564
    tt2908564  could not be retrieved from api
    tt0348894
    tt6959064
    tt6959064  could not be retrieved from api
    tt1737565
    tt1454730
    tt0468999
    tt1495163
    tt2514488
    tt2390003
    tt0293725
    tt0293725  could not be retrieved from api
    tt0092362
    tt0092362  could not be retrieved from api
    tt0818895
    tt0818895  could not be retrieved from api
    tt1509653
    tt1509653  could not be retrieved from api
    tt1809909
    tt1809909  could not be retrieved from api
    tt1796975
    tt1796975  could not be retrieved from api
    tt6501522
    tt6501522  could not be retrieved from api
    tt0424611
    tt0424611  could not be retrieved from api
    tt0439932
    tt0439932  could not be retrieved from api
    tt4671004
    tt0471048
    tt0471048  could not be retrieved from api
    tt1156526
    tt1156526  could not be retrieved from api
    tt0264226
    tt0264226  could not be retrieved from api
    tt1170222
    tt1170222  could not be retrieved from api
    tt2689384
    tt0295081
    tt0295081  could not be retrieved from api
    tt4369244
    tt4369244  could not be retrieved from api
    tt2781594
    tt2781594  could not be retrieved from api
    tt4662374
    tt1105316
    tt1105316  could not be retrieved from api
    tt3840030
    tt3840030  could not be retrieved from api
    tt2579722
    tt0072546
    tt4628790
    tt0046590
    tt2184509
    tt0497854
    tt0363323
    tt1458207
    tt0439356
    tt0377146
    tt0954318
    tt2214505
    tt2435530
    tt0473419
    tt0768151
    tt0439365
    tt0278177
    tt1299440
    tt2083701
    tt1933836
    tt6473824
    tt6473824  could not be retrieved from api
    tt0187632
    tt0187632  could not be retrieved from api
    tt4033696
    tt0391666
    tt0391666  could not be retrieved from api
    tt0465344
    tt0465344  could not be retrieved from api
    tt2170392
    tt4390084
    tt2189892
    tt2189892  could not be retrieved from api
    tt6586510
    tt6586510  could not be retrieved from api
    tt3174316
    tt2374870
    tt2374870  could not be retrieved from api
    tt2366111
    tt2111994
    tt2111994  could not be retrieved from api
    tt4588734
    tt4588734  could not be retrieved from api
    tt0863047
    tt0863047  could not be retrieved from api
    tt1495648
    tt1579108
    tt1579108  could not be retrieved from api
    tt1159610
    tt0984168
    tt0984168  could not be retrieved from api
    tt6752226
    tt6752226  could not be retrieved from api
    tt0856723
    tt0856723  could not be retrieved from api
    tt0416347
    tt0416347  could not be retrieved from api
    tt5571740
    tt5571740  could not be retrieved from api
    tt1552185
    tt1552185  could not be retrieved from api
    tt3595870
    tt1728864
    tt1062185
    tt0380949
    tt1013861
    tt0848174
    tt0321000
    tt1855738
    tt0363335
    tt0420381
    tt1814550
    tt1987353
    tt0187654
    tt1461569
    tt1850160
    tt0954661
    tt0198095
    tt4012388
    tt0482028
    tt0176381
    tt0419307
    tt1684732
    tt5154762
    tt3139774
    tt0819708
    tt0819708  could not be retrieved from api
    tt0888280
    tt0888280  could not be retrieved from api
    tt6021260
    tt6021260  could not be retrieved from api
    tt0185065
    tt0185065  could not be retrieved from api
    tt4123482
    tt1491299
    tt1492090
    tt6059298
    tt6059298  could not be retrieved from api
    tt1826951
    tt0273025
    tt0273025  could not be retrieved from api
    tt1888795
    tt1888795  could not be retrieved from api
    tt1821879
    tt1821879  could not be retrieved from api
    tt2497788
    tt0476038
    tt0476038  could not be retrieved from api
    tt1830924
    tt1830924  could not be retrieved from api
    tt1368470
    tt1368470  could not be retrieved from api
    tt1361721
    tt1361721  could not be retrieved from api
    tt2647792
    tt2647792  could not be retrieved from api
    tt3148194
    tt0302163
    tt0302163  could not be retrieved from api
    tt5515342
    tt0292859
    tt0292859  could not be retrieved from api
    tt0243082
    tt0243082  could not be retrieved from api
    tt4654650
    tt4654650  could not be retrieved from api
    tt0298682
    tt0298682  could not be retrieved from api
    tt1534856
    tt1534856  could not be retrieved from api
    tt3097134
    tt3097134  could not be retrieved from api
    tt2582840
    tt2582840  could not be retrieved from api
    tt4605154
    tt1478217
    tt1478217  could not be retrieved from api
    tt0374366
    tt1631948
    tt0368494
    tt1721347
    tt5319670
    tt1684855
    tt5209280
    tt6217260
    tt6842890
    tt5040090
    tt3501210
    tt0367323
    tt0397012
    tt0954837
    tt1784056
    tt3228548
    tt0861753
    tt0933898
    tt0433705
    tt0287845
    tt0329816
    tt0329816  could not be retrieved from api
    tt2815342
    tt3548386
    tt3548386  could not be retrieved from api
    tt0410958
    tt0410958  could not be retrieved from api
    tt0057740
    tt0057740  could not be retrieved from api
    tt5583124
    tt5583124  could not be retrieved from api
    tt1440045
    tt1440045  could not be retrieved from api
    tt0810737
    tt0810737  could not be retrieved from api
    tt0989753
    tt0989753  could not be retrieved from api
    tt1313075
    tt1313075  could not be retrieved from api
    tt1073528
    tt1073528  could not be retrieved from api
    tt0310516
    tt0310516  could not be retrieved from api
    tt1642103
    tt1642103  could not be retrieved from api
    tt0448973
    tt0448973  could not be retrieved from api
    tt0302098
    tt0302098  could not be retrieved from api
    tt0805368
    tt0805368  could not be retrieved from api
    tt1124662
    tt1124662  could not be retrieved from api
    tt0324891
    tt0324891  could not be retrieved from api
    tt0423631
    tt0423631  could not be retrieved from api
    tt2226096
    tt2226096  could not be retrieved from api
    tt0773264
    tt1798695
    tt1307083
    tt4845734
    tt0046641
    tt0046641  could not be retrieved from api
    tt1519575
    tt1519575  could not be retrieved from api
    tt0853078
    tt0853078  could not be retrieved from api
    tt0118423
    tt0118423  could not be retrieved from api
    tt0284767
    tt4052124
    tt4052124  could not be retrieved from api
    tt0878801
    tt3703500



```python
# Oops,  We've hit the API to hard.  A second attempt to pull low rated show information
# will be needed, with a time delay to stay within API limitations.

```


```python
# This shape is misleading, as many of the rows simply contain a message that the API limit 
# had been exceeded

losers.shape
```




    (229, 22)




```python
# This is accurate, 235 shows from the top show list were obtained
shows.shape
```




    (235, 20)




```python
DO_NOT_RUN = True  # Be sure to check the file name to write before enabling execution on this block

if not DO_NOT_RUN:
    pickle.dump( losers, open( "save_losers_df.p", "wb" ) )
```


```python
# read data back in from the saved file
losers2 = pickle.load( open( "save_losers_df.p", "rb" ) )
```

### This is the start of a second attempt to pull more TV shows with low ratings

This is needed.  After the first pull, and after cleanup, there were only 10 Shows left in the low rating category with complete information.   The cells below collect more data from the API for additional low rated shows.


```python
losers.loc[0:9]['externals']
```




    0    {u'thetvdb': 283995, u'tvrage': 40425, u'imdb'...
    1    {u'thetvdb': 299234, u'tvrage': 50418, u'imdb'...
    2    {u'thetvdb': 118021, u'tvrage': None, u'imdb':...
    3    {u'thetvdb': 274705, u'tvrage': 31580, u'imdb'...
    4    {u'thetvdb': 246161, u'tvrage': None, u'imdb':...
    5    {u'thetvdb': 75638, u'tvrage': None, u'imdb': ...
    6    {u'thetvdb': 260183, u'tvrage': 31024, u'imdb'...
    7    {u'thetvdb': None, u'tvrage': None, u'imdb': u...
    8    {u'thetvdb': 299688, u'tvrage': None, u'imdb':...
    9    {u'thetvdb': 222481, u'tvrage': None, u'imdb':...
    Name: externals, dtype: object




```python
# In the first attempt, there were a number of shows where data was not returned becuase of two many api calls
# in quick succession. In order to re-submit those show ids, it is necessary to get a list of ids that were
# returned successfully, and then to remove them from the original list of ids before resubmitting.  
# losers_pulled is a list of ids that were successful on the previous attempt.

losers_pulled = []
no_imdb_at_idx = []
for i in range(len(losers)):
    try:
        losers_pulled.append(losers.loc[i,'externals']['imdb'])
    except:
        no_imdb_at_idx.append(i)
print no_imdb_at_idx
print
print losers_pulled
print len(losers_pulled)
```

    [11, 35, 36, 37, 38, 39, 40, 41, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 228]
    
    [u'tt2058498', u'tt4566242', u'tt1364951', u'tt2341819', u'tt1893088', u'tt0445890', u'tt2279571', u'tt1733734', u'tt1372153', u'tt1817054', u'tt2879822', u'tt0190177', u'tt5047510', u'tt5338860', u'tt1848281', u'tt6839504', u'tt2114184', u'tt1674023', u'tt1971860', u'tt3100634', u'tt1936732', u'tt1197582', u'tt1485389', u'tt2288050', u'tt1612920', u'tt0402701', u'tt0491735', u'tt2369946', u'tt0990403', u'tt6148376', u'tt1429551', u'tt1586637', u'tt4873032', u'tt3293566', u'tt2235190', u'tt1086761', u'tt4214468', u'tt5937940', u'tt0305056', u'tt1833558', u'tt0123366', u'tt3481544', u'tt5335110', u'tt1332074', u'tt1259798', u'tt2953706', u'tt1244881', u'tt1232190', u'tt3859844', u'tt2262354', u'tt0283714', u'tt0313038', u'tt2453016', u'tt1277979', u'tt1024814', u'tt1770959', u'tt3731648', u'tt4410468', u'tt0348894', u'tt1737565', u'tt1454730', u'tt0468999', u'tt1495163', u'tt2514488', u'tt2390003', u'tt4671004', u'tt2689384', u'tt4662374', u'tt1299440', u'tt2083701', u'tt1933836', u'tt4033696', u'tt2170392', u'tt4390084', u'tt3174316', u'tt2366111', u'tt1495648', u'tt1159610', u'tt4123482', u'tt1491299', u'tt1492090', u'tt1826951', u'tt2497788', u'tt3148194', u'tt5515342', u'tt4605154', u'tt2815342', u'tt0773264', u'tt1798695', u'tt1307083', u'tt4845734', u'tt0284767', u'tt0878801']
    93



```python
# There were that do not even include their own imdb number, and indicator that the pull was unsuccessful
# While a few of these might have been successful but have only limited data, most are unusuable.
# Thus all will be re-requested at a slower rate and any duplicates removed when the data is merged.

print len(no_imdb_at_idx)
```

    136



```python
# This generates a list of the original requests that were not successfully returned from the api.   
# First the will be requested again, using a time delay to avoid requesting more than the server
# will willingly return.  They will also be batched in groups of 100 ids

missing_losers = [x for x in loser_list if x not in losers_pulled]
missing_losers
```




    ['tt0465347',
     'tt4427122',
     'tt1015682',
     'tt2505738',
     'tt2402465',
     'tt0278236',
     'tt0268066',
     'tt4813760',
     'tt1526001',
     'tt1243976',
     'tt3897284',
     'tt3665690',
     'tt4132180',
     'tt0824229',
     'tt0314990',
     'tt5423750',
     'tt5423664',
     'tt2175125',
     'tt0404593',
     'tt4160422',
     'tt4552562',
     'tt5804854',
     'tt0886666',
     'tt5423824',
     'tt3500210',
     'tt0285357',
     'tt0280234',
     'tt1863530',
     'tt0280349',
     'tt2660922',
     'tt0292776',
     'tt0264230',
     'tt1102523',
     'tt3333790',
     'tt0320863',
     'tt0830848',
     'tt0939270',
     'tt1459294',
     'tt6026132',
     'tt1443593',
     'tt0354267',
     'tt0147749',
     'tt0161180',
     'tt4733812',
     'tt0367362',
     'tt5626868',
     'tt7268752',
     'tt0464767',
     'tt3550770',
     'tt6422012',
     'tt3154248',
     'tt5016274',
     'tt1715229',
     'tt0489426',
     'tt5798754',
     'tt2022182',
     'tt0303564',
     'tt3462252',
     'tt0329849',
     'tt5074180',
     'tt3900878',
     'tt3887402',
     'tt0149408',
     'tt1360544',
     'tt1718355',
     'tt2364950',
     'tt0285374',
     'tt5267590',
     'tt0314993',
     'tt0300870',
     'tt7036530',
     'tt5657014',
     'tt0149488',
     'tt1204865',
     'tt1182860',
     'tt0423626',
     'tt4223864',
     'tt1773440',
     'tt0872067',
     'tt0428172',
     'tt0817379',
     'tt1210720',
     'tt3855028',
     'tt1611594',
     'tt5822004',
     'tt6524930',
     'tt1902032',
     'tt0466201',
     'tt1757293',
     'tt1807575',
     'tt0332896',
     'tt3140278',
     'tt1176297',
     'tt0285406',
     'tt6680212',
     'tt0200336',
     'tt0385483',
     'tt3534894',
     'tt1108281',
     'tt3855016',
     'tt0787948',
     'tt1292967',
     'tt1466565',
     'tt0435565',
     'tt1229266',
     'tt0364837',
     'tt0477409',
     'tt0875097',
     'tt1227542',
     'tt1131289',
     'tt0355135',
     'tt1418598',
     'tt0290970',
     'tt0184124',
     'tt0490736',
     'tt0439354',
     'tt1157935',
     'tt1425641',
     'tt2830404',
     'tt0835397',
     'tt0880581',
     'tt1078463',
     'tt1234506',
     'tt0323463',
     'tt5168468',
     'tt0296322',
     'tt3911254',
     'tt3827516',
     'tt0364899',
     'tt4204032',
     'tt0259768',
     'tt0287880',
     'tt0270763',
     'tt0846349',
     'tt2699648',
     'tt3616368',
     'tt2672920',
     'tt0813074',
     'tt1694422',
     'tt0472241',
     'tt0202186',
     'tt1297366',
     'tt3919918',
     'tt1564985',
     'tt3336800',
     'tt2254454',
     'tt0824737',
     'tt1288431',
     'tt1705811',
     'tt0968726',
     'tt2058840',
     'tt3857708',
     'tt0315030',
     'tt2337185',
     'tt0775356',
     'tt0244356',
     'tt2338400',
     'tt0220047',
     'tt0341789',
     'tt0197151',
     'tt0222529',
     'tt6086050',
     'tt1625263',
     'tt2289244',
     'tt0278229',
     'tt0429438',
     'tt1410490',
     'tt5588910',
     'tt3670858',
     'tt0397182',
     'tt1911975',
     'tt0420366',
     'tt3079034',
     'tt0859270',
     'tt0050070',
     'tt0300798',
     'tt5915502',
     'tt6697244',
     'tt1776388',
     'tt0424639',
     'tt1119204',
     'tt1744868',
     'tt1588824',
     'tt3696798',
     'tt0301123',
     'tt1018436',
     'tt0815776',
     'tt0407462',
     'tt0198147',
     'tt0997412',
     'tt5047494',
     'tt5368216',
     'tt3356610',
     'tt1454750',
     'tt5891726',
     'tt4286824',
     'tt0476926',
     'tt5167034',
     'tt0056759',
     'tt3622818',
     'tt0887788',
     'tt4588620',
     'tt0258341',
     'tt0489430',
     'tt2567210',
     'tt4674178',
     'tt0125638',
     'tt5146640',
     'tt0196284',
     'tt3075154',
     'tt0436003',
     'tt1538090',
     'tt1728226',
     'tt3796070',
     'tt1381395',
     'tt0190199',
     'tt0855213',
     'tt0358890',
     'tt3484986',
     'tt2208507',
     'tt4896052',
     'tt0217211',
     'tt0430836',
     'tt1291098',
     'tt0399968',
     'tt2909920',
     'tt3164276',
     'tt0926012',
     'tt1305560',
     'tt1291488',
     'tt0428088',
     'tt1057469',
     'tt3807326',
     'tt0410964',
     'tt1579186',
     'tt0271931',
     'tt6519752',
     'tt1417358',
     'tt4568130',
     'tt1705611',
     'tt0244328',
     'tt0459155',
     'tt1890984',
     'tt0460381',
     'tt0439069',
     'tt0329817',
     'tt1805082',
     'tt0468985',
     'tt1071166',
     'tt1634699',
     'tt0170930',
     'tt1024887',
     'tt7062438',
     'tt4411548',
     'tt0105970',
     'tt0348949',
     'tt2309197',
     'tt0327271',
     'tt1729597',
     'tt0428108',
     'tt3144026',
     'tt0292770',
     'tt0077041',
     'tt1489024',
     'tt0458269',
     'tt1020924',
     'tt0444578',
     'tt0787980',
     'tt0249275',
     'tt1280868',
     'tt0462121',
     'tt3136086',
     'tt1908157',
     'tt0055714',
     'tt0781991',
     'tt0224517',
     'tt0426804',
     'tt0484508',
     'tt0186742',
     'tt0460081',
     'tt0320809',
     'tt0798631',
     'tt3119834',
     'tt3804586',
     'tt0479614',
     'tt0780447',
     'tt3975956',
     'tt0471990',
     'tt6846846',
     'tt0381741',
     'tt6208480',
     'tt0829040',
     'tt1761662',
     'tt0103411',
     'tt0356281',
     'tt4628798',
     'tt1147702',
     'tt0780444',
     'tt1981147',
     'tt0756524',
     'tt0312095',
     'tt0260645',
     'tt1728958',
     'tt4688354',
     'tt1296242',
     'tt1062211',
     'tt1500453',
     'tt0358320',
     'tt1118205',
     'tt0480781',
     'tt0303490',
     'tt0278256',
     'tt0812148',
     'tt0892683',
     'tt1562042',
     'tt0218767',
     'tt2265901',
     'tt1456074',
     'tt1978967',
     'tt5437800',
     'tt5209238',
     'tt7165310',
     'tt0362379',
     'tt0348512',
     'tt0065343',
     'tt3976016',
     'tt1459376',
     'tt4629950',
     'tt0443361',
     'tt1320317',
     'tt6212410',
     'tt5872774',
     'tt0196232',
     'tt3693866',
     'tt6295148',
     'tt0804424',
     'tt0458252',
     'tt2933730',
     'tt5690306',
     'tt3038492',
     'tt0854912',
     'tt0426740',
     'tt0364787',
     'tt1033281',
     'tt0473416',
     'tt5423592',
     'tt2064427',
     'tt1208634',
     'tt0402660',
     'tt1566044',
     'tt0292845',
     'tt2633208',
     'tt1685317',
     'tt0421158',
     'tt1176154',
     'tt3099832',
     'tt0396337',
     'tt0337790',
     'tt0287847',
     'tt0421343',
     'tt0408364',
     'tt0346300',
     'tt2908564',
     'tt6959064',
     'tt0293725',
     'tt0092362',
     'tt0818895',
     'tt1509653',
     'tt1809909',
     'tt1796975',
     'tt6501522',
     'tt0424611',
     'tt0439932',
     'tt0471048',
     'tt1156526',
     'tt0264226',
     'tt1170222',
     'tt0295081',
     'tt4369244',
     'tt2781594',
     'tt1105316',
     'tt3840030',
     'tt2579722',
     'tt0072546',
     'tt4628790',
     'tt0046590',
     'tt2184509',
     'tt0497854',
     'tt0363323',
     'tt1458207',
     'tt0439356',
     'tt0377146',
     'tt0954318',
     'tt2214505',
     'tt2435530',
     'tt0473419',
     'tt0768151',
     'tt0439365',
     'tt0278177',
     'tt6473824',
     'tt0187632',
     'tt0391666',
     'tt0465344',
     'tt2189892',
     'tt6586510',
     'tt2374870',
     'tt2111994',
     'tt4588734',
     'tt0863047',
     'tt1579108',
     'tt0984168',
     'tt6752226',
     'tt0856723',
     'tt0416347',
     'tt5571740',
     'tt1552185',
     'tt3595870',
     'tt1728864',
     'tt1062185',
     'tt0380949',
     'tt1013861',
     'tt0848174',
     'tt0321000',
     'tt1855738',
     'tt0363335',
     'tt0420381',
     'tt1814550',
     'tt1987353',
     'tt0187654',
     'tt1461569',
     'tt1850160',
     'tt0954661',
     'tt0198095',
     'tt4012388',
     'tt0482028',
     'tt0176381',
     'tt0419307',
     'tt1684732',
     'tt5154762',
     'tt3139774',
     'tt0819708',
     'tt0888280',
     'tt6021260',
     'tt0185065',
     'tt6059298',
     'tt0273025',
     'tt1888795',
     'tt1821879',
     'tt0476038',
     'tt1830924',
     'tt1368470',
     'tt1361721',
     'tt2647792',
     'tt0302163',
     'tt0292859',
     'tt0243082',
     'tt4654650',
     'tt0298682',
     'tt1534856',
     'tt3097134',
     'tt2582840',
     'tt1478217',
     'tt0374366',
     'tt1631948',
     'tt0368494',
     'tt1721347',
     'tt5319670',
     'tt1684855',
     'tt5209280',
     'tt6217260',
     'tt6842890',
     'tt5040090',
     'tt3501210',
     'tt0367323',
     'tt0397012',
     'tt0954837',
     'tt1784056',
     'tt3228548',
     'tt0861753',
     'tt0933898',
     'tt0433705',
     'tt0287845',
     'tt0329816',
     'tt3548386',
     'tt0410958',
     'tt0057740',
     'tt5583124',
     'tt1440045',
     'tt0810737',
     'tt0989753',
     'tt1313075',
     'tt1073528',
     'tt0310516',
     'tt1642103',
     'tt0448973',
     'tt0302098',
     'tt0805368',
     'tt1124662',
     'tt0324891',
     'tt0423631',
     'tt2226096',
     'tt0046641',
     'tt1519575',
     'tt0853078',
     'tt0118423',
     'tt4052124',
     'tt3703500']




```python
# This processes the oringinal list of 600 ids, minus the ones that were successfully pulled, 
# into groups of 100 + 7 in last list
# break up the missing list into groups of 100
subset_loser_list = []
print len(missing_losers)
for i in range(len(missing_losers)/100):
    temp_list = []
    for j in range(100):
        temp_list.append(missing_losers[i*100 + j])
    subset_loser_list.append(temp_list)    

# get last 7
for j in range(500, len(missing_losers)):
    temp_list = []
    for j in range(500, len(missing_losers)):
        temp_list.append(missing_losers[j])
```

    507



```python
# After reprocessing the first list of ids a 2nd time,  there are still not enough samples of low rated shows
# A third list of 600 low rated shows was scraped from IMDB, and this list is broken into subsets of 100 here

subset_loser_list2 = []
print len(loser_list)
for i in range(len(loser_list)/100):
    temp_list = []
    for j in range(100):
        temp_list.append(loser_list[i*100 + j])
    subset_loser_list2.append(temp_list)    

```

    600



```python
subset_loser_list2[0]

```




    ['tt0773264',
     'tt1798695',
     'tt1307083',
     'tt4845734',
     'tt0046641',
     'tt1519575',
     'tt0853078',
     'tt0118423',
     'tt0284767',
     'tt4052124',
     'tt0878801',
     'tt3703500',
     'tt1105170',
     'tt4363582',
     'tt3155428',
     'tt0362350',
     'tt0287196',
     'tt2766052',
     'tt0405545',
     'tt0262975',
     'tt0367278',
     'tt7134262',
     'tt1695352',
     'tt0421470',
     'tt2466890',
     'tt0343305',
     'tt1002739',
     'tt1615697',
     'tt0274262',
     'tt0465320',
     'tt1388381',
     'tt0358889',
     'tt1085789',
     'tt1011591',
     'tt0364804',
     'tt1489335',
     'tt3612584',
     'tt0363377',
     'tt0111930',
     'tt0401913',
     'tt0808086',
     'tt0309212',
     'tt5464192',
     'tt0080250',
     'tt4533338',
     'tt4741696',
     'tt1922810',
     'tt1793868',
     'tt4789316',
     'tt0185054',
     'tt1079622',
     'tt1786048',
     'tt0790508',
     'tt1716372',
     'tt0295098',
     'tt3409706',
     'tt0222574',
     'tt2171325',
     'tt0442643',
     'tt2142117',
     'tt0371433',
     'tt0138244',
     'tt1002010',
     'tt0495557',
     'tt1811817',
     'tt5529996',
     'tt1352053',
     'tt0439346',
     'tt0940147',
     'tt3075138',
     'tt1974439',
     'tt2693842',
     'tt0092325',
     'tt6772826',
     'tt1563069',
     'tt0489598',
     'tt0142055',
     'tt1566154',
     'tt0338592',
     'tt0167515',
     'tt2330327',
     'tt1576464',
     'tt2389845',
     'tt0186747',
     'tt0355096',
     'tt1821877',
     'tt0112033',
     'tt1792654',
     'tt0472243',
     'tt6453018',
     'tt3648886',
     'tt1599374',
     'tt2946482',
     'tt4672020',
     'tt1016283',
     'tt2649480',
     'tt1229945',
     'tt2390606',
     'tt1876612',
     'tt0140732']




```python

```


```python
# This block calls the API.   It is run repeatedly with each new sublist of 100 show ids,  sleeping 10
# seconds between each request.  There is a do not run flag that will prevent running this block if the 
# notebook is restarted.  The first time it was executed, a new dataframe called "more_losers" was initialized,
# and then commented out for subsequent executions so the data returned in eacn subsequent data request will
# be appended to the bottom of the dataframe.

# After collection is complete, set flag to prevent running this block unnecessarily if notebook is restarted

import time
DO_NOT_RUN = True

if not DO_NOT_RUN:
#     responses = []
#     more_losers = pd.DataFrame()
    for loser_id in subset_loser_list2[0]:   # change the index and re-run to accesses each set of 100 ids
        time.sleep(10)    
        try: 
            # Get the tv show info from the api
            url = "http://api.tvmaze.com/lookup/shows?imdb=" + loser_id
            r = requests.get(url)

            # convert the return data to a dictionary
            json_data = r.json()

            # load a temp datafram with the dictionary, then append to the composite dataframe
            temp_df = pd.DataFrame.from_dict(json_data, orient='index', dtype=None)
            ttemp_df = temp_df.T     # Was not able to load json in column orientation, so must transpose
            more_losers = more_losers.append(ttemp_df, ignore_index=True)
            stat = ''
        except: 
            stat = 'failed'

        print loser_id, stat, r.status_code
        res = [loser_id, stat, r.status_code]
        responses.append(res)
        
    losers.head()    


```

    tt0773264  200
    tt1798695  200
    tt1307083  200
    tt4845734  200
    tt0046641 failed 404
    tt1519575 failed 404
    tt0853078 failed 404
    tt0118423 failed 404
    tt0284767  200
    tt4052124 failed 404
    tt0878801  200
    tt3703500  200
    tt1105170 failed 404
    tt4363582 failed 404
    tt3155428  200
    tt0362350 failed 404
    tt0287196  200
    tt2766052  200
    tt0405545 failed 404
    tt0262975  200
    tt0367278 failed 404
    tt7134262 failed 404
    tt1695352 failed 404
    tt0421470 failed 404
    tt2466890 failed 404
    tt0343305 failed 404
    tt1002739 failed 404
    tt1615697 failed 404
    tt0274262 failed 404
    tt0465320 failed 404
    tt1388381  200
    tt0358889  200
    tt1085789 failed 404
    tt1011591  200
    tt0364804 failed 404
    tt1489335 failed 404
    tt3612584  200
    tt0363377 failed 404
    tt0111930 failed 404
    tt0401913 failed 404
    tt0808086 failed 404
    tt0309212 failed 404
    tt5464192  200
    tt0080250 failed 404
    tt4533338 failed 404
    tt4741696  200
    tt1922810 failed 404
    tt1793868 failed 404
    tt4789316 failed 404
    tt0185054 failed 404
    tt1079622 failed 404
    tt1786048 failed 404
    tt0790508 failed 404
    tt1716372 failed 404
    tt0295098 failed 404
    tt3409706 failed 404
    tt0222574 failed 404
    tt2171325 failed 404
    tt0442643 failed 404
    tt2142117 failed 404
    tt0371433 failed 404
    tt0138244 failed 404
    tt1002010 failed 404
    tt0495557 failed 404
    tt1811817 failed 404
    tt5529996 failed 404
    tt1352053 failed 404
    tt0439346 failed 404
    tt0940147 failed 404
    tt3075138 failed 404
    tt1974439  200
    tt2693842 failed 404
    tt0092325  200
    tt6772826  200
    tt1563069  200
    tt0489598  200
    tt0142055 failed 404
    tt1566154  200
    tt0338592  200
    tt0167515  200
    tt2330327  200
    tt1576464 failed 404
    tt2389845 failed 404
    tt0186747  200
    tt0355096 failed 404
    tt1821877  200
    tt0112033 failed 404
    tt1792654 failed 404
    tt0472243 failed 404
    tt6453018 failed 404
    tt3648886 failed 404
    tt1599374  200
    tt2946482  200
    tt4672020 failed 404
    tt1016283 failed 404
    tt2649480  200
    tt1229945  200
    tt2390606 failed 404
    tt1876612  200
    tt0140732 failed 404



```python
len(responses)
```




    1000




```python
for i in range(len(more_losers)):
    print more_losers.loc[i, 'externals']
```

    {u'thetvdb': 279947, u'tvrage': 37045, u'imdb': u'tt3595870'}
    {u'thetvdb': None, u'tvrage': 13173, u'imdb': u'tt0848174'}
    {u'thetvdb': 72157, u'tvrage': None, u'imdb': u'tt0374366'}
    {u'thetvdb': 218241, u'tvrage': None, u'imdb': u'tt1684855'}
    {u'thetvdb': 327908, u'tvrage': None, u'imdb': u'tt6842890'}
    {u'thetvdb': 279810, u'tvrage': None, u'imdb': u'tt3501210'}
    {u'thetvdb': 283658, u'tvrage': None, u'imdb': u'tt0367323'}
    {u'thetvdb': 271341, u'tvrage': 33650, u'imdb': u'tt2633208'}
    {u'thetvdb': 260677, u'tvrage': None, u'imdb': u'tt2579722'}
    {u'thetvdb': 77616, u'tvrage': None, u'imdb': u'tt0072546'}
    {u'thetvdb': 74419, u'tvrage': None, u'imdb': u'tt0458269'}
    {u'thetvdb': None, u'tvrage': None, u'imdb': u'tt0249275'}
    {u'thetvdb': 282527, u'tvrage': 42189, u'imdb': u'tt2815184'}
    {u'thetvdb': 246631, u'tvrage': None, u'imdb': u'tt1753229'}
    {u'thetvdb': 82500, u'tvrage': None, u'imdb': u'tt1240534'}
    {u'thetvdb': 206381, u'tvrage': 26873, u'imdb': u'tt1999642'}
    {u'thetvdb': 284259, u'tvrage': None, u'imdb': u'tt3784176'}
    {u'thetvdb': 250186, u'tvrage': None, u'imdb': u'tt1958848'}
    {u'thetvdb': 320679, u'tvrage': None, u'imdb': u'tt5684430'}
    {u'thetvdb': 74181, u'tvrage': 6494, u'imdb': u'tt0134269'}
    {u'thetvdb': 84159, u'tvrage': 19672, u'imdb': u'tt1252370'}
    {u'thetvdb': 300105, u'tvrage': 48178, u'imdb': u'tt3824018'}
    {u'thetvdb': 264850, u'tvrage': None, u'imdb': u'tt2555880'}
    {u'thetvdb': 277020, u'tvrage': 35629, u'imdb': u'tt3310544'}
    {u'thetvdb': 254524, u'tvrage': 31887, u'imdb': u'tt2125758'}
    {u'thetvdb': 271916, u'tvrage': None, u'imdb': u'tt1973047'}
    {u'thetvdb': 82005, u'tvrage': None, u'imdb': u'tt0934701'}
    {u'thetvdb': 250472, u'tvrage': None, u'imdb': u'tt2059031'}
    {u'thetvdb': 81491, u'tvrage': None, u'imdb': u'tt1056536'}
    {u'thetvdb': 137691, u'tvrage': None, u'imdb': u'tt1618950'}
    {u'thetvdb': 74395, u'tvrage': 3883, u'imdb': u'tt0115206'}
    {u'thetvdb': 298860, u'tvrage': 50010, u'imdb': u'tt4575056'}
    {u'thetvdb': 269115, u'tvrage': 33511, u'imdb': u'tt2889104'}
    {u'thetvdb': 285008, u'tvrage': None, u'imdb': u'tt2644204'}
    {u'thetvdb': 82237, u'tvrage': None, u'imdb': u'tt1210781'}
    {u'thetvdb': 314998, u'tvrage': None, u'imdb': u'tt0048898'}
    {u'thetvdb': 276337, u'tvrage': None, u'imdb': u'tt3398108'}
    {u'thetvdb': 221621, u'tvrage': None, u'imdb': u'tt1252620'}
    {u'thetvdb': 269059, u'tvrage': 35857, u'imdb': u'tt2901828'}
    {u'thetvdb': 273303, u'tvrage': 35560, u'imdb': u'tt3006666'}
    {u'thetvdb': 260473, u'tvrage': 30918, u'imdb': u'tt2197994'}
    {u'thetvdb': 83313, u'tvrage': None, u'imdb': u'tt1263594'}
    {u'thetvdb': 80117, u'tvrage': 7218, u'imdb': u'tt0497079'}
    {u'thetvdb': 174991, u'tvrage': 25843, u'imdb': u'tt1755893'}
    {u'thetvdb': 71424, u'tvrage': None, u'imdb': u'tt0329824'}
    {u'thetvdb': 258632, u'tvrage': 31545, u'imdb': u'tt2245937'}
    {u'thetvdb': 259235, u'tvrage': None, u'imdb': u'tt2147632'}
    {u'thetvdb': 297209, u'tvrage': 38100, u'imdb': u'tt3218114'}
    {u'thetvdb': 185651, u'tvrage': None, u'imdb': u'tt1583417'}
    {u'thetvdb': 250370, u'tvrage': 28934, u'imdb': u'tt1963853'}
    {u'thetvdb': 129051, u'tvrage': None, u'imdb': u'tt1520150'}
    {u'thetvdb': 76370, u'tvrage': None, u'imdb': u'tt0236907'}
    {u'thetvdb': 316174, u'tvrage': None, u'imdb': u'tt5865052'}
    {u'thetvdb': 82304, u'tvrage': 19011, u'imdb': u'tt1231448'}
    {u'thetvdb': 289640, u'tvrage': 46963, u'imdb': u'tt4287478'}
    {u'thetvdb': 249750, u'tvrage': None, u'imdb': u'tt1874006'}
    {u'thetvdb': 250959, u'tvrage': 28442, u'imdb': u'tt2006560'}
    {u'thetvdb': 281375, u'tvrage': 38313, u'imdb': u'tt3565412'}
    {u'thetvdb': 274414, u'tvrage': None, u'imdb': u'tt3396736'}
    {u'thetvdb': 271820, u'tvrage': None, u'imdb': u'tt0855313'}
    {u'thetvdb': 250955, u'tvrage': None, u'imdb': u'tt2309561'}
    {u'thetvdb': 273130, u'tvrage': 36774, u'imdb': u'tt3136814'}
    {u'thetvdb': 84669, u'tvrage': 18525, u'imdb': u'tt1191056'}
    {u'thetvdb': 74697, u'tvrage': 3348, u'imdb': u'tt0235917'}
    {u'thetvdb': 76708, u'tvrage': None, u'imdb': u'tt0111892'}
    {u'thetvdb': 266934, u'tvrage': None, u'imdb': u'tt2643770'}
    {u'thetvdb': 79896, u'tvrage': None, u'imdb': u'tt0423657'}
    {u'thetvdb': 303252, u'tvrage': None, u'imdb': u'tt5327970'}
    {u'thetvdb': 256806, u'tvrage': None, u'imdb': u'tt2190731'}
    {u'thetvdb': 78409, u'tvrage': None, u'imdb': u'tt0101041'}
    {u'thetvdb': 274820, u'tvrage': None, u'imdb': u'tt3317020'}
    {u'thetvdb': 296474, u'tvrage': 45813, u'imdb': u'tt4732076'}
    {u'thetvdb': 285651, u'tvrage': 41593, u'imdb': u'tt3828162'}
    {u'thetvdb': 315767, u'tvrage': None, u'imdb': u'tt5819414'}
    {u'thetvdb': 287534, u'tvrage': 42884, u'imdb': u'tt4180738'}
    {u'thetvdb': 76621, u'tvrage': None, u'imdb': u'tt0300802'}
    {u'thetvdb': 280683, u'tvrage': 34278, u'imdb': u'tt2649738'}
    {u'thetvdb': 280256, u'tvrage': 41644, u'imdb': u'tt3181412'}
    {u'thetvdb': 79496, u'tvrage': 2677, u'imdb': u'tt0382400'}
    {u'thetvdb': 271514, u'tvrage': None, u'imdb': u'tt2168240'}
    {u'thetvdb': 271826, u'tvrage': None, u'imdb': u'tt2560966'}
    {u'thetvdb': None, u'tvrage': None, u'imdb': u'tt0375440'}
    {u'thetvdb': 282253, u'tvrage': 44602, u'imdb': u'tt4081326'}
    {u'thetvdb': None, u'tvrage': None, u'imdb': u'tt6664486'}
    {u'thetvdb': 70734, u'tvrage': 14443, u'imdb': u'tt0247094'}
    {u'thetvdb': 70852, u'tvrage': 5323, u'imdb': u'tt0320969'}
    {u'thetvdb': 267185, u'tvrage': None, u'imdb': u'tt2720144'}
    {u'thetvdb': 265320, u'tvrage': 33976, u'imdb': u'tt2287380'}
    {u'thetvdb': 252485, u'tvrage': None, u'imdb': u'tt2010634'}
    {u'thetvdb': 271722, u'tvrage': 36787, u'imdb': u'tt3084090'}
    {u'thetvdb': 260126, u'tvrage': 30877, u'imdb': u'tt2392683'}
    {u'thetvdb': 251033, u'tvrage': 28408, u'imdb': u'tt1628058'}
    {u'thetvdb': None, u'tvrage': None, u'imdb': u'tt1837169'}
    {u'thetvdb': 260341, u'tvrage': 31462, u'imdb': u'tt2404111'}
    {u'thetvdb': 89831, u'tvrage': 22647, u'imdb': u'tt1411598'}
    {u'thetvdb': 70609, u'tvrage': 5102, u'imdb': u'tt0106123'}
    {u'thetvdb': 245071, u'tvrage': 26645, u'imdb': u'tt1740718'}
    {u'thetvdb': 73230, u'tvrage': 6188, u'imdb': u'tt0362153'}
    {u'thetvdb': 163671, u'tvrage': None, u'imdb': u'tt1637756'}
    {u'thetvdb': 259478, u'tvrage': 31194, u'imdb': u'tt2328067'}
    {u'thetvdb': 294774, u'tvrage': None, u'imdb': u'tt0057741'}
    {u'thetvdb': 282993, u'tvrage': None, u'imdb': u'tt1261356'}
    {u'thetvdb': 268795, u'tvrage': 36420, u'imdb': u'tt2559390'}
    {u'thetvdb': 72048, u'tvrage': 4056, u'imdb': u'tt0083433'}
    {u'thetvdb': 256513, u'tvrage': 31344, u'imdb': u'tt2330453'}
    {u'thetvdb': None, u'tvrage': None, u'imdb': u'tt0804423'}
    {u'thetvdb': 159351, u'tvrage': None, u'imdb': u'tt1118131'}
    {u'thetvdb': 300384, u'tvrage': None, u'imdb': u'tt4016700'}
    {u'thetvdb': 264239, u'tvrage': None, u'imdb': u'tt0950199'}
    {u'thetvdb': 106801, u'tvrage': None, u'imdb': u'tt1477137'}
    {u'thetvdb': 87131, u'tvrage': None, u'imdb': u'tt1176156'}
    {u'thetvdb': 173981, u'tvrage': None, u'imdb': u'tt1545453'}
    {u'thetvdb': None, u'tvrage': None, u'imdb': u'tt1240983'}
    {u'thetvdb': 264762, u'tvrage': 31404, u'imdb': u'tt1415000'}
    {u'thetvdb': 72180, u'tvrage': None, u'imdb': u'tt0144701'}
    {u'thetvdb': 307473, u'tvrage': None, u'imdb': u'tt4718304'}
    {u'thetvdb': 147701, u'tvrage': None, u'imdb': u'tt1095213'}
    {u'thetvdb': 98371, u'tvrage': None, u'imdb': u'tt1453090'}
    {u'thetvdb': 72141, u'tvrage': None, u'imdb': u'tt0168372'}
    {u'thetvdb': 75567, u'tvrage': 12949, u'imdb': u'tt0425725'}
    {u'thetvdb': 275787, u'tvrage': None, u'imdb': u'tt3300126'}
    {u'thetvdb': 308457, u'tvrage': 51439, u'imdb': u'tt5459976'}
    {u'thetvdb': 285286, u'tvrage': 44525, u'imdb': u'tt4041694'}
    {u'thetvdb': 261287, u'tvrage': 32847, u'imdb': u'tt2322264'}
    {u'thetvdb': 250325, u'tvrage': None, u'imdb': u'tt1441005'}
    {u'thetvdb': 72133, u'tvrage': None, u'imdb': u'tt0365991'}
    {u'thetvdb': 72488, u'tvrage': None, u'imdb': u'tt0364807'}
    {u'thetvdb': 149371, u'tvrage': 25246, u'imdb': u'tt1591375'}
    {u'thetvdb': 291820, u'tvrage': None, u'imdb': u'tt3562462'}
    {u'thetvdb': 96071, u'tvrage': None, u'imdb': u'tt1372127'}
    {u'thetvdb': 287516, u'tvrage': None, u'imdb': u'tt2088493'}
    {u'thetvdb': 295059, u'tvrage': 48857, u'imdb': u'tt4658248'}
    {u'thetvdb': 250280, u'tvrage': None, u'imdb': u'tt1973659'}
    {u'thetvdb': 272357, u'tvrage': None, u'imdb': u'tt2849552'}
    {u'thetvdb': 282130, u'tvrage': None, u'imdb': u'tt3774098'}
    {u'thetvdb': None, u'tvrage': 18611, u'imdb': u'tt1151434'}
    {u'thetvdb': 271067, u'tvrage': None, u'imdb': u'tt2993514'}
    {u'thetvdb': 80311, u'tvrage': None, u'imdb': u'tt0773264'}
    {u'thetvdb': 260189, u'tvrage': 32126, u'imdb': u'tt1798695'}
    {u'thetvdb': 139481, u'tvrage': 20203, u'imdb': u'tt1307083'}
    {u'thetvdb': 297960, u'tvrage': 49841, u'imdb': u'tt4845734'}
    {u'thetvdb': 70656, u'tvrage': None, u'imdb': u'tt0284767'}
    {u'thetvdb': 80694, u'tvrage': 15758, u'imdb': u'tt0878801'}
    {u'thetvdb': 282654, u'tvrage': 39954, u'imdb': u'tt3703500'}
    {u'thetvdb': 272737, u'tvrage': 37535, u'imdb': u'tt3155428'}
    {u'thetvdb': 76237, u'tvrage': None, u'imdb': u'tt0287196'}
    {u'thetvdb': 270469, u'tvrage': 34560, u'imdb': u'tt2766052'}
    {u'thetvdb': 301235, u'tvrage': None, u'imdb': u'tt0262975'}
    {u'thetvdb': 126811, u'tvrage': None, u'imdb': u'tt1388381'}
    {u'thetvdb': 307480, u'tvrage': None, u'imdb': u'tt0358889'}
    {u'thetvdb': 83326, u'tvrage': None, u'imdb': u'tt1011591'}
    {u'thetvdb': 279772, u'tvrage': None, u'imdb': u'tt3612584'}
    {u'thetvdb': 305936, u'tvrage': None, u'imdb': u'tt5464192'}
    {u'thetvdb': 267921, u'tvrage': None, u'imdb': u'tt4741696'}
    {u'thetvdb': 95351, u'tvrage': None, u'imdb': u'tt1974439'}
    {u'thetvdb': 79838, u'tvrage': 5631, u'imdb': u'tt0092325'}
    {u'thetvdb': None, u'tvrage': None, u'imdb': u'tt6772826'}
    {u'thetvdb': 127351, u'tvrage': 24425, u'imdb': u'tt1563069'}
    {u'thetvdb': 79550, u'tvrage': 6890, u'imdb': u'tt0489598'}
    {u'thetvdb': 148561, u'tvrage': 24465, u'imdb': u'tt1566154'}
    {u'thetvdb': 70905, u'tvrage': 3150, u'imdb': u'tt0338592'}
    {u'thetvdb': 70829, u'tvrage': None, u'imdb': u'tt0167515'}
    {u'thetvdb': 262883, u'tvrage': 31271, u'imdb': u'tt2330327'}
    {u'thetvdb': 84208, u'tvrage': None, u'imdb': u'tt0186747'}
    {u'thetvdb': 239961, u'tvrage': 27826, u'imdb': u'tt1821877'}
    {u'thetvdb': 216741, u'tvrage': None, u'imdb': u'tt1599374'}
    {u'thetvdb': 270465, u'tvrage': 35836, u'imdb': u'tt2946482'}
    {u'thetvdb': 268600, u'tvrage': 35103, u'imdb': u'tt2649480'}
    {u'thetvdb': 82550, u'tvrage': None, u'imdb': u'tt1229945'}
    {u'thetvdb': 248039, u'tvrage': 23213, u'imdb': u'tt1876612'}



```python
more_losers
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
      <th>status</th>
      <th>rating</th>
      <th>genres</th>
      <th>weight</th>
      <th>updated</th>
      <th>name</th>
      <th>language</th>
      <th>schedule</th>
      <th>url</th>
      <th>officialSite</th>
      <th>externals</th>
      <th>premiered</th>
      <th>summary</th>
      <th>_links</th>
      <th>image</th>
      <th>webChannel</th>
      <th>runtime</th>
      <th>type</th>
      <th>id</th>
      <th>network</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ended</td>
      <td>{u'average': None}</td>
      <td>[]</td>
      <td>0</td>
      <td>1449178946</td>
      <td>Famous in 12</td>
      <td>English</td>
      <td>{u'days': [u'Tuesday'], u'time': u'20:00'}</td>
      <td>http://www.tvmaze.com/shows/9024/famous-in-12</td>
      <td>None</td>
      <td>{u'thetvdb': 279947, u'tvrage': 37045, u'imdb'...</td>
      <td>2014-06-03</td>
      <td>&lt;p&gt;&lt;i&gt;&lt;b&gt;"Famous in 12"&lt;/b&gt;&lt;/i&gt;, the new unscr...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>60</td>
      <td>Reality</td>
      <td>9024</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ended</td>
      <td>{u'average': None}</td>
      <td>[Comedy, Family]</td>
      <td>14</td>
      <td>1497059695</td>
      <td>The Sharon Osbourne Show</td>
      <td>English</td>
      <td>{u'days': [u'Monday', u'Tuesday', u'Wednesday'...</td>
      <td>http://www.tvmaze.com/shows/19004/the-sharon-o...</td>
      <td>None</td>
      <td>{u'thetvdb': None, u'tvrage': 13173, u'imdb': ...</td>
      <td>2006-08-29</td>
      <td>&lt;p&gt;Daily talk show hosted by Sharon Osbourne.&lt;/p&gt;</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>60</td>
      <td>Talk Show</td>
      <td>19004</td>
      <td>{u'country': {u'timezone': u'Europe/London', u...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ended</td>
      <td>{u'average': None}</td>
      <td>[Comedy]</td>
      <td>0</td>
      <td>1503083428</td>
      <td>Steve Harvey's Big Time Challenge</td>
      <td>English</td>
      <td>{u'days': [u'Sunday'], u'time': u'21:00'}</td>
      <td>http://www.tvmaze.com/shows/29202/steve-harvey...</td>
      <td>None</td>
      <td>{u'thetvdb': 72157, u'tvrage': None, u'imdb': ...</td>
      <td>2003-09-11</td>
      <td>&lt;p&gt;&lt;b&gt;Steve Harvey's Big Time Challenge&lt;/b&gt;, a...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>60</td>
      <td>Talk Show</td>
      <td>29202</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Ended</td>
      <td>{u'average': None}</td>
      <td>[]</td>
      <td>0</td>
      <td>1475183910</td>
      <td>The Spin Crowd</td>
      <td>English</td>
      <td>{u'days': [u'Sunday'], u'time': u'22:30'}</td>
      <td>http://www.tvmaze.com/shows/21619/the-spin-crowd</td>
      <td>None</td>
      <td>{u'thetvdb': 218241, u'tvrage': None, u'imdb':...</td>
      <td>2010-08-22</td>
      <td>&lt;p&gt;Nobody knows how to make stars shine bright...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>30</td>
      <td>Reality</td>
      <td>21619</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Running</td>
      <td>{u'average': 1}</td>
      <td>[]</td>
      <td>0</td>
      <td>1495714601</td>
      <td>Babushka</td>
      <td>English</td>
      <td>{u'days': [u'Monday', u'Tuesday', u'Wednesday'...</td>
      <td>http://www.tvmaze.com/shows/25450/babushka</td>
      <td>http://www.itv.com/beontv/shows/babushka</td>
      <td>{u'thetvdb': 327908, u'tvrage': None, u'imdb':...</td>
      <td>2017-05-01</td>
      <td>&lt;p&gt;&lt;b&gt;Babushka&lt;/b&gt; is a brand new game show wh...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>60</td>
      <td>Game Show</td>
      <td>25450</td>
      <td>{u'country': {u'timezone': u'Europe/London', u...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Ended</td>
      <td>{u'average': None}</td>
      <td>[]</td>
      <td>0</td>
      <td>1483745416</td>
      <td>Chrome Underground</td>
      <td>English</td>
      <td>{u'days': [u'Tuesday'], u'time': u'22:00'}</td>
      <td>http://www.tvmaze.com/shows/24213/chrome-under...</td>
      <td>http://www.discovery.com/tv-shows/chrome-under...</td>
      <td>{u'thetvdb': 279810, u'tvrage': None, u'imdb':...</td>
      <td>2014-05-23</td>
      <td>&lt;p&gt;Two international classic car dealers searc...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>60</td>
      <td>Reality</td>
      <td>24213</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Ended</td>
      <td>{u'average': None}</td>
      <td>[]</td>
      <td>0</td>
      <td>1495602919</td>
      <td>Fear Factor</td>
      <td>English</td>
      <td>{u'days': [u'Sunday'], u'time': u''}</td>
      <td>http://www.tvmaze.com/shows/26838/fear-factor</td>
      <td>None</td>
      <td>{u'thetvdb': 283658, u'tvrage': None, u'imdb':...</td>
      <td>2002-09-10</td>
      <td>&lt;p&gt;This version has two teams of three contest...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>60</td>
      <td>Game Show</td>
      <td>26838</td>
      <td>{u'country': {u'timezone': u'Europe/London', u...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Ended</td>
      <td>{u'average': None}</td>
      <td>[]</td>
      <td>0</td>
      <td>1495254081</td>
      <td>Owner's Manual</td>
      <td>English</td>
      <td>{u'days': [u'Thursday'], u'time': u'22:00'}</td>
      <td>http://www.tvmaze.com/shows/9261/owners-manual</td>
      <td>None</td>
      <td>{u'thetvdb': 271341, u'tvrage': 33650, u'imdb'...</td>
      <td>2013-08-15</td>
      <td>&lt;p&gt;&lt;b&gt;Owner's Manual&lt;/b&gt; will test one of the ...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>30</td>
      <td>Reality</td>
      <td>9261</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Ended</td>
      <td>{u'average': None}</td>
      <td>[]</td>
      <td>0</td>
      <td>1487011574</td>
      <td>The Shire</td>
      <td>English</td>
      <td>{u'days': [u'Monday'], u'time': u'21:45'}</td>
      <td>http://www.tvmaze.com/shows/25288/the-shire</td>
      <td>None</td>
      <td>{u'thetvdb': 260677, u'tvrage': None, u'imdb':...</td>
      <td>2012-07-16</td>
      <td>&lt;p&gt;The series follows the lives and love of a ...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>25</td>
      <td>Reality</td>
      <td>25288</td>
      <td>{u'country': {u'timezone': u'Australia/Sydney'...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Ended</td>
      <td>{u'average': None}</td>
      <td>[Comedy]</td>
      <td>0</td>
      <td>1483143763</td>
      <td>The Montefuscos</td>
      <td>English</td>
      <td>{u'days': [u'Thursday'], u'time': u'20:00'}</td>
      <td>http://www.tvmaze.com/shows/24079/the-montefuscos</td>
      <td>None</td>
      <td>{u'thetvdb': 77616, u'tvrage': None, u'imdb': ...</td>
      <td>1975-09-04</td>
      <td>&lt;p&gt;The trials and tribulations of three genera...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>30</td>
      <td>Scripted</td>
      <td>24079</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Ended</td>
      <td>{u'average': None}</td>
      <td>[]</td>
      <td>0</td>
      <td>1464030266</td>
      <td>I Want to Be a Hilton</td>
      <td>English</td>
      <td>{u'days': [u'Tuesday'], u'time': u'21:00'}</td>
      <td>http://www.tvmaze.com/shows/17541/i-want-to-be...</td>
      <td>None</td>
      <td>{u'thetvdb': 74419, u'tvrage': None, u'imdb': ...</td>
      <td>2005-06-21</td>
      <td>&lt;p&gt;Kathy Hilton, onetime actress and mother of...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>None</td>
      <td>None</td>
      <td>60</td>
      <td>Reality</td>
      <td>17541</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Ended</td>
      <td>{u'average': None}</td>
      <td>[]</td>
      <td>20</td>
      <td>1478379662</td>
      <td>ABC's Nightlife</td>
      <td>English</td>
      <td>{u'days': [u'Monday', u'Tuesday', u'Wednesday'...</td>
      <td>http://www.tvmaze.com/shows/22597/abcs-nightlife</td>
      <td>None</td>
      <td>{u'thetvdb': None, u'tvrage': None, u'imdb': u...</td>
      <td>1964-11-09</td>
      <td>&lt;p&gt;&lt;b&gt;ABC's Nightlife&lt;/b&gt; is a late night dail...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>None</td>
      <td>None</td>
      <td>105</td>
      <td>Talk Show</td>
      <td>22597</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Running</td>
      <td>{u'average': None}</td>
      <td>[]</td>
      <td>0</td>
      <td>1454050022</td>
      <td>Untying the Knot</td>
      <td>English</td>
      <td>{u'days': [u'Monday'], u'time': u'22:00'}</td>
      <td>http://www.tvmaze.com/shows/6843/untying-the-knot</td>
      <td>http://www.bravotv.com/untying-the-knot</td>
      <td>{u'thetvdb': 282527, u'tvrage': 42189, u'imdb'...</td>
      <td>2014-06-04</td>
      <td>&lt;p&gt;Vikki Ziegler, known as the Divorce Diva, i...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>60</td>
      <td>Reality</td>
      <td>6843</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Ended</td>
      <td>{u'average': None}</td>
      <td>[Action]</td>
      <td>0</td>
      <td>1495406329</td>
      <td>Wipeout Canada</td>
      <td>English</td>
      <td>{u'days': [u'Sunday'], u'time': u'20:00'}</td>
      <td>http://www.tvmaze.com/shows/12998/wipeout-canada</td>
      <td>None</td>
      <td>{u'thetvdb': 246631, u'tvrage': None, u'imdb':...</td>
      <td>2011-04-03</td>
      <td>&lt;p&gt;&lt;b&gt;Wipeout Canada&lt;/b&gt; is a hilarious game s...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>60</td>
      <td>Game Show</td>
      <td>12998</td>
      <td>{u'country': {u'timezone': u'Canada/Atlantic',...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Ended</td>
      <td>{u'average': None}</td>
      <td>[]</td>
      <td>0</td>
      <td>1464363967</td>
      <td>Hurl!</td>
      <td>English</td>
      <td>{u'days': [u'Tuesday'], u'time': u'21:00'}</td>
      <td>http://www.tvmaze.com/shows/17705/hurl</td>
      <td>None</td>
      <td>{u'thetvdb': 82500, u'tvrage': None, u'imdb': ...</td>
      <td>2008-07-15</td>
      <td>&lt;p&gt;Get ready to get grossed out with G4's off-...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>None</td>
      <td>None</td>
      <td>30</td>
      <td>Reality</td>
      <td>17705</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Ended</td>
      <td>{u'average': None}</td>
      <td>[]</td>
      <td>0</td>
      <td>1457450255</td>
      <td>Meet the Parents</td>
      <td>English</td>
      <td>{u'days': [u'Thursday'], u'time': u'21:30'}</td>
      <td>http://www.tvmaze.com/shows/13973/meet-the-par...</td>
      <td>http://www.channel4.com/programmes/meet-the-pa...</td>
      <td>{u'thetvdb': 206381, u'tvrage': 26873, u'imdb'...</td>
      <td>2010-11-18</td>
      <td>&lt;p&gt;&lt;i&gt;Meet the Parents&lt;/i&gt; is a reality series...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>30</td>
      <td>Reality</td>
      <td>13973</td>
      <td>{u'country': {u'timezone': u'Europe/London', u...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Ended</td>
      <td>{u'average': None}</td>
      <td>[Drama, Action]</td>
      <td>0</td>
      <td>1481553637</td>
      <td>4th and Loud</td>
      <td>English</td>
      <td>{u'days': [u'Tuesday'], u'time': u'22:00'}</td>
      <td>http://www.tvmaze.com/shows/11854/4th-and-loud</td>
      <td>http://www.amc.com/shows/4th-and-loud</td>
      <td>{u'thetvdb': 284259, u'tvrage': None, u'imdb':...</td>
      <td>2014-08-12</td>
      <td>&lt;p&gt;&lt;b&gt;4th and Loud&lt;/b&gt; will follow the LA KISS...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>60</td>
      <td>Reality</td>
      <td>11854</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Ended</td>
      <td>{u'average': None}</td>
      <td>[]</td>
      <td>0</td>
      <td>1495496078</td>
      <td>It's Worth What?</td>
      <td>English</td>
      <td>{u'days': [u'Tuesday'], u'time': u'20:00'}</td>
      <td>http://www.tvmaze.com/shows/17619/its-worth-what</td>
      <td>None</td>
      <td>{u'thetvdb': 250186, u'tvrage': None, u'imdb':...</td>
      <td>2011-07-19</td>
      <td>&lt;p&gt;&lt;b&gt;It's Worth What? &lt;/b&gt;stars Cedric the En...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>60</td>
      <td>Game Show</td>
      <td>17619</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>To Be Determined</td>
      <td>{u'average': 6.6}</td>
      <td>[Drama, Thriller, Adult]</td>
      <td>92</td>
      <td>1497788418</td>
      <td>The Deleted</td>
      <td>English</td>
      <td>{u'days': [], u'time': u''}</td>
      <td>http://www.tvmaze.com/shows/19884/the-deleted</td>
      <td>https://www.fullscreen.com/series/the-deleted</td>
      <td>{u'thetvdb': 320679, u'tvrage': None, u'imdb':...</td>
      <td>2016-12-04</td>
      <td>&lt;p&gt;When escapees from a mysterious cult start ...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
      <td>15</td>
      <td>Scripted</td>
      <td>19884</td>
      <td>None</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Ended</td>
      <td>{u'average': 7.3}</td>
      <td>[Comedy, Action, Crime]</td>
      <td>14</td>
      <td>1500877446</td>
      <td>V.I.P.</td>
      <td>English</td>
      <td>{u'days': [u'Saturday'], u'time': u''}</td>
      <td>http://www.tvmaze.com/shows/1885/vip</td>
      <td>None</td>
      <td>{u'thetvdb': 74181, u'tvrage': 6494, u'imdb': ...</td>
      <td>1998-09-26</td>
      <td>&lt;p&gt;A campy syndicated series about Vallery Iro...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>60</td>
      <td>Scripted</td>
      <td>1885</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Running</td>
      <td>{u'average': 6}</td>
      <td>[Drama]</td>
      <td>63</td>
      <td>1496679327</td>
      <td>The Real Housewives of Atlanta</td>
      <td>English</td>
      <td>{u'days': [u'Sunday'], u'time': u'20:00'}</td>
      <td>http://www.tvmaze.com/shows/597/the-real-house...</td>
      <td>None</td>
      <td>{u'thetvdb': 84159, u'tvrage': 19672, u'imdb':...</td>
      <td>2008-10-07</td>
      <td>&lt;p&gt;An up-close and personal look at life in Ho...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>60</td>
      <td>Reality</td>
      <td>597</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Running</td>
      <td>{u'average': None}</td>
      <td>[Comedy, Children]</td>
      <td>0</td>
      <td>1475116665</td>
      <td>Pickle and Peanut</td>
      <td>English</td>
      <td>{u'days': [u'Monday'], u'time': u'18:30'}</td>
      <td>http://www.tvmaze.com/shows/3019/pickle-and-pe...</td>
      <td>http://disneyxd.disney.com/pickle-and-peanut</td>
      <td>{u'thetvdb': 300105, u'tvrage': 48178, u'imdb'...</td>
      <td>2015-09-07</td>
      <td>&lt;p&gt;&lt;i&gt;&lt;b&gt;"Pickle &amp;amp; Peanut"&lt;/b&gt;&lt;/i&gt; is abou...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
      <td>15</td>
      <td>Animation</td>
      <td>3019</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Ended</td>
      <td>{u'average': None}</td>
      <td>[Drama, Comedy, Romance]</td>
      <td>0</td>
      <td>1501880843</td>
      <td>Buckwild</td>
      <td>English</td>
      <td>{u'days': [u'Thursday'], u'time': u'22:00'}</td>
      <td>http://www.tvmaze.com/shows/25036/buckwild</td>
      <td>http://www.mtv.com/shows/buckwild</td>
      <td>{u'thetvdb': 264850, u'tvrage': None, u'imdb':...</td>
      <td>2013-01-03</td>
      <td>&lt;p&gt;The show follows the lives of nine young ad...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>30</td>
      <td>Reality</td>
      <td>25036</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Ended</td>
      <td>{u'average': 3}</td>
      <td>[]</td>
      <td>0</td>
      <td>1486506841</td>
      <td>Mystery Girls</td>
      <td>English</td>
      <td>{u'days': [u'Wednesday'], u'time': u'20:30'}</td>
      <td>http://www.tvmaze.com/shows/3950/mystery-girls</td>
      <td>http://abcfamily.go.com/shows/mystery-girls</td>
      <td>{u'thetvdb': 277020, u'tvrage': 35629, u'imdb'...</td>
      <td>2014-06-25</td>
      <td>&lt;p&gt;Two former detective TV show starlets broug...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>30</td>
      <td>Scripted</td>
      <td>3950</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Running</td>
      <td>{u'average': None}</td>
      <td>[Family]</td>
      <td>12</td>
      <td>1450883412</td>
      <td>Celebrity Wife Swap</td>
      <td>English</td>
      <td>{u'days': [u'Wednesday'], u'time': u'22:00'}</td>
      <td>http://www.tvmaze.com/shows/1783/celebrity-wif...</td>
      <td>http://abc.go.com/shows/celebrity-wife-swap/ab...</td>
      <td>{u'thetvdb': 254524, u'tvrage': 31887, u'imdb'...</td>
      <td>2012-01-02</td>
      <td>&lt;p&gt;The spouses in two celebrity families with ...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>60</td>
      <td>Reality</td>
      <td>1783</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Running</td>
      <td>{u'average': 7}</td>
      <td>[Comedy]</td>
      <td>0</td>
      <td>1472855087</td>
      <td>Dish Nation</td>
      <td>English</td>
      <td>{u'days': [u'Monday', u'Tuesday', u'Wednesday'...</td>
      <td>http://www.tvmaze.com/shows/9199/dish-nation</td>
      <td>http://www.reelz.com/dish-nation/</td>
      <td>{u'thetvdb': 271916, u'tvrage': None, u'imdb':...</td>
      <td>2011-07-25</td>
      <td>&lt;p&gt;&lt;i&gt;Dish Nation&lt;/i&gt; is a nightly syndicated ...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>30</td>
      <td>Scripted</td>
      <td>9199</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Ended</td>
      <td>{u'average': None}</td>
      <td>[Children]</td>
      <td>0</td>
      <td>1502544202</td>
      <td>Ni Hao, Kai-lan</td>
      <td>English</td>
      <td>{u'days': [u'Monday', u'Tuesday', u'Wednesday'...</td>
      <td>http://www.tvmaze.com/shows/13161/ni-hao-kai-lan</td>
      <td>None</td>
      <td>{u'thetvdb': 82005, u'tvrage': None, u'imdb': ...</td>
      <td>2008-02-07</td>
      <td>&lt;p&gt;Ni Hao, Kai-lan , which is Mandarin for "He...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>30</td>
      <td>Animation</td>
      <td>13161</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Running</td>
      <td>{u'average': None}</td>
      <td>[Comedy, Family]</td>
      <td>0</td>
      <td>1502948333</td>
      <td>Scaredy Squirrel</td>
      <td>English</td>
      <td>{u'days': [u'Monday', u'Tuesday', u'Wednesday'...</td>
      <td>http://www.tvmaze.com/shows/20564/scaredy-squi...</td>
      <td>http://www.scaredysquirrel.com</td>
      <td>{u'thetvdb': 250472, u'tvrage': None, u'imdb':...</td>
      <td>2011-04-01</td>
      <td>&lt;p&gt;&lt;b&gt;Scaredy Squirrel &lt;/b&gt;follows the adventu...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>10</td>
      <td>Animation</td>
      <td>20564</td>
      <td>{u'country': {u'timezone': u'Canada/Atlantic',...</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Running</td>
      <td>{u'average': None}</td>
      <td>[]</td>
      <td>76</td>
      <td>1502312151</td>
      <td>Big Brother After Dark</td>
      <td>English</td>
      <td>{u'days': [u'Monday', u'Tuesday', u'Wednesday'...</td>
      <td>http://www.tvmaze.com/shows/18240/big-brother-...</td>
      <td>http://poptv.com/big_brother_after_dark</td>
      <td>{u'thetvdb': 81491, u'tvrage': None, u'imdb': ...</td>
      <td>2007-07-05</td>
      <td>&lt;p&gt;&lt;b&gt;Big Brother After Dark&lt;/b&gt; is the live, ...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>180</td>
      <td>Reality</td>
      <td>18240</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Ended</td>
      <td>{u'average': 1}</td>
      <td>[Action, Adventure]</td>
      <td>0</td>
      <td>1474827145</td>
      <td>American Paranormal</td>
      <td>English</td>
      <td>{u'days': [u'Sunday'], u'time': u'20:00'}</td>
      <td>http://www.tvmaze.com/shows/19115/american-par...</td>
      <td>None</td>
      <td>{u'thetvdb': 137691, u'tvrage': None, u'imdb':...</td>
      <td>2010-01-24</td>
      <td>&lt;p&gt;Whether it is the existence of aliens, the ...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>60</td>
      <td>Reality</td>
      <td>19115</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>107</th>
      <td>To Be Determined</td>
      <td>{u'average': None}</td>
      <td>[]</td>
      <td>0</td>
      <td>1495420105</td>
      <td>Who's Doing the Dishes?</td>
      <td>English</td>
      <td>{u'days': [u'Monday', u'Tuesday', u'Wednesday'...</td>
      <td>http://www.tvmaze.com/shows/8612/whos-doing-th...</td>
      <td>None</td>
      <td>{u'thetvdb': 300384, u'tvrage': None, u'imdb':...</td>
      <td>2014-09-01</td>
      <td>&lt;p&gt;&lt;b&gt;Who's Doing the Dishes?&lt;/b&gt; is a UK game...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>60</td>
      <td>Game Show</td>
      <td>8612</td>
      <td>{u'country': {u'timezone': u'Europe/London', u...</td>
    </tr>
    <tr>
      <th>108</th>
      <td>Ended</td>
      <td>{u'average': None}</td>
      <td>[]</td>
      <td>0</td>
      <td>1474499818</td>
      <td>I'm a Celebrity, Get Me Out of Here! NOW!</td>
      <td>English</td>
      <td>{u'days': [u'Monday', u'Tuesday', u'Wednesday'...</td>
      <td>http://www.tvmaze.com/shows/8558/im-a-celebrit...</td>
      <td>http://www.itv.com/imacelebrity/itv2-now</td>
      <td>{u'thetvdb': 264239, u'tvrage': None, u'imdb':...</td>
      <td>2011-11-13</td>
      <td>&lt;p&gt;&lt;i&gt;"I'm a Celebrity...Get Me Out of Here! N...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>60</td>
      <td>Reality</td>
      <td>8558</td>
      <td>{u'country': {u'timezone': u'Europe/London', u...</td>
    </tr>
    <tr>
      <th>109</th>
      <td>Ended</td>
      <td>{u'average': None}</td>
      <td>[Romance]</td>
      <td>0</td>
      <td>1474764176</td>
      <td>More to Love</td>
      <td>English</td>
      <td>{u'days': [u'Tuesday'], u'time': u'21:00'}</td>
      <td>http://www.tvmaze.com/shows/21467/more-to-love</td>
      <td>None</td>
      <td>{u'thetvdb': 106801, u'tvrage': None, u'imdb':...</td>
      <td>2009-07-28</td>
      <td>&lt;p&gt;Follows one regular guy's search for love a...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>60</td>
      <td>Reality</td>
      <td>21467</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
    </tr>
    <tr>
      <th>110</th>
      <td>Ended</td>
      <td>{u'average': None}</td>
      <td>[]</td>
      <td>0</td>
      <td>1467307858</td>
      <td>I Want to Work for Diddy</td>
      <td>English</td>
      <td>{u'days': [u'Monday'], u'time': u'22:00'}</td>
      <td>http://www.tvmaze.com/shows/18829/i-want-to-wo...</td>
      <td>None</td>
      <td>{u'thetvdb': 87131, u'tvrage': None, u'imdb': ...</td>
      <td>2008-08-04</td>
      <td>&lt;p&gt;Diddy. He only needs one name, but he needs...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>60</td>
      <td>Reality</td>
      <td>18829</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
    </tr>
    <tr>
      <th>111</th>
      <td>Ended</td>
      <td>{u'average': None}</td>
      <td>[]</td>
      <td>0</td>
      <td>1490997318</td>
      <td>Donald J. Trump Presents: The Ultimate Merger</td>
      <td>English</td>
      <td>{u'days': [u'Thursday'], u'time': u'22:00'}</td>
      <td>http://www.tvmaze.com/shows/26564/donald-j-tru...</td>
      <td>None</td>
      <td>{u'thetvdb': 173981, u'tvrage': None, u'imdb':...</td>
      <td>2010-06-17</td>
      <td>&lt;p&gt;Through a series of challenges, both relati...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>60</td>
      <td>Reality</td>
      <td>26564</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
    </tr>
    <tr>
      <th>112</th>
      <td>Running</td>
      <td>{u'average': None}</td>
      <td>[]</td>
      <td>0</td>
      <td>1477193039</td>
      <td>America's Election Headquarters</td>
      <td>English</td>
      <td>{u'days': [u'Monday', u'Tuesday', u'Wednesday'...</td>
      <td>http://www.tvmaze.com/shows/11837/americas-ele...</td>
      <td>http://www.foxnews.com/on-air/americas-news-hq...</td>
      <td>{u'thetvdb': None, u'tvrage': None, u'imdb': u...</td>
      <td>2008-04-22</td>
      <td>&lt;p&gt;&lt;b&gt;America's Election Headquarters&lt;/b&gt; is a...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>60</td>
      <td>Talk Show</td>
      <td>11837</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
    </tr>
    <tr>
      <th>113</th>
      <td>Running</td>
      <td>{u'average': None}</td>
      <td>[]</td>
      <td>45</td>
      <td>1502693229</td>
      <td>BBC Weekend News</td>
      <td>English</td>
      <td>{u'days': [u'Saturday', u'Sunday'], u'time': u''}</td>
      <td>http://www.tvmaze.com/shows/7333/bbc-weekend-news</td>
      <td>http://www.bbc.co.uk/programmes/b009m51q</td>
      <td>{u'thetvdb': 264762, u'tvrage': 31404, u'imdb'...</td>
      <td>1954-07-05</td>
      <td>&lt;p&gt;&lt;b&gt;BBC Weekend News&lt;/b&gt; is the national new...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>None</td>
      <td>News</td>
      <td>7333</td>
      <td>{u'country': {u'timezone': u'Europe/London', u...</td>
    </tr>
    <tr>
      <th>114</th>
      <td>Ended</td>
      <td>{u'average': None}</td>
      <td>[Comedy, Children]</td>
      <td>0</td>
      <td>1477293529</td>
      <td>Barney &amp; Friends</td>
      <td>English</td>
      <td>{u'days': [u'Monday', u'Tuesday', u'Wednesday'...</td>
      <td>http://www.tvmaze.com/shows/15482/barney-friends</td>
      <td>None</td>
      <td>{u'thetvdb': 72180, u'tvrage': None, u'imdb': ...</td>
      <td>1992-04-06</td>
      <td>&lt;p&gt;&lt;b&gt;Barney &amp;amp; Friends&lt;/b&gt; is an American ...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>30</td>
      <td>Scripted</td>
      <td>15482</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
    </tr>
    <tr>
      <th>115</th>
      <td>Running</td>
      <td>{u'average': 7}</td>
      <td>[Comedy]</td>
      <td>89</td>
      <td>1503147213</td>
      <td>The Powerpuff Girls</td>
      <td>English</td>
      <td>{u'days': [u'Sunday'], u'time': u'17:30'}</td>
      <td>http://www.tvmaze.com/shows/6771/the-powerpuff...</td>
      <td>None</td>
      <td>{u'thetvdb': 307473, u'tvrage': None, u'imdb':...</td>
      <td>2016-04-04</td>
      <td>&lt;p&gt;The city of Townsville may be a beautiful, ...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>15</td>
      <td>Animation</td>
      <td>6771</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
    </tr>
    <tr>
      <th>116</th>
      <td>Running</td>
      <td>{u'average': None}</td>
      <td>[]</td>
      <td>0</td>
      <td>1497251730</td>
      <td>TMZ</td>
      <td>English</td>
      <td>{u'days': [u'Monday', u'Tuesday', u'Wednesday'...</td>
      <td>http://www.tvmaze.com/shows/24857/tmz</td>
      <td>http://www.tmz.com/when-its-on?adid=tmz_web_na...</td>
      <td>{u'thetvdb': 147701, u'tvrage': None, u'imdb':...</td>
      <td>2011-11-02</td>
      <td>&lt;p&gt;&lt;b&gt;TMZ &lt;/b&gt;(also known simply as &lt;i&gt;TMZ&lt;/i&gt;...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>30</td>
      <td>Talk Show</td>
      <td>24857</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
    </tr>
    <tr>
      <th>117</th>
      <td>Ended</td>
      <td>{u'average': 6}</td>
      <td>[]</td>
      <td>0</td>
      <td>1476263385</td>
      <td>Kendra</td>
      <td>English</td>
      <td>{u'days': [u'Sunday'], u'time': u'22:00'}</td>
      <td>http://www.tvmaze.com/shows/21952/kendra</td>
      <td>None</td>
      <td>{u'thetvdb': 98371, u'tvrage': None, u'imdb': ...</td>
      <td>2009-06-07</td>
      <td>&lt;p&gt;Kendra Wilkinson finds herself at a crossro...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>30</td>
      <td>Reality</td>
      <td>21952</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
    </tr>
    <tr>
      <th>118</th>
      <td>Ended</td>
      <td>{u'average': None}</td>
      <td>[]</td>
      <td>0</td>
      <td>1465065870</td>
      <td>The Roseanne Show</td>
      <td>English</td>
      <td>{u'days': [u'Monday', u'Tuesday', u'Wednesday'...</td>
      <td>http://www.tvmaze.com/shows/12252/the-roseanne...</td>
      <td>None</td>
      <td>{u'thetvdb': 72141, u'tvrage': None, u'imdb': ...</td>
      <td>1998-09-14</td>
      <td>&lt;p&gt;&lt;i&gt;&lt;b&gt;"The Roseanne Show"&lt;/b&gt;&lt;/i&gt; is a synd...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>60</td>
      <td>Talk Show</td>
      <td>12252</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
    </tr>
    <tr>
      <th>119</th>
      <td>Running</td>
      <td>{u'average': 1}</td>
      <td>[Music]</td>
      <td>0</td>
      <td>1484368515</td>
      <td>The Xtra Factor Live</td>
      <td>English</td>
      <td>{u'days': [u'Saturday', u'Sunday'], u'time': u''}</td>
      <td>http://www.tvmaze.com/shows/3764/the-xtra-fact...</td>
      <td>None</td>
      <td>{u'thetvdb': 75567, u'tvrage': 12949, u'imdb':...</td>
      <td>2004-09-04</td>
      <td>&lt;p&gt;Thousands audition. Only one can win. The s...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>60</td>
      <td>Reality</td>
      <td>3764</td>
      <td>{u'country': {u'timezone': u'Europe/London', u...</td>
    </tr>
    <tr>
      <th>120</th>
      <td>Ended</td>
      <td>{u'average': 7}</td>
      <td>[Comedy]</td>
      <td>0</td>
      <td>1488031177</td>
      <td>But I'm Chris Jericho!</td>
      <td>English</td>
      <td>{u'days': [u'Tuesday'], u'time': u'22:00'}</td>
      <td>http://www.tvmaze.com/shows/13150/but-im-chris...</td>
      <td>http://butimchrisjericho.com</td>
      <td>{u'thetvdb': 275787, u'tvrage': None, u'imdb':...</td>
      <td>2013-10-29</td>
      <td>&lt;p&gt;&lt;b&gt;But I'm Chris Jericho!&lt;/b&gt; is an interac...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
      <td>8</td>
      <td>Scripted</td>
      <td>13150</td>
      <td>{u'country': {u'timezone': u'Canada/Atlantic',...</td>
    </tr>
    <tr>
      <th>121</th>
      <td>Ended</td>
      <td>{u'average': 6}</td>
      <td>[Comedy]</td>
      <td>0</td>
      <td>1466802381</td>
      <td>Party Over Here</td>
      <td>English</td>
      <td>{u'days': [u'Saturday'], u'time': u'23:00'}</td>
      <td>http://www.tvmaze.com/shows/12662/party-over-here</td>
      <td>http://www.fox.com/party-over-here</td>
      <td>{u'thetvdb': 308457, u'tvrage': 51439, u'imdb'...</td>
      <td>2016-03-12</td>
      <td>&lt;p&gt;A new late-night half-hour sketch comedy se...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>30</td>
      <td>Variety</td>
      <td>12662</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
    </tr>
    <tr>
      <th>122</th>
      <td>Ended</td>
      <td>{u'average': None}</td>
      <td>[Action, Adventure, Horror]</td>
      <td>0</td>
      <td>1500043650</td>
      <td>Alaska Monsters</td>
      <td>English</td>
      <td>{u'days': [u'Saturday'], u'time': u'22:00'}</td>
      <td>http://www.tvmaze.com/shows/3124/alaska-monsters</td>
      <td>http://www.destinationamerica.com/tv-shows/ala...</td>
      <td>{u'thetvdb': 285286, u'tvrage': 44525, u'imdb'...</td>
      <td>2014-09-12</td>
      <td>&lt;p&gt;Treacherous terrain and unforgiving natural...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>60</td>
      <td>Reality</td>
      <td>3124</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
    </tr>
    <tr>
      <th>123</th>
      <td>Ended</td>
      <td>{u'average': None}</td>
      <td>[Drama, Children]</td>
      <td>0</td>
      <td>1495726406</td>
      <td>Abby's Ultimate Dance Competition</td>
      <td>English</td>
      <td>{u'days': [u'Tuesday'], u'time': u'21:00'}</td>
      <td>http://www.tvmaze.com/shows/9420/abbys-ultimat...</td>
      <td>http://www.mylifetime.com/shows/abbys-ultimate...</td>
      <td>{u'thetvdb': 261287, u'tvrage': 32847, u'imdb'...</td>
      <td>2012-10-09</td>
      <td>&lt;p&gt;Lifetime has picked-up the reality series &lt;...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>60</td>
      <td>Game Show</td>
      <td>9420</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
    </tr>
    <tr>
      <th>124</th>
      <td>Ended</td>
      <td>{u'average': None}</td>
      <td>[Children, Mystery, Supernatural]</td>
      <td>0</td>
      <td>1502934987</td>
      <td>The Othersiders</td>
      <td>English</td>
      <td>{u'days': [u'Wednesday'], u'time': u'21:00'}</td>
      <td>http://www.tvmaze.com/shows/9593/the-othersiders</td>
      <td>None</td>
      <td>{u'thetvdb': 250325, u'tvrage': None, u'imdb':...</td>
      <td>2009-06-17</td>
      <td>&lt;p&gt;&lt;b&gt;The Othersiders&lt;/b&gt; was an American para...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>30</td>
      <td>Reality</td>
      <td>9593</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
    </tr>
    <tr>
      <th>125</th>
      <td>Ended</td>
      <td>{u'average': None}</td>
      <td>[]</td>
      <td>0</td>
      <td>1449520834</td>
      <td>Canadian Idol</td>
      <td>English</td>
      <td>{u'days': [], u'time': u'20:00'}</td>
      <td>http://www.tvmaze.com/shows/9674/canadian-idol</td>
      <td>None</td>
      <td>{u'thetvdb': 72133, u'tvrage': None, u'imdb': ...</td>
      <td>2003-06-09</td>
      <td>&lt;p&gt;&lt;i&gt;&lt;b&gt;"Canadian Idol"&lt;/b&gt;&lt;/i&gt; is a Canadian...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>60</td>
      <td>Reality</td>
      <td>9674</td>
      <td>{u'country': {u'timezone': u'Canada/Atlantic',...</td>
    </tr>
    <tr>
      <th>126</th>
      <td>Ended</td>
      <td>{u'average': None}</td>
      <td>[]</td>
      <td>0</td>
      <td>1474314323</td>
      <td>Extreme Makeover</td>
      <td>English</td>
      <td>{u'days': [u'Thursday'], u'time': u'21:00'}</td>
      <td>http://www.tvmaze.com/shows/21134/extreme-make...</td>
      <td>None</td>
      <td>{u'thetvdb': 72488, u'tvrage': None, u'imdb': ...</td>
      <td>2002-12-11</td>
      <td>&lt;p&gt;Three people are chosen to receive the make...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>60</td>
      <td>Reality</td>
      <td>21134</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
    </tr>
    <tr>
      <th>127</th>
      <td>Ended</td>
      <td>{u'average': None}</td>
      <td>[]</td>
      <td>0</td>
      <td>1469556547</td>
      <td>Pretty Wild</td>
      <td>English</td>
      <td>{u'days': [u'Sunday'], u'time': u'22:30'}</td>
      <td>http://www.tvmaze.com/shows/19522/pretty-wild</td>
      <td>None</td>
      <td>{u'thetvdb': 149371, u'tvrage': 25246, u'imdb'...</td>
      <td>2010-03-14</td>
      <td></td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>30</td>
      <td>Reality</td>
      <td>19522</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
    </tr>
    <tr>
      <th>128</th>
      <td>Running</td>
      <td>{u'average': None}</td>
      <td>[Comedy]</td>
      <td>0</td>
      <td>1502570683</td>
      <td>Just for Laughs: All Access</td>
      <td>English</td>
      <td>{u'days': [u'Saturday'], u'time': u'22:00'}</td>
      <td>http://www.tvmaze.com/shows/18044/just-for-lau...</td>
      <td>http://www.thecomedynetwork.ca/Shows/JustForLa...</td>
      <td>{u'thetvdb': 291820, u'tvrage': None, u'imdb':...</td>
      <td>2012-10-12</td>
      <td>&lt;p&gt;Comedians celebrate the 30th anniversary of...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>60</td>
      <td>Variety</td>
      <td>18044</td>
      <td>{u'country': {u'timezone': u'Canada/Atlantic',...</td>
    </tr>
    <tr>
      <th>129</th>
      <td>Ended</td>
      <td>{u'average': None}</td>
      <td>[]</td>
      <td>0</td>
      <td>1455387941</td>
      <td>Jesse James is a Dead Man</td>
      <td>English</td>
      <td>{u'days': [u'Sunday'], u'time': u'22:00'}</td>
      <td>http://www.tvmaze.com/shows/12951/jesse-james-...</td>
      <td>None</td>
      <td>{u'thetvdb': 96071, u'tvrage': None, u'imdb': ...</td>
      <td>2009-05-31</td>
      <td>&lt;p&gt;Jesse James takes on the role of a modern-d...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>60</td>
      <td>Reality</td>
      <td>12951</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
    </tr>
    <tr>
      <th>130</th>
      <td>Ended</td>
      <td>{u'average': None}</td>
      <td>[]</td>
      <td>0</td>
      <td>1488221218</td>
      <td>Secretly Pregnant</td>
      <td>English</td>
      <td>{u'days': [u'Thursday'], u'time': u'22:00'}</td>
      <td>http://www.tvmaze.com/shows/25580/secretly-pre...</td>
      <td>http://www.discoverylife.com/tv-shows/secretly...</td>
      <td>{u'thetvdb': 287516, u'tvrage': None, u'imdb':...</td>
      <td>2011-10-13</td>
      <td>&lt;p&gt;The stories of women who, for various reaso...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>60</td>
      <td>Reality</td>
      <td>25580</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
    </tr>
    <tr>
      <th>131</th>
      <td>Running</td>
      <td>{u'average': None}</td>
      <td>[Family]</td>
      <td>13</td>
      <td>1455319657</td>
      <td>The Briefcase</td>
      <td>English</td>
      <td>{u'days': [], u'time': u'20:00'}</td>
      <td>http://www.tvmaze.com/shows/1831/the-briefcase</td>
      <td>None</td>
      <td>{u'thetvdb': 295059, u'tvrage': 48857, u'imdb'...</td>
      <td>2015-05-27</td>
      <td>&lt;p&gt;The show features a social experiment eleme...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>60</td>
      <td>Reality</td>
      <td>1831</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
    </tr>
    <tr>
      <th>132</th>
      <td>Ended</td>
      <td>{u'average': None}</td>
      <td>[Comedy]</td>
      <td>0</td>
      <td>1492370917</td>
      <td>PrankStars</td>
      <td>English</td>
      <td>{u'days': [], u'time': u''}</td>
      <td>http://www.tvmaze.com/shows/27206/prankstars</td>
      <td>None</td>
      <td>{u'thetvdb': 250280, u'tvrage': None, u'imdb':...</td>
      <td>2011-07-15</td>
      <td>&lt;p&gt;A hidden-camera series where unsuspecting t...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>30</td>
      <td>Reality</td>
      <td>27206</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
    </tr>
    <tr>
      <th>133</th>
      <td>Ended</td>
      <td>{u'average': None}</td>
      <td>[]</td>
      <td>0</td>
      <td>1485549026</td>
      <td>Cash Dome</td>
      <td>English</td>
      <td>{u'days': [u'Tuesday'], u'time': u'21:30'}</td>
      <td>http://www.tvmaze.com/shows/24751/cash-dome</td>
      <td>None</td>
      <td>{u'thetvdb': 272357, u'tvrage': None, u'imdb':...</td>
      <td>2013-08-13</td>
      <td>&lt;p&gt;For a quarter century, &lt;b&gt;Cash Dome&lt;/b&gt; Jew...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>30</td>
      <td>Reality</td>
      <td>24751</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
    </tr>
    <tr>
      <th>134</th>
      <td>Ended</td>
      <td>{u'average': None}</td>
      <td>[Comedy]</td>
      <td>0</td>
      <td>1502593090</td>
      <td>CeeLo Green's The Good Life</td>
      <td>English</td>
      <td>{u'days': [u'Monday'], u'time': u'22:30'}</td>
      <td>http://www.tvmaze.com/shows/25900/ceelo-greens...</td>
      <td>None</td>
      <td>{u'thetvdb': 282130, u'tvrage': None, u'imdb':...</td>
      <td>2014-06-23</td>
      <td>&lt;p&gt;Follow CeeLo as he tackles not only a packe...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>30</td>
      <td>Reality</td>
      <td>25900</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
    </tr>
    <tr>
      <th>135</th>
      <td>Ended</td>
      <td>{u'average': None}</td>
      <td>[]</td>
      <td>0</td>
      <td>1477193480</td>
      <td>America's Prom Queen</td>
      <td>English</td>
      <td>{u'days': [u'Monday'], u'time': u'21:00'}</td>
      <td>http://www.tvmaze.com/shows/16384/americas-pro...</td>
      <td>None</td>
      <td>{u'thetvdb': None, u'tvrage': 18611, u'imdb': ...</td>
      <td>2008-03-17</td>
      <td>&lt;p&gt;&lt;b&gt;America's Prom Queen&lt;/b&gt; is a reality TV...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>60</td>
      <td>Reality</td>
      <td>16384</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
    </tr>
    <tr>
      <th>136</th>
      <td>Ended</td>
      <td>{u'average': None}</td>
      <td>[]</td>
      <td>0</td>
      <td>1461445299</td>
      <td>Hollywood Me</td>
      <td>English</td>
      <td>{u'days': [u'Wednesday'], u'time': u'20:00'}</td>
      <td>http://www.tvmaze.com/shows/15972/hollywood-me</td>
      <td>None</td>
      <td>{u'thetvdb': 271067, u'tvrage': None, u'imdb':...</td>
      <td>2013-06-19</td>
      <td>&lt;p&gt;Martyn Lawrence Bullard's normal clients in...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>30</td>
      <td>Reality</td>
      <td>15972</td>
      <td>{u'country': {u'timezone': u'Europe/London', u...</td>
    </tr>
  </tbody>
</table>
<p>137 rows × 20 columns</p>
</div>




```python
# Create a backup occasionally, and pickle after we've pulled the data
more_losers_backup = more_losers.copy()

```


```python
DO_NOT_RUN = True  # Be sure to check the file name to write before enabling execution on this block

if not DO_NOT_RUN:
    pickle.dump( more_losers, open( "save_more_losers_df.p", "wb" ) )
```

### Add a column to both shows (good) and losers (bad) to classify the rows as winners or losers 


```python
# All the data pulled from api and placed in dataframes was pickled and written to disk.
# Reading it all back in and adding a column to indicate if it was a winner or loser
# then will clean up and begin the analysis.
# $ ls *.p
# save_losers_df.p    save_more_losers_df.p     save_shows_df.p
```


```python
# read data back in from the saved file
winners = pickle.load( open( "save_shows_df.p", "rb" ) )
losers1 = pickle.load( open( "save_losers_df.p", "rb" ) )
losers2 = pickle.load( open( "save_more_losers_df.p", "rb" ) )
```


```python
print " Winners:", winners.shape
print " Losers1:", losers1.shape
print " Losers2:", losers2.shape
```

     Winners: (235, 20)
     Losers1: (229, 22)
     Losers2: (170, 20)



```python
# Investigate why Losers1 has 22 columns, must have been pickled after a change.   
losers1.columns
```




    Index([u'_links', u'code', u'externals', u'genres', u'id', u'image',
           u'language', u'message', u'name', u'network', u'officialSite',
           u'premiered', u'rating', u'runtime', u'schedule', u'status', u'summary',
           u'type', u'updated', u'url', u'webChannel', u'weight'],
          dtype='object')




```python
losers2.columns
```




    Index([u'status', u'rating', u'genres', u'weight', u'updated', u'name',
           u'language', u'schedule', u'url', u'officialSite', u'externals',
           u'premiered', u'summary', u'_links', u'image', u'webChannel',
           u'runtime', u'type', u'id', u'network'],
          dtype='object')




```python
winners.columns
```




    Index([u'status', u'rating', u'genres', u'weight', u'updated', u'name',
           u'language', u'schedule', u'url', u'officialSite', u'externals',
           u'premiered', u'summary', u'_links', u'image', u'webChannel',
           u'runtime', u'type', u'id', u'network'],
          dtype='object')




```python
# Correct the issue by copying correct columns from losers1 into new_losers1
cols = losers2.columns
new_losers1 = losers1[cols]
```


```python
new_losers1.shape
```




    (229, 20)




```python
# check that all three dataframes have same data in same order
winners.head(2)
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
      <th>status</th>
      <th>rating</th>
      <th>genres</th>
      <th>weight</th>
      <th>updated</th>
      <th>name</th>
      <th>language</th>
      <th>schedule</th>
      <th>url</th>
      <th>officialSite</th>
      <th>externals</th>
      <th>premiered</th>
      <th>summary</th>
      <th>_links</th>
      <th>image</th>
      <th>webChannel</th>
      <th>runtime</th>
      <th>type</th>
      <th>id</th>
      <th>network</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ended</td>
      <td>{u'average': 9.4}</td>
      <td>[Nature]</td>
      <td>87</td>
      <td>1490631396</td>
      <td>Planet Earth II</td>
      <td>English</td>
      <td>{u'days': [u'Sunday'], u'time': u'20:00'}</td>
      <td>http://www.tvmaze.com/shows/22036/planet-earth-ii</td>
      <td>http://www.bbc.co.uk/programmes/p02544td</td>
      <td>{u'thetvdb': 318408, u'tvrage': None, u'imdb':...</td>
      <td>2016-11-06</td>
      <td>&lt;p&gt;David Attenborough presents a documentary s...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>60</td>
      <td>Documentary</td>
      <td>22036</td>
      <td>{u'country': {u'timezone': u'Europe/London', u...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ended</td>
      <td>{u'average': 9.4}</td>
      <td>[Drama, Action, War, History]</td>
      <td>86</td>
      <td>1492651730</td>
      <td>Band of Brothers</td>
      <td>English</td>
      <td>{u'days': [u'Sunday'], u'time': u'20:00'}</td>
      <td>http://www.tvmaze.com/shows/465/band-of-brothers</td>
      <td>http://www.hbo.com/band-of-brothers</td>
      <td>{u'thetvdb': 74205, u'tvrage': 2708, u'imdb': ...</td>
      <td>2001-09-09</td>
      <td>&lt;p&gt;Drawn from interviews with survivors of Eas...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>60</td>
      <td>Scripted</td>
      <td>465</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
    </tr>
  </tbody>
</table>
</div>




```python
new_losers1.head(2)
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
      <th>status</th>
      <th>rating</th>
      <th>genres</th>
      <th>weight</th>
      <th>updated</th>
      <th>name</th>
      <th>language</th>
      <th>schedule</th>
      <th>url</th>
      <th>officialSite</th>
      <th>externals</th>
      <th>premiered</th>
      <th>summary</th>
      <th>_links</th>
      <th>image</th>
      <th>webChannel</th>
      <th>runtime</th>
      <th>type</th>
      <th>id</th>
      <th>network</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Running</td>
      <td>{u'average': None}</td>
      <td>[]</td>
      <td>63</td>
      <td>1463447317</td>
      <td>The Bill Cunningham Show</td>
      <td>English</td>
      <td>{u'days': [u'Monday', u'Tuesday', u'Wednesday'...</td>
      <td>http://www.tvmaze.com/shows/6068/the-bill-cunn...</td>
      <td>http://www.thebillcunninghamshow.com/</td>
      <td>{u'thetvdb': 283995, u'tvrage': 40425, u'imdb'...</td>
      <td>2011-09-19</td>
      <td>&lt;p&gt;&lt;i&gt;&lt;b&gt;"The Bill Cunningham Show"&lt;/b&gt;,&lt;/i&gt; T...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>60</td>
      <td>Talk Show</td>
      <td>6068</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>To Be Determined</td>
      <td>{u'average': None}</td>
      <td>[Comedy, Music]</td>
      <td>0</td>
      <td>1477139892</td>
      <td>Six Degrees of Everything</td>
      <td>English</td>
      <td>{u'days': [u'Tuesday'], u'time': u'23:00'}</td>
      <td>http://www.tvmaze.com/shows/2821/six-degrees-o...</td>
      <td>http://www.trutv.com/shows/six-degrees-of-ever...</td>
      <td>{u'thetvdb': 299234, u'tvrage': 50418, u'imdb'...</td>
      <td>2015-08-18</td>
      <td>&lt;p&gt;&lt;b&gt;Six Degrees of Everything&lt;/b&gt; is a fast-...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>30</td>
      <td>Variety</td>
      <td>2821</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
    </tr>
  </tbody>
</table>
</div>




```python
losers2.head(2)
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
      <th>status</th>
      <th>rating</th>
      <th>genres</th>
      <th>weight</th>
      <th>updated</th>
      <th>name</th>
      <th>language</th>
      <th>schedule</th>
      <th>url</th>
      <th>officialSite</th>
      <th>externals</th>
      <th>premiered</th>
      <th>summary</th>
      <th>_links</th>
      <th>image</th>
      <th>webChannel</th>
      <th>runtime</th>
      <th>type</th>
      <th>id</th>
      <th>network</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ended</td>
      <td>{u'average': None}</td>
      <td>[]</td>
      <td>0</td>
      <td>1449178946</td>
      <td>Famous in 12</td>
      <td>English</td>
      <td>{u'days': [u'Tuesday'], u'time': u'20:00'}</td>
      <td>http://www.tvmaze.com/shows/9024/famous-in-12</td>
      <td>None</td>
      <td>{u'thetvdb': 279947, u'tvrage': 37045, u'imdb'...</td>
      <td>2014-06-03</td>
      <td>&lt;p&gt;&lt;i&gt;&lt;b&gt;"Famous in 12"&lt;/b&gt;&lt;/i&gt;, the new unscr...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>60</td>
      <td>Reality</td>
      <td>9024</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ended</td>
      <td>{u'average': None}</td>
      <td>[Comedy, Family]</td>
      <td>14</td>
      <td>1497059695</td>
      <td>The Sharon Osbourne Show</td>
      <td>English</td>
      <td>{u'days': [u'Monday', u'Tuesday', u'Wednesday'...</td>
      <td>http://www.tvmaze.com/shows/19004/the-sharon-o...</td>
      <td>None</td>
      <td>{u'thetvdb': None, u'tvrage': 13173, u'imdb': ...</td>
      <td>2006-08-29</td>
      <td>&lt;p&gt;Daily talk show hosted by Sharon Osbourne.&lt;/p&gt;</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>60</td>
      <td>Talk Show</td>
      <td>19004</td>
      <td>{u'country': {u'timezone': u'Europe/London', u...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Add a column to classify the shows as winners or losers (not winners)
winners['winner'] = 1
new_losers1['winner'] = 0
losers2['winner'] = 0

```

    /Users/erhepp/anaconda/lib/python2.7/site-packages/ipykernel_launcher.py:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      This is separate from the ipykernel package so we can avoid doing imports until



```python

```

### Merge into one dataframe called shows


```python
# now concatenate the loser data to the winner data, the result is the dataframe shows
shows = pd.DataFrame()
shows = winners.copy()
shows = shows.append(new_losers1, ignore_index=True)
shows = shows.append(losers2, ignore_index=True)
shows.shape
```




    (634, 21)




```python
shows.head()
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
      <th>status</th>
      <th>rating</th>
      <th>genres</th>
      <th>weight</th>
      <th>updated</th>
      <th>name</th>
      <th>language</th>
      <th>schedule</th>
      <th>url</th>
      <th>officialSite</th>
      <th>...</th>
      <th>premiered</th>
      <th>summary</th>
      <th>_links</th>
      <th>image</th>
      <th>webChannel</th>
      <th>runtime</th>
      <th>type</th>
      <th>id</th>
      <th>network</th>
      <th>winner</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Ended</td>
      <td>{u'average': 9.4}</td>
      <td>[Nature]</td>
      <td>87</td>
      <td>1490631396</td>
      <td>Planet Earth II</td>
      <td>English</td>
      <td>{u'days': [u'Sunday'], u'time': u'20:00'}</td>
      <td>http://www.tvmaze.com/shows/22036/planet-earth-ii</td>
      <td>http://www.bbc.co.uk/programmes/p02544td</td>
      <td>...</td>
      <td>2016-11-06</td>
      <td>&lt;p&gt;David Attenborough presents a documentary s...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>60</td>
      <td>Documentary</td>
      <td>22036</td>
      <td>{u'country': {u'timezone': u'Europe/London', u...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Ended</td>
      <td>{u'average': 9.4}</td>
      <td>[Drama, Action, War, History]</td>
      <td>86</td>
      <td>1492651730</td>
      <td>Band of Brothers</td>
      <td>English</td>
      <td>{u'days': [u'Sunday'], u'time': u'20:00'}</td>
      <td>http://www.tvmaze.com/shows/465/band-of-brothers</td>
      <td>http://www.hbo.com/band-of-brothers</td>
      <td>...</td>
      <td>2001-09-09</td>
      <td>&lt;p&gt;Drawn from interviews with survivors of Eas...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>60</td>
      <td>Scripted</td>
      <td>465</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ended</td>
      <td>{u'average': 9.2}</td>
      <td>[Nature]</td>
      <td>82</td>
      <td>1502854135</td>
      <td>Planet Earth</td>
      <td>English</td>
      <td>{u'days': [u'Sunday'], u'time': u'21:00'}</td>
      <td>http://www.tvmaze.com/shows/768/planet-earth</td>
      <td>http://www.bbc.co.uk/programmes/b006mywy</td>
      <td>...</td>
      <td>2006-03-05</td>
      <td>&lt;p&gt;David Attenborough celebrates the amazing v...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>60</td>
      <td>Documentary</td>
      <td>768</td>
      <td>{u'country': {u'timezone': u'Europe/London', u...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Running</td>
      <td>{u'average': 9.3}</td>
      <td>[Drama, Adventure, Fantasy]</td>
      <td>100</td>
      <td>1502955537</td>
      <td>Game of Thrones</td>
      <td>English</td>
      <td>{u'days': [u'Sunday'], u'time': u'21:00'}</td>
      <td>http://www.tvmaze.com/shows/82/game-of-thrones</td>
      <td>http://www.hbo.com/game-of-thrones</td>
      <td>...</td>
      <td>2011-04-17</td>
      <td>&lt;p&gt;Based on the bestselling book series &lt;i&gt;A S...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>60</td>
      <td>Scripted</td>
      <td>82</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ended</td>
      <td>{u'average': 9.3}</td>
      <td>[Drama, Crime, Thriller]</td>
      <td>97</td>
      <td>1502331382</td>
      <td>Breaking Bad</td>
      <td>English</td>
      <td>{u'days': [u'Sunday'], u'time': u'22:00'}</td>
      <td>http://www.tvmaze.com/shows/169/breaking-bad</td>
      <td>http://www.amc.com/shows/breaking-bad</td>
      <td>...</td>
      <td>2008-01-20</td>
      <td>&lt;p&gt;&lt;b&gt;Breaking Bad&lt;/b&gt; follows protagonist Wal...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>60</td>
      <td>Scripted</td>
      <td>169</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
# Check id column for any duplicates. There will be some from the losers for two reasons:
#    During first pull, the API limitions were not known, so some were returned with message,
#       "Too Many Requests"  rather tahn data, these need to be removed
#    Some did not contain their own imdb number in the data, so when the list of imdb #s to recheck was generated, 
#        these had to be included in the 2nd attempt as they could not be identified as being in the first pull.  

shows = shows[shows['name'] != 'Too Many Requests']
print shows.shape

print "Duplicate show IDs", shows.duplicated('id').sum()

# Display the duplicates to visually examine before dropping
# shows[shows.isin(shows[shows.duplicated()])].sort("ID")
shows[shows.duplicated('id')]
```

    (498, 21)
    Duplicate show IDs 6





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
      <th>status</th>
      <th>rating</th>
      <th>genres</th>
      <th>weight</th>
      <th>updated</th>
      <th>name</th>
      <th>language</th>
      <th>schedule</th>
      <th>url</th>
      <th>officialSite</th>
      <th>...</th>
      <th>premiered</th>
      <th>summary</th>
      <th>_links</th>
      <th>image</th>
      <th>webChannel</th>
      <th>runtime</th>
      <th>type</th>
      <th>id</th>
      <th>network</th>
      <th>winner</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>601</th>
      <td>Ended</td>
      <td>{u'average': None}</td>
      <td>[]</td>
      <td>0</td>
      <td>1477683583</td>
      <td>Tyler Perry's House of Payne</td>
      <td>English</td>
      <td>{u'days': [u'Friday'], u'time': u'20:00'}</td>
      <td>http://www.tvmaze.com/shows/14013/tyler-perrys...</td>
      <td>None</td>
      <td>...</td>
      <td>2007-06-06</td>
      <td>&lt;p&gt;&lt;b&gt;Tyler Perry's House of Payne&lt;/b&gt; is a co...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>30</td>
      <td>Scripted</td>
      <td>14013</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>602</th>
      <td>Ended</td>
      <td>{u'average': 3.3}</td>
      <td>[Comedy]</td>
      <td>4</td>
      <td>1502774582</td>
      <td>The Inbetweeners</td>
      <td>English</td>
      <td>{u'days': [u'Monday'], u'time': u'22:30'}</td>
      <td>http://www.tvmaze.com/shows/1138/the-inbetweeners</td>
      <td>None</td>
      <td>...</td>
      <td>2012-08-20</td>
      <td>&lt;p&gt;&lt;b&gt;The Inbetweeners&lt;/b&gt; takes a comedic loo...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>30</td>
      <td>Scripted</td>
      <td>1138</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>603</th>
      <td>Ended</td>
      <td>{u'average': 6}</td>
      <td>[Family]</td>
      <td>0</td>
      <td>1497646938</td>
      <td>19 Kids and Counting</td>
      <td>English</td>
      <td>{u'days': [u'Tuesday'], u'time': u'21:00'}</td>
      <td>http://www.tvmaze.com/shows/969/19-kids-and-co...</td>
      <td>http://www.tlc.com/tv-shows/19-kids-and-counting/</td>
      <td>...</td>
      <td>2008-09-29</td>
      <td>&lt;p&gt;&lt;b&gt;19 Kids and Counting&lt;/b&gt; follows Michell...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>30</td>
      <td>Reality</td>
      <td>969</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>604</th>
      <td>Ended</td>
      <td>{u'average': 9}</td>
      <td>[Comedy, Food, Family]</td>
      <td>0</td>
      <td>1463627692</td>
      <td>Talia in the Kitchen</td>
      <td>English</td>
      <td>{u'days': [u'Monday', u'Tuesday', u'Wednesday'...</td>
      <td>http://www.tvmaze.com/shows/2369/talia-in-the-...</td>
      <td>http://www.nick.com/talia-in-the-kitchen/</td>
      <td>...</td>
      <td>2015-07-06</td>
      <td>&lt;p&gt;When 14-year-old Talia visits her grandmoth...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>30</td>
      <td>Scripted</td>
      <td>2369</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>605</th>
      <td>Running</td>
      <td>{u'average': 3.8}</td>
      <td>[]</td>
      <td>48</td>
      <td>1497310190</td>
      <td>The Factor</td>
      <td>English</td>
      <td>{u'days': [u'Monday', u'Tuesday', u'Wednesday'...</td>
      <td>http://www.tvmaze.com/shows/9066/the-factor</td>
      <td>http://www.foxnews.com/shows/the-oreilly-facto...</td>
      <td>...</td>
      <td>1996-10-07</td>
      <td>&lt;p&gt;&lt;b&gt;The Factor&lt;/b&gt;, originally titled &lt;i&gt;The...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>60</td>
      <td>News</td>
      <td>9066</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>606</th>
      <td>Ended</td>
      <td>{u'average': None}</td>
      <td>[Drama, Comedy, Music]</td>
      <td>0</td>
      <td>1462214107</td>
      <td>Viva Laughlin</td>
      <td>English</td>
      <td>{u'days': [u'Sunday'], u'time': u'20:00'}</td>
      <td>http://www.tvmaze.com/shows/6924/viva-laughlin</td>
      <td>None</td>
      <td>...</td>
      <td>2007-10-18</td>
      <td>&lt;p&gt;A remake of the British series &lt;i&gt;Blackpool...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>60</td>
      <td>Scripted</td>
      <td>6924</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>6 rows × 21 columns</p>
</div>




```python
# validate that these are really dups by looking at both rows with the duplicate id
shows[shows['id'] == 6924]
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
      <th>status</th>
      <th>rating</th>
      <th>genres</th>
      <th>weight</th>
      <th>updated</th>
      <th>name</th>
      <th>language</th>
      <th>schedule</th>
      <th>url</th>
      <th>officialSite</th>
      <th>...</th>
      <th>premiered</th>
      <th>summary</th>
      <th>_links</th>
      <th>image</th>
      <th>webChannel</th>
      <th>runtime</th>
      <th>type</th>
      <th>id</th>
      <th>network</th>
      <th>winner</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>462</th>
      <td>Ended</td>
      <td>{u'average': None}</td>
      <td>[Drama, Comedy, Music]</td>
      <td>0</td>
      <td>1462214107</td>
      <td>Viva Laughlin</td>
      <td>English</td>
      <td>{u'days': [u'Sunday'], u'time': u'20:00'}</td>
      <td>http://www.tvmaze.com/shows/6924/viva-laughlin</td>
      <td>None</td>
      <td>...</td>
      <td>2007-10-18</td>
      <td>&lt;p&gt;A remake of the British series &lt;i&gt;Blackpool...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>60</td>
      <td>Scripted</td>
      <td>6924</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>606</th>
      <td>Ended</td>
      <td>{u'average': None}</td>
      <td>[Drama, Comedy, Music]</td>
      <td>0</td>
      <td>1462214107</td>
      <td>Viva Laughlin</td>
      <td>English</td>
      <td>{u'days': [u'Sunday'], u'time': u'20:00'}</td>
      <td>http://www.tvmaze.com/shows/6924/viva-laughlin</td>
      <td>None</td>
      <td>...</td>
      <td>2007-10-18</td>
      <td>&lt;p&gt;A remake of the British series &lt;i&gt;Blackpool...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>60</td>
      <td>Scripted</td>
      <td>6924</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 21 columns</p>
</div>




```python
# All 6 of these check out as true duplicates, so remove the 2nd instance of each
shows = shows.drop_duplicates(subset='id')
```


```python
shows.shape
```




    (492, 21)




```python
# make a copy, so there's a backup without having to re-pull shows info from api or from pickle and recombine
df_shows = shows.copy()
```


```python
# Subdivide the columns so we can fit sections of the dataframe in notebook windows to see what we have
first_cols = df_shows.columns[1:10]
second_cols = df_shows.columns[10:17]
third_cols = df_shows.columns[17:]
```


```python
df_shows[first_cols].head()
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
      <th>rating</th>
      <th>genres</th>
      <th>weight</th>
      <th>updated</th>
      <th>name</th>
      <th>language</th>
      <th>schedule</th>
      <th>url</th>
      <th>officialSite</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>{u'average': 9.4}</td>
      <td>[Nature]</td>
      <td>87</td>
      <td>1490631396</td>
      <td>Planet Earth II</td>
      <td>English</td>
      <td>{u'days': [u'Sunday'], u'time': u'20:00'}</td>
      <td>http://www.tvmaze.com/shows/22036/planet-earth-ii</td>
      <td>http://www.bbc.co.uk/programmes/p02544td</td>
    </tr>
    <tr>
      <th>1</th>
      <td>{u'average': 9.4}</td>
      <td>[Drama, Action, War, History]</td>
      <td>86</td>
      <td>1492651730</td>
      <td>Band of Brothers</td>
      <td>English</td>
      <td>{u'days': [u'Sunday'], u'time': u'20:00'}</td>
      <td>http://www.tvmaze.com/shows/465/band-of-brothers</td>
      <td>http://www.hbo.com/band-of-brothers</td>
    </tr>
    <tr>
      <th>2</th>
      <td>{u'average': 9.2}</td>
      <td>[Nature]</td>
      <td>82</td>
      <td>1502854135</td>
      <td>Planet Earth</td>
      <td>English</td>
      <td>{u'days': [u'Sunday'], u'time': u'21:00'}</td>
      <td>http://www.tvmaze.com/shows/768/planet-earth</td>
      <td>http://www.bbc.co.uk/programmes/b006mywy</td>
    </tr>
    <tr>
      <th>3</th>
      <td>{u'average': 9.3}</td>
      <td>[Drama, Adventure, Fantasy]</td>
      <td>100</td>
      <td>1502955537</td>
      <td>Game of Thrones</td>
      <td>English</td>
      <td>{u'days': [u'Sunday'], u'time': u'21:00'}</td>
      <td>http://www.tvmaze.com/shows/82/game-of-thrones</td>
      <td>http://www.hbo.com/game-of-thrones</td>
    </tr>
    <tr>
      <th>4</th>
      <td>{u'average': 9.3}</td>
      <td>[Drama, Crime, Thriller]</td>
      <td>97</td>
      <td>1502331382</td>
      <td>Breaking Bad</td>
      <td>English</td>
      <td>{u'days': [u'Sunday'], u'time': u'22:00'}</td>
      <td>http://www.tvmaze.com/shows/169/breaking-bad</td>
      <td>http://www.amc.com/shows/breaking-bad</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_shows[second_cols].head()
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
      <th>externals</th>
      <th>premiered</th>
      <th>summary</th>
      <th>_links</th>
      <th>image</th>
      <th>webChannel</th>
      <th>runtime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>{u'thetvdb': 318408, u'tvrage': None, u'imdb':...</td>
      <td>2016-11-06</td>
      <td>&lt;p&gt;David Attenborough presents a documentary s...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>60</td>
    </tr>
    <tr>
      <th>1</th>
      <td>{u'thetvdb': 74205, u'tvrage': 2708, u'imdb': ...</td>
      <td>2001-09-09</td>
      <td>&lt;p&gt;Drawn from interviews with survivors of Eas...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>60</td>
    </tr>
    <tr>
      <th>2</th>
      <td>{u'thetvdb': 79257, u'tvrage': 8077, u'imdb': ...</td>
      <td>2006-03-05</td>
      <td>&lt;p&gt;David Attenborough celebrates the amazing v...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>60</td>
    </tr>
    <tr>
      <th>3</th>
      <td>{u'thetvdb': 121361, u'tvrage': 24493, u'imdb'...</td>
      <td>2011-04-17</td>
      <td>&lt;p&gt;Based on the bestselling book series &lt;i&gt;A S...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>60</td>
    </tr>
    <tr>
      <th>4</th>
      <td>{u'thetvdb': 81189, u'tvrage': 18164, u'imdb':...</td>
      <td>2008-01-20</td>
      <td>&lt;p&gt;&lt;b&gt;Breaking Bad&lt;/b&gt; follows protagonist Wal...</td>
      <td>{u'previousepisode': {u'href': u'http://api.tv...</td>
      <td>{u'medium': u'http://static.tvmaze.com/uploads...</td>
      <td>None</td>
      <td>60</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_shows[third_cols].head()
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
      <th>type</th>
      <th>id</th>
      <th>network</th>
      <th>winner</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Documentary</td>
      <td>22036</td>
      <td>{u'country': {u'timezone': u'Europe/London', u...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Scripted</td>
      <td>465</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Documentary</td>
      <td>768</td>
      <td>{u'country': {u'timezone': u'Europe/London', u...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Scripted</td>
      <td>82</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Scripted</td>
      <td>169</td>
      <td>{u'country': {u'timezone': u'America/New_York'...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## Cleanup and Organization of the DataFrame


```python
# Cleanup and Organization

# The genres column is generally a list of strings, but is missing some values, and has empty lists for others.
#   !. Change all NaN to []
#   2. Convert all to strings
#   3. Use Count Vectorizer to make new columns for each genre
#   4. Remove existing genres column

df_shows['genres'] = df_shows['genres'].fillna(0).map(lambda x: [] if x == 0 else x)
df_shows['genres'] = df_shows['genres'].map(lambda x: ','.join(x))
```


```python
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(
    binary=True,
    tokenizer=(lambda x: x.split(','))
    )
cvfit = cv.fit_transform(df_shows['genres']).todense()
genre_cols = pd.DataFrame(cvfit, columns=cv.get_feature_names())
genre_cols.rename(columns={'' : 'unknown'}, inplace=True)
genre_cols.columns
```




    Index([        u'unknown',          u'action',           u'adult',
                 u'adventure',           u'anime',        u'children',
                    u'comedy',           u'crime',           u'drama',
                 u'espionage',          u'family',         u'fantasy',
                      u'food',         u'history',          u'horror',
                     u'legal',         u'medical',           u'music',
                   u'mystery',          u'nature',         u'romance',
           u'science-fiction',          u'sports',    u'supernatural',
                  u'thriller',          u'travel',             u'war',
                   u'western'],
          dtype='object')




```python
new_genre_columns = []
for item in genre_cols:
    new_genre_columns.append('gn_' + item)
genre_cols.columns = new_genre_columns
genre_cols.head()
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
      <th>gn_unknown</th>
      <th>gn_action</th>
      <th>gn_adult</th>
      <th>gn_adventure</th>
      <th>gn_anime</th>
      <th>gn_children</th>
      <th>gn_comedy</th>
      <th>gn_crime</th>
      <th>gn_drama</th>
      <th>gn_espionage</th>
      <th>...</th>
      <th>gn_mystery</th>
      <th>gn_nature</th>
      <th>gn_romance</th>
      <th>gn_science-fiction</th>
      <th>gn_sports</th>
      <th>gn_supernatural</th>
      <th>gn_thriller</th>
      <th>gn_travel</th>
      <th>gn_war</th>
      <th>gn_western</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
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
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>2</th>
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
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
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
      <td>...</td>
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
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
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
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>




```python
# Add the new genre columns to the df_shows dataframe
df_shows = pd.concat([df_shows, genre_cols], axis=1, join_axes=[df_shows.index])
df_shows = df_shows.drop('genres', 1)
```


```python
# Genre information is missing for 69 loser shows and 13 winner shows

df_shows[df_shows['gn_unknown'] ==1][['gn_unknown', 'winner']].groupby(['winner']).count()
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
      <th>gn_unknown</th>
    </tr>
    <tr>
      <th>winner</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>69</td>
    </tr>
    <tr>
      <th>1</th>
      <td>13</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_shows.columns
```




    Index([            u'status',             u'rating',             u'weight',
                      u'updated',               u'name',           u'language',
                     u'schedule',                u'url',       u'officialSite',
                    u'externals',          u'premiered',            u'summary',
                       u'_links',              u'image',         u'webChannel',
                      u'runtime',               u'type',                 u'id',
                      u'network',             u'winner',         u'gn_unknown',
                    u'gn_action',           u'gn_adult',       u'gn_adventure',
                     u'gn_anime',        u'gn_children',          u'gn_comedy',
                     u'gn_crime',           u'gn_drama',       u'gn_espionage',
                    u'gn_family',         u'gn_fantasy',            u'gn_food',
                   u'gn_history',          u'gn_horror',           u'gn_legal',
                   u'gn_medical',           u'gn_music',         u'gn_mystery',
                    u'gn_nature',         u'gn_romance', u'gn_science-fiction',
                    u'gn_sports',    u'gn_supernatural',        u'gn_thriller',
                    u'gn_travel',             u'gn_war',         u'gn_western'],
          dtype='object')




```python
# Convert the rating to a number
# sometimes the rating column is NaN, and sometimes the value for 'average' in the dictionary is Nan
# so the NaNs must be handled twice, once for each case
# This code first fills the missing dictionarys with -1 (value chosen to signify no rating)
# It then sets the column to the average value in the rating dictionary, and if that is NaN converts to -1

df_shows['rating'] = df_shows['rating'].fillna(-1).map(lambda x: -1 if x == -1 else x['average']).fillna(-1)
```


```python
# Rating information is missing for 192 loser shows and 6 winner shows
df_shows[df_shows['rating'] == -1][['rating', 'winner']].groupby(['winner']).count()
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
      <th>rating</th>
    </tr>
    <tr>
      <th>winner</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>192</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Unpack 'schedule' into days treating NaN in a similar way, 
df_shows['sched_day'] = df_shows['schedule'].fillna(0).map(lambda x: [] if x == 0 else x)
df_shows['sched_day'] = df_shows['sched_day'].map(lambda x: x if x == [] else x['days'])
df_shows['sched_day'] = df_shows['sched_day'].map(lambda x: ','.join(x))
```


```python
cv = CountVectorizer(
    binary=True,
    tokenizer=(lambda x: x.split(','))
    )
cvfit = cv.fit_transform(df_shows['sched_day']).todense()
day_cols = pd.DataFrame(cvfit, columns=cv.get_feature_names())
day_cols.rename(columns={'' : 'unknown'}, inplace=True)
day_cols.columns
```




    Index([  u'unknown',    u'friday',    u'monday',  u'saturday',    u'sunday',
            u'thursday',   u'tuesday', u'wednesday'],
          dtype='object')




```python
new_day_columns = []
for item in day_cols:
    new_day_columns.append('sched_' + item)
day_cols.columns = new_day_columns
day_cols.head()
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
      <th>sched_unknown</th>
      <th>sched_friday</th>
      <th>sched_monday</th>
      <th>sched_saturday</th>
      <th>sched_sunday</th>
      <th>sched_thursday</th>
      <th>sched_tuesday</th>
      <th>sched_wednesday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Add the new genre columns to the df_shows dataframe
df_shows = pd.concat([df_shows, day_cols], axis=1, join_axes=[df_shows.index])

```


```python
df_shows = df_shows.drop('sched_day', 1)
```


```python
# Scheduled Day information is missing for 15 loser shows and 45 winner shows

df_shows[df_shows['sched_unknown'] ==1][['sched_unknown', 'winner']].groupby(['winner']).count()
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
      <th>sched_unknown</th>
    </tr>
    <tr>
      <th>winner</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>45</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Unpack 'schedule' into times treating NaN in a similar way.
# Samples with a valid show time will be HH:MM and missing values will be :
df_shows['sched_time'] = df_shows['schedule'].fillna(':').map(lambda x: x if x == ':' else x['time'])
df_shows['sched_time'] = df_shows['sched_time'].map(lambda x: ':' if x == '' else x)
```


```python
# Scheduled Time information is missing for 35 loser shows and 61 winner shows

print len(df_shows[df_shows['sched_time'] == ':'])
df_shows[df_shows['sched_time'] == ':'][['sched_time', 'winner']].groupby(['winner']).count()
```

    96





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
      <th>sched_time</th>
    </tr>
    <tr>
      <th>winner</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>35</td>
    </tr>
    <tr>
      <th>1</th>
      <td>61</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Sched time is in HH:MM format as a string. I will leave this as string, and count vectorize it
print type(df_shows.loc[0,'sched_time'])

cv = CountVectorizer(
    binary=True,
    tokenizer=(lambda x: x.split(','))
    )
cvfit = cv.fit_transform(df_shows['sched_time']).todense()
time_cols = pd.DataFrame(cvfit, columns=cv.get_feature_names())
time_cols.rename(columns={':' : 'unknown'}, inplace=True)
time_cols.columns
```

    <type 'unicode'>





    Index([  u'00:00',   u'00:30',   u'00:50',   u'00:55',   u'01:00',   u'01:05',
             u'01:30',   u'01:35',   u'02:00',   u'02:05',   u'08:00',   u'10:00',
             u'11:00',   u'12:00',   u'13:00',   u'13:30',   u'14:00',   u'14:30',
             u'15:00',   u'15:15',   u'16:00',   u'16:30',   u'17:00',   u'17:15',
             u'17:30',   u'18:00',   u'18:30',   u'19:00',   u'19:30',   u'19:45',
             u'20:00',   u'20:15',   u'20:30',   u'20:40',   u'20:45',   u'20:55',
             u'21:00',   u'21:10',   u'21:15',   u'21:30',   u'21:45',   u'22:00',
             u'22:10',   u'22:30',   u'22:35',   u'23:00',   u'23:02',   u'23:15',
             u'23:30', u'unknown'],
          dtype='object')




```python
new_time_columns = []
for item in time_cols:
    new_time_columns.append('sched_time_' + item)
time_cols.columns = new_time_columns
time_cols.head()
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
      <th>sched_time_00:00</th>
      <th>sched_time_00:30</th>
      <th>sched_time_00:50</th>
      <th>sched_time_00:55</th>
      <th>sched_time_01:00</th>
      <th>sched_time_01:05</th>
      <th>sched_time_01:30</th>
      <th>sched_time_01:35</th>
      <th>sched_time_02:00</th>
      <th>sched_time_02:05</th>
      <th>...</th>
      <th>sched_time_21:45</th>
      <th>sched_time_22:00</th>
      <th>sched_time_22:10</th>
      <th>sched_time_22:30</th>
      <th>sched_time_22:35</th>
      <th>sched_time_23:00</th>
      <th>sched_time_23:02</th>
      <th>sched_time_23:15</th>
      <th>sched_time_23:30</th>
      <th>sched_time_unknown</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
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
      <td>...</td>
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
    </tr>
    <tr>
      <th>1</th>
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
      <td>...</td>
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
    </tr>
    <tr>
      <th>2</th>
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
      <td>...</td>
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
    </tr>
    <tr>
      <th>3</th>
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
      <td>...</td>
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
    </tr>
    <tr>
      <th>4</th>
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
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 50 columns</p>
</div>




```python
# Add the new genre columns to the df_shows dataframe
df_shows = pd.concat([df_shows, time_cols], axis=1, join_axes=[df_shows.index])

```


```python
df_shows = df_shows.drop('schedule', 1)
```


```python
df_shows = df_shows.drop('sched_time', 1)
```


```python
print df_shows.columns
```

    Index([            u'status',             u'rating',             u'weight',
                      u'updated',               u'name',           u'language',
                          u'url',       u'officialSite',          u'externals',
                    u'premiered',
           ...
             u'sched_time_21:45',   u'sched_time_22:00',   u'sched_time_22:10',
             u'sched_time_22:30',   u'sched_time_22:35',   u'sched_time_23:00',
             u'sched_time_23:02',   u'sched_time_23:15',   u'sched_time_23:30',
           u'sched_time_unknown'],
          dtype='object', length=105)



```python

```


```python
# Print out a network dictionary to learn how to unpack the structure
df_shows.loc[0,'network']
```




    {u'country': {u'code': u'GB',
      u'name': u'United Kingdom',
      u'timezone': u'Europe/London'},
     u'id': 12,
     u'name': u'BBC One'}




```python
# 25 shows have no network info,  might need to drop these, but dummied for now
df_shows['network'].isnull().sum()
```




    25




```python
# Unpack 'network' into country code, country name, timezone,  treating NaN in a similar way, 
df_shows['country_code'] = df_shows['network'].fillna('').map(lambda x: x if x == '' else x['country'])
df_shows['country_code'] = df_shows['country_code'].map(lambda x: x if x == '' else x['code'])

df_shows['country_name'] = df_shows['network'].fillna('').map(lambda x: x if x == '' else x['country'])
df_shows['country_name'] = df_shows['country_name'].map(lambda x: x if x == '' else x['name'])

df_shows['country_tz'] = df_shows['network'].fillna('').map(lambda x: x if x == '' else x['country'])
df_shows['country_tz'] = df_shows['country_tz'].map(lambda x: x if x == '' else x['timezone'])

df_shows['network_id'] = df_shows['network'].fillna('').map(lambda x: x if x == '' else x['id'])
df_shows['network_name'] = df_shows['network'].fillna('').map(lambda x: x if x == '' else x['name'])

```


```python
df_shows = df_shows.drop(['network'], 1)
```


```python
# Country and network information is missing for 4 loser shows and 21 winner shows

df_shows[df_shows['country_code'] == ''] [['country_code', 'winner']].groupby(['winner']).count()
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
      <th>country_code</th>
    </tr>
    <tr>
      <th>winner</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_shows[['country_code', 'country_name', 'country_tz', 'network_id', 'network_name']].head()
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
      <th>country_code</th>
      <th>country_name</th>
      <th>country_tz</th>
      <th>network_id</th>
      <th>network_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>GB</td>
      <td>United Kingdom</td>
      <td>Europe/London</td>
      <td>12</td>
      <td>BBC One</td>
    </tr>
    <tr>
      <th>1</th>
      <td>US</td>
      <td>United States</td>
      <td>America/New_York</td>
      <td>8</td>
      <td>HBO</td>
    </tr>
    <tr>
      <th>2</th>
      <td>GB</td>
      <td>United Kingdom</td>
      <td>Europe/London</td>
      <td>12</td>
      <td>BBC One</td>
    </tr>
    <tr>
      <th>3</th>
      <td>US</td>
      <td>United States</td>
      <td>America/New_York</td>
      <td>8</td>
      <td>HBO</td>
    </tr>
    <tr>
      <th>4</th>
      <td>US</td>
      <td>United States</td>
      <td>America/New_York</td>
      <td>20</td>
      <td>AMC</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_shows[['updated', 'premiered']].head()
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
      <th>updated</th>
      <th>premiered</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1490631396</td>
      <td>2016-11-06</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1492651730</td>
      <td>2001-09-09</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1502854135</td>
      <td>2006-03-05</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1502955537</td>
      <td>2011-04-17</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1502331382</td>
      <td>2008-01-20</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Updated date is complete, premiered date is missing 6 values

print df_shows['updated'].isnull().sum()
print df_shows['premiered'].isnull().sum()

```

    0
    6



```python
# Must represent updated as a real date time object, currently is seconds from epoch (1970)
# Convert string to int, then int to datetime
import datetime
print type(df_shows.loc[0,'updated'])

df_shows['updated'] = df_shows['updated'].fillna(0).apply(lambda x: x if x == 0 else datetime.datetime.fromtimestamp(x))

```

    <type 'int'>



```python
# Turn premiered into real date time object, currently this is a string, need to convert to date
print type(df_shows.loc[0,'premiered'])
df_shows['premiered'] = df_shows['premiered'].fillna(0).apply(lambda x: x if x == 0 else datetime.datetime.strptime(x, '%Y-%m-%d'))
```

    <type 'unicode'>



```python
df_shows[['updated', 'premiered']].head()
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
      <th>updated</th>
      <th>premiered</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017-03-27 12:16:36</td>
      <td>2016-11-06 00:00:00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017-04-19 21:28:50</td>
      <td>2001-09-09 00:00:00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017-08-15 23:28:55</td>
      <td>2006-03-05 00:00:00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017-08-17 03:38:57</td>
      <td>2011-04-17 00:00:00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017-08-09 22:16:22</td>
      <td>2008-01-20 00:00:00</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Updated date is complete, premiered date is missing 6 values, all from loser shows

df_shows[df_shows['premiered'] == 0] [['premiered', 'winner']].groupby(['winner']).count()
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
      <th>premiered</th>
    </tr>
    <tr>
      <th>winner</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Drop columns not useful for analysis

# webChannel has no or insufficient useful information, can drop
print "webChannel null count:", df_shows['webChannel'].isnull().sum()

# url, officialSite, externals, _links, image, webChannel
df_shows = df_shows.drop(['url', 'officialSite', 'externals', '_links', 'image', 'webChannel', ], 1)
```

    webChannel null count: 464



```python
# Looks like runtime is already an integer number of minutes
# runtime is missing 9 values, 5 winners and 4 losers
print type(df_shows.loc[0,'runtime'])
print df_shows['runtime'].isnull().sum(), " null values"
# df_shows['runtime'].value_counts()


```

    <type 'int'>
    9  null values



```python
df_shows[df_shows['runtime'].isnull()][['runtime', 'winner']]
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
      <th>runtime</th>
      <th>winner</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17</th>
      <td>None</td>
      <td>1</td>
    </tr>
    <tr>
      <th>65</th>
      <td>None</td>
      <td>1</td>
    </tr>
    <tr>
      <th>137</th>
      <td>None</td>
      <td>1</td>
    </tr>
    <tr>
      <th>144</th>
      <td>None</td>
      <td>1</td>
    </tr>
    <tr>
      <th>198</th>
      <td>None</td>
      <td>1</td>
    </tr>
    <tr>
      <th>544</th>
      <td>None</td>
      <td>0</td>
    </tr>
    <tr>
      <th>556</th>
      <td>None</td>
      <td>0</td>
    </tr>
    <tr>
      <th>577</th>
      <td>None</td>
      <td>0</td>
    </tr>
    <tr>
      <th>609</th>
      <td>None</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Contains html tags, otherwise a string, html tags will be removed in text processing steps during analysis
print df_shows.loc[0,'summary']
print df_shows['summary'].isnull().sum(), " null values"
```

    <p>David Attenborough presents a documentary series exploring how animals meet the challenges of surviving in the most iconic habitats on earth.</p>
    1  null values



```python
df_shows[df_shows['summary'].isnull()]
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
      <th>status</th>
      <th>rating</th>
      <th>weight</th>
      <th>updated</th>
      <th>name</th>
      <th>language</th>
      <th>premiered</th>
      <th>summary</th>
      <th>runtime</th>
      <th>type</th>
      <th>...</th>
      <th>sched_time_23:00</th>
      <th>sched_time_23:02</th>
      <th>sched_time_23:15</th>
      <th>sched_time_23:30</th>
      <th>sched_time_unknown</th>
      <th>country_code</th>
      <th>country_name</th>
      <th>country_tz</th>
      <th>network_id</th>
      <th>network_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>570</th>
      <td>Ended</td>
      <td>-1.0</td>
      <td>75</td>
      <td>2017-04-18 17:55:28</td>
      <td>Chop Socky Chooks</td>
      <td>English</td>
      <td>2008-03-07 00:00:00</td>
      <td>None</td>
      <td>11</td>
      <td>Animation</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>US</td>
      <td>United States</td>
      <td>America/New_York</td>
      <td>11</td>
      <td>Cartoon Network</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 104 columns</p>
</div>




```python
# This one with the missing summary, Chop Socky Chooks, is missing other information also, and will be dropped.
# Too bad,  looks like a truly dreadful one that would be good for the very bottom of the losers list.
df_shows = df_shows[df_shows['summary'].notnull()]
df_shows.shape
```




    (491, 104)




```python
# Use textacy to clean the html tags, punctuation, etc. from the summary text
from textacy.preprocess import preprocess_text

df_shows['summary'] = df_shows['summary'].map(lambda x: preprocess_text(x, fix_unicode=True, lowercase=True, \
                              transliterate=False, no_contractions = True,
                              no_urls=True, no_emails=True, no_phone_numbers=True, no_currency_symbols=True,
                              no_punct=True, no_accents=True))
```


```python
print df_shows.loc[1,'summary']
print
print df_shows.loc[2,'summary']
```

    <p>drawn from interviews with survivors of easy company as well as their journals and letters <b>band of brothers<b> chronicles the experiences of these men from paratrooper training in georgia through the end of the war as an elite rifle company parachuting into normandy early on dday morning participants in the battle of the bulge and witness to the horrors of war the men of easy knew extraordinary bravery and extraordinary fear and became the stuff of legend based on stephen e ambroses acclaimed book of the same name<p>
    
    <p>david attenborough celebrates the amazing variety of the natural world in this epic documentary series filmed over four years across 64 different countries<p>



```python
# Looks like all the summaries have html paragraph <p> and break <b> tags, and textacy hasn't removed them. 
# These lambda function knock them out

import string
df_shows['summary'] = df_shows['summary'].map(lambda x: x.replace('<p>',''))
df_shows['summary'] = df_shows['summary'].map(lambda x: x.replace('<b>',''))
```


```python
# This looks better for analysis
print df_shows.loc[1,'summary']
```

    drawn from interviews with survivors of easy company as well as their journals and letters band of brothers chronicles the experiences of these men from paratrooper training in georgia through the end of the war as an elite rifle company parachuting into normandy early on dday morning participants in the battle of the bulge and witness to the horrors of war the men of easy knew extraordinary bravery and extraordinary fear and became the stuff of legend based on stephen e ambroses acclaimed book of the same name



```python
df_shows[df_shows.isnull().any(axis=1)]
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
      <th>status</th>
      <th>rating</th>
      <th>weight</th>
      <th>updated</th>
      <th>name</th>
      <th>language</th>
      <th>premiered</th>
      <th>summary</th>
      <th>runtime</th>
      <th>type</th>
      <th>...</th>
      <th>sched_time_23:00</th>
      <th>sched_time_23:02</th>
      <th>sched_time_23:15</th>
      <th>sched_time_23:30</th>
      <th>sched_time_unknown</th>
      <th>country_code</th>
      <th>country_name</th>
      <th>country_tz</th>
      <th>network_id</th>
      <th>network_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17</th>
      <td>Ended</td>
      <td>9.0</td>
      <td>0</td>
      <td>1455913373</td>
      <td>The Decalogue</td>
      <td>Polish</td>
      <td>1989-12-10 00:00:00</td>
      <td>&lt;p&gt;Ten television drama films, each one based ...</td>
      <td>None</td>
      <td>Variety</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>PL</td>
      <td>Poland</td>
      <td>Europe/Warsaw</td>
      <td>336</td>
      <td>TVP1</td>
    </tr>
    <tr>
      <th>65</th>
      <td>Ended</td>
      <td>9.0</td>
      <td>85</td>
      <td>1501781828</td>
      <td>Sherlock Holmes</td>
      <td>English</td>
      <td>1984-04-24 00:00:00</td>
      <td>&lt;p&gt;&lt;b&gt;Sherlock Holmes&lt;/b&gt; is one of the world'...</td>
      <td>None</td>
      <td>Scripted</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>GB</td>
      <td>United Kingdom</td>
      <td>Europe/London</td>
      <td>35</td>
      <td>ITV</td>
    </tr>
    <tr>
      <th>137</th>
      <td>Running</td>
      <td>8.7</td>
      <td>98</td>
      <td>1489944935</td>
      <td>Taboo</td>
      <td>English</td>
      <td>2017-01-07 00:00:00</td>
      <td>&lt;p&gt;1814: James Keziah Delaney returns to Londo...</td>
      <td>None</td>
      <td>Scripted</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>GB</td>
      <td>United Kingdom</td>
      <td>Europe/London</td>
      <td>12</td>
      <td>BBC One</td>
    </tr>
    <tr>
      <th>144</th>
      <td>Ended</td>
      <td>8.6</td>
      <td>42</td>
      <td>1494693177</td>
      <td>The New Batman Adventures</td>
      <td>English</td>
      <td>1997-09-13 00:00:00</td>
      <td>&lt;p&gt;The New Batman Adventures comes from the cr...</td>
      <td>None</td>
      <td>Animation</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>US</td>
      <td>United States</td>
      <td>America/New_York</td>
      <td>71</td>
      <td>The WB</td>
    </tr>
    <tr>
      <th>198</th>
      <td>Ended</td>
      <td>9.0</td>
      <td>0</td>
      <td>1491564027</td>
      <td>The Larry Sanders Show</td>
      <td>English</td>
      <td>1992-08-15 00:00:00</td>
      <td>&lt;p&gt;Comic Garry Shandling draws upon his own ta...</td>
      <td>None</td>
      <td>Scripted</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>US</td>
      <td>United States</td>
      <td>America/New_York</td>
      <td>8</td>
      <td>HBO</td>
    </tr>
    <tr>
      <th>492</th>
      <td>Running</td>
      <td>-1.0</td>
      <td>76</td>
      <td>1502312151</td>
      <td>Big Brother After Dark</td>
      <td>English</td>
      <td>2007-07-05 00:00:00</td>
      <td>&lt;p&gt;&lt;b&gt;Big Brother After Dark&lt;/b&gt; is the live, ...</td>
      <td>180</td>
      <td>Reality</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>US</td>
      <td>United States</td>
      <td>America/New_York</td>
      <td>88</td>
      <td>Pop</td>
    </tr>
    <tr>
      <th>493</th>
      <td>Ended</td>
      <td>1.0</td>
      <td>0</td>
      <td>1474827145</td>
      <td>American Paranormal</td>
      <td>English</td>
      <td>2010-01-24 00:00:00</td>
      <td>&lt;p&gt;Whether it is the existence of aliens, the ...</td>
      <td>60</td>
      <td>Reality</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>US</td>
      <td>United States</td>
      <td>America/New_York</td>
      <td>42</td>
      <td>National Geographic Channel</td>
    </tr>
    <tr>
      <th>494</th>
      <td>Ended</td>
      <td>-1.0</td>
      <td>11</td>
      <td>1469108505</td>
      <td>Homeboys in Outer Space</td>
      <td>English</td>
      <td>1996-08-27 00:00:00</td>
      <td>&lt;p&gt;The plot centers around two astronauts, Tyb...</td>
      <td>30</td>
      <td>Scripted</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>US</td>
      <td>United States</td>
      <td>America/New_York</td>
      <td>70</td>
      <td>UPN</td>
    </tr>
    <tr>
      <th>495</th>
      <td>Ended</td>
      <td>-1.0</td>
      <td>0</td>
      <td>1485097253</td>
      <td>Gainesville: Friends Are Family</td>
      <td>English</td>
      <td>2015-08-20 00:00:00</td>
      <td>&lt;p&gt;&lt;i&gt;&lt;b&gt;"Gainesville: Friends Are Family"&lt;/b&gt;...</td>
      <td>30</td>
      <td>Documentary</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>US</td>
      <td>United States</td>
      <td>America/New_York</td>
      <td>173</td>
      <td>CMT</td>
    </tr>
    <tr>
      <th>496</th>
      <td>Ended</td>
      <td>-1.0</td>
      <td>0</td>
      <td>1449234102</td>
      <td>The Show with Vinny</td>
      <td>English</td>
      <td>2013-05-01 00:00:00</td>
      <td>&lt;p&gt;Vinny Guadagnino invites musicians, TV star...</td>
      <td>30</td>
      <td>Talk Show</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>US</td>
      <td>United States</td>
      <td>America/New_York</td>
      <td>22</td>
      <td>MTV</td>
    </tr>
    <tr>
      <th>497</th>
      <td>Ended</td>
      <td>-1.0</td>
      <td>0</td>
      <td>1457985576</td>
      <td>Gormiti Nature Unleashed</td>
      <td>French</td>
      <td>2013-04-01 00:00:00</td>
      <td>&lt;p&gt;Gormiti Nature Unleashed is an Italian CGI ...</td>
      <td>25</td>
      <td>Animation</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>FR</td>
      <td>France</td>
      <td>Europe/Paris</td>
      <td>1050</td>
      <td>Canal J</td>
    </tr>
    <tr>
      <th>498</th>
      <td>Ended</td>
      <td>-1.0</td>
      <td>23</td>
      <td>1483294279</td>
      <td>Denise Richards: It's Complicated</td>
      <td>English</td>
      <td>2008-05-26 00:00:00</td>
      <td>&lt;p&gt;&lt;b&gt;Denise Richards: It's Complicated&lt;/b&gt; is...</td>
      <td>30</td>
      <td>Reality</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>US</td>
      <td>United States</td>
      <td>America/New_York</td>
      <td>43</td>
      <td>E!</td>
    </tr>
    <tr>
      <th>499</th>
      <td>Ended</td>
      <td>-1.0</td>
      <td>0</td>
      <td>1482875019</td>
      <td>Stanley</td>
      <td>English</td>
      <td>1956-09-24 00:00:00</td>
      <td>&lt;p&gt;&lt;b&gt;Stanley&lt;/b&gt; revolved around the adventur...</td>
      <td>30</td>
      <td>Scripted</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>US</td>
      <td>United States</td>
      <td>America/New_York</td>
      <td>1</td>
      <td>NBC</td>
    </tr>
    <tr>
      <th>500</th>
      <td>Ended</td>
      <td>1.0</td>
      <td>0</td>
      <td>1468782928</td>
      <td>Uncovering Aliens</td>
      <td>English</td>
      <td>2013-12-15 00:00:00</td>
      <td>&lt;p&gt;Across America, there are more UFO sighting...</td>
      <td>60</td>
      <td>Documentary</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>US</td>
      <td>United States</td>
      <td>America/New_York</td>
      <td>92</td>
      <td>Animal Planet</td>
    </tr>
    <tr>
      <th>501</th>
      <td>Ended</td>
      <td>-1.0</td>
      <td>0</td>
      <td>1477142177</td>
      <td>Bulging Brides</td>
      <td>English</td>
      <td>2008-01-31 00:00:00</td>
      <td>&lt;p&gt;&lt;b&gt;Bulging Brides&lt;/b&gt; is a television serie...</td>
      <td>30</td>
      <td>Reality</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>CA</td>
      <td>Canada</td>
      <td>Canada/Atlantic</td>
      <td>472</td>
      <td>Slice</td>
    </tr>
    <tr>
      <th>502</th>
      <td>Running</td>
      <td>6.7</td>
      <td>0</td>
      <td>1502923678</td>
      <td>Never Ever Do This at Home</td>
      <td>English</td>
      <td>2013-05-06 00:00:00</td>
      <td>&lt;p&gt;&lt;b&gt;Never Ever Do This at Home&lt;/b&gt; is a come...</td>
      <td>30</td>
      <td>Reality</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>CA</td>
      <td>Canada</td>
      <td>Canada/Atlantic</td>
      <td>298</td>
      <td>Discovery Channel</td>
    </tr>
    <tr>
      <th>503</th>
      <td>Ended</td>
      <td>-1.0</td>
      <td>0</td>
      <td>1465987779</td>
      <td>Hello Ross</td>
      <td>English</td>
      <td>2013-09-06 00:00:00</td>
      <td>&lt;p&gt;&lt;i&gt;&lt;b&gt;"Hello Ross"&lt;/b&gt;&lt;/i&gt; is the new weekl...</td>
      <td>30</td>
      <td>Talk Show</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>US</td>
      <td>United States</td>
      <td>America/New_York</td>
      <td>43</td>
      <td>E!</td>
    </tr>
    <tr>
      <th>504</th>
      <td>Ended</td>
      <td>-1.0</td>
      <td>0</td>
      <td>1499803314</td>
      <td>3</td>
      <td>English</td>
      <td>2012-07-26 00:00:00</td>
      <td>&lt;p&gt;&lt;b&gt;3&lt;/b&gt; is a new relationship series in wh...</td>
      <td>60</td>
      <td>Reality</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>US</td>
      <td>United States</td>
      <td>America/New_York</td>
      <td>2</td>
      <td>CBS</td>
    </tr>
    <tr>
      <th>505</th>
      <td>Ended</td>
      <td>-1.0</td>
      <td>0</td>
      <td>1495568447</td>
      <td>Trexx and Flipside</td>
      <td>English</td>
      <td>0</td>
      <td>&lt;p&gt;Wannabe hip hop stars but their music label...</td>
      <td>30</td>
      <td>Scripted</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>GB</td>
      <td>United Kingdom</td>
      <td>Europe/London</td>
      <td>49</td>
      <td>BBC Three</td>
    </tr>
    <tr>
      <th>506</th>
      <td>Running</td>
      <td>8.5</td>
      <td>96</td>
      <td>1503483430</td>
      <td>The Real Housewives of Orange County</td>
      <td>English</td>
      <td>2006-03-21 00:00:00</td>
      <td>&lt;p&gt;These ladies show no signs of slowing down ...</td>
      <td>60</td>
      <td>Reality</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>US</td>
      <td>United States</td>
      <td>America/New_York</td>
      <td>52</td>
      <td>Bravo</td>
    </tr>
    <tr>
      <th>507</th>
      <td>Ended</td>
      <td>5.3</td>
      <td>16</td>
      <td>1479782037</td>
      <td>Skins</td>
      <td>English</td>
      <td>2011-01-17 00:00:00</td>
      <td>&lt;p&gt;&lt;b&gt;Skins&lt;/b&gt; is about the lives and loves o...</td>
      <td>60</td>
      <td>Scripted</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>US</td>
      <td>United States</td>
      <td>America/New_York</td>
      <td>22</td>
      <td>MTV</td>
    </tr>
    <tr>
      <th>508</th>
      <td>Running</td>
      <td>-1.0</td>
      <td>73</td>
      <td>1503490679</td>
      <td>Dr. Phil</td>
      <td>English</td>
      <td>2002-09-16 00:00:00</td>
      <td>&lt;p&gt;The &lt;b&gt;Dr. Phil&lt;/b&gt; show provides the most ...</td>
      <td>60</td>
      <td>Talk Show</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>US</td>
      <td>United States</td>
      <td>America/New_York</td>
      <td>72</td>
      <td>Syndication</td>
    </tr>
    <tr>
      <th>509</th>
      <td>Running</td>
      <td>7.5</td>
      <td>50</td>
      <td>1497449904</td>
      <td>My Big Fat American Gypsy Wedding</td>
      <td>English</td>
      <td>2012-04-29 00:00:00</td>
      <td>&lt;p&gt;Going inside the hidden world of American G...</td>
      <td>60</td>
      <td>Reality</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>US</td>
      <td>United States</td>
      <td>America/New_York</td>
      <td>80</td>
      <td>TLC</td>
    </tr>
    <tr>
      <th>510</th>
      <td>Running</td>
      <td>1.0</td>
      <td>0</td>
      <td>1479731918</td>
      <td>Mystery Diners</td>
      <td>English</td>
      <td>2012-05-20 00:00:00</td>
      <td>&lt;p&gt;When a restaurant owner suspects employees ...</td>
      <td>30</td>
      <td>Reality</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>US</td>
      <td>United States</td>
      <td>America/New_York</td>
      <td>81</td>
      <td>Food Network</td>
    </tr>
    <tr>
      <th>511</th>
      <td>Running</td>
      <td>-1.0</td>
      <td>0</td>
      <td>1498393231</td>
      <td>Pig Goat Banana Cricket</td>
      <td>English</td>
      <td>2015-07-18 00:00:00</td>
      <td>&lt;p&gt;&lt;b&gt;Pig Goat Banana Cricket&lt;/b&gt; features a s...</td>
      <td>30</td>
      <td>Animation</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>US</td>
      <td>United States</td>
      <td>America/New_York</td>
      <td>73</td>
      <td>nicktoons</td>
    </tr>
    <tr>
      <th>512</th>
      <td>Ended</td>
      <td>-1.0</td>
      <td>9</td>
      <td>1460230772</td>
      <td>Jerseylicious</td>
      <td>English</td>
      <td>2010-03-21 00:00:00</td>
      <td>&lt;p&gt;Jerseylicious is a reality show which takes...</td>
      <td>60</td>
      <td>Reality</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>US</td>
      <td>United States</td>
      <td>America/New_York</td>
      <td>184</td>
      <td>Esquire Network</td>
    </tr>
    <tr>
      <th>513</th>
      <td>Ended</td>
      <td>-1.0</td>
      <td>38</td>
      <td>1501384818</td>
      <td>South Beach Tow</td>
      <td>English</td>
      <td>2011-07-20 00:00:00</td>
      <td>&lt;p&gt;The &lt;b&gt;South Beach Tow&lt;/b&gt; crew returns to ...</td>
      <td>30</td>
      <td>Reality</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>US</td>
      <td>United States</td>
      <td>America/New_York</td>
      <td>84</td>
      <td>truTV</td>
    </tr>
    <tr>
      <th>514</th>
      <td>Ended</td>
      <td>-1.0</td>
      <td>0</td>
      <td>1466882679</td>
      <td>Starhyke</td>
      <td>English</td>
      <td>2009-11-30 00:00:00</td>
      <td>&lt;p&gt;It's the year 3034. Everyone on Earth has b...</td>
      <td>30</td>
      <td>Scripted</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>GB</td>
      <td>United Kingdom</td>
      <td>Europe/London</td>
      <td>324</td>
      <td>Showcase TV</td>
    </tr>
    <tr>
      <th>515</th>
      <td>Ended</td>
      <td>-1.0</td>
      <td>0</td>
      <td>1496675604</td>
      <td>Making the Band</td>
      <td>English</td>
      <td>2000-03-24 00:00:00</td>
      <td>&lt;p&gt;&lt;b&gt;Making the Band&lt;/b&gt; was the brainchild o...</td>
      <td>30</td>
      <td>Reality</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>US</td>
      <td>United States</td>
      <td>America/New_York</td>
      <td>22</td>
      <td>MTV</td>
    </tr>
    <tr>
      <th>516</th>
      <td>Running</td>
      <td>4.5</td>
      <td>68</td>
      <td>1480821374</td>
      <td>Second Jen</td>
      <td>English</td>
      <td>2016-08-28 00:00:00</td>
      <td>&lt;p&gt;&lt;b&gt;Second Jen&lt;/b&gt; is a ground-breaking scri...</td>
      <td>30</td>
      <td>Scripted</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>CA</td>
      <td>Canada</td>
      <td>Canada/Atlantic</td>
      <td>151</td>
      <td>City</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>598</th>
      <td>Ended</td>
      <td>-1.0</td>
      <td>0</td>
      <td>1502593090</td>
      <td>CeeLo Green's The Good Life</td>
      <td>English</td>
      <td>2014-06-23 00:00:00</td>
      <td>&lt;p&gt;Follow CeeLo as he tackles not only a packe...</td>
      <td>30</td>
      <td>Reality</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>US</td>
      <td>United States</td>
      <td>America/New_York</td>
      <td>32</td>
      <td>TBS</td>
    </tr>
    <tr>
      <th>599</th>
      <td>Ended</td>
      <td>-1.0</td>
      <td>0</td>
      <td>1477193480</td>
      <td>America's Prom Queen</td>
      <td>English</td>
      <td>2008-03-17 00:00:00</td>
      <td>&lt;p&gt;&lt;b&gt;America's Prom Queen&lt;/b&gt; is a reality TV...</td>
      <td>60</td>
      <td>Reality</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>US</td>
      <td>United States</td>
      <td>America/New_York</td>
      <td>26</td>
      <td>FreeForm</td>
    </tr>
    <tr>
      <th>600</th>
      <td>Ended</td>
      <td>-1.0</td>
      <td>0</td>
      <td>1461445299</td>
      <td>Hollywood Me</td>
      <td>English</td>
      <td>2013-06-19 00:00:00</td>
      <td>&lt;p&gt;Martyn Lawrence Bullard's normal clients in...</td>
      <td>30</td>
      <td>Reality</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>GB</td>
      <td>United Kingdom</td>
      <td>Europe/London</td>
      <td>45</td>
      <td>Channel 4</td>
    </tr>
    <tr>
      <th>607</th>
      <td>Ended</td>
      <td>-1.0</td>
      <td>52</td>
      <td>1490313454</td>
      <td>Utopia</td>
      <td>English</td>
      <td>2014-09-07 00:00:00</td>
      <td>&lt;p&gt;Get ready to witness the birth of a brave n...</td>
      <td>60</td>
      <td>Reality</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>US</td>
      <td>United States</td>
      <td>America/New_York</td>
      <td>4</td>
      <td>FOX</td>
    </tr>
    <tr>
      <th>608</th>
      <td>Running</td>
      <td>-1.0</td>
      <td>93</td>
      <td>1499236738</td>
      <td>Storage Wars: Canada</td>
      <td>English</td>
      <td>2013-08-29 00:00:00</td>
      <td>&lt;p&gt;On a daily basis, high-stakes buyers descen...</td>
      <td>30</td>
      <td>Reality</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>CA</td>
      <td>Canada</td>
      <td>Canada/Atlantic</td>
      <td>350</td>
      <td>OLN</td>
    </tr>
    <tr>
      <th>609</th>
      <td>Ended</td>
      <td>-1.0</td>
      <td>0</td>
      <td>1502217322</td>
      <td>Big Brother</td>
      <td>English</td>
      <td>2001-04-23 00:00:00</td>
      <td>&lt;p&gt;&lt;b&gt;Big Brother Australia&lt;/b&gt; is based on th...</td>
      <td>None</td>
      <td>Reality</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>AU</td>
      <td>Australia</td>
      <td>Australia/Sydney</td>
      <td>120</td>
      <td>Nine Network</td>
    </tr>
    <tr>
      <th>610</th>
      <td>Ended</td>
      <td>-1.0</td>
      <td>0</td>
      <td>1497307824</td>
      <td>The Vineyard</td>
      <td>English</td>
      <td>2013-07-23 00:00:00</td>
      <td>&lt;p&gt;ABC Family's newest original docu-series, &lt;...</td>
      <td>60</td>
      <td>Reality</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>US</td>
      <td>United States</td>
      <td>America/New_York</td>
      <td>26</td>
      <td>FreeForm</td>
    </tr>
    <tr>
      <th>611</th>
      <td>Running</td>
      <td>-1.0</td>
      <td>0</td>
      <td>1503655502</td>
      <td>Na dobre i na złe</td>
      <td>Polish</td>
      <td>1999-11-07 00:00:00</td>
      <td></td>
      <td>60</td>
      <td>Scripted</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>PL</td>
      <td>Poland</td>
      <td>Europe/Warsaw</td>
      <td>333</td>
      <td>TVP2</td>
    </tr>
    <tr>
      <th>612</th>
      <td>Ended</td>
      <td>-1.0</td>
      <td>0</td>
      <td>1477348482</td>
      <td>Big Top</td>
      <td>English</td>
      <td>2009-12-02 00:00:00</td>
      <td>&lt;p&gt;&lt;b&gt;Big Top&lt;/b&gt; was a sit-com that aired on ...</td>
      <td>30</td>
      <td>Scripted</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>GB</td>
      <td>United Kingdom</td>
      <td>Europe/London</td>
      <td>12</td>
      <td>BBC One</td>
    </tr>
    <tr>
      <th>613</th>
      <td>Running</td>
      <td>9.0</td>
      <td>0</td>
      <td>1468322551</td>
      <td>MTV Suspect</td>
      <td>English</td>
      <td>2016-02-23 00:00:00</td>
      <td>&lt;p&gt;Across America, people are hiding deep secr...</td>
      <td>60</td>
      <td>Reality</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>US</td>
      <td>United States</td>
      <td>America/New_York</td>
      <td>22</td>
      <td>MTV</td>
    </tr>
    <tr>
      <th>614</th>
      <td>Ended</td>
      <td>-1.0</td>
      <td>0</td>
      <td>1497305713</td>
      <td>Kimora Life in the Fab Lane</td>
      <td>English</td>
      <td>2007-08-05 00:00:00</td>
      <td>&lt;p&gt;A glimpse into the life of former model Kim...</td>
      <td>30</td>
      <td>Reality</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>US</td>
      <td>United States</td>
      <td>America/New_York</td>
      <td>43</td>
      <td>E!</td>
    </tr>
    <tr>
      <th>615</th>
      <td>Ended</td>
      <td>-1.0</td>
      <td>0</td>
      <td>1490293113</td>
      <td>Celebrities Undercover</td>
      <td>English</td>
      <td>2014-03-18 00:00:00</td>
      <td>&lt;p&gt;Celebrities are used to transforming into o...</td>
      <td>30</td>
      <td>Reality</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>US</td>
      <td>United States</td>
      <td>America/New_York</td>
      <td>79</td>
      <td>Oxygen</td>
    </tr>
    <tr>
      <th>616</th>
      <td>Running</td>
      <td>-1.0</td>
      <td>0</td>
      <td>1458216770</td>
      <td>Recipe for Deception</td>
      <td>English</td>
      <td>2016-01-21 00:00:00</td>
      <td>&lt;p&gt;Bravo Media cooks up a battle of secrets an...</td>
      <td>60</td>
      <td>Reality</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>US</td>
      <td>United States</td>
      <td>America/New_York</td>
      <td>52</td>
      <td>Bravo</td>
    </tr>
    <tr>
      <th>617</th>
      <td>Ended</td>
      <td>-1.0</td>
      <td>0</td>
      <td>1481538915</td>
      <td>16 Kids and Counting</td>
      <td>English</td>
      <td>2013-01-11 00:00:00</td>
      <td>&lt;p&gt;What's life like when you have enough child...</td>
      <td>60</td>
      <td>Documentary</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>GB</td>
      <td>United Kingdom</td>
      <td>Europe/London</td>
      <td>45</td>
      <td>Channel 4</td>
    </tr>
    <tr>
      <th>618</th>
      <td>Ended</td>
      <td>-1.0</td>
      <td>0</td>
      <td>1484475919</td>
      <td>A Poet's Guide to Britain</td>
      <td>English</td>
      <td>2009-05-04 00:00:00</td>
      <td>&lt;p&gt;Poet and author Owen Sheers presents a seri...</td>
      <td>30</td>
      <td>Documentary</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>GB</td>
      <td>United Kingdom</td>
      <td>Europe/London</td>
      <td>51</td>
      <td>BBC Four</td>
    </tr>
    <tr>
      <th>619</th>
      <td>Running</td>
      <td>-1.0</td>
      <td>94</td>
      <td>1502640953</td>
      <td>The Bold and the Beautiful</td>
      <td>English</td>
      <td>1987-03-23 00:00:00</td>
      <td>&lt;p&gt;They created a dynasty where passion rules,...</td>
      <td>30</td>
      <td>Scripted</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>US</td>
      <td>United States</td>
      <td>America/New_York</td>
      <td>2</td>
      <td>CBS</td>
    </tr>
    <tr>
      <th>620</th>
      <td>Running</td>
      <td>-1.0</td>
      <td>99</td>
      <td>1502485797</td>
      <td>Life of Kylie</td>
      <td>English</td>
      <td>2017-08-06 00:00:00</td>
      <td>&lt;p&gt;&lt;b&gt;Life of Kylie&lt;/b&gt; will follow Kylie Jenn...</td>
      <td>30</td>
      <td>Reality</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>US</td>
      <td>United States</td>
      <td>America/New_York</td>
      <td>43</td>
      <td>E!</td>
    </tr>
    <tr>
      <th>621</th>
      <td>Ended</td>
      <td>6.0</td>
      <td>0</td>
      <td>1502487937</td>
      <td>Jersey Shore</td>
      <td>English</td>
      <td>2009-12-03 00:00:00</td>
      <td>&lt;p&gt;Grab your hair gel, wax that Cadillac and g...</td>
      <td>60</td>
      <td>Reality</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>US</td>
      <td>United States</td>
      <td>America/New_York</td>
      <td>43</td>
      <td>E!</td>
    </tr>
    <tr>
      <th>622</th>
      <td>Ended</td>
      <td>-1.0</td>
      <td>0</td>
      <td>1485103110</td>
      <td>The Hills</td>
      <td>English</td>
      <td>2006-05-31 00:00:00</td>
      <td>&lt;p&gt;In the final season of &lt;b&gt;The Hills&lt;/b&gt; - K...</td>
      <td>30</td>
      <td>Reality</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>US</td>
      <td>United States</td>
      <td>America/New_York</td>
      <td>22</td>
      <td>MTV</td>
    </tr>
    <tr>
      <th>623</th>
      <td>Running</td>
      <td>2.7</td>
      <td>91</td>
      <td>1500442171</td>
      <td>Teen Mom</td>
      <td>English</td>
      <td>2009-12-08 00:00:00</td>
      <td>&lt;p&gt;In 16 and Pregnant, they were moms-to-be. N...</td>
      <td>60</td>
      <td>Reality</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>US</td>
      <td>United States</td>
      <td>America/New_York</td>
      <td>22</td>
      <td>MTV</td>
    </tr>
    <tr>
      <th>624</th>
      <td>Ended</td>
      <td>5.7</td>
      <td>66</td>
      <td>1489774713</td>
      <td>Coupling</td>
      <td>English</td>
      <td>2003-09-25 00:00:00</td>
      <td>&lt;p&gt;&lt;b&gt;Coupling&lt;/b&gt; is an American remake of th...</td>
      <td>30</td>
      <td>Scripted</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>US</td>
      <td>United States</td>
      <td>America/New_York</td>
      <td>1</td>
      <td>NBC</td>
    </tr>
    <tr>
      <th>625</th>
      <td>Running</td>
      <td>-1.0</td>
      <td>0</td>
      <td>1486846250</td>
      <td>Access Hollywood Live</td>
      <td>English</td>
      <td>1996-09-09 00:00:00</td>
      <td>&lt;p&gt;&lt;b&gt;Access Hollywood Live&lt;/b&gt; is a weekday t...</td>
      <td>60</td>
      <td>Variety</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>US</td>
      <td>United States</td>
      <td>America/New_York</td>
      <td>75</td>
      <td>REELZ</td>
    </tr>
    <tr>
      <th>626</th>
      <td>To Be Determined</td>
      <td>-1.0</td>
      <td>0</td>
      <td>1462596807</td>
      <td>The First Family</td>
      <td>English</td>
      <td>2012-09-17 00:00:00</td>
      <td>&lt;p&gt;&lt;i&gt;&lt;b&gt;"The First Family"&lt;/b&gt;&lt;/i&gt; is an Amer...</td>
      <td>30</td>
      <td>Scripted</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>US</td>
      <td>United States</td>
      <td>America/New_York</td>
      <td>5</td>
      <td>The CW</td>
    </tr>
    <tr>
      <th>627</th>
      <td>Ended</td>
      <td>10.0</td>
      <td>0</td>
      <td>1502461972</td>
      <td>Garbage Pail Kids</td>
      <td>English</td>
      <td>0</td>
      <td>&lt;p&gt;From deep within the historic TV animation ...</td>
      <td>25</td>
      <td>Animation</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>US</td>
      <td>United States</td>
      <td>America/New_York</td>
      <td>2</td>
      <td>CBS</td>
    </tr>
    <tr>
      <th>628</th>
      <td>Ended</td>
      <td>-1.0</td>
      <td>50</td>
      <td>1497743151</td>
      <td>Khloé &amp; Lamar</td>
      <td>English</td>
      <td>2011-04-10 00:00:00</td>
      <td>&lt;p&gt;In &lt;b&gt;Khloé &amp;amp; Lamar&lt;/b&gt;, cameras will f...</td>
      <td>30</td>
      <td>Reality</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>US</td>
      <td>United States</td>
      <td>America/New_York</td>
      <td>43</td>
      <td>E!</td>
    </tr>
    <tr>
      <th>629</th>
      <td>Ended</td>
      <td>-1.0</td>
      <td>0</td>
      <td>1482948423</td>
      <td>The Paul Reiser Show</td>
      <td>English</td>
      <td>2011-04-14 00:00:00</td>
      <td>&lt;p&gt;Paul Reiser plays a fictional version of hi...</td>
      <td>30</td>
      <td>Scripted</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>US</td>
      <td>United States</td>
      <td>America/New_York</td>
      <td>1</td>
      <td>NBC</td>
    </tr>
    <tr>
      <th>630</th>
      <td>Ended</td>
      <td>-1.0</td>
      <td>0</td>
      <td>1485719969</td>
      <td>Pretty Wicked Moms</td>
      <td>English</td>
      <td>2013-06-04 00:00:00</td>
      <td>&lt;p&gt;Six Atlanta moms give a whole new meaning t...</td>
      <td>60</td>
      <td>Reality</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>US</td>
      <td>United States</td>
      <td>America/New_York</td>
      <td>18</td>
      <td>Lifetime</td>
    </tr>
    <tr>
      <th>631</th>
      <td>Ended</td>
      <td>-1.0</td>
      <td>0</td>
      <td>1502430474</td>
      <td>The Wright Way</td>
      <td>English</td>
      <td>2013-04-23 00:00:00</td>
      <td>&lt;p&gt;Gerald Wright runs the Baselricky Council H...</td>
      <td>30</td>
      <td>Scripted</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>GB</td>
      <td>United Kingdom</td>
      <td>Europe/London</td>
      <td>12</td>
      <td>BBC One</td>
    </tr>
    <tr>
      <th>632</th>
      <td>Ended</td>
      <td>-1.0</td>
      <td>0</td>
      <td>1474119411</td>
      <td>High School Musical: Get in the Picture</td>
      <td>English</td>
      <td>2008-07-20 00:00:00</td>
      <td>&lt;p&gt;A group of teenagers are invited to partici...</td>
      <td>60</td>
      <td>Reality</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>US</td>
      <td>United States</td>
      <td>America/New_York</td>
      <td>3</td>
      <td>ABC</td>
    </tr>
    <tr>
      <th>633</th>
      <td>Ended</td>
      <td>-1.0</td>
      <td>0</td>
      <td>1477283569</td>
      <td>Audrina</td>
      <td>English</td>
      <td>2011-04-17 00:00:00</td>
      <td>&lt;p&gt;Besides Audrina's blossoming career and tum...</td>
      <td>30</td>
      <td>Reality</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>US</td>
      <td>United States</td>
      <td>America/New_York</td>
      <td>55</td>
      <td>VH1</td>
    </tr>
  </tbody>
</table>
<p>141 rows × 103 columns</p>
</div>




```python
# What do we have that is mostly complete
print df_shows[~df_shows.isnull().any(axis=1)]['winner'].value_counts()
df_shows_notnull = df_shows[~df_shows.isnull().any(axis=1)]
```

    1    209
    0    117
    Name: winner, dtype: int64



```python
# In the processing above, NaNs were replaced by other values for some columns.  This block creates a new
# dataframe where all rows with these coded values representing missing data have been removed.

df_shows_complete = df_shows_notnull[(df_shows_notnull['rating'] != -1) & \
                                     (df_shows_notnull['gn_unknown'] != 1) & \
                                     (df_shows_notnull['sched_unknown'] != 1) & \
                                     (df_shows_notnull['sched_time_unknown'] != 1) & \
                                     (df_shows_notnull['country_code'] != '') & \
                                     (df_shows_notnull['country_name'] != '') & \
                                     (df_shows_notnull['country_tz'] != '') & \
                                     (df_shows_notnull['network_id'] != '') & \
                                     (df_shows_notnull['network_name'] != '') & \
                                     (df_shows_notnull['premiered'] != 0)]
```


```python
df_shows_complete.shape
```




    (157, 103)




```python
# Cool, at least not missing any summaries for samples that are otherwise complete
df_shows_complete['summary'].isnull().sum()
```




    0




```python
df_shows[['summary', 'winner']].groupby(['winner']).count()
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
      <th>summary</th>
    </tr>
    <tr>
      <th>winner</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>256</td>
    </tr>
    <tr>
      <th>1</th>
      <td>235</td>
    </tr>
  </tbody>
</table>
</div>



## Modeling Section

  -- Note:  Cells in this section must be run sequentially to obtain correct results as some variables are reused in the various modeling sections

### Vectorize summary text in different ways


```python
# I'll first try a model with just the summary text, that is available for 491 shows, 256 loosers and 235 winners


# Use NLP techniques to create lots of factors
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from collections import Counter

# Use different Vectorizers to find ngrams for us
tfidf = TfidfVectorizer(ngram_range=(2,4), max_features=2000, stop_words='english')
cvec = CountVectorizer(ngram_range=(2,4), max_features=2000, stop_words='english')
hvec = HashingVectorizer(ngram_range=(2,4), n_features=2000, stop_words='english')

X_tfidf = tfidf.fit_transform(df_shows['summary']).todense()
X_cvec = cvec.fit_transform(df_shows['summary']).todense()
X_hvec = hvec.fit_transform(df_shows['summary']).todense()

y = df_shows['winner'].values

print '\ntfidf shape:', X_tfidf.shape
print '\ncvec shape:', X_cvec.shape
print '\nhvec shape:', X_hvec.shape
print len(y)

```

    
    tfidf shape: (491, 2000)
    
    cvec shape: (491, 2000)
    
    hvec shape: (491, 2000)
    491


## Model on summary text using Count Vectorizer

- results were best when Count Vectorizer scores were modeled with Gaussian Naive Bayes


Features:     2000  
Train Set Accuracy:   0.905  
CrossVal Accuracy:     0.644 +/- 0.028   
Test Set Accuracy:   0.626   

**n-grams with higest cumulative sum of tf-idf scores for winners: **   'drama series', 'david attenborough', 'tells story', 'young boy', 'anthology series', 'documentary series', 'years later', 'main character', 'trials tribulations', 'crime drama', 'serial killer', 'tv history', 'super hero', 'story starts goku', 'starts goku', 'story starts', 'american television', 'fictional town', 'television drama', 'american crime'
  
**n-grams with higest cumulative sum of tf-idf scores for losers: ** 'real housewives', 'television series', 'reality series', 'follows lives', 'series produced', 'pop culture', 'reality television', 'reality television series', 'animated series', 'come true', 'aired abc', 'reality tv', 'series debuted', 'real housewives orange county', 'real housewives orange', 'housewives orange', 'housewives orange county', 'talk hosted', 'studio audience', 'cash prize'

  


```python
# Baseline for training set
winner_avg = y.mean()
baseline = max(winner_avg, 1-winner_avg)
print baseline
```

    0.521384928717



```python
# Test Train Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_cvec, y, test_size=0.25)
```


```python
print X_train.shape,  len(y_train)
print X_test.shape,  len(y_test)
```

    (368, 2000) 368
    (123, 2000) 123



```python
#  Standardize - 

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

Xs_train = ss.fit_transform(X_train)
Xs_test = ss.transform(X_test)


```


```python
# Run lots of classifiers on this and see which perform the best
# Import all the modeling libraries

from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, \
                                    KFold, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt
```


```python
# prepare configuration for cross validation test harness
seed = 42

# prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('QDA', QuadraticDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('RFST', RandomForestClassifier()))
models.append(('GB', GradientBoostingClassifier()))
models.append(('ADA', AdaBoostClassifier()))
models.append(('SVM', SVC()))
models.append(('GNB', GaussianNB()))
models.append(('MNB', MultinomialNB()))
models.append(('BNB', BernoulliNB()))


# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'

print "\n{}:   {:0.3} ".format('Baseline', baseline, cv_results.std())
print "\n{:5.5}:  {:10.8}  {:20.18}  {:20.17}  {:20.17}".format\
        ("Model", "Features", "Train Set Accuracy", "CrossVal Accuracy", "Test Set Accuracy")

for name, model in models:
    try:
        kfold = KFold(n_splits=3, shuffle=True, random_state=seed)
        cv_results = cross_val_score(model, Xs_train, y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        this_model = model
        this_model.fit(X_train,y_train)
        print "{:5.5}     {:}         {:0.3f}               {:0.3f} +/- {:0.3f}         {:0.3f} ".format\
                (name, X_train.shape[1], metrics.accuracy_score(y_train, this_model.predict(Xs_train)), \
                 cv_results.mean(), cv_results.std(), metrics.accuracy_score(y_test, this_model.predict(Xs_test)))
    except:
        print "    {:5.5}:   {} ".format(name, 'failed on this input dataset')

        
                
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
ax.axhline(y=baseline, color='grey', linestyle='--')
plt.show()
```

    
    Baseline:   0.521 
    
    Model:  Features    Train Set Accuracy    CrossVal Accuracy     Test Set Accuracy   
    LR        2000         0.938               0.660 +/- 0.037         0.626 
    LDA       2000         0.938               0.544 +/- 0.054         0.593 
    QDA       2000         0.549               0.399 +/- 0.034         0.390 
    KNN       2000         0.758               0.500 +/- 0.010         0.528 
    CART      2000         0.943               0.576 +/- 0.028         0.585 
    RFST      2000         0.940               0.636 +/- 0.046         0.626 
    GB        2000         0.826               0.546 +/- 0.020         0.585 
    ADA       2000         0.769               0.552 +/- 0.042         0.545 
    SVM       2000         0.519               0.519 +/- 0.018         0.528 
    GNB       2000         0.913               0.688 +/- 0.038         0.561 
        MNB  :   failed on this input dataset 
    BNB       2000         0.902               0.625 +/- 0.023         0.602 



![png](/images/tv-show-info_files/tv-show-info_121_1.png)



```python
# Which words are most common in the winner summaries ?
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

# We can use the TfidfVectorizer to find ngrams for us
vect = CountVectorizer(ngram_range=(2,4), stop_words='english')

# Pulls all of trumps tweet text's into one giant string
summaries = "".join(df_shows[df_shows['winner'] == 1]['summary'])
ngrams_summaries = vect.build_analyzer()(summaries)

Counter(ngrams_summaries).most_common(20)
```




    [(u'new york', 11),
     (u'drama series', 8),
     (u'york city', 6),
     (u'high school', 6),
     (u'men women', 5),
     (u'tv series', 5),
     (u'series based', 5),
     (u'video game', 5),
     (u'bugs bunny', 5),
     (u'new york city', 5),
     (u'tells story', 4),
     (u'young boy', 4),
     (u'comedy series', 4),
     (u'main character', 4),
     (u'united states', 4),
     (u'life new', 4),
     (u'series follows', 4),
     (u'anthology series', 3),
     (u'mr bean', 3),
     (u'prisoner cell', 3)]




```python
# Which words are most common in the loser summaries ?
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

# We can use the TfidfVectorizer to find ngrams for us
vect = CountVectorizer(ngram_range=(2,4), stop_words='english')

# Pulls all of trumps tweet text's into one giant string
summaries = "".join(df_shows[df_shows['winner'] == 0]['summary'])
ngrams_summaries = vect.build_analyzer()(summaries)

Counter(ngrams_summaries).most_common(20)
```




    [(u'real housewives', 12),
     (u'television series', 12),
     (u'los angeles', 11),
     (u'pop culture', 10),
     (u'series follows', 9),
     (u'new york', 9),
     (u'animated series', 7),
     (u'cartoon network', 7),
     (u'big brother', 6),
     (u'dance moms', 6),
     (u'reality series', 6),
     (u'best friend', 6),
     (u'high school', 5),
     (u'late night', 5),
     (u'best friends', 5),
     (u'nick jr', 5),
     (u'reality television series', 5),
     (u'plastic surgery', 5),
     (u'access hollywood', 5),
     (u'comedy series', 5)]




```python
# Sum matrix columns to see what has the most overall importance ?

print "Highest sum Count Vectoror score for n_grams in winner shows"

cvec_results = pd.DataFrame(Xs_train, columns=cvec.get_feature_names())
cvec_results['winners'] = y_train

winner_results = pd.DataFrame(cvec_results[cvec_results['winners'] ==1].sum(), columns=['cvec_sum'])


high = winner_results.drop(['winners']).sort_values('cvec_sum', axis=0, ascending=False).head(20).index
print  [str(r) for r in high]

winner_results.drop(['winners']).sort_values('cvec_sum', axis=0, ascending=False).head(20)

```

    Highest sum Count Vectoror score for n_grams in winner shows
    ['drama series', 'david attenborough', 'main character', 'tells story', 'years later', 'years ago', 'fictional town', 'anthology series', 'documentary series', 'provocative series', 'makes effort', 'standup comedian', 'set world', 'time 13yearold', 'based manga', 'highs lows', 'set fictional', 'sherlock holmes', 'series takes', 'seaside town']





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
      <th>cvec_sum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>drama series</th>
      <td>21.615324</td>
    </tr>
    <tr>
      <th>david attenborough</th>
      <td>20.022240</td>
    </tr>
    <tr>
      <th>main character</th>
      <td>20.022240</td>
    </tr>
    <tr>
      <th>tells story</th>
      <td>20.022240</td>
    </tr>
    <tr>
      <th>years later</th>
      <td>17.315999</td>
    </tr>
    <tr>
      <th>years ago</th>
      <td>17.315999</td>
    </tr>
    <tr>
      <th>fictional town</th>
      <td>17.315999</td>
    </tr>
    <tr>
      <th>anthology series</th>
      <td>17.315999</td>
    </tr>
    <tr>
      <th>documentary series</th>
      <td>17.315999</td>
    </tr>
    <tr>
      <th>provocative series</th>
      <td>14.119126</td>
    </tr>
    <tr>
      <th>makes effort</th>
      <td>14.119126</td>
    </tr>
    <tr>
      <th>standup comedian</th>
      <td>14.119126</td>
    </tr>
    <tr>
      <th>set world</th>
      <td>14.119126</td>
    </tr>
    <tr>
      <th>time 13yearold</th>
      <td>14.119126</td>
    </tr>
    <tr>
      <th>based manga</th>
      <td>14.119126</td>
    </tr>
    <tr>
      <th>highs lows</th>
      <td>14.119126</td>
    </tr>
    <tr>
      <th>set fictional</th>
      <td>14.119126</td>
    </tr>
    <tr>
      <th>sherlock holmes</th>
      <td>14.119126</td>
    </tr>
    <tr>
      <th>series takes</th>
      <td>14.119126</td>
    </tr>
    <tr>
      <th>seaside town</th>
      <td>14.119126</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Sum matrix columns to see what has the most overall importance ?

print "Highest sum Count Vectoror score for n_grams in loser shows"

cvec_results = pd.DataFrame(Xs_train, columns=cvec.get_feature_names())
cvec_results['winners'] = y_train

winner_results = pd.DataFrame(cvec_results[cvec_results['winners'] ==0].sum(), columns=['cvec_sum'])


high = winner_results.drop(['winners']).sort_values('cvec_sum', axis=0, ascending=False).head(20).index
print  [str(r) for r in high]

winner_results.drop(['winners']).sort_values('cvec_sum', axis=0, ascending=False).head(20)

```

    Highest sum Count Vectoror score for n_grams in loser shows
    ['reality series', 'television series', 'series produced', 'real housewives', 'series debuted', 'family friends', 'follows lives', 'series features', 'animated series', 'los angeles', 'reality television', 'reality television series', 'bros television distribution', 'bros television', 'warner bros television', 'warner bros television distribution', 'television series debuted', 'cash prize', 'new series', 'news channel']





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
      <th>cvec_sum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>reality series</th>
      <td>22.787391</td>
    </tr>
    <tr>
      <th>television series</th>
      <td>21.394240</td>
    </tr>
    <tr>
      <th>series produced</th>
      <td>20.773274</td>
    </tr>
    <tr>
      <th>real housewives</th>
      <td>19.851334</td>
    </tr>
    <tr>
      <th>series debuted</th>
      <td>18.554642</td>
    </tr>
    <tr>
      <th>family friends</th>
      <td>18.554642</td>
    </tr>
    <tr>
      <th>follows lives</th>
      <td>18.554642</td>
    </tr>
    <tr>
      <th>series features</th>
      <td>18.554642</td>
    </tr>
    <tr>
      <th>animated series</th>
      <td>18.554642</td>
    </tr>
    <tr>
      <th>los angeles</th>
      <td>18.313960</td>
    </tr>
    <tr>
      <th>reality television</th>
      <td>17.522176</td>
    </tr>
    <tr>
      <th>reality television series</th>
      <td>17.522176</td>
    </tr>
    <tr>
      <th>bros television distribution</th>
      <td>16.046764</td>
    </tr>
    <tr>
      <th>bros television</th>
      <td>16.046764</td>
    </tr>
    <tr>
      <th>warner bros television</th>
      <td>16.046764</td>
    </tr>
    <tr>
      <th>warner bros television distribution</th>
      <td>16.046764</td>
    </tr>
    <tr>
      <th>television series debuted</th>
      <td>16.046764</td>
    </tr>
    <tr>
      <th>cash prize</th>
      <td>16.046764</td>
    </tr>
    <tr>
      <th>new series</th>
      <td>16.046764</td>
    </tr>
    <tr>
      <th>news channel</th>
      <td>16.046764</td>
    </tr>
  </tbody>
</table>
</div>



## Model on summary text using TF-IDF Vectorizer

- results were best when tf-idf scores were modeled with Gaussian Naive Bayes



Features:     2000  
Train Set Accuracy:   0.924  
CrossVal Accuracy:     0.609 +/- 0.034   
Test Set Accuracy:   0.609 +/- 0.034   

**n-grams with higest cumulative sum of tf-idf scores for winners: **   'david attenborough', 'drama series', 'men women', 'new york', 'documentary series', 'new york city', 'york city', 'quest save', 'tv series', 'world know', 'television drama', 'sitcom set', 'young boy', 'comedy series', 'series created', 'tells story', '21st century', 'super hero', 'cable news', 'best friends'  
  
**n-grams with higest cumulative sum of tf-idf scores for losers: ** 'real housewives', 'series follows', 'television series', 'best friends', 'best friend', 'los angeles', 'things just', 'group teenagers', 'series features', 'restaurant industry', 'children ages', 'animated series', 'big brother', 'cartoon network', 'recent divorce', 'american women', 'high school', 'reality series', 'follows lives', 'lives loves'  

  


```python
# Baseline for training set
winner_avg = y.mean()
baseline = max(winner_avg, 1-winner_avg)
print baseline

```

    0.521384928717



```python
# Test Train Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.25)
```


```python
print X_train.shape,  len(y_train)
print X_test.shape,  len(y_test)
```

    (368, 2000) 368
    (123, 2000) 123



```python
#  Standardize - 

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

Xs_train = ss.fit_transform(X_train)
Xs_test = ss.transform(X_test)


```


```python
# prepare configuration for cross validation test harness
seed = 42

# prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('QDA', QuadraticDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('RFST', RandomForestClassifier()))
models.append(('GB', GradientBoostingClassifier()))
models.append(('ADA', AdaBoostClassifier()))
models.append(('SVM', SVC()))
models.append(('GNB', GaussianNB()))
models.append(('MNB', MultinomialNB()))
models.append(('BNB', BernoulliNB()))


# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'

print "\n{}:   {:0.3} ".format('Baseline', baseline, cv_results.std())
print "\n{:5.5}:  {:10.8}  {:20.18}  {:20.17}  {:20.17}".format\
        ("Model", "Features", "Train Set Accuracy", "CrossVal Accuracy", "Test Set Accuracy")

for name, model in models:
    try:
        kfold = KFold(n_splits=3, shuffle=True, random_state=seed)
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        this_model = model
        this_model.fit(X_train,y_train)
        print "{:5.5}     {:}         {:0.3f}               {:0.3f} +/- {:0.3f}         {:0.3f} ".format\
                (name, X_train.shape[1], metrics.accuracy_score(y_train, this_model.predict(X_train)), \
                 cv_results.mean(), cv_results.std(), metrics.accuracy_score(y_test, this_model.predict(X_test)))
    except:
        print "    {:5.5}:   {} ".format(name, 'failed on this input dataset')

        
                
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
ax.axhline(y=baseline, color='grey', linestyle='--')
plt.show()
```

    
    Baseline:   0.521 
    
    Model:  Features    Train Set Accuracy    CrossVal Accuracy     Test Set Accuracy   
    LR        2000         0.957               0.620 +/- 0.020         0.634 
    LDA       2000         0.959               0.658 +/- 0.035         0.610 
    QDA       2000         0.671               0.437 +/- 0.013         0.431 
    KNN       2000         0.647               0.519 +/- 0.031         0.488 
    CART      2000         0.959               0.554 +/- 0.026         0.496 
    RFST      2000         0.957               0.581 +/- 0.020         0.561 
    GB        2000         0.872               0.598 +/- 0.034         0.545 
    ADA       2000         0.772               0.541 +/- 0.047         0.504 
    SVM       2000         0.505               0.492 +/- 0.009         0.569 
    GNB       2000         0.943               0.668 +/- 0.028         0.634 
    MNB       2000         0.927               0.658 +/- 0.029         0.642 
    BNB       2000         0.932               0.641 +/- 0.048         0.593 



![png](/images/tv-show-info_files/tv-show-info_131_1.png)



```python
# Which words are most common in the winner summaries ?
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

# We can use the TfidfVectorizer to find ngrams for us
vect = TfidfVectorizer(ngram_range=(2,4), stop_words='english')

# Pulls all of trumps tweet text's into one giant string
summaries = "".join(df_shows[df_shows['winner'] == 1]['summary'])
ngrams_summaries = vect.build_analyzer()(summaries)

Counter(ngrams_summaries).most_common(20)
```




    [(u'new york', 11),
     (u'drama series', 8),
     (u'york city', 6),
     (u'high school', 6),
     (u'men women', 5),
     (u'tv series', 5),
     (u'series based', 5),
     (u'video game', 5),
     (u'bugs bunny', 5),
     (u'new york city', 5),
     (u'tells story', 4),
     (u'young boy', 4),
     (u'comedy series', 4),
     (u'main character', 4),
     (u'united states', 4),
     (u'life new', 4),
     (u'series follows', 4),
     (u'anthology series', 3),
     (u'mr bean', 3),
     (u'prisoner cell', 3)]




```python
# Which words are most common in the loser summaries ?
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

# We can use the TfidfVectorizer to find ngrams for us
vect = TfidfVectorizer(ngram_range=(2,4), stop_words='english')

# Pulls all of trumps tweet text's into one giant string
summaries = "".join(df_shows[df_shows['winner'] == 0]['summary'])
ngrams_summaries = vect.build_analyzer()(summaries)

Counter(ngrams_summaries).most_common(20)
```




    [(u'real housewives', 12),
     (u'television series', 12),
     (u'los angeles', 11),
     (u'pop culture', 10),
     (u'series follows', 9),
     (u'new york', 9),
     (u'animated series', 7),
     (u'cartoon network', 7),
     (u'big brother', 6),
     (u'dance moms', 6),
     (u'reality series', 6),
     (u'best friend', 6),
     (u'high school', 5),
     (u'late night', 5),
     (u'best friends', 5),
     (u'nick jr', 5),
     (u'reality television series', 5),
     (u'plastic surgery', 5),
     (u'access hollywood', 5),
     (u'comedy series', 5)]




```python
# Sum matrix columns to see what has the most overall importance ?

print "Highest cumulative tfidf score for n_grams in winner shows"

tfidf_results = pd.DataFrame(X_train, columns= tfidf.get_feature_names())
tfidf_results['winners'] = y_train

winner_results = pd.DataFrame(tfidf_results[tfidf_results['winners'] ==1].sum(), columns=['tfidf_sum'])


high = winner_results.drop(['winners']).sort_values('tfidf_sum', axis=0, ascending=False).head(20).index
print  [str(r) for r in high]

winner_results.drop(['winners']).sort_values('tfidf_sum', axis=0, ascending=False).head(20)

```

    Highest cumulative tfidf score for n_grams in winner shows
    ['new york', 'men women', 'documentary series', 'york city', 'new york city', 'high school', 'drama series', 'tells story', 'years ago', 'david attenborough', 'series created', 'years later', 'young man', 'comedy series', 'main character', '21st century', 'tv series', 'andrew davies', 'cable news', 'series based']





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
      <th>tfidf_sum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>new york</th>
      <td>2.835972</td>
    </tr>
    <tr>
      <th>men women</th>
      <td>2.484786</td>
    </tr>
    <tr>
      <th>documentary series</th>
      <td>2.171334</td>
    </tr>
    <tr>
      <th>york city</th>
      <td>1.897754</td>
    </tr>
    <tr>
      <th>new york city</th>
      <td>1.897754</td>
    </tr>
    <tr>
      <th>high school</th>
      <td>1.743989</td>
    </tr>
    <tr>
      <th>drama series</th>
      <td>1.716267</td>
    </tr>
    <tr>
      <th>tells story</th>
      <td>1.685216</td>
    </tr>
    <tr>
      <th>years ago</th>
      <td>1.634339</td>
    </tr>
    <tr>
      <th>david attenborough</th>
      <td>1.522240</td>
    </tr>
    <tr>
      <th>series created</th>
      <td>1.484294</td>
    </tr>
    <tr>
      <th>years later</th>
      <td>1.484223</td>
    </tr>
    <tr>
      <th>young man</th>
      <td>1.474759</td>
    </tr>
    <tr>
      <th>comedy series</th>
      <td>1.401982</td>
    </tr>
    <tr>
      <th>main character</th>
      <td>1.366767</td>
    </tr>
    <tr>
      <th>21st century</th>
      <td>1.358933</td>
    </tr>
    <tr>
      <th>tv series</th>
      <td>1.307707</td>
    </tr>
    <tr>
      <th>andrew davies</th>
      <td>1.304276</td>
    </tr>
    <tr>
      <th>cable news</th>
      <td>1.261358</td>
    </tr>
    <tr>
      <th>series based</th>
      <td>1.258897</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Sum matrix columns to see what has the most overall importance ?

print "Highest cumulative tfidf score for n_grams in loser shows"

tfidf_results = pd.DataFrame(X_train, columns= tfidf.get_feature_names())
tfidf_results['winners'] = y_train

winner_results = pd.DataFrame(tfidf_results[tfidf_results['winners'] == 0].sum(), columns=['tfidf_sum'])

low = winner_results.drop(['winners']).sort_values('tfidf_sum', axis=0, ascending=False).head(20).index

print  [str(r) for r in low]
winner_results.drop(['winners']).sort_values('tfidf_sum', axis=0, ascending=False).head(20)
```

    Highest cumulative tfidf score for n_grams in loser shows
    ['reality series', 'television series', 'los angeles', 'real housewives', 'things just', 'best friend', 'group teenagers', 'series follows', 'restaurant industry', 'new york', 'high school', 'children ages', 'big brother', 'recent divorce', 'series features', 'cartoon network', 'football team', 'plastic surgery', 'bizarre adventures', 'nick jr']





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
      <th>tfidf_sum</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>reality series</th>
      <td>3.535573</td>
    </tr>
    <tr>
      <th>television series</th>
      <td>2.715189</td>
    </tr>
    <tr>
      <th>los angeles</th>
      <td>2.582930</td>
    </tr>
    <tr>
      <th>real housewives</th>
      <td>2.160853</td>
    </tr>
    <tr>
      <th>things just</th>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>best friend</th>
      <td>1.958176</td>
    </tr>
    <tr>
      <th>group teenagers</th>
      <td>1.791283</td>
    </tr>
    <tr>
      <th>series follows</th>
      <td>1.772954</td>
    </tr>
    <tr>
      <th>restaurant industry</th>
      <td>1.707107</td>
    </tr>
    <tr>
      <th>new york</th>
      <td>1.662150</td>
    </tr>
    <tr>
      <th>high school</th>
      <td>1.657534</td>
    </tr>
    <tr>
      <th>children ages</th>
      <td>1.648007</td>
    </tr>
    <tr>
      <th>big brother</th>
      <td>1.591352</td>
    </tr>
    <tr>
      <th>recent divorce</th>
      <td>1.473112</td>
    </tr>
    <tr>
      <th>series features</th>
      <td>1.443284</td>
    </tr>
    <tr>
      <th>cartoon network</th>
      <td>1.441719</td>
    </tr>
    <tr>
      <th>football team</th>
      <td>1.414214</td>
    </tr>
    <tr>
      <th>plastic surgery</th>
      <td>1.382887</td>
    </tr>
    <tr>
      <th>bizarre adventures</th>
      <td>1.368894</td>
    </tr>
    <tr>
      <th>nick jr</th>
      <td>1.366875</td>
    </tr>
  </tbody>
</table>
</div>



## Model using data other than the TV show summary text 


```python
# Get list of columns for the useful non-summary data.  Dropping the "unknown" columns will solve
# the colinearity issue with dummied columns, as these will be the dropped dummies.  
# Dropping premiered as it is a datatime and standardize can't handle it.  Also dropping
# weight as it is not understood, and rating and winner as they are the targets

cols = [x for x in df_shows.columns if x not in ['rating', 'weight', 'updated', 'premiered', 'summary', 'id', \
                                                 'gn_unknown', 'sched_unknown', 'sched_time_unknown', \
                                                 'country_name', 'country_tz', 'network_name', 'name', 'winner']]
cols
```




    [u'status',
     u'language',
     u'runtime',
     u'type',
     u'network',
     u'gn_action',
     u'gn_adult',
     u'gn_adventure',
     u'gn_anime',
     u'gn_children',
     u'gn_comedy',
     u'gn_crime',
     u'gn_drama',
     u'gn_espionage',
     u'gn_family',
     u'gn_fantasy',
     u'gn_food',
     u'gn_history',
     u'gn_horror',
     u'gn_legal',
     u'gn_medical',
     u'gn_music',
     u'gn_mystery',
     u'gn_nature',
     u'gn_romance',
     u'gn_science-fiction',
     u'gn_sports',
     u'gn_supernatural',
     u'gn_thriller',
     u'gn_travel',
     u'gn_war',
     u'gn_western',
     u'sched_friday',
     u'sched_monday',
     u'sched_saturday',
     u'sched_sunday',
     u'sched_thursday',
     u'sched_tuesday',
     u'sched_wednesday',
     u'sched_time_00:00',
     u'sched_time_00:30',
     u'sched_time_00:50',
     u'sched_time_00:55',
     u'sched_time_01:00',
     u'sched_time_01:05',
     u'sched_time_01:30',
     u'sched_time_01:35',
     u'sched_time_02:00',
     u'sched_time_02:05',
     u'sched_time_08:00',
     u'sched_time_10:00',
     u'sched_time_11:00',
     u'sched_time_12:00',
     u'sched_time_13:00',
     u'sched_time_13:30',
     u'sched_time_14:00',
     u'sched_time_14:30',
     u'sched_time_15:00',
     u'sched_time_15:15',
     u'sched_time_16:00',
     u'sched_time_16:30',
     u'sched_time_17:00',
     u'sched_time_17:15',
     u'sched_time_17:30',
     u'sched_time_18:00',
     u'sched_time_18:30',
     u'sched_time_19:00',
     u'sched_time_19:30',
     u'sched_time_19:45',
     u'sched_time_20:00',
     u'sched_time_20:15',
     u'sched_time_20:30',
     u'sched_time_20:40',
     u'sched_time_20:45',
     u'sched_time_20:55',
     u'sched_time_21:00',
     u'sched_time_21:10',
     u'sched_time_21:15',
     u'sched_time_21:30',
     u'sched_time_21:45',
     u'sched_time_22:00',
     u'sched_time_22:10',
     u'sched_time_22:30',
     u'sched_time_22:35',
     u'sched_time_23:00',
     u'sched_time_23:02',
     u'sched_time_23:15',
     u'sched_time_23:30',
     'country_code',
     'network_id']




```python
# Dummy country code, network id, status, language, and type
df_showsd = pd.get_dummies(df_shows, columns=['network_id'], prefix='NW', prefix_sep='_')

df_showsd = df_showsd.drop('NW_', 1)
df_showsd = df_showsd.drop('network', 1)

```


```python
df_showsd = pd.get_dummies(df_showsd, columns=['country_code'], prefix='C', prefix_sep='_', drop_first=True)
df_showsd = pd.get_dummies(df_showsd, columns=['status'], prefix='ST', prefix_sep='_', drop_first=True)
df_showsd = pd.get_dummies(df_showsd, columns=['language'], prefix='L', prefix_sep='_', drop_first=True)
df_showsd = pd.get_dummies(df_showsd, columns=['type'], prefix='T', prefix_sep='_', drop_first=True)

```


```python
# Handle any NaN values that remain
shows_clean = df_showsd.dropna()
```


```python
# We have 326 total samples left, about 1/3 loser and 2/3 winner
# Seems reasonable to proceed with a classification model

print "Number winner samples:", shows_clean['winner'].sum()
print "Number loser samples:", len(shows_clean[shows_clean['winner'] == 0])
```

    Number winner samples: 230
    Number loser samples: 121



```python
cols = [x for x in shows_clean.columns if x not in ['rating', 'weight', 'updated', 'premiered', 'summary', 'id', \
                                                 'gn_unknown', 'sched_unknown', 'sched_time_unknown', \
                                                 'country_name', 'country_tz', 'network_name', 'name', 'winner']]
cols
```




    [u'runtime',
     u'gn_action',
     u'gn_adult',
     u'gn_adventure',
     u'gn_anime',
     u'gn_children',
     u'gn_comedy',
     u'gn_crime',
     u'gn_drama',
     u'gn_espionage',
     u'gn_family',
     u'gn_fantasy',
     u'gn_food',
     u'gn_history',
     u'gn_horror',
     u'gn_legal',
     u'gn_medical',
     u'gn_music',
     u'gn_mystery',
     u'gn_nature',
     u'gn_romance',
     u'gn_science-fiction',
     u'gn_sports',
     u'gn_supernatural',
     u'gn_thriller',
     u'gn_travel',
     u'gn_war',
     u'gn_western',
     u'sched_friday',
     u'sched_monday',
     u'sched_saturday',
     u'sched_sunday',
     u'sched_thursday',
     u'sched_tuesday',
     u'sched_wednesday',
     u'sched_time_00:00',
     u'sched_time_00:30',
     u'sched_time_00:50',
     u'sched_time_00:55',
     u'sched_time_01:00',
     u'sched_time_01:05',
     u'sched_time_01:30',
     u'sched_time_01:35',
     u'sched_time_02:00',
     u'sched_time_02:05',
     u'sched_time_08:00',
     u'sched_time_10:00',
     u'sched_time_11:00',
     u'sched_time_12:00',
     u'sched_time_13:00',
     u'sched_time_13:30',
     u'sched_time_14:00',
     u'sched_time_14:30',
     u'sched_time_15:00',
     u'sched_time_15:15',
     u'sched_time_16:00',
     u'sched_time_16:30',
     u'sched_time_17:00',
     u'sched_time_17:15',
     u'sched_time_17:30',
     u'sched_time_18:00',
     u'sched_time_18:30',
     u'sched_time_19:00',
     u'sched_time_19:30',
     u'sched_time_19:45',
     u'sched_time_20:00',
     u'sched_time_20:15',
     u'sched_time_20:30',
     u'sched_time_20:40',
     u'sched_time_20:45',
     u'sched_time_20:55',
     u'sched_time_21:00',
     u'sched_time_21:10',
     u'sched_time_21:15',
     u'sched_time_21:30',
     u'sched_time_21:45',
     u'sched_time_22:00',
     u'sched_time_22:10',
     u'sched_time_22:30',
     u'sched_time_22:35',
     u'sched_time_23:00',
     u'sched_time_23:02',
     u'sched_time_23:15',
     u'sched_time_23:30',
     'NW_1',
     'NW_2',
     'NW_3',
     'NW_4',
     'NW_5',
     'NW_6',
     'NW_8',
     'NW_9',
     'NW_10',
     'NW_11',
     'NW_12',
     'NW_13',
     'NW_14',
     'NW_16',
     'NW_17',
     'NW_18',
     'NW_19',
     'NW_20',
     'NW_22',
     'NW_23',
     'NW_24',
     'NW_25',
     'NW_26',
     'NW_27',
     'NW_29',
     'NW_30',
     'NW_32',
     'NW_34',
     'NW_35',
     'NW_36',
     'NW_37',
     'NW_41',
     'NW_42',
     'NW_43',
     'NW_44',
     'NW_45',
     'NW_47',
     'NW_48',
     'NW_49',
     'NW_51',
     'NW_52',
     'NW_54',
     'NW_55',
     'NW_56',
     'NW_59',
     'NW_63',
     'NW_66',
     'NW_70',
     'NW_71',
     'NW_72',
     'NW_73',
     'NW_75',
     'NW_76',
     'NW_77',
     'NW_78',
     'NW_79',
     'NW_80',
     'NW_81',
     'NW_84',
     'NW_85',
     'NW_88',
     'NW_91',
     'NW_92',
     'NW_107',
     'NW_109',
     'NW_114',
     'NW_115',
     'NW_118',
     'NW_120',
     'NW_122',
     'NW_125',
     'NW_131',
     'NW_132',
     'NW_137',
     'NW_144',
     'NW_149',
     'NW_151',
     'NW_155',
     'NW_157',
     'NW_158',
     'NW_159',
     'NW_163',
     'NW_173',
     'NW_177',
     'NW_184',
     'NW_185',
     'NW_206',
     'NW_224',
     'NW_231',
     'NW_239',
     'NW_248',
     'NW_251',
     'NW_270',
     'NW_286',
     'NW_298',
     'NW_309',
     'NW_324',
     'NW_333',
     'NW_336',
     'NW_349',
     'NW_350',
     'NW_360',
     'NW_376',
     'NW_409',
     'NW_472',
     'NW_551',
     'NW_553',
     'NW_639',
     'NW_652',
     'NW_714',
     'NW_809',
     'NW_813',
     'NW_821',
     'NW_870',
     'NW_976',
     'NW_1027',
     'NW_1050',
     'NW_1485',
     u'C_AU',
     u'C_CA',
     u'C_DE',
     u'C_DK',
     u'C_FR',
     u'C_GB',
     u'C_IT',
     u'C_JP',
     u'C_KR',
     u'C_NO',
     u'C_NZ',
     u'C_PL',
     u'C_RU',
     u'C_SE',
     u'C_TR',
     u'C_US',
     u'ST_Running',
     u'ST_To Be Determined',
     u'L_English',
     u'L_French',
     u'L_German',
     u'L_Hindi',
     u'L_Italian',
     u'L_Japanese',
     u'L_Korean',
     u'L_Norwegian',
     u'L_Polish',
     u'L_Russian',
     u'L_Swedish',
     u'L_Turkish',
     u'T_Documentary',
     u'T_Game Show',
     u'T_News',
     u'T_Panel Show',
     u'T_Reality',
     u'T_Scripted',
     u'T_Talk Show',
     u'T_Variety']




```python
# Generate X matrix and y target

X = shows_clean[cols]
y = shows_clean['winner'].values
```


```python
# Baseline 
winner_avg = y.mean()
baseline = max(winner_avg, 1-winner_avg)
print baseline

```

    0.655270655271



```python
# Test Train Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
```


```python
print X_train.shape,  len(y_train)
print X_test.shape,  len(y_test)
```

    (263, 240) 263
    (88, 240) 88



```python
#  Standardize - 

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

Xs_train = ss.fit_transform(X_train)
Xs_test = ss.transform(X_test)# Test Train Split


```


```python
# Gridsearch for best C and penalty
gs_params = {
    'penalty':['l1', 'l2'],
    'solver':['liblinear'],
    'C':np.logspace(-5,5,100)
}
from sklearn.model_selection import GridSearchCV
lr_gridsearch = GridSearchCV(LogisticRegression(), gs_params, cv=3, verbose=1, n_jobs=-1)

```


```python
lr_gridsearch.fit(Xs_train, y_train)
```

    Fitting 3 folds for each of 200 candidates, totalling 600 fits


    [Parallel(n_jobs=-1)]: Done 196 tasks      | elapsed:    0.8s
    [Parallel(n_jobs=-1)]: Done 600 out of 600 | elapsed:    1.4s finished





    GridSearchCV(cv=3, error_score='raise',
           estimator=LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False),
           fit_params={}, iid=True, n_jobs=-1,
           param_grid={'penalty': ['l1', 'l2'], 'C': array([  1.00000e-05,   1.26186e-05, ...,   7.92483e+04,   1.00000e+05]), 'solver': ['liblinear']},
           pre_dispatch='2*n_jobs', refit=True, return_train_score=True,
           scoring=None, verbose=1)




```python
# best score on the training data:
lr_gridsearch.best_score_
```




    0.9125475285171103




```python
# best parameters on the training data:
lr_gridsearch.best_params_
```




    {'C': 0.068926121043496949, 'penalty': 'l2', 'solver': 'liblinear'}




```python
# assign the best estimator to a variable:
best_lr = lr_gridsearch.best_estimator_
```


```python
# Score it on the testing data:
best_lr.score(Xs_test, y_test)
```




    0.88636363636363635




```python
# Much better than baseline, and we can find the most important factors and run all the classifiers using
# those factors.
```


```python
coef_df = pd.DataFrame({
        'features': X.columns,
        'log odds': best_lr.coef_[0],
        'percentage change in odds': np.round(np.exp(best_lr.coef_[0])*100-100,2)
    })
```


```python
coef_df.sort_values(by='percentage change in odds', ascending=0)
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
      <th>features</th>
      <th>log odds</th>
      <th>percentage change in odds</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>gn_comedy</td>
      <td>0.500129</td>
      <td>64.89</td>
    </tr>
    <tr>
      <th>237</th>
      <td>T_Scripted</td>
      <td>0.477513</td>
      <td>61.21</td>
    </tr>
    <tr>
      <th>232</th>
      <td>T_Documentary</td>
      <td>0.268693</td>
      <td>30.83</td>
    </tr>
    <tr>
      <th>8</th>
      <td>gn_drama</td>
      <td>0.254062</td>
      <td>28.93</td>
    </tr>
    <tr>
      <th>3</th>
      <td>gn_adventure</td>
      <td>0.246904</td>
      <td>28.01</td>
    </tr>
    <tr>
      <th>90</th>
      <td>NW_8</td>
      <td>0.242839</td>
      <td>27.49</td>
    </tr>
    <tr>
      <th>94</th>
      <td>NW_12</td>
      <td>0.207017</td>
      <td>23.00</td>
    </tr>
    <tr>
      <th>114</th>
      <td>NW_37</td>
      <td>0.201048</td>
      <td>22.27</td>
    </tr>
    <tr>
      <th>7</th>
      <td>gn_crime</td>
      <td>0.192888</td>
      <td>21.27</td>
    </tr>
    <tr>
      <th>83</th>
      <td>sched_time_23:30</td>
      <td>0.178175</td>
      <td>19.50</td>
    </tr>
    <tr>
      <th>21</th>
      <td>gn_science-fiction</td>
      <td>0.177248</td>
      <td>19.39</td>
    </tr>
    <tr>
      <th>23</th>
      <td>gn_supernatural</td>
      <td>0.177209</td>
      <td>19.39</td>
    </tr>
    <tr>
      <th>18</th>
      <td>gn_mystery</td>
      <td>0.167534</td>
      <td>18.24</td>
    </tr>
    <tr>
      <th>95</th>
      <td>NW_13</td>
      <td>0.139862</td>
      <td>15.01</td>
    </tr>
    <tr>
      <th>218</th>
      <td>ST_Running</td>
      <td>0.139006</td>
      <td>14.91</td>
    </tr>
    <tr>
      <th>0</th>
      <td>runtime</td>
      <td>0.138903</td>
      <td>14.90</td>
    </tr>
    <tr>
      <th>11</th>
      <td>gn_fantasy</td>
      <td>0.137532</td>
      <td>14.74</td>
    </tr>
    <tr>
      <th>64</th>
      <td>sched_time_19:45</td>
      <td>0.137084</td>
      <td>14.69</td>
    </tr>
    <tr>
      <th>176</th>
      <td>NW_270</td>
      <td>0.126593</td>
      <td>13.50</td>
    </tr>
    <tr>
      <th>87</th>
      <td>NW_4</td>
      <td>0.122341</td>
      <td>13.01</td>
    </tr>
    <tr>
      <th>143</th>
      <td>NW_85</td>
      <td>0.115952</td>
      <td>12.29</td>
    </tr>
    <tr>
      <th>13</th>
      <td>gn_history</td>
      <td>0.111519</td>
      <td>11.80</td>
    </tr>
    <tr>
      <th>24</th>
      <td>gn_thriller</td>
      <td>0.102413</td>
      <td>10.78</td>
    </tr>
    <tr>
      <th>30</th>
      <td>sched_saturday</td>
      <td>0.102229</td>
      <td>10.76</td>
    </tr>
    <tr>
      <th>207</th>
      <td>C_GB</td>
      <td>0.101888</td>
      <td>10.73</td>
    </tr>
    <tr>
      <th>1</th>
      <td>gn_action</td>
      <td>0.090291</td>
      <td>9.45</td>
    </tr>
    <tr>
      <th>14</th>
      <td>gn_horror</td>
      <td>0.088511</td>
      <td>9.25</td>
    </tr>
    <tr>
      <th>62</th>
      <td>sched_time_19:00</td>
      <td>0.086923</td>
      <td>9.08</td>
    </tr>
    <tr>
      <th>98</th>
      <td>NW_17</td>
      <td>0.085510</td>
      <td>8.93</td>
    </tr>
    <tr>
      <th>225</th>
      <td>L_Japanese</td>
      <td>0.085179</td>
      <td>8.89</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>53</th>
      <td>sched_time_15:00</td>
      <td>-0.106270</td>
      <td>-10.08</td>
    </tr>
    <tr>
      <th>153</th>
      <td>NW_122</td>
      <td>-0.113304</td>
      <td>-10.71</td>
    </tr>
    <tr>
      <th>132</th>
      <td>NW_71</td>
      <td>-0.113592</td>
      <td>-10.74</td>
    </tr>
    <tr>
      <th>201</th>
      <td>NW_1485</td>
      <td>-0.115251</td>
      <td>-10.89</td>
    </tr>
    <tr>
      <th>186</th>
      <td>NW_376</td>
      <td>-0.115363</td>
      <td>-10.90</td>
    </tr>
    <tr>
      <th>169</th>
      <td>NW_185</td>
      <td>-0.116553</td>
      <td>-11.00</td>
    </tr>
    <tr>
      <th>106</th>
      <td>NW_26</td>
      <td>-0.117644</td>
      <td>-11.10</td>
    </tr>
    <tr>
      <th>220</th>
      <td>L_English</td>
      <td>-0.119882</td>
      <td>-11.30</td>
    </tr>
    <tr>
      <th>105</th>
      <td>NW_25</td>
      <td>-0.126560</td>
      <td>-11.89</td>
    </tr>
    <tr>
      <th>55</th>
      <td>sched_time_16:00</td>
      <td>-0.128590</td>
      <td>-12.07</td>
    </tr>
    <tr>
      <th>29</th>
      <td>sched_monday</td>
      <td>-0.134095</td>
      <td>-12.55</td>
    </tr>
    <tr>
      <th>192</th>
      <td>NW_652</td>
      <td>-0.143685</td>
      <td>-13.38</td>
    </tr>
    <tr>
      <th>217</th>
      <td>C_US</td>
      <td>-0.154332</td>
      <td>-14.30</td>
    </tr>
    <tr>
      <th>117</th>
      <td>NW_43</td>
      <td>-0.155093</td>
      <td>-14.37</td>
    </tr>
    <tr>
      <th>110</th>
      <td>NW_32</td>
      <td>-0.157596</td>
      <td>-14.58</td>
    </tr>
    <tr>
      <th>158</th>
      <td>NW_144</td>
      <td>-0.158937</td>
      <td>-14.69</td>
    </tr>
    <tr>
      <th>49</th>
      <td>sched_time_13:00</td>
      <td>-0.159660</td>
      <td>-14.76</td>
    </tr>
    <tr>
      <th>140</th>
      <td>NW_80</td>
      <td>-0.160574</td>
      <td>-14.83</td>
    </tr>
    <tr>
      <th>33</th>
      <td>sched_tuesday</td>
      <td>-0.174971</td>
      <td>-16.05</td>
    </tr>
    <tr>
      <th>203</th>
      <td>C_CA</td>
      <td>-0.179060</td>
      <td>-16.39</td>
    </tr>
    <tr>
      <th>93</th>
      <td>NW_11</td>
      <td>-0.179608</td>
      <td>-16.44</td>
    </tr>
    <tr>
      <th>17</th>
      <td>gn_music</td>
      <td>-0.189299</td>
      <td>-17.25</td>
    </tr>
    <tr>
      <th>179</th>
      <td>NW_309</td>
      <td>-0.216452</td>
      <td>-19.46</td>
    </tr>
    <tr>
      <th>124</th>
      <td>NW_52</td>
      <td>-0.224190</td>
      <td>-20.08</td>
    </tr>
    <tr>
      <th>238</th>
      <td>T_Talk Show</td>
      <td>-0.233104</td>
      <td>-20.79</td>
    </tr>
    <tr>
      <th>233</th>
      <td>T_Game Show</td>
      <td>-0.244726</td>
      <td>-21.71</td>
    </tr>
    <tr>
      <th>78</th>
      <td>sched_time_22:30</td>
      <td>-0.251273</td>
      <td>-22.22</td>
    </tr>
    <tr>
      <th>102</th>
      <td>NW_22</td>
      <td>-0.268289</td>
      <td>-23.53</td>
    </tr>
    <tr>
      <th>67</th>
      <td>sched_time_20:30</td>
      <td>-0.300379</td>
      <td>-25.95</td>
    </tr>
    <tr>
      <th>236</th>
      <td>T_Reality</td>
      <td>-0.557917</td>
      <td>-42.76</td>
    </tr>
  </tbody>
</table>
<p>240 rows × 3 columns</p>
</div>




```python
# Create a subset of "coef_df" DataFrame with most important coefficients
imp_coefs = pd.concat([coef_df.sort_values(by='percentage change in odds', ascending=0).head(10),
                     coef_df.sort_values(by='percentage change in odds', ascending=0).tail(10)])
```


```python
imp_coefs.set_index('features', inplace=True)
```


```python
imp_coefs
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
      <th>log odds</th>
      <th>percentage change in odds</th>
    </tr>
    <tr>
      <th>features</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>gn_comedy</th>
      <td>0.500129</td>
      <td>64.89</td>
    </tr>
    <tr>
      <th>T_Scripted</th>
      <td>0.477513</td>
      <td>61.21</td>
    </tr>
    <tr>
      <th>T_Documentary</th>
      <td>0.268693</td>
      <td>30.83</td>
    </tr>
    <tr>
      <th>gn_drama</th>
      <td>0.254062</td>
      <td>28.93</td>
    </tr>
    <tr>
      <th>gn_adventure</th>
      <td>0.246904</td>
      <td>28.01</td>
    </tr>
    <tr>
      <th>NW_8</th>
      <td>0.242839</td>
      <td>27.49</td>
    </tr>
    <tr>
      <th>NW_12</th>
      <td>0.207017</td>
      <td>23.00</td>
    </tr>
    <tr>
      <th>NW_37</th>
      <td>0.201048</td>
      <td>22.27</td>
    </tr>
    <tr>
      <th>gn_crime</th>
      <td>0.192888</td>
      <td>21.27</td>
    </tr>
    <tr>
      <th>sched_time_23:30</th>
      <td>0.178175</td>
      <td>19.50</td>
    </tr>
    <tr>
      <th>NW_11</th>
      <td>-0.179608</td>
      <td>-16.44</td>
    </tr>
    <tr>
      <th>gn_music</th>
      <td>-0.189299</td>
      <td>-17.25</td>
    </tr>
    <tr>
      <th>NW_309</th>
      <td>-0.216452</td>
      <td>-19.46</td>
    </tr>
    <tr>
      <th>NW_52</th>
      <td>-0.224190</td>
      <td>-20.08</td>
    </tr>
    <tr>
      <th>T_Talk Show</th>
      <td>-0.233104</td>
      <td>-20.79</td>
    </tr>
    <tr>
      <th>T_Game Show</th>
      <td>-0.244726</td>
      <td>-21.71</td>
    </tr>
    <tr>
      <th>sched_time_22:30</th>
      <td>-0.251273</td>
      <td>-22.22</td>
    </tr>
    <tr>
      <th>NW_22</th>
      <td>-0.268289</td>
      <td>-23.53</td>
    </tr>
    <tr>
      <th>sched_time_20:30</th>
      <td>-0.300379</td>
      <td>-25.95</td>
    </tr>
    <tr>
      <th>T_Reality</th>
      <td>-0.557917</td>
      <td>-42.76</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plot important coefficients
imp_coefs['percentage change in odds'].plot(kind = "barh")
plt.title("Percentage change in odds with Ridge regularization")
plt.show()
```


![png](/images/tv-show-info_files/tv-show-info_160_0.png)



```python
df_shows[df_shows['network_id'] == 309]
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
      <th>status</th>
      <th>rating</th>
      <th>weight</th>
      <th>updated</th>
      <th>name</th>
      <th>language</th>
      <th>premiered</th>
      <th>summary</th>
      <th>runtime</th>
      <th>type</th>
      <th>...</th>
      <th>sched_time_23:00</th>
      <th>sched_time_23:02</th>
      <th>sched_time_23:15</th>
      <th>sched_time_23:30</th>
      <th>sched_time_unknown</th>
      <th>country_code</th>
      <th>country_name</th>
      <th>country_tz</th>
      <th>network_id</th>
      <th>network_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>279</th>
      <td>Running</td>
      <td>8.7</td>
      <td>40</td>
      <td>2017-08-22 19:26:37</td>
      <td>I Live with Models</td>
      <td>English</td>
      <td>2015-02-23 00:00:00</td>
      <td>tommy heads to new york city with scarlet as t...</td>
      <td>30</td>
      <td>Scripted</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>GB</td>
      <td>United Kingdom</td>
      <td>Europe/London</td>
      <td>309</td>
      <td>Comedy Central</td>
    </tr>
    <tr>
      <th>535</th>
      <td>To Be Determined</td>
      <td>6.5</td>
      <td>0</td>
      <td>2016-01-13 15:12:17</td>
      <td>Brotherhood</td>
      <td>English</td>
      <td>2015-06-02 00:00:00</td>
      <td>twentysomethings dan and toby are in over thei...</td>
      <td>30</td>
      <td>Scripted</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>GB</td>
      <td>United Kingdom</td>
      <td>Europe/London</td>
      <td>309</td>
      <td>Comedy Central</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 104 columns</p>
</div>




```python
# Get list of features and re-run model with just the 20 most important features
imp_features = imp_coefs.index
```


```python
# Set up X and y
X = shows_clean[imp_features]
y = shows_clean['winner'].values
```


```python
# Baseline
winner_avg = y.mean()
baseline = max(winner_avg, 1-winner_avg)
print baseline
```

    0.655270655271



```python
# Test Train Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
```


```python
print X_train.shape,  len(y_train)
print X_test.shape,  len(y_test)
```

    (263, 20) 263
    (88, 20) 88



```python
#  Standardize - 

from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

Xs_train = ss.fit_transform(X_train)
Xs_test = ss.transform(X_test)# Test Train Split

```


```python
# prepare configuration for cross validation test harness
seed = 42

# prepare models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('QDA', QuadraticDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('RFST', RandomForestClassifier()))
models.append(('GB', GradientBoostingClassifier()))
models.append(('ADA', AdaBoostClassifier()))
models.append(('SVM', SVC()))
models.append(('GNB', GaussianNB()))
models.append(('MNB', MultinomialNB()))
models.append(('BNB', BernoulliNB()))


# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'

print "\n{}:   {:0.3} ".format('Baseline', baseline, cv_results.std())
print "\n{:5.5}:  {:10.8}  {:20.18}  {:20.17}  {:20.17}".format\
        ("Model", "Features", "Train Set Accuracy", "CrossVal Accuracy", "Test Set Accuracy")

for name, model in models:
    try:
        kfold = KFold(n_splits=3, shuffle=True, random_state=seed)
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        this_model = model
        this_model.fit(X_train,y_train)
        print "{:5.5}     {:}         {:0.3f}               {:0.3f} +/- {:0.3f}         {:0.3f} ".format\
                (name, X_train.shape[1], metrics.accuracy_score(y_train, this_model.predict(X_train)), \
                 cv_results.mean(), cv_results.std(), metrics.accuracy_score(y_test, this_model.predict(X_test)))
    except:
        print "    {:5.5}:   {} ".format(name, 'failed on this input dataset')

        
                
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
ax.axhline(y=baseline, color='grey', linestyle='--')
plt.show()
```

    
    Baseline:   0.655 
    
    Model:  Features    Train Set Accuracy    CrossVal Accuracy     Test Set Accuracy   
    LR        20         0.905               0.874 +/- 0.010         0.909 
    LDA       20         0.901               0.890 +/- 0.020         0.875 
    QDA       20         0.669               0.448 +/- 0.082         0.682 
    KNN       20         0.909               0.905 +/- 0.005         0.886 
    CART      20         0.943               0.894 +/- 0.019         0.898 
    RFST      20         0.939               0.909 +/- 0.000         0.886 
    GB        20         0.943               0.905 +/- 0.020         0.898 
    ADA       20         0.916               0.897 +/- 0.032         0.920 
    SVM       20         0.863               0.848 +/- 0.014         0.818 
    GNB       20         0.616               0.673 +/- 0.057         0.614 
    MNB       20         0.886               0.886 +/- 0.010         0.898 
    BNB       20         0.905               0.901 +/- 0.010         0.920 



![png](/images/tv-show-info_files/tv-show-info_168_1.png)



```python

```
