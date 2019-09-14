#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from urllib.request import urlopen
import calendar


# In[ ]:


# AFSNT 파일로 date 뽑기
AFSNT = pd.read_csv('data/AFSNT.csv', encoding='CP949')
AFSNT_date = list(set(AFSNT['DATE']))
AFSNT_DLY = pd.read_csv('data/AFSNT_DLY.csv', encoding='CP949')
AFSNT_DLY_date = list(set(AFSNT['DATE']))
date = AFSNT_date + AFSNT_DLY_date
date.sort()
date_df = pd.DataFrame({'date':date})


# In[ ]:


AFSNT_ALL = pd.concat([AFSNT, AFSNT_DLY], ignore_index=True)
AFSNT_ALL = AFSNT_ALL[['DATE','SDT_DY']].drop_duplicates().reset_index(drop=True)
AFSNT_ALL.head()


# In[ ]:


def findmdidx(s):
    try:
        idx1 = s.index("월")
        idx2 = s.index("일")
    return idx1,idx2


# In[ ]:


pat = "{0:0>4}-{1:0>2}-{2:0>2}"
year = [2017, 2018, 2019]
holi = []
for y in range(len(year)):
    url = 'https://search.naver.com/search.naver?sm=top_hty&fbm=1&ie=utf8&query='+str(year[y])+'년+공휴일'
    req = requests.get(url)
    html = req.text
    soup = BeautifulSoup(html, 'html.parser')
    result = soup.select('#dss_free_uio_cont1 > div > div > table > tbody')
    
    st = list(result[0].strings)
    tmp = []
    for i in range(len(st)):
        try:
            int(st[i][0])
        except:
            continue
    if "월" in st[i]:
        tmp.append(st[i])
          
    for i in range(len(tmp)):
        if '~' in tmp[i]:
            t = tmp[i].split(' ~ ')
            m1, d1 = findmdidx(t[0])
        try:
            m2, d2 = findmdidx(t[1])
        except:
            m2 = m1
            d2 = -1
        
        mon1 = int(t[0][:m1])
        if len(t[1])>3:
            mon2 = int(t[1][:m2])
        else:
            mon2 = mon1
        
        day1 = int(t[0][m1+2:d1])
        try:
            day2 = int(t[1][m2+2:d2])
        except:
            day2 = int(t[1][:-1])
        
        if mon1 != mon2:
            l1, last = calendar.monthrange(year[y], mon1)
            holi.append(pat.format(year[y], mon1, day1))
            holi.append(pat.format(year[y], mon2, day2))
        if last != day1:
            holi.append(pat.format(year[y], mon1, last))
        else:
            holi.append(pat.format(year[y], mon2, day2-1))
    else:
        for k in range(day1, day2+1):
            holi.append(pat.format(year[y], mon1, k))
          
    else:
        m, d = findmdidx(tmp[i])
        holi.append(pat.format(year[y], tmp[i][:m], tmp[i][m+2:d]))


# In[ ]:


holi.sort()
holi


# In[ ]:


holi_bfr = [0]*len(AFSNT_ALL)
holi_day = [0]*len(AFSNT_ALL)
holi_aft = [0]*len(AFSNT_ALL)
for i in range(len(AFSNT_ALL)):
    if AFSNT_ALL['DATE'][i] in holi:
        holi_day[i] = 1
        if i-1 > 0 and AFSNT_ALL['DATE'][i-1] not in holi:
            if AFSNT_ALL['SDT_DY'][i-1] not in ['토', '일']:
                holi_bfr[i-1] = 1
        if i+1 < len(AFSNT_ALL) and AFSNT_ALL['DATE'][i+1] not in holi and AFSNT_ALL['SDT_DY'][i+1] not in ['토', '일']:
            holi_aft[i+1] = 1


# In[ ]:


AFSNT_ALL['holi_bfr'] = holi_bfr
AFSNT_ALL['holi_day'] = holi_day
AFSNT_ALL['holi_aft'] = holi_aft


# In[ ]:


AFSNT_ALL


# In[ ]:


len(AFSNT_ALL)


# #### 징검다리연휴처리

# In[ ]:


holi_bfr_bigcon = holi_bfr
holi_day_bigcon = holi_day
holi_aft_bigcon = holi_aft


# In[ ]:


for i in range(len(AFSNT_ALL)):
    if AFSNT_ALL['holi_bfr'][i] == 1 and AFSNT_ALL['holi_aft'][i] == 1:
        holi_bfr_bigcon[i]=0
        holi_day_bigcon[i]=1
        holi_aft_bigcon[i]=0
    if i-1 > 0:
        if AFSNT_ALL['SDT_DY'][i-1] == '일' and AFSNT_ALL['holi_bfr'][i] == 1:
            holi_day_bigcon[i]=1
            holi_bfr_bigcon[i]=0
        if AFSNT_ALL['SDT_DY'][i] == '토' and AFSNT_ALL['holi_day'][i-1] == 1 and AFSNT_ALL['holi_day'][i+2] == 1:
            holi_day_bigcon[i]=1
            holi_bfr_bigcon[i]=0
            holi_aft_bigcon[i]=0
            holi_day_bigcon[i+1]=1
            holi_bfr_bigcon[i+1]=0
            holi_aft_bigcon[i+1]=0
    if i+1 < len(AFSNT_ALL):
        if AFSNT_ALL['SDT_DY'][i+1] == '토' and AFSNT_ALL['holi_aft'][i] == 1:
            holi_day_bigcon[i]=1
            holi_aft_bigcon[i]=0


# In[ ]:


AFSNT_ALL_bigcon = AFSNT_ALL[['DATE', 'SDT_DY']]
AFSNT_ALL_bigcon['holi_bfr'] = holi_bfr_bigcon
AFSNT_ALL_bigcon['holi_day'] = holi_day_bigcon
AFSNT_ALL_bigcon['holi_aft'] = holi_aft_bigcon


# In[ ]:


AFSNT_ALL_bigcon.to_csv("data/holiday.csv", index=False, encoding='CP949')

