import os
import pandas as pd
import pylab as pl
from numpy import *


f = {}
for file in os.listdir('.'):
     try:
        a=pd.read_csv(file,header=5,index_col=1)

        p =[]
        for key in a.T.keys():
            try:
                q=map(lambda x:
                float(x.replace(",",'')),a.T[key].ix[5:12])
                if len(q) == 7: p.append(q)
            except: pass
        f[file.split('-')[-1].split('.')[0]] = p
        print file.split('-')[-1].split('.')[0]
     except Exception,e: print e

print len(f),f.keys(),len(f[f.keys()[0]])

pl.pcolor(corrcoef(array(f[f.keys()[0]])))
pl.colorbar()
pl.show()
