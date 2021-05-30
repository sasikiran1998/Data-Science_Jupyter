#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
poke=pd.read_csv('sem_mark.csv')
print(poke)


# In[ ]:





# In[2]:


import pandas as pd
df_xlsx=pd.read_excel('sems.xlsx')
print(df_xlsx.head(2))


# In[4]:


import pandas as pd
poke=pd.read_csv('log2.csv')
print(poke.head(3))
print(poke.tail(3))


# In[1]:


import pandas as pd
res=pd.read_csv('Future50.csv')
print(res)


# In[2]:


res.sort_values(['Restaurant','Location'],ascending=[0,0])


# In[3]:


res.sort_values(['Restaurant'],ascending=[0])


# In[4]:


res.sort_values(['Restaurant'],ascending=[0])
res.head(5)


# In[5]:


res.sort_values(['Restaurant'],ascending=[0])
res.head(10)


# In[6]:


res.sort_values(['Restaurant'],ascending=[1])
res.head(5)


# In[7]:


res.sort_values(['Restaurant','Location'],ascending=[0,1])


# In[8]:


res.sort_values(['Restaurant','Location'],ascending=[0,0])


# In[9]:


res.sort_values(['Restaurant'],ascending=0)


# In[10]:


res.sort_values(['Location'],ascending=1)


# In[11]:


res.sort_values(['Restaurant','Location'],ascending=[1,0])


# In[12]:


res['Total']=res['Sales']+res['Units']+res['Unit_Volume']


# In[13]:


res['Total']=res['Sales']+res['Units']+res['Unit_Volume']
res.head(10)


# In[14]:


res['Total']=res.iloc[:,4,9]. sum(axis=1)
res.head(10)


# In[15]:


res['Total']=res.iloc[:,4:9]. sum(axis=1)
res.head(10)


# In[16]:


res['Total']=res.iloc[:,4:9]. sum(axis=0)
res.head(10)


# In[17]:


res['Total']=res.iloc[:,4:10]. sum(axis=1)
cols=list(res.columns)
res=res[cols[0:4]+[cols[-1]]+cols[4:12]]
res.head(10)


# In[18]:


res=res.drop(columns=['Total'])
res.head(10)


# In[19]:


res.loc[(res['Location']=='New York, N.Y.') & (res['Franchising']=='No')]
res.to_excel('future_new.xlsx',index=False)


# In[20]:


res.loc[(res['Location']=='New York, N.Y.')]
res.to_excel('future_new.xlsx',index=False)


# In[21]:


pip install openpyxl


# In[22]:


import openpyxl


# In[23]:


res.loc[(res['Location']=='New York, N.Y.')]
res.to_excel('future_new.xlsx',index=False)


# In[26]:


res.loc[(res['Sales']>30)]
res.to_excel('future_new.xlsx',index= False)


# In[27]:


res.loc[(res['Units']==105)]
res.to_excel('future_new.xlsx',index= False)


# In[28]:


new_res=res.loc[(res['Location']=='New York, N.Y.') & (res['Franchising']=='No')]
new_res.to_excel('new.xlsx',index=False)


# In[29]:


print(new_res)


# In[30]:


import re
res.loc[res['Location']. str.contains('Columbus, Ohio|New York, N.Y.',regex=True)]


# In[31]:


res = res[res['Restaurant'].str.contains('^t[a-z]*t', regex = True)]


# In[32]:


res


# In[33]:


import re
res.loc[res['Location']. str.contains('Columbus, Ohio|New York, N.Y.',regex=True)]


# In[34]:


import re
res.loc[res['Location']. str.contains('Columbus, Ohio|New York, N.Y.',regex=True)]


# In[35]:


import re
res.loc[res['Location']. str.contains('Columbus, Ohio|New York, N.Y.',regex=True)]


# In[36]:


print(res)


# In[37]:


import pandas as pd
res=pd.read_csv('Future50.csv')
print(res)


# In[40]:


import re
res.loc[res['Location']. str.contains('Columbus, Ohio|New York, N.Y.',regex=True)]


# In[39]:


import re
res.loc[res['Restaurant'].str.contains('^The[a-z]*',flags=re.I,regex=True)]


# In[41]:


import seaborn as sn


# In[43]:


pip install seaborn


# In[44]:


import seaborn as sn


# In[45]:


res = sn.load_dataset("iris") 
sn.lineplot(x="sepal_length", y="sepal_width", data=data)


# In[47]:


import seaborn as sn
res = sn.load_dataset("iris") 
sn.lineplot(x="sepal_length", y="sepal_width", res=res)


# In[48]:


import seaborn as sn
res = sn.load_dataset("iris") 
sn.heatmap(iris.corr(),cmap="YlGnBu",linecolor='white',linewidths=1)


# In[49]:


import seaborn as sn
iris = sn.load_dataset("iris") 
sn.heatmap(iris.corr(),cmap="YlGnBu",linecolor='white',linewidths=1)


# In[50]:


res = sn.load_dataset("iris") 
  
# draw lineplot 
sn.lineplot(x="sepal_length", y="sepal_width", res=res)


# In[51]:


g=sn.pairplot(data,hue="class")


# In[52]:


sn.heatmap(iris.corr(),cmap="YlGnBu",linecolor='white',linewidths=1,annot=True)


# In[53]:


data.corr(method='pearson')


# In[54]:


res.corr(method='pearson')


# In[55]:


g=sn.pairplot(res,hue="class")


# In[56]:


g=sn.pairplot(data,hue="Franchising")


# In[57]:


g=sn.pairplot(res,hue="Franchising")


# In[58]:


sn.lineplot(x="total_bill", y="size", hue="sex", style="sex", res=res)


# In[59]:


sb.lineplot(x="total_bill", y="size", hue="Franchising", style="sex", data=data)


# In[60]:


sn.lineplot(x="total_bill", y="size", hue="Franchising", style="sex", data=data)


# In[61]:


sn.lineplot(x="total_bill", y="size", hue="Franchising", style="sex", res=res)


# In[62]:


sn.lineplot(x="Location", y="Restaurant", hue="Franchising", style="sex", res=res)


# In[63]:


sn.lineplot(x="Restaurant", y="Location", hue="Franchising",res=res)


# In[64]:


import seaborn as sb


# In[65]:


data = sb.load_dataset("iris") 
  
# draw lineplot 
sb.lineplot(x="sepal_length", y="sepal_width", data=data) 


# In[66]:


data = sb.load_dataset("tips") 

sb.lineplot(x="total_bill", y="size", hue="sex", style="sex", data=data) 


# In[67]:


sb.set()


# In[68]:


plt.plot(x, y)
plt.legend('ABCDEF', ncol=2, loc='upper left');


# In[69]:


import matplotlib.pyplot as plt
plt.plot(x, y)
plt.legend('ABCDEF', ncol=2, loc='upper left');


# In[70]:


import matplotlib.pyplot as plt
plt.style.use('classic')
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd


# In[71]:


rng = np.random.RandomState(0)
x = np.linspace(0, 10, 500)
y = np.cumsum(rng.randn(500, 6), 0)


# In[72]:


plt.plot(x, y)
plt.legend('ABCDEF', ncol=2, loc='upper left');


# In[73]:


import seaborn as sns
sns.set()


# In[74]:


plt.plot(x, y)
plt.legend('ABCDEF', ncol=2, loc='upper left');


# In[75]:


data = np.random.multivariate_normal([0, 0], [[5, 2], [2, 2]], size=2000)
data = pd.DataFrame(data, columns=['x', 'y'])

for col in 'xy':
    plt.hist(data[col], normed=True, alpha=0.5)


# In[76]:


for col in 'xy':
    sns.kdeplot(data[col], shade=True)


# In[77]:


sns.distplot(data['x'])
sns.distplot(data['y']);


# In[78]:


sns.kdeplot(data);


# In[79]:


with sns.axes_style('white'):
    sns.jointplot("x", "y", data, kind='kde');


# In[80]:


iris = sns.load_dataset("iris")
iris.head()


# In[81]:


sns.pairplot(iris, hue='species', size=2.5);


# In[82]:


sns.jointplot("total_bill", "tip", data=tips, kind='reg');


# In[ ]:





# In[83]:


planets = sns.load_dataset('planets')
planets.head()


# In[84]:


with sns.axes_style('white'):
    g = sns.factorplot("year", data=planets, aspect=2,
                       kind="count", color='steelblue')
    g.set_xticklabels(step=5)


# In[85]:


with sns.axes_style('white'):
    g = sns.factorplot("year", data=planets, aspect=4.0, kind='count',
                       hue='method', order=range(2001, 2015))
    g.set_ylabels('Number of Planets Discovered')


# In[ ]:




