# coding=utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("-----------------Creating a Series by passing a list of values, letting pandas create a default integer index---------------------")
s = pd.Series([1,3,5,np.nan,6,8])
print(s)

print("-----------------Creating a DataFrame by passing a numpy array, with a datetime index and labeled columns---------------------")
dates = pd.date_range('2017-03-23',periods=6)
print(dates)
df = pd.DataFrame(np.random.rand(6,4),index=dates,columns=list('ABCD'))
print(df)
print(df.dtypes)

print("-----------------Creating a DataFrame by passing a dict of objects that can be converted to series-like.---------------------")
df2 = pd.DataFrame({ 'A' : 1.,'B' : pd.Timestamp('20130102'),'C' : pd.Series(1,index=list(range(4)),dtype='float32'),'D' : np.array([3] * 4,dtype='int32'),'E' : pd.Categorical(["test","train","test","train"]),'F' : 'foo' })
print(df2)

print("-----------------Having specific dtypes---------------------")
print(df2.dtypes)

print("-----------------See the top & bottom rows of the frame---------------------")
print(df.head())
print(df.tail(3))

print("-----------------Display the index, columns, and the underlying numpy data---------------------")
print(df.index)
print(df.values)
print(df.columns)

print("-----------------Describe shows a quick statistic summary of your data---------------------")
print(df.describe())

print("-----------------Transposing your data---------------------")
print(df.T)

print("-----------------Sorting by index---------------------")
print(df.sort_index(axis=1,ascending=False))

print("-----------------Sorting by values---------------------")
print(df.sort_values(by="B",ascending=False))

print("-----------------selection---------------------")
print("-----------------Selecting a single column, which yields a Series, equivalent to df.A---------------------")
print(df['A'])

print("-----------------Selecting via [], which slices the rows.---------------------")
print(df[0:3])
print(df['2017-03-24':'2017-03-27'])

print("-----------------Selection by Label---------------------")
print("-----------------For getting a cross section using a label---------------------")
print(df)
print(dates)
print(df.loc[dates[0]])
print("-----------------Selecting on a multi-axis by label---------------------")
print(df.loc['2017-03-24':'2017-03-27',['A','B']])

print("-----------------Showing label slicing, both endpoints are included---------------------")
print(df.loc['20130102':'20130104',['A','B']])

print("-----------------For getting fast access to a scalar (equiv to the prior method)---------------------")
print(df.at[dates[0],'A'])

print("-----------------Selection by Position---------------------")
print("-----------------Select via the position of the passed integers---------------------")
print(df.iloc[3])

print("-----------------By integer slices, acting similar to numpy/python---------------------")
print(df.iloc[3:5,0:2])

print("-----------------For getting a value explicitly---------------------")
print(df.iloc[0,0])
print(df.iat[0,0])

print("-----------------Boolean Indexing---------------------")
print("-----------------Using a single column’s values to select data---------------------")
print(df[df.A > 0.5])

print("-----------------A where operation for getting.---------------------")
print(df[df > 0.5])

print("-----------------Using the isin() method for filtering:---------------------")
df2 = df.copy()
df2['E'] = ['one', 'one','two','three','four','three']
print(df2)

print(df2[df2['E'].isin(['two','four'])])

print("-----------------Setting---------------------")
print("-----------------Setting a new column automatically aligns the data by the indexes---------------------")
s1 = pd.Series([1,2,3,4,5,6],index=pd.date_range('2017-03-23',periods=6))
print(s1)
df['F'] = s1
print(df)

print("-----------------Setting values by label---------------------")
df.at[dates[0],'A'] = '666'
print(df)
print("-----------------Setting values by position---------------------")
df.iat[0,0] = '555'
print(df)
print("-----------------Setting by assigning with a numpy array---------------------")
df.loc[:,'D'] = np.array([5]*len(df))
print(df)

print("-----------------A where operation with setting.---------------------")
df2 = df.copy()
df2[df2 > 0] = -df2
print(df2)

print("-----------------Missing Data---------------------")
print("-----------------pandas primarily uses the value np.nan to represent missing data. It is by default not included in computations. See the Missing Data section---------------------")
df1 = df.reindex(index=dates[0:4],columns=list(df.columns) + ['E'])
df1.loc[dates[0]:dates[1],'E'] = 1
print(df1)

print("-----------------To drop any rows that have missing data.---------------------")
print(df1.dropna(how='any'))

print("-----------------Filling missing data---------------------")
print(df1.fillna(value=5))

print("-----------------To get the boolean mask where values are nan---------------------")
print(pd.isnull(df1))

print("-----------------Operations---------------------")
print("-----------------Operations in general exclude missing data.Performing a descriptive statistic---------------------")

print(df.mean)
print("-----------------Same operation on the other axis---------------------")
print("-----------------axis 0---------------------")
print(df.mean(0))
print("-----------------axis 1---------------------")
print(df.mean(1))

print("-----------------Operating with objects that have different dimensionality and need alignment. In addition, pandas automatically broadcasts along the specified dimension.---------------------")
s = pd.Series([1,3,5,np.nan,6,8],index=dates).shift(2)
print(s)
print(df.sub(s,axis='index'))

print("-----------------Apply---------------------")
print(df)
print("-----------------df.apply(np.cumsum)---------------------")
print(df.apply(np.cumsum))

print("-----------------df.apply(lambda x: x.max() - x.min())---------------------")
print(df.apply(lambda x: x.max() - x.min()))

print("-----------------Histogramming---------------------")
s = pd.Series(np.random.randint(0, 7, size=10))
print(s)
print(s.value_counts())

print("-----------------String Methods---------------------")
s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
print(s.str.lower())


print("-----------------Merge---------------------")
print("-----------------Concat---------------------")
df = pd.DataFrame(np.random.randn(10, 4))
print(df)
print("-----------------break it into pieces---------------------")
pieces = [df[:3], df[3:7], df[7:]]
pd.concat(pieces)
print(pieces)

print("-----------------Join---------------------")
print("-----------------SQL style merges.---------------------")
left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})
print(left)
print(right)
print(pd.merge(left,right,on='key'))

print("-----------------Another example that can be given is:---------------------")
left = pd.DataFrame({'key': ['foo', 'bar'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'bar'], 'rval': [4, 5]})
print(left)
print(right)
print(pd.merge(left,right,on='key'))

print("-----------------Append---------------------")
print("-----------------Append rows to a dataframe.---------------------")
df = pd.DataFrame(np.random.randn(8, 4), columns=['A','B','C','D'])
print(df)
s = df.iloc[3]
df.append(s, ignore_index=True)
print(df)
# @TODO 结果不对,dataFrame append series无结果

print("-----------------Grouping---------------------")
# By “group by” we are referring to a process involving one or more of the following steps
#
# Splitting the data into groups based on some criteria
# Applying a function to each group independently
# Combining the results into a data structure
df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar','foo', 'bar', 'foo', 'foo'],'B' : ['one', 'one', 'two', 'three','two', 'two', 'one', 'three'],'C' : np.random.randn(8),'D' : np.random.randn(8)})
print(df)

print("-----------------Grouping and then applying a function sum to the resulting groups.---------------------")
print(df.groupby('A').sum())

print("-----------------Grouping by multiple columns forms a hierarchical index, which we then apply the function.---------------------")
print(df.groupby(['A','B']).sum())


print("-----------------Reshaping---------------------")
print("-----------------Stack---------------------")
tuples = list(zip(*[['bar', 'bar', 'baz', 'baz','foo', 'foo', 'qux', 'qux'],['one', 'two', 'one', 'two','one', 'two', 'one', 'two']]))
print(tuples)
index = pd.MultiIndex.from_tuples(tuples, names=['first', 'second'])
print(index)
df = pd.DataFrame(np.random.randn(8, 2), index=index, columns=['A', 'B'])
df2 = df[:4]
print(df2)

print("-----------------The stack() method “compresses” a level in the DataFrame’s columns.---------------------")
stacked = df2.stack()
print(stacked)

print("----------------- the inverse operation of stack() is unstack(), which by default unstacks the last level:---------------------")
print(stacked.unstack())
print(stacked.unstack(1))
print(stacked.unstack(0))

print("-----------------Pivot Tables---------------------")
df = pd.DataFrame({'A' : ['one', 'one', 'two', 'three'] * 3,'B' : ['A', 'B', 'C'] * 4,'C' : ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 2,'D' : np.random.randn(12),'E' : np.random.randn(12)})
print(df)
print("-----------------We can produce pivot tables from this data very easily---------------------")
print(pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C']))

print("-----------------Time Series---------------------")
rng = pd.date_range('1/1/2012', periods=100, freq='S')
print(rng)
ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
print(ts)
print(ts.resample('5Min').sum())

print("-----------------Time zone representation---------------------")
rng = pd.date_range('3/6/2012 00:00', periods=5, freq='D')
ts = pd.Series(np.random.randn(len(rng)), rng)
print(ts)
ts_utc = ts.tz_localize('UTC')
print(ts_utc)

print("-----------------Convert to another time zone---------------------")
print(ts_utc.tz_convert('Asia/Shanghai'))

print("-----------------Converting between time span representations---------------------")
rng = pd.date_range('1/1/2012', periods=5, freq='M')
print(rng)
ts = pd.Series(np.random.randn(len(rng)), index=rng)
print(ts)
ps = ts.to_period()
print(ps)
print(ps.to_timestamp())

# Converting between period and timestamp enables some convenient arithmetic functions to be used. In the following example, we convert a quarterly frequency with year ending in November to 9am of the end of the month following the quarter end:

print("-----------------Converting between period and timestamp---------------------")
prng = pd.period_range('1990Q1', '2000Q4', freq='Q-NOV')
print(prng)
ts = pd.Series(np.random.randn(len(prng)), prng)
print(ts)
ts.index = (prng.asfreq('M', 'e') + 1).asfreq('H', 's') + 9
print(ts.index)
print(ts.head())

print("-----------------selection---------------------")
print("-----------------selection---------------------")
print("-----------------selection---------------------")