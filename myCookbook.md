
# My cookbook
##### 1 How to make bins and categories


```
dframe_sorted =dframe_nona.sort_values("Age", ascending=False)
age_cat = pd.cut(dframe_sorted["Age"].values.tolist(), 10)

age_to_agerank_arr= np.arange(len(age_cat.categories)) #0-9 arr
age_to_agerank_arr = np.linspace(1 ,len(age_cat.categories),len(age_cat.categories) )

navals = dframe.groupby(['Age', pd.cut(dframe.Age, age_cat.categories)])


navals.size().unstack()
```

##### 2 Pivoting stacking unstacking
```
xmpl = male_frame[["Sex","age_grp","Survived"]]
xmpl = xmpl.dropna()
xmpl.pivot("Sex","age_grp","Survived")

dframe.groupby("Sex").get_group('female').head()
```


##### 3 Detect outliers
Code taken from [this stackoverflow entry](https://stackoverflow.com/questions/35827863/remove-outliers-in-pandas-dataframe-using-percentiles/35828995)

If one want to slice out outliers he can only do so by first, make a percentile dataframe as below.

```
low = .05
high = .95
quant_df = filt_df.quantile([low, high])
```

and then use apply on a the current dataframe to slice out relevent values

(important mark- when applying a series one applies each value seperatly, in order to apply the whole series on need to define it as a dataframe as such :[[df.column]])
```
df = df.apply(lambda x: x[(x>quant_df.loc[low,x.name]) &
                                    (x < quant_df.loc[high,x.name])])
```


#### Links
Visualisation - https://pandas.pydata.org/pandas-docs/stable/visualization.html

Machine learning tutorial - https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy

Bsaic Machine learning tutorial - https://machinelearningmastery.com/machine-learning-in-python-step-by-step/

Group By: split-apply-combine - https://pandas.pydata.org/pandas-docs/stable/groupby.html

Merge, join, and concatenate - https://pandas.pydata.org/pandas-docs/stable/merging.html#