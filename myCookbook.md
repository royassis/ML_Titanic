
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

##### 2 Pivoting
```
xmpl = male_frame[["Sex","age_grp","Survived"]]
xmpl = xmpl.dropna()
xmpl.pivot("Sex","age_grp","Survived")


dframe.groupby("Sex").get_group('female').head()
```



#### Links
https://pandas.pydata.org/pandas-docs/stable/visualization.html vis