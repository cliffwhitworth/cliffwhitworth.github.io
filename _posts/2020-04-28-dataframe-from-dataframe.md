---
layout: post
title: "Dataframe from Dataframe"
date: 2020-04-28 15:53:00 
comments: false
---

Create a new dataframe from existing dataframe

```
# Keep the first column
turnover_df = pd.DataFrame(data=turnover.iloc[:, 0].values, columns=['Turnover_Date'])
```

Add columns based on some text in columns
```
QLD_cols = [col for col in turnover.columns.values if 'Queensland' in col and 'Clothing retailing' in col]
turnover_df[turnover[QLD_cols].columns] = turnover[QLD_cols]
print(turnover_df.shape)
turnover_df.head()
```

Rename columns
```
# Rename columns
t = turnover_df.rename(columns={turnover_df.columns[1]: 'NSW',
                               turnover_df.columns[2]: 'QLD',
                               turnover_df.columns[3]: 'VIC',
                               turnover_df.columns[4]: 'WA'})

t.head()
```

Save df to csv
```
turnover_df.to_csv(r'./turnover.csv', index = False, header = True)
```
