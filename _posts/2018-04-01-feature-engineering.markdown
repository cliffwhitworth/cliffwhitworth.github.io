---
layout: post
title:  "Feature Engineering"
date:   2018-04-01 07
categories: More
---
<br />
Hat tip to
<a href="https://www.udemy.com/user/soledad-galli/">
Soledad Galli
</a>
<p>Replace values based on training set for both training and test sets</p>

<br />
<h4>Missing Values</h4>
<p>Missing Completely at Random, Missing at Random, Missing Not at Random</p>

{% highlight ruby %}

# Sample data
dataframe = pd.read_csv(
    'some.csv', usecols=use_cols).sample(
        10000, random_state=1)

# Inspet Values
dataframe.column.dropna().unique()

# Values distribution
dataframe.column.hist(bins=100)

# Creating a binary variable based on values in another column
dataframe['binary_column'] = np.where(dataframe.column.isin(['Some Value']), 1, 0)

# Count and plot values
dataframe['some column'].value_counts().plot.bar()

# Percentage of count values
dataframe.Column.value_counts() / len(dataframe)

# Count isnull per column
dataframe.isnull().sum()

# Find precentages
dataframe.groupby(['Target'])['Feature'].mean()

# Missing count and percentages based on missing
missing = len(dataframe[dataframe.Column.isnull()])
dataframe[dataframe.Column.isnull()].groupby(['ColumnToCompare'])['Column'].count().sort_values() / missing

{% endhighlight %}

<br />
<h4>Outliers</h4>

{% highlight ruby %}

# Simulate outlier
import seaborn as sns
sns.distplot(data.Age.fillna(some outlier value))

# Show distribution of values
fig = dataframe.Column.hist(bins=100)
fig.set_title('Distribution')
fig.set_xlabel('X')
fig.set_ylabel('Y')

# Boxplot of distribution
fig = dataframe.boxplot(column='Column')
fig.set_title('Boxplot')
fig.set_xlabel('X')
fig.set_ylabel('Y')

# Describe and define IQR, Lower_fence, Upper_fence
dataframe.Column.describe()
IQR = dataframe.Column.quantile(0.75) - dataframe.Column.quantile(0.25)
Lower_fence = dataframe.Column.quantile(0.25) - (IQR * 1.5) # sometimes 3 is used
Upper_fence = dataframe.Column.quantile(0.75) + (IQR * 1.5) # instead of 1.5

# Add a column with redefined outliers and run algorithms to see impact
# Upper boundary values, capping, top-coding

{% endhighlight %}

<br />
<h4>Labels</h4>

{% highlight ruby %}

# Reduce cardinality of variable
dataframe['New Column'] = dataframe['Column'].astype(str).str[0]


{% endhighlight %}

<br />
<h4>Rare Values</h4>

{% highlight ruby %}

# Show relationships
relationships = pd.Series(dataframe['Feature'].value_counts() / len(dataframe)).reset_index
relationships.columns = ['Feature', 'Feature_Percent']

# Show means and merge with relationships
means = data.groupby(['Feature'])['NonBinaryTarget'].mean().reset_index()
relationships = relationships.merge(means, on='Feature', how='left')

# Regroup rare labels
dataframe[dataframe >= 0.1].index
group = {
    l: ('rare' if l not in dataframe[dataframe >= 0.1].index else l)
    for l in dataframe.index
}
dataframe['Grouped_Feature'] = dataframe['Feature'].map(group)

# Lables unique to train or test sets
unique_train = [
    x for x in X_train['Feature'].unique() if x not in X_test['Feature'].unique()
]

{% endhighlight %}

<br />
<h4>Missingness</h4>
<p>Capture the importance of missingness by creating an additional variable indicating whether the data was missing for that observation (1) or not (0)</p>

{% highlight ruby %}

X_train['Feature_NaN'] = np.where(X_train['Feature'].isnull(), 1, 0)

{% endhighlight %}

<br />
<h4>Imputation</h4>
<p>Imputation alters variance of original distribution and should be done over the training set, and then propagated to the test set.</p>

{% highlight ruby %}

# Missing values
dataframe.isnull().mean()

# Separate into training and testing set
# Impute mean and zero to training set
# Mean if Gaussian, median if not
X_train[variable+' median'] = X_train[variable].fillna(train_median)
X_train[variable+' zero'] = X_train[variable].fillna(0)
X_test[variable+' median'] = X_test[variable].fillna(train_median)

# Random sample
# Use random_state to repeat values
X_train[variable].dropna().sample(X_train[variable].isnull().sum(), random_state=0)

# End of distribution
X_train.Column.mean()+1.5*X_train.Column.std()

# Arbitrary value
# Adding a missingness category

{% endhighlight %}

<br />
<h4>Encoding</h4>

{% highlight ruby %}

# One hot encoding
# Ordinal encoding
# Count or frequency encoding
# Target guided encoding
# Mean encoding
# Weight of evidence encoding
# Probabiliy ration encoding
prob_df = X_train.groupby(['Feature'])['Target'].mean()
prob_df = pd.DataFrame(prob_df)
prob_df['FeatureCompliment'] = 1-prob_df.Feature
prob_df['ratio'] = prob_df.Feature/prob_df.FeatureCompliment
label_ratios = prob_df['ratio'].to_dict()
X_train['Feature_ratio'] = X_train.Feature.map(label_ratios)
X_test['Feature_ratio'] = X_test.Feature.map(label_ratios)

{% endhighlight %}

<br />
<h4>Misc Functions</h4>

{% highlight ruby %}

# Convert series to data to_frame
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.to_frame.html
Series.to_frame())

{% endhighlight %}
