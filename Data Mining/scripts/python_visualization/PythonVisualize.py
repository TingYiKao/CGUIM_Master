# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 11:48:23 2020

@author: Chun-Hsien Chen
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('/Users/kao900531/Desktop/iris.csv')
df.head()
df.describe()

sns.distplot(df['sepalwidth'])
plt.show()

# Remove NaN by dropna()
sns.distplot(df['sepalwidth'].dropna())
plt.savefig('/Users/kao900531/Desktop/irisDistplot_sepalwidth.png')
"""
filename='irisDistplot_sepalwidth.png'
ff=%utils
plt.savefig('%s/%s' % (utils.get_save_image_path(), filename))
"""
plt.show()

# Setting bin #
sns.distplot(df['sepalwidth'].dropna(), bins=20)
plt.savefig('/Users/kao900531/Desktop/irisDistplot_sepalwidth.png')
plt.show()

# Scatter Plot
plt.scatter(df['petalwidth'], df['petallength'], marker='o', c="m")
plt.xlabel("petalwidth")
plt.ylabel("petallength")
plt.savefig('/Users/kao900531/Desktop/irisScatterplot.png')
plt.show()

# Scatter Plot with regression
sns.regplot(x="petalwidth", y="petallength", data=df, marker=".")
plt.savefig('/Users/kao900531/Desktop/RegressionPlot.png')
plt.show()

# Scatter Plot with color based on class label
df1=df[df['class']=="Iris-setosa"]
df2=df[df['class']=="Iris-versicolor"]
df3=df[df['class']=="Iris-virginica"]
plt.scatter(df1['petalwidth'], df1['petallength'], marker='^', c="m")
plt.scatter(df2['petalwidth'], df2['petallength'], marker='o', c="b")
plt.scatter(df3['petalwidth'], df3['petallength'], marker='+', c="g")
plt.xlabel("petalwidth")
plt.ylabel("petallength")
plt.legend(loc='upper left')
plt.savefig('/Users/kao900531/Desktop/irisColorScatterPlot.png')
plt.show()

# Pair Plot with color based on class label
sns.pairplot(df)
plt.show()

sns.pairplot(df, hue="class")
plt.savefig('/Users/kao900531/Desktop/irisPairPlot.png')
plt.show()

# Bar Chart
dfSum = df.groupby('class').size().reset_index(name='sampleNo')
sns.barplot(x=dfSum['class'],y=dfSum['sampleNo'])
plt.savefig('/Users/kao900531/Desktop/irisBarPlot.png')
plt.show()

#Box Plot
sns.boxplot(x="class", y="petalwidth", data=df)
plt.savefig('/Users/kao900531/Desktop/irisBoxPlot.png')
plt.show()

# Pie Chart
dfSum = df.groupby('class').size().reset_index(name='sampleNo')
plt.pie(x=dfSum['sampleNo'],labels=dfSum['class'],autopct='%1.1f%%', 
        shadow=False, explode = (0, 0.1, 0))
plt.axis('equal')
plt.title('Sample Percentage of Each Class')
plt.savefig('irisPiePlot.png')
plt.show()

dfHeatMap = df[['sepallength','sepalwidth','petallength','petalwidth']]
#dfHeatMap = (dfHeatMap - dfHeatMap.mean())/dfHeatMap.std()
dfHeatMap = dfHeatMap.transpose()
dfHeatMap
ax=sns.heatmap(dfHeatMap)
plt.savefig('irisHeatMap1.png')

# correlation 計算
corr = df[['sepallength','sepalwidth','petallength','petalwidth']].corr()
plt.figure(figsize=(5,5))
ax=sns.heatmap(corr, square=True, annot=True, cmap="RdBu_r")
plt.savefig('irisHeatMap2.png')





