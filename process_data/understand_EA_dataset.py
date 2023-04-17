import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import datetime

### Setup ###
df = pd.read_csv('EA_trimmed.csv')
datasetName = "EA"

# print(df.info())
# print(df.describe())

df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)
# print(df.head())

plotName = ["Close_Price", "Volume", "Nasdaq_100", "SP_500", "Ten_Year_Treasury_Rate", "PE_Ratio", "ROA", "GDP", "High_Price", "Low_Price", "Open_Price", "news", "Days_After_Last_News"]

### Date vs. Features Plot ###
# for i in plotName:
#     plt.figure(figsize=(11, 8)) # resizing the plot
#     df[i].plot()
#     name = i + " History"
#     plt.title(name) # title
#     plt.xlabel("Date") # x-label
#     plt.ylabel(i) # y-label
#     # f = plt.figure()
#     path = "plot/" + datasetName + "/single/" + name + ".jpg"
#     # f.savefig(path)
#     plt.savefig(path)



### Feature Boxplot ###
# for i in plotName:
#     plt.figure(figsize=(11, 8)) # resizing the plot
#     df.boxplot(column=[i])
#     name = i + " Boxplot"
#     plt.title(name)
#     path = "plot/" + datasetName + "/boxplot/" + name + ".jpg"
#     plt.savefig(path)



### Take a look on GDP ###
# print(df[df["GDP"] < 0]) # 187 rows hmm

### Generate a bar diagram for years ###
# print(df.loc["2010-01-04"])
# year_dic = {}
# for year in df.index.year.tolist():
#     year_dic[year] = year_dic.get(year, 0) + 1
# # print(year_dic)
# years = list(year_dic.keys())
# counts = list(year_dic.values())

# plt.figure(figsize=(11, 8)) # resizing the plot

# plt.bar(years, counts, color="blue", width =0.4)
# plt.xlabel("Years")
# plt.xticks(years)
# plt.ylabel("No. information")
# plt.title("No. of information in different year")
# name = "plot/" + datasetName + "/barplot/Year_Bar.jpg"
# plt.savefig(name)



### Generate a bar graph for news ###
# year_dic = {}
# print(df[df["Days_After_Last_News"] == 0].head())
# for year in df[df["Days_After_Last_News"] == 0].index.year.tolist():
#     year_dic[year] = year_dic.get(year, 0) + 1
# # print(year_dic)
# years = list(year_dic.keys())
# counts = list(year_dic.values())

# plt.figure(figsize=(11, 8)) # resizing the plot

# plt.bar(years, counts, color="blue", width =0.4)
# plt.xlabel("Years")
# plt.xticks(np.arange(2010,2020))
# plt.ylabel("No. news")
# plt.title("No. of news in different year")
# name = "plot/" + datasetName + "/barplot/Num_News_Bar.jpg"
# plt.savefig(name)



### Take a look on stock price ###
# cols = ['Open_Price', 'Close_Price', 'Volume', 'High_Price', 'Low_Price']
# axes = df[cols].plot(figsize=(11, 9), subplots = True)
# name = "plot/" + datasetName + "/Stock_Price.jpg"
# plt.savefig(name)



### Correlation Graph ###
# print(df.corr())

# define the mask to set the values in the upper triangle to True
# plt.figure(figsize=(16, 6))
# mask = np.triu(np.ones_like(df.corr(), dtype=np.bool_))
# heatmap = sns.heatmap(df.corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')
# heatmap.set_title('Triangle Correlation Heatmap', fontdict={'fontsize':18}, pad=16);
# # save heatmap as .png file
# # dpi - sets the resolution of the saved image in dots/inches
# # bbox_inches - when set to 'tight' - does not allow the labels to be cropped
# name = "plot/" + datasetName + '/correlation.png'
# plt.savefig(name, dpi=300, bbox_inches='tight')


### histogram ###
columns = df.columns
for col in columns:
    plt.figure(figsize=(11, 8)) # resizing the plot
    print("col:", col)
    df[col].hist()
    name = col + " Histogram"
    plt.title(name)
    path = "../plot/" + datasetName + '/histplot/' + col + ".jpg"
    plt.savefig(path)


### generate EA_selected dataset ###
'''
For all of our models, 
Response: Close_Price
Features: Volume, Nasdaq_100, SP_500, Ten_Year_Treasury_Rate, PE_Ratio, ROA, GDP, news
'''
# df = df[["Close_Price","Volume", "Nasdaq_100", "SP_500", "Ten_Year_Treasury_Rate", "PE_Ratio", "ROA", "GDP", "news"]]
# print(df.head())
df.to_csv("../dataset/EA_dataset.csv", index=True)




# plt.show()

