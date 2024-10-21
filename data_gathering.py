import pandas as pd
import openpyxl
import numpy as np
import matplotlib.pyplot as plt
import time

df = pd.read_excel("D:\IA\Data cat personality and predation Cordonnier et al.xlsx")
races = {}
for i in df["Race"]:
    races[i] = races.get(i, 0) + 1
categories = {}
for i in df:
    if i != 'Plus' and i != 'Row.names' and i != 'Horodateur':
        categories[i] = {}
        for j in df[i]:
            categories[i][j] = categories[i].get(j, 0) + 1
race_data = {}
for k in races:
    race_data[k] = {}
    for i in df:
        if i != 'Plus' and i != 'Row.names' and i != 'Horodateur':
            race_data[k][i] = {}
            for j in df[i][df['Race'] == k]:
                race_data[k][i][j] = race_data[k][i].get(j, 0) + 1
values = np.array(list(races.values()))
names = list(races.keys())
correlations = {}
for k in races:
    correlations[k] = {}
    for i in df:
        if i != 'Plus' and i != 'Row.names' and i != 'Horodateur' and i != 'Race':
            for j in race_data[k][i].keys():
                if race_data[k][i][j] / sum(race_data[k][i].values()) > 2/len(race_data[k][i].values()):
                    correlations[k][i] = {}
                    correlations[k][i][j] = round(race_data[k][i][j] / sum(race_data[k][i].values()) * 100, 3)
print(len(correlations.keys()))
for i in correlations.keys():
    print(i)
    print(correlations[i])
# Generates histograms at dataset level
'''for i in categories:
    heights, bins, a = plt.hist(df[i], bins=len(df[i]), color=['blue'], edgecolor='black', rwidth=0.8)
    plt.title(i)
    width = bins[1] - bins[0]
    for j in range(len(df[i])):
        plt.text(bins[j] + width / 2, heights[j] + 1, str(heights[j]), ha='center')
    plt.show()'''
# Generates histograms at race level
for k in races:
    for i in df:
        if i != 'Plus' and i != 'Row.names' and i != 'Horodateur' and i != 'Race':
            heights, bins, a = plt.hist(df[i][df['Race'] == k], bins=len(race_data[k][i].keys()), color=['blue'], edgecolor='black', rwidth=0.8)
            plt.title(k + i)
            width = bins[1] - bins[0]
            bin_centers = (bins[:-1] + bins[1:]) / 2
            plt.xticks(bin_centers, race_data[k][i].keys())
            for j in range(len(race_data[k][i].keys())):
                plt.text(bins[j] + width / 2, heights[j] + 1, round(heights[j] / races[k] * 100, 3), ha='center')
            plt.show()
