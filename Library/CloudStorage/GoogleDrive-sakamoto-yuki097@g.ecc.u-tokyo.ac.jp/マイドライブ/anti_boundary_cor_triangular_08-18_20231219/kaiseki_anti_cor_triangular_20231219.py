import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def isSame(x,y,eps=1e-4):
    return (np.abs(x-y) < eps)

path = './correlation_*.csv'
flist = glob.glob(path)
# df_result = pd.DataFrame(index=[], columns=["r", "K", "ave_payoff"])
number_r = 101
r_list = np.linspace(0.8, 1.8, number_r)
number_K = 51
K_list = np.logspace(-3., 2., number_K)
array_result = np.zeros((number_K, number_r))


for file in flist:
    print(file)
    df = pd.read_csv(file)
    r = df.iloc[0]["r"]
    # K = df.iloc[0]["K"]
    w = df.iloc[0]["w"]
    df = df[["K", "iterations", "correlation"]]
    iterations = df["iterations"].max()
    
    for id_r in range(len(r_list)):
        if not isSame(r_list[id_r], r):
            continue
        print("r =",r)
        for id_K in range(len(K_list)):
            K = K_list[id_K]
            array_result[id_K, id_r] = df[isSame(df["K"], K)]["correlation"].mean()
            print(r, K, df["correlation"].mean())


#np.set_printoptions(precision=1)
index_list=["{:.4f}".format(i) for i in K_list]
colums_list=["{:.4f}".format(i) for i in r_list]
df_result = pd.DataFrame(data=array_result, index=index_list, columns=colums_list)
df_result.to_csv("./df_anti_cor_triangular_20231219.csv")
# for index in index_list:
#     if index[-1] != "0":
#         index = ""
# for colum in colums_list:
#     if not colums_list[-1] in ["0", "5"]:
#         colum = ""
# sns.heatmap(df_result,xticklabels=index_list, yticklabels=colums_list)
# plt.xlabel("Cost-to-Benefit ave_payoff")
# plt.ylabel("温度係数")
# plt.show()
print(0)
print(1)