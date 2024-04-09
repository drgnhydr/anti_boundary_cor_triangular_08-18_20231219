import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import ScalarFormatter

number_r = 101
r_list = np.linspace(0.8, 1.8, number_r)
# number_K = 51
# K_list = np.linspace(0., 5., number_K)[::-1]


#np.set_printoptions(precision=1)
# index_list=["{:.2f}".format(i) for i in K_list]
col_list=["{:.2f}".format(i) for i in r_list]
files = glob.glob("df*.csv")
file = files[0]
df_result = pd.read_csv(file,index_col=0).iloc[::-1] #reversed
if "0.0750" in df_result.columns:
    df_result = df_result.drop(columns=["0.0750","0.0800"])
# for index in index_list:
#     if index[-2] != "0":
#         index = ""
for i, col in enumerate(r_list):
    if not i in [0,20,40,60,80,100,120]:
        col_list[i] = ""

    else:
        col_list[i] = "{:.01f}".format(r_list[i])
fs = 20
fs_text=24
fig = plt.figure(figsize=[6.4,7.5])
ax = fig.add_subplot(111)
ax.tick_params(labelsize=fs)
fig.subplots_adjust(top=0.98, bottom=0.23, left=0.13,right=0.96)
sns.heatmap(df_result, xticklabels=col_list, yticklabels=10, ax=ax, cbar_kws={'label':'Correlation of Neighboring Pairs', 'location':'top'})
# ax.set_xticklabels(col_list,rotation=90)
ax.xaxis.set_tick_params(rotation=90)
# ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
# ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
# plt.gca().get_xaxis().get_major_formatter().set_powerlimits([-3,3])
# ax.set_xticks([2*i for i in range(number_r//2 + 1)])
# ax.set_xticklabels(r_list[::2],rotation=0)
# ax.set_xticklabels(r_list)
ax.figure.axes[-1].xaxis.label.set_size(fs_text)
ax.figure.axes[-1].tick_params(labelsize=20)
ax.set_xlabel("$r$",fontsize=fs_text)
ax.set_ylabel("Temperature",fontsize=fs_text)

# ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

#plt.show()
# title = '(h) $k=24$'
# fig.text(0.40, 0.02, title, fontsize=fs_text)
fig.savefig("heatmap_revised_cor_" + file.split(".")[0] + ".png")
print(0)
print(1)