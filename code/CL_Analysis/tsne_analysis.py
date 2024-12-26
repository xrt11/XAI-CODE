import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import manifold



parser = argparse.ArgumentParser()
parser.add_argument('--latentAB_save_path',type=str,default='/code/CL_Analysis/results/testAB_CL_codes_extraction_results.csv')  ###where the class-associated codes extracted from the images and the images label are recorded
parser.add_argument('--tsne_save_path',type=str,default='/results/tsne_analysis_result.png')


opts = parser.parse_args()

latentAB_save_path=opts.latentAB_save_path

tsne_save_path=opts.tsne_save_path

df = pd.read_csv(latentAB_save_path)

feature_names = [c for c in df.columns if c not in ["image_name","label"]]
X = np.array(df[feature_names])
y = np.array(df["label"])


'''t-SNE'''
tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
X_tsne = tsne.fit_transform(X)

print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))


x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)  # normalize

plt.figure(figsize=(8, 8))
for i in range(X_norm.shape[0]):
    plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]),
             fontdict={'weight': 'bold', 'size': 9})

label_AB_list=[]
for i in range(y.shape[0]):
    if y[i] not in label_AB_list:
        label_AB_list.append(y[i])

X_norm_x_mean=[]
X_norm_y_mean=[]
X_norm_label_num=[]
for i in range(len(label_AB_list)):
    X_norm_x_mean.append(0)
    X_norm_y_mean.append(0)
    X_norm_label_num.append(0)
for i in range(X_norm.shape[0]):
    X_norm_x_mean[label_AB_list.index(y[i])]=X_norm_x_mean[label_AB_list.index(y[i])]+X_norm[i, 0]
    X_norm_y_mean[label_AB_list.index(y[i])] = X_norm_y_mean[label_AB_list.index(y[i])] + X_norm[i, 1]
    X_norm_label_num[label_AB_list.index(y[i])]=X_norm_label_num[label_AB_list.index(y[i])]+1
for i in range(len(label_AB_list)):
    X_norm_x_mean[i]=X_norm_x_mean[i]/X_norm_label_num[i]
    X_norm_y_mean[i] = X_norm_y_mean[i] / X_norm_label_num[i]
for i in range(len(label_AB_list)):
    plt.text(X_norm_x_mean[i], X_norm_y_mean[i], str(label_AB_list[i]), color=plt.cm.Set1(label_AB_list[i]),
                 fontdict={'weight': 'bold', 'size': 40})


plt.axis('off')
plt.savefig(tsne_save_path)
plt.close()




