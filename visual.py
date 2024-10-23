import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd
from sklearn import manifold
from sklearn.manifold import TSNE
from matplotlib.lines import Line2D

def get_fer_data(data_path="data_embed_npy.npy",
                 label_path="label_npu.npy"):
    """
	该函数读取上一步保存的两个npy文件，返回data和label数据
    Args:
        data_path:
        label_path:

    Returns:
        data: 样本特征数据，shape=(BS,embed)
        label: 样本标签数据，shape=(BS,)
        n_samples :样本个数
        n_features：样本的特征维度

    """
    data = np.load(data_path)
    # print(data.shape)
    label = np.load(label_path)
    print(label.shape)
    n_samples, n_features = data.shape

    return data, label, n_samples, n_features


# def get_fer_data(D1_data_path,D1_label_path,
#                  D2_data_path,D2_label_path):
#     """
# 	该函数读取上一步保存的两个npy文件，返回data和label数据
#     Args:
#         data_path:
#         label_path:
#
#     Returns:
#         data: 样本特征数据，shape=(BS,embed)
#         label: 样本标签数据，shape=(BS,)
#         n_samples :样本个数
#         n_features：样本的特征维度
#
#     """
#     D1data = np.load(D1_data_path)
#     D1label = np.load(D1_label_path)
#     D2data = np.load(D2_data_path)
#     D2label = np.load(D2_label_path)
#     data = np.concatenate((D1data,D2data), axis=0)
#     label = np.hstack((D1label , D2label))
#     print(data.shape)
#     print(label.shape)
#     n_samples, n_features = data.shape
#
#     return data, label, n_samples, n_features



color_map = ['r','darkorange','tan','y','g' ,'c' ,'deepskyblue' , 'b' ,'m' ,'pink']  # 14个类，准备14种颜色
# 设置散点形状
maker = ['o', 's', '^', 's', 'p', '*', '<', '>', 'D', 'd', ]


def plot_embedding_2D(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure(figsize=(5, 4),dpi=500)
    for i in range(data.shape[0]):
        # plt.plot(data[i, 0], data[i, 1],  marker=maker[label[i]], markersize=1, color=color_map[label[i]],alpha=0.65)
        # plt.scatter(data[i, 0],data[i, 1], cmap=plt.cm.Spectral, s=100, marker=maker[label[i]], c=color_map[label[i]], edgecolors=color_map[label[i]], alpha=1)
        plt.scatter(data[i, 0], data[i, 1],marker='o', c=color_map[label[i]] , edgecolors='black', linewidths=0.2,cmap=plt.cm.Spectral,s=10, alpha=1 )

        # print(label[i])
    plt.xticks([])
    plt.yticks([])

    # myHandle = [
    #     Line2D([], [], marker='o', color=color_map[0], markersize=10, linestyle='None'),
    #     Line2D([], [], marker='o', color=color_map[1], markersize=10, linestyle='None'),
    #     Line2D([], [], marker='o', color=color_map[2], markersize=10, linestyle='None'),
    #     Line2D([], [], marker='o', color=color_map[3], markersize=10, linestyle='None'),
    #     Line2D([], [], marker='o', color=color_map[4], markersize=10, linestyle='None'),
    #     Line2D([], [], marker='o', color=color_map[5], markersize=10, linestyle='None'),
    #     Line2D([], [], marker='o', color=color_map[6], markersize=10, linestyle='None'),
    #     Line2D([], [], marker='o', color=color_map[7], markersize=10, linestyle='None'),
    #     Line2D([], [], marker='o', color=color_map[8], markersize=10, linestyle='None'),
    #     Line2D([], [], marker='o', color=color_map[9], markersize=10, linestyle='None'),
    #     # Line2D([], [], marker='o', color=color_map[0], markersize=10, linestyle='None'),
    #
    # ]



    # plt.legend(handles= myHandle ,labels=['0','1', '2', '3', '4', '5','6','7','8','9'] ,loc='upper right')
    # plt.title(title)
    return fig


def main(title):
    data, label, n_samples, n_features = get_fer_data(data_path=f"data/CWRU/plot_Tsne/D2_{title}_data_embed_npy.npy",
    label_path=f"data/CWRU/plot_Tsne/D2_{title}_label_npu.npy" )  # 根据自己的路径合理更改

    # #
    # # 调用t-SNE对高维的data进行降维，得到的2维的result_2D，shape=(samples,2)
    # tsne_2D = TSNE(n_components=2, init='pca', random_state=0)
    # result_2D = tsne_2D.fit_transform(data)
    #
    # fig1 = plot_embedding_2D(result_2D, label, title)  # 将二维数据用plt绘制出来
    # # plt.title('(a)', fontsize=32, fontweight='normal', pad=20)
    # plt.savefig(f'data/CWRU/{title}',bbox_inches='tight',dpi=500)
    # fig1.show()
    # plt.pause(50)



if __name__ == '__main__':
    main('WCamba')


