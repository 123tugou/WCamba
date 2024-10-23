import time
import torch
import torch.nn as nn
import utils
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import csv
import numpy as np
import numpy
from einops import rearrange
from tqdm.auto import tqdm
from joblib import dump, load
print(torch.cuda.is_available())

p = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_plot(file_name ='plot.csv' ,name='mamba',data=[]):
    with open(file_name,'a',newline='',encoding='utf-8') as file:
        writer = csv.writer(file)
        data.append(name)
        writer.writerow(data)

# 看下这个网络结构总共有多少个参数
def count_parameters(model):
    params = [p.numel() for p in model.parameters() if p.requires_grad]
    # for item in params:
    #     print(f'{item:>6}')
    print(f'______\n{sum(params):>6}')



def intraclass_covariance(test_data, label, sum_n, classes):
    """
    test_data:输出特征
    label:真实标签
    sum_n: 总测试个数
    classes:总的类别数
    """
    # Nk = sum_n//classes
    mk = []
    x = test_data
    y = label.reshape(sum_n)
    # print(y)
    Sb, Sw = 0, 0
    for i in range(classes):
        Nk = len(x[y == i])
        cur_mean = np.sum(x[y == i], axis=0) / Nk
        mk.append(cur_mean)
    m = np.mean(x, axis=0)

    for j in range(classes):
        Nk = len(x[y == j])
        Sb += Nk * np.linalg.norm((mk[j] - m))
        x_class = x[y == j]
        for k in range(Nk):
            Sw += np.linalg.norm((x_class[k] - mk[j]))

    J1 = Sb / Sw
    return Sb, Sw, J1
import numpy as np

def compute_J(encode_images, test_y):
    """
    compute the sb、 sw  and  J .
    :param encode_images: the images after encoding
    :param test_y: the label.
    :return:trace(sw) 、 trace(sb) and J
    """

    # 获取行数row(样本数)和列数column(类别数)
    _ , n_feature = encode_images.shape
    print(test_y.shape)
    row=test_y.shape
    column = 3

    test=np.zeros(shape=(column,300,n_feature))
    P=np.zeros(shape=(column,))
    m=np.zeros(shape=(column,n_feature))
    index= test_y
    print(index)
    sw=0
    #  类内散度矩阵sw+各类的均值m
    for i in range(column):
        test[i] = encode_images[index==i]
        P[i]=len(test[i])/row
        m[i]=np.mean(test[i],axis=0)
        sw=sw+P[i]*np.cov(test[i],rowvar=0)    #改成以行为独立的变量

    #   类间散度矩阵sb
    for i in range(column):
        m[i]=P[i]*m[i]              #每个类的均值乘类的比列
    #   总体的均值m0
    m0=np.sum(m,axis=0)

    sb=0
    for i in range(column):
        t1=(m[i]-m0).reshape(1,n_feature)
        t2 =(m[i]-m0).reshape(n_feature,1)
        sb=sb+P[i]*np.dot(t2,t1)

    J=np.trace(sb)/np.trace(sw)

    return  J

def model_train(batch_size, epochs, model, optimizer, loss_function,  train_loader, val_loader ,LR , i):

    model = model.to(device)
    # 样本长度
    train_size = len(train_loader) * batch_size
    val_size = len(val_loader) * batch_size

    # 最高准确率  最佳模型
    best_accuracy = 0.0
    best_model = model
    best_model.to(device)
    train_loss = []  # 记录在训练集上每个epoch的loss的变化情况
    train_acc = []  # 记录在训练集上每个epoch的准确率的变化情况
    validate_acc = []
    validate_loss = []

    # 计算模型运行时间
    start_time = time.time()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # for epoch in range(epochs):
    for epoch in tqdm(range(epochs)):
        # 训练
        model.train()

        loss_epoch = 0.  # 保存当前epoch的loss和
        correct_epoch = 0  # 保存当前epoch的正确个数和
        for seq, labels in train_loader:
            # print(seq.size(), labels.size())   torch.Size([32, 512]) torch.Size([32, 1])
            seq, labels = seq.to(device), labels.to(device)
            # print(seq.is_cuda)
            # print(seq.size(), labels.size()) torch.Size([32, 7, 1024]) torch.Size([32])
            # 每次更新参数前都梯度归零和初始化
            optimizer.zero_grad()
            # 前向传播
            y_pred = model(seq)  # torch.Size([16, 10])
            # 对模型输出进行softmax操作，得到概率分布
            probabilities = F.softmax(y_pred, dim=1)
            # 得到预测的类别
            predicted_labels = torch.argmax(probabilities, dim=1)
            # 与真实标签进行比较，计算预测正确的样本数量  # 计算当前batch预测正确个数
            correct_epoch += (predicted_labels == labels).sum().item()
            # 损失计算
            loss = loss_function(y_pred, labels)
            loss_epoch += loss.item()
            # 反向传播和参数更新
            loss.backward()
            optimizer.step()
        if LR == True:
            scheduler.step()
        # break
        # 计算准确率
        train_Accuracy = correct_epoch / train_size
        train_loss.append(loss_epoch / train_size)
        train_acc.append(train_Accuracy)
        # print(f'Epoch: {epoch + 1:2} train_Loss: {loss_epoch / train_size:10.8f} train_Accuracy:{train_Accuracy:4.4f}')
        # 每一个epoch结束后，在验证集上验证实验结果。
        with torch.no_grad():
            loss_validate = 0.
            correct_validate = 0
            for data, label in val_loader:
                model.eval()
                data, label = data.to(device), label.to(device)
                pre = model(data)
                # 对模型输出进行softmax操作，得到概率分布
                probabilities = F.softmax(pre, dim=1)
                # 得到预测的类别
                predicted_labels = torch.argmax(probabilities, dim=1)
                # 与真实标签进行比较，计算预测正确的样本数量  # 计算当前batch预测正确个数
                correct_validate += (predicted_labels == label).sum().item()
                loss = loss_function(pre, label)
                loss_validate += loss.item()
            # print(f'validate_sum:{loss_validate},  validate_Acc:{correct_validate}')
            val_accuracy = correct_validate / val_size
            # print(f'Epoch: {epoch + 1:2} val_Loss:{loss_validate / val_size:10.8f},  validate_Acc:{val_accuracy:4.4f}')
            validate_loss.append(loss_validate / val_size)
            validate_acc.append(val_accuracy)
            # 如果当前模型的准确率优于之前的最佳准确率，则更新最佳模型
            # 保存当前最优模型参数
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_model = model  # 更新最佳模型的参数
    # 保存最好的参数
    root = ['CWRU', 'PU', 'su', 'xj','ottawa']
    save_parament = root[i]+'best_model.pt'

    torch.save(best_model,save_parament)

    print(f'\nDuration: {time.time() - start_time:.2f} seconds')
    # plt.plot(range(epochs), train_loss, color='b', label='train_loss')
    # plt.plot(range(epochs), train_acc, color='g', label='train_acc')
    # plt.plot(range(epochs), validate_loss, color='y', label='validate_loss')
    # plt.plot(range(epochs), validate_acc, color='r', label='validate_acc')
    # plt.legend()
    # plt.show()  # 显示 lable
    print("best_accuracy :", best_accuracy)


    # name = 'mamba_4'
    # save_plot(file_name ='plot/Mtrain_loss.csv' ,name=name,data=train_loss)
    # save_plot(file_name='plot/Mtrain_acc.csv', name=name, data=train_acc)
    # save_plot(file_name='plot/Mvalidate_loss.csv', name=name, data=validate_loss)
    # save_plot(file_name='plot/Mvalidate_acc.csv', name=name, data=validate_acc)



    return train_loss,train_acc,validate_loss, validate_acc
    # print('train_loss:{}\ntrain_acc:{}\nvalidate_loss:{}\nvalidate_acc{}\n'.format(train_loss,train_acc,validate_loss,validate_acc))



def model_test(test_loader,i,k):
    # 得出每一类的分类准确率
    root = ['CWRU', 'PU', 'su', 'xj','ottawa']
    save_parament = root[i]+'best_model.pt'
    model = torch.load(save_parament)
    model = model.to(device)
    data_embed_collect = []
    label_collect = []
    # 使用测试集数据进行推断并计算每一类的分类准确率
    class_labels = []  # 存储类别标签
    predicted_labels = []  # 存储预测的标签
    test_start = time.time()
    with torch.no_grad():
        for test_data, test_label in test_loader:
            # 将模型设置为评估模式
            model.eval()
            test_data = test_data.to(device)
            test_output = model(test_data)

            data_embed_collect.append(test_output)
            label_collect.append(test_label)

            probabilities = F.softmax(test_output, dim=1)
            predicted = torch.argmax(probabilities, dim=1)
            class_labels.extend(test_label.tolist())
            predicted_labels.extend(predicted.tolist())
    test_time = time.time() - test_start
    data_embed_npy = torch.cat(data_embed_collect, axis=0).cpu().numpy()
    label_npu = torch.cat(label_collect, axis=0).cpu().numpy()
    # print(data_embed_npy)
    # model_name = ['WCamba','Liconformer','clformer','conformer','wcnn','transformer']
    # np.save(f"data/CWRU/plot_Tsne/D2_{model_name[k]}_data_embed_npy.npy", data_embed_npy)
    # np.save(f"data/CWRU/plot_Tsne/D2_{model_name[k]}_label_npu.npy", label_npu)
    # 混淆矩阵
    confusion_mat = confusion_matrix(class_labels, predicted_labels)

    # 计算每一类的分类准确率
    report = classification_report(class_labels, predicted_labels, digits=4 , zero_division=1)
    # print(report)
    # # 原始标签和自定义标签的映射
    label_mapping = {
        0: "0", 1: "1", 2: "2", 3: "3", 4: "4",
        5: "5", 6: "6", 7: "7", 8: "8", 9:"9",
        10:"10",11:"11",12:"12",14:"13",
    }
    cmap = ['Blues' ,'BuGn','Purples','Oranges','Reds']
    # # 绘制混淆矩阵
    plt.figure(figsize=(8, 6), dpi=600)
    sns.heatmap(confusion_mat ,xticklabels=label_mapping.values(), yticklabels=label_mapping.values(), annot=True,
                fmt='g', cmap=cmap[k])
    # plt.xlabel('Predicted Labels')
    # plt.ylabel('True Labels')
    # plt.title('Confusion Matrix')
    plt.savefig(f'Matrix/{k}', bbox_inches='tight', dpi=600)
    plt.show()
    print('if', test_time)
    return float(report.split()[-3])


