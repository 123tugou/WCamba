import numpy as np
from train_test import model_train,model_test,count_parameters ,save_plot
import torch.nn as nn
import torch
from data_set import dataloader, gen_data
from model.wdcnn import WDCNN
from model.resnet import resnet18
from model.transformers import DSCTransformer
from model.Kamba import mambaModel
from model.Liconvformer import Liconvformer
from model.CLFormer import CLFormer
from model.Convformer_NSE import convoformer_v1_small
from model.MCSwinT import mcswint
from model.MK_ResCNN import MSResNet
from model.MobileNet import MobileNet
import visual
import csv
from scipy import stats
# 基本参数
learn_rate = 0.001
batch_size = 32
epochs = 50
i = 1
j = 0
T = 0
root = ['CWRU', 'PU', 'su', 'xj','ottawa']
data_name = ['/standard', '/noise6', '/noise4', '/noise2', '/noise0', '/noise-2', '/noise-4', '/noise-6','/D_D']
few = ['/T118','T217','T316','T415']
file_path = './data/' + root[i] + data_name[j]
print("数据：", root[i] + data_name[j])

# 数据生成
gen_data(i, j, time_steps = 1024, shuffle=False,split_rate=[0.6,0.2,0.2] ,overlap_ratio = 0,normal=False)


if i == 0 : output = 10
if i == 1 : output = 14
if i == 2 : output = 5
if i == 3 : output = 7
if i == 4 : output = 3

loss_function = nn.CrossEntropyLoss(reduction='sum')  # loss
train_loader,val_loader,test_loader = dataloader(file_path, batch_size,shuffle =True)
print()


'''1.wcamba模型参数'''
input_dim = 1024  # 输入维度
output_dim = output  # 输出维度
num_layers = 2  # 编码器层数
# 加载模型
m_model = mambaModel(input_dim, num_layers, output_dim, d_state=16, d_conv=4, expand=2)

'''2.clformer'''
c_model = CLFormer(in_channel=1, out_channel=output)

'''3.WDCNN'''
in_channel = 1
out_channel = output
# 加载模型
w_model = WDCNN(in_channel, out_channel)

#
'''4.resnet18'''
num_classes = output
r_model = resnet18(batch_size=batch_size,num_classes=num_classes)

#
'''5.transformer'''
# 定义参数
N = 2  # 编码器个数
input_dim = 1024
seq_len = 32  # 句子长度
d_model = 64  # 词嵌入维度
d_ff = 256  # 全连接层维度
head = 4  # 注意力头数
dropout = 0.1
num_classes = output
# 加载模型
t_model = DSCTransformer(input_dim=input_dim, num_classes=num_classes, dim=d_model, depth=N,
                       heads=head, mlp_dim=d_ff, dim_head=d_model, emb_dropout=dropout, dropout=dropout)
''' 6.Liconvformer'''
Li_model = Liconvformer(in_channel=1, out_channel=output)

'''7.conformer'''
con_model = convoformer_v1_small(in_channel=1, out_channel=output)

'''8.MCSwinT'''
mc_model = mcswint(in_channel=1, out_channel=output)

'''9.mk_rescnn'''
ms_model = MSResNet(in_channel=1, out_channel=output)

'''10.mobilenet'''
mo_model = MobileNet(in_channel=1, out_channel=output)

def save_txt(name,train_loss,train_acc,validate_loss, validate_acc):
    save_plot(file_name ='plot/pu/train_loss.csv' ,name=name,data=train_loss)
    save_plot(file_name='plot/pu/train_acc.csv', name=name, data=train_acc)
    save_plot(file_name='plot/pu/validate_loss.csv', name=name, data=validate_loss)
    save_plot(file_name='plot/pu/validate_acc.csv', name=name, data=validate_acc)



# 模型训练
m = []
c = []
w = []
r = []
t = []
li = []
con = []
mc = []
ms =[]
mo=[]
hz = []
for k in range(1):
    # train_loader, val_loader, test_loader = dataloader(file_path, batch_size, shuffle=True)

    print('开始训练第{}轮'.format(k+1))
    optimizer = torch.optim.AdamW(m_model.parameters(), lr=0.001)  # 优化器
    # optimizer = torch.optim.AdamW(m_model.parameters(), learn_rate, weight_decay=0.001)
    train_loss,train_acc,validate_loss, validate_acc = model_train(batch_size, epochs, m_model, optimizer, loss_function, train_loader, val_loader, LR=False,i=i)
    save_txt('wcamba', train_loss, train_acc, validate_loss, validate_acc)
    accury = model_test(test_loader,i=i,k=0)
    m.append(accury)
    hz.append(accury)
    visual.main('WCamba')
    print('1.wcamba---end:',accury)
    #
    optimizer = torch.optim.AdamW(Li_model.parameters(),lr=0.001)  # 优化器 #weight_decay=0.001
    train_loss,train_acc,validate_loss, validate_acc=model_train(batch_size, epochs, Li_model, optimizer, loss_function, train_loader, val_loader, LR=False, i=i)
    save_txt('Liconformer', train_loss, train_acc, validate_loss, validate_acc)
    accury = model_test(test_loader, i=i,k=1)
    li.append(accury)
    hz.append(accury)
    visual.main('Liconformer')
    print('2.Liconformer---end', accury)

    optimizer = torch.optim.Adam(c_model.parameters(), learn_rate, weight_decay=0.001)
    train_loss, train_acc, validate_loss, validate_acc = model_train(batch_size, epochs, c_model, optimizer, loss_function, train_loader, val_loader, LR=True,i=i)
    accury = model_test(test_loader,i=i,k=2)
    c.append(accury)
    hz.append(accury)
    visual.main('clformer')
    save_txt('clformer', train_loss, train_acc, validate_loss, validate_acc)
    print('3.clformer---end',accury)

    optimizer = torch.optim.AdamW(con_model.parameters(), lr=0.001)  # 优化器 #weight_decay=0.001
    train_loss, train_acc, validate_loss, validate_acc =model_train(batch_size, epochs, con_model, optimizer, loss_function, train_loader, val_loader, LR=False, i=i)
    accury = model_test(test_loader, i=i,k=3)
    con.append(accury)
    hz.append(accury)
    visual.main('conformer')
    save_txt('conformer', train_loss, train_acc, validate_loss, validate_acc)
    print('4.conformer---end', accury)


    optimizer = torch.optim.AdamW(w_model.parameters())
    train_loss, train_acc, validate_loss, validate_acc = model_train(batch_size, epochs, w_model, optimizer, loss_function, train_loader, val_loader, LR=False,i=i)
    accury = model_test(test_loader,i=i,k=4)
    w.append(accury)
    hz.append(accury)
    visual.main('wcnn')
    save_txt('wcnn', train_loss, train_acc, validate_loss, validate_acc)
    print('5.wcnn---end',accury)

    optimizer = torch.optim.AdamW(t_model.parameters(), lr=0.001)
    train_loss, train_acc, validate_loss, validate_acc=model_train(batch_size, epochs, t_model, optimizer, loss_function, train_loader, val_loader, LR=False, i=i)
    accury = model_test(test_loader,i=i,k=5)
    t.append(accury)
    hz.append(accury)
    visual.main('transformer')
    save_txt('transformer', train_loss, train_acc, validate_loss, validate_acc)
    print('6.transformer---end',accury)




# print([i*100 for i in hz])
hz = [round(i*100,2) for i in hz]
file_name = file_path+'accuray.csv'
with open(file_name, 'a', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(hz)


# print('........................')
# print("数据：", root[i] + data_name[j])
# print("mamba:{}均值{:.2f}+{:.2f} {:.2f},标准差：{:.2f}" .format(m,100*np.mean(m),100*(np.max(m)-np.mean(m)),100*(np.min(m)-np.mean(m)),100*np.std(m, ddof=1)))
# print("clformer:{}均值{:.2f}+{:.2f} {:.2f},标准差：{:.2f}" .format(c,100*np.mean(c),100*(np.max(c)-np.mean(c)),100*(np.min(c)-np.mean(c)),100*np.std(c, ddof=1)))
# print("wd:{}均值{:.2f}+{:.2f} {:.2f},标准差：{:.2f}" .format(w,100*np.mean(w),100*(np.max(w)-np.mean(w)),100*(np.min(w)-np.mean(w)),100*np.std(w, ddof=1)))
# # print("resnet:{}均值{:.2f}+{:.2f} {:.2f}" .format(r,100*np.mean(r),100*(np.max(r)-np.mean(r)),100*(np.min(r)-np.mean(r))))
# print("trans:{}均值{:.2f}+{:.2f} {:.2f},标准差：{:.2f}" .format(t,100*np.mean(t),100*(np.max(t)-np.mean(t)),100*(np.min(t)-np.mean(t)),100*np.std(t, ddof=1)))
# print("Liconformer:{}均值{:.2f}+{:.2f} {:.2f},标准差：{:.2f}" .format(li,100*np.mean(li),100*(np.max(li)-np.mean(li)),100*(np.min(li)-np.mean(li)),100*np.std(li, ddof=1)))
# print("conformer:{}均值{:.2f}+{:.2f} {:.2f},标准差：{:.2f}" .format(con,100*np.mean(con),100*(np.max(con)-np.mean(con)),100*(np.min(con)-np.mean(con)),100*np.std(con, ddof=1)))
# print("mcswint:{}均值{:.2f}+{:.2f} {:.2f},标准差：{:.2f}" .format(mc,100*np.mean(mc),100*(np.max(mc)-np.mean(mc)),100*(np.min(mc)-np.mean(mc)),100*np.std(con, ddof=1)))
# print("MSResNet:{}均值{:.2f}+{:.2f} {:.2f}" .format(ms,100*np.mean(ms),100*(np.max(ms)-np.mean(ms)),100*(np.min(ms)-np.mean(ms))))
# print("mobilenet:{}均值{:.2f}+{:.2f} {:.2f}" .format(mo,100*np.mean(mo),100*(np.max(mo)-np.mean(mo)),100*(np.min(mo)-np.mean(mo))))


# t_stat, p_value = stats.ttest_ind(m, c)
# print('clformer的P值：',p_value)
# t_stat, p_value = stats.ttest_ind(m, w)
# print('wd的P值：',p_value)
# t_stat, p_value = stats.ttest_ind(m, t)
# print('trans的P值：',p_value)
# t_stat, p_value = stats.ttest_ind(m, li)
# print('licon的P值：',p_value)
# t_stat, p_value = stats.ttest_ind(m, con)
# print('conformer的P值：',p_value)





# count_parameters(model)