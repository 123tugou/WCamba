import pandas as pd
from scipy.io import loadmat
import sklearn
import torch
import torch.utils.data as Data
import numpy as np
from scipy.fftpack import fft
import pywt
import ewtpy
import torchvision.transforms as transforms
from sklearn.preprocessing import MinMaxScaler



file_path='_data.csv'


def wgn(x, snr, seed=7):
    '''
    加入高斯白噪声 Additive White Gaussian Noise
    :param x: 原始信号
    :param snr: 信噪比
    :return: 加入噪声后的信号
    '''
    np.random.seed(seed)  # 设置随机种子
    t_snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / len(x)
    npower = xpower / t_snr
    noise = np.random.randn(len(x)) * np.sqrt(npower)
    return x + noise



def normalize(data):
    ''' (0,1)归一化
        参数:一维时间序列数据
    '''
    s = (data-min(data)) / (max(data)-min(data))
    return s



def CWT(data):
    # 设置连续小波变换参数
    wavelet = 'morl'  # 小波函数选择Morlet小波
    scales = np.arange(1, 127)  # 尺度参数范围

    # 进行连续小波变换
    coefficients, frequencies = pywt.cwt(data, scales, wavelet, 1024)
    return np.abs(coefficients)

def FFT(Fs, data):
    """
    对输入信号进行FFT
    :param Fs:  采样频率
    :param data:待FFT的序列
    :return:
    """
    L = len(data)  # 信号长度
    N = np.power(2, np.ceil(np.log2(L)))  # 下一个最近二次幂，也即N个点的FFT
    result = np.abs(fft(x=data, n=int(N))) / L * 2  # N点FFT
    axisFreq = np.arange(int(N / 2)) * Fs / N  # 频率坐标
    result = result[range(int(N / 2))]  # 因为图形对称，所以取一半
    return  result

def EWT(data):
    ewt, mfb, boundaries = ewtpy.EWT1D(data, N=3)
    return ewt

def read_data(dm='cwru'):

    data_12k_10c = pd.DataFrame()
    if dm == 'cwru':
        #  西储数据
        file_names = ['0_0.mat', '7_1.mat', '7_2.mat', '7_3.mat', '14_1.mat', '14_2.mat', '14_3.mat', '21_1.mat', '21_2.mat',
                      '21_3.mat']
        # for file in file_names:
        #     data = loadmat(f'./dataset/matfiles/{file}')
        #     print(list(data.keys()))

        data_columns = ['X097_DE_time', 'X105_DE_time', 'X118_DE_time', 'X130_DE_time', 'X169_DE_time',
                        'X185_DE_time', 'X197_DE_time', 'X209_DE_time', 'X222_DE_time', 'X234_DE_time']
        columns_name = ['de_normal', 'de_7_inner', 'de_7_ball', 'de_7_outer', 'de_14_inner', 'de_14_ball', 'de_14_outer',
                        'de_21_inner', 'de_21_ball', 'de_21_outer']

        # 读取MAT文件
        for index in range(len(file_names)):
            data = loadmat(f'./dataset/matfiles/{file_names[index]}')  # 西储
            dataList = data[data_columns[index]].reshape(-1)
            data_12k_10c[columns_name[index]] = dataList[:102400]
        print('cwru数据总量：', data_12k_10c.shape)
        data_12k_10c.set_index('de_normal', inplace=True)




    if dm == 'jn':
        # 江南数据
        file_names = ['ib600.mat', 'n600.mat', 'ob600.mat', 'tb600.mat']
        data_columns = ['ib600', 'n600', 'ob600', 'tb600']
        columns_name = ['ib600', 'n600', 'ob600', 'tb600']
        # 读取MAT文件
        for index in range(len(file_names)):
            data = loadmat(f'./dataset/jiangnan/{file_names[index]}')  # 西储
            dataList = data[data_columns[index]].reshape(-1)
            data_12k_10c[columns_name[index]] = dataList[:102400]
        print('jn数据总量：', data_12k_10c.shape)
        data_12k_10c.set_index('ib600', inplace=True)



    if dm == 'xj':
        #  西交数据
        file_names = ['B_0.mat', 'B_1.mat', 'IR_0.mat', 'IR_1.mat', 'Normal.mat', 'OR_0.mat', 'OR_1.mat']
        data_columns = ['B_0', 'B_1', 'IR_0', 'IR_1', 'Normal', 'OR_0', 'OR_1']
        columns_name = ['B_0', 'B_1', 'IR_0', 'IR_1', 'Normal', 'OR_0', 'OR_1']
        # 读取MAT文件
        for index in range(len(file_names)):
            data = loadmat(f'./dataset/mat_xijiao/{file_names[index]}')  # 西储
            dataList = data[data_columns[index]].reshape(-1)
            data_12k_10c[columns_name[index]] = dataList[:112400]
        print('xj数据总量：', data_12k_10c.shape)
        data_12k_10c.set_index('B_0', inplace=True)

    if dm == 'su':
        # 东南
        file_names = ['ball_20.mat', 'comb_20.mat', 'health.mat', 'inner_20.mat', 'ourter_20.mat']
        data_columns = ['ball_20', 'comb_20', 'health_20', 'inner_20', 'ourter_20']
        columns_name = ['ball_20', 'comb_20', 'health_20', 'inner_20', 'ourter_20']
        # 读取MAT文件
        for index in range(len(file_names)):
            data = loadmat(f'./dataset/su/{file_names[index]}')
            dataList = data[data_columns[index]].reshape(-1)
            data_12k_10c[columns_name[index]] = dataList[:102400]
        print('su数据总量：', data_12k_10c.shape)
        data_12k_10c.set_index('ball_20', inplace=True)

    if dm == 'pu':
        # pu
        file_names = ['N15_M07_F10_K001_20.mat', 'N15_M07_F10_KA01_20.mat', 'N15_M07_F10_KA03_20.mat', 'N15_M07_F10_KA04_20.mat',
                      'N15_M07_F10_KA07_20.mat', 'N15_M07_F10_KA15_20.mat', 'N15_M07_F10_KA16_20.mat', 'N15_M07_F10_KI01_20.mat',
                      'N15_M07_F10_KI03_20.mat', 'N15_M07_F10_KI04_20.mat', 'N15_M07_F10_KI07_20.mat', 'N15_M07_F10_KI16_20.mat']

        data_columns = ['N15_M07_F10_K001', 'N15_M07_F10_KA01','N15_M07_F10_KA03','N15_M07_F10_KA04',
                        'N15_M07_F10_KA07','N15_M07_F10_KA15','N15_M07_F10_KA16','N15_M07_F10_KI01',
                        'N15_M07_F10_KI03', 'N15_M07_F10_KI04', 'N15_M07_F10_KI07', 'N15_M07_F10_KI16']

        columns_name = ['N15_M07_F10_K001', 'N15_M07_F10_KA01','N15_M07_F10_KA03','N15_M07_F10_KA04',
                        'N15_M07_F10_KA07','N15_M07_F10_KA15','N15_M07_F10_KA16','N15_M07_F10_KI01',
                        'N15_M07_F10_KI03', 'N15_M07_F10_KI04', 'N15_M07_F10_KI07', 'N15_M07_F10_KI16']
        # 读取MAT文件
        for index in range(len(file_names)):
            data = loadmat(f'./dataset/PU/{file_names[index]}')  # 西储
            dataList = data[data_columns[index]].reshape(-1)
            data_12k_10c[columns_name[index]] = dataList[:102400]
        print('pu数据总量：', data_12k_10c.shape)
        data_12k_10c.set_index('N15_M07_F10_K001', inplace=True)

    print(data_12k_10c)
    data_12k_10c.to_csv(dm+file_path)

def split_data_with_overlap(data, time_steps, lable, overlap_ratio):
    """
        data:要切分的时间序列数据,可以是一个一维数组或列表。
        time_steps:切分的时间步长,表示每个样本包含的连续时间步数。
        lable: 表示切分数据对应 类别标签
        overlap_ratio:前后帧切分时的重叠率,取值范围为 0 到 1,表示重叠的比例。
    """
    stride = int(time_steps * (1 - overlap_ratio))  # 计算步幅
    samples = (len(data) - time_steps) // stride + 1  # 计算样本数
    # 用于存储生成的数据
    Clasiffy_dataFrame = pd.DataFrame(columns=[x for x in range(time_steps + 1)])
    # 记录数据行数(量)
    index_count = 0
    data_list = []
    for i in range(samples):
        start_idx = i * stride
        end_idx = start_idx + time_steps

        temp_data = data[start_idx:end_idx]

        temp_data = temp_data.tolist()

        temp_data.append(lable)  # 对应哪一类

        data_list.append(temp_data)
    Clasiffy_dataFrame = pd.DataFrame(data_list)

    return Clasiffy_dataFrame

# def make_datasets(data_file_csv, split_rate=[0.7, 0.2, 0.1], overlap_ratio=0, steps=512 ):
#     """
#         参数:
#         data_file_csv: 故障分类的数据集,csv格式
#         label_list: 故障分类标签
#         split_rate: 训练集、验证集、测试集划分比例
#
#         返回:
#         train_set: 训练集数据
#         val_set: 验证集数据
#         test_set: 测试集数据
#     """
#     read_data()
#     # 1.读取数据
#     origin_data = pd.read_csv(data_file_csv)
#     # 2.分割样本点
#     time_steps = steps  # 时间步长
#     # 用于存储生成的数据# 10个样本集合
#
#     samples_data = pd.DataFrame(columns=[x for x in range(time_steps + 1)])
#     # 记录类别标签
#     label = 0
#     # 使用iteritems()方法遍历每一列
#     for column_name, column_data in origin_data.items():
#         # # 对数据集的每一维进行归一化
#         # column_data = normalize(column_data)
#         # 加入噪声
#         # column_data = wgn(column_data, 0)
#         # 划分样本点  window = 512  overlap_ratio = 0.5  samples = 467 每个类有467个样本
#         split_data = split_data_with_overlap(column_data, time_steps, label, overlap_ratio)
#
#
#         label += 1  # 类别标签递增
#         samples_data = pd.concat([samples_data, split_data])
#         # 随机打乱样本点顺序
#         samples_data = sklearn.utils.shuffle(samples_data)  # 设置随机种子 保证每次实验数据一致
#
#     # 3.分割训练集-、验证集、测试集
#     sample_len = len(samples_data)  # 每一类样本数量
#     train_len = int(sample_len * split_rate[0])  # 向下取整
#     val_len = int(sample_len * split_rate[1])
#     train_set = samples_data.iloc[0:train_len, :]
#     val_set = samples_data.iloc[train_len:train_len + val_len, :]
#     test_set = samples_data.iloc[train_len + val_len:sample_len, :]
#
#     print('训练集数量：{};验证集数量：{}；测试集数量：{}'.format(len(train_set), len(val_set), len(test_set)))
#
#     return train_set, val_set, test_set

def make_datasets(dm, split_rate=[0.7, 0.2, 0.1], overlap_ratio=0, steps=512 , noise = True ,snr = -6 , normal = False):
    """
        参数:
        data_file_csv: 故障分类的数据集,csv格式
        label_list: 故障分类标签
        split_rate: 训练集、验证集、测试集划分比例

        返回:
        train_set: 训练集数据
        val_set: 验证集数据
        test_set: 测试集数据
    """
    data_file_csv = dm+file_path
    read_data(dm)
    # 1.读取数据
    origin_data = pd.read_csv(data_file_csv)
    # 2.分割样本点
    time_steps = steps  # 时间步长
    # 用于存储生成的数据# 10个样本集合
    Train_set = pd.DataFrame(columns=[x for x in range(time_steps + 1)])
    Val_set = pd.DataFrame(columns=[x for x in range(time_steps + 1)])
    Test_set = pd.DataFrame(columns=[x for x in range(time_steps + 1)])
    # 记录类别标签
    label = 0
    # 使用iteritems()方法遍历每一列
    for column_name, column_data in origin_data.items():
        if noise == True :
            # 加入噪声
            column_data = wgn(column_data, snr)
        if normal == True :
            # 对数据集的每一维进行归一化
            column_data = normalize(column_data)
        # 划分样本点  window = 512  overlap_ratio = 0.5  samples = 467 每个类有467个样本
        split_data = split_data_with_overlap(column_data, time_steps, label, overlap_ratio)
        label += 1  # 类别标签递增
        samples_data = sklearn.utils.shuffle(split_data)  # 设置随机种子 保证每次实验数据一致
        # 3.分割训练集-、验证集、测试集
        sample_len = len(samples_data)  # 每一类样本数量
        train_len = int(sample_len * split_rate[0])  # 向下取整
        val_len = int(sample_len * split_rate[1])
        train_set = samples_data.iloc[0:train_len, :]
        val_set = samples_data.iloc[train_len:train_len + val_len, :]
        test_set = samples_data.iloc[train_len + val_len:sample_len, :]
        print(column_name)
        print(len(test_set))
        Train_set = pd.concat([Train_set, train_set])
        Val_set = pd.concat([Val_set, val_set])
        Test_set = pd.concat([Test_set, test_set])

        # 随机打乱样本点顺序

    print('训练集数量：{};验证集数量：{}；测试集数量：{}'.format(len(Train_set), len(Val_set), len(Test_set)))

    return Train_set, Val_set, Test_set


def make_data_labels(dataframe):
    """
        参数 dataframe: 数据框
        返回 x_data: 数据集     torch.tensor
            y_label: 对应标签值  torch.tensor
    """
    # 信号值
    x_data = dataframe.iloc[:, 0:-1]
    # x_data = x_data.astype(float)  # numpy强制类型转换

    # 标签值
    y_label = dataframe.iloc[:, -1]
    label1 = 0
    # for aa in y_label:
    #     if aa==1:
    #         label1 = label1+1
    # print(label1)
    x_data = torch.tensor(x_data.values).float()
    y_label = torch.tensor(y_label.values.astype('int64'))  # 指定了这些张量的数据类型为64位整数，通常用于分类任务的类别标签
    return x_data, y_label


def dataloader(dm,batch_size=32, workers=0,split_rate=[0.7, 0.2, 0.1], overlap_ratio= 0.5, steps=1024 ,  noise = True ,snr = -6 , normal = False):

    train_set, val_set, test_set = make_datasets(dm, split_rate, overlap_ratio=overlap_ratio,
                                                 steps=steps ,  noise = noise ,snr = snr , normal = normal)
    # 制作标签
    train_xdata, train_ylabel = make_data_labels(train_set)
    val_xdata, val_ylabel = make_data_labels(val_set)
    test_xdata, test_ylabel = make_data_labels(test_set)
    print(train_xdata.size(), train_ylabel.size())
    print(val_xdata.size(), val_ylabel.size())
    print(test_xdata.size(), test_ylabel.size())

    # 加载数据
    train_loader = Data.DataLoader(dataset=Data.TensorDataset(train_xdata, train_ylabel),
                                   batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    val_loader = Data.DataLoader(dataset=Data.TensorDataset(val_xdata, val_ylabel),
                                 batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    test_loader = Data.DataLoader(dataset=Data.TensorDataset(test_xdata, test_ylabel),
                                  batch_size=batch_size, shuffle=True, num_workers=workers,drop_last=True)
    # print(test_loader)

    return train_loader, val_loader, test_loader


if __name__  ==  '__main__':
    dm = 'xj'
    # split_rate = [0.7, 0.2, 0.1]
    # train_set, val_set, test_set = make_datasets(dm, split_rate, overlap_ratio= 0.5,steps=1024 , noise = True ,snr = -6 , normal = False)
    # 制作标签
    # train_xdata, train_ylabel = make_data_labels(train_set)
    # val_xdata, val_ylabel = make_data_labels(val_set)
    # test_xdata, test_ylabel = make_data_labels(test_set)
    batch_size = 32
    # # 加载数据
    train_loader, val_loader, test_loader = dataloader(dm,batch_size, workers=0,split_rate=[0.15, 0.5, 0.85],
                                                       overlap_ratio= 0, steps=1024,  noise = True ,snr = -6 , normal = False)