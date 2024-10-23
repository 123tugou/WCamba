from scipy.io import loadmat
import pandas as pd
from joblib import dump, load
import torch.utils.data as Data
import torch
import numpy as np
import os

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


def read_data(data_file_csv,dm='CWRU'):

    data_csv = pd.DataFrame()

    if dm == 'CWRU':
        # 设置文件夹路径
        folder_path = './dataset/matfiles/2HP/'
        columns_name = ['de_7_b', 'de_14_b', 'de_21_b', 'de_7_ir', 'de_14_ir',
                        'de_21_ir','de_7_or', 'de_14_or', 'de_21_or','de_normal',]
        # 获取文件夹下所有文件名，并按字母顺序排序
        file_names = sorted(os.listdir(folder_path))
        print(file_names)
        data_columns = []
        for file in file_names:
            # print(file)
            data = loadmat(f'{folder_path}{file}')
            # print(list(data.keys()))
            if folder_path == './dataset/matfiles/0HP/':
                if file == '12k_Drive_End_IR007_0_105.mat' or file == '12k_Drive_End_IR014_0_169.mat' \
                        or file == '12k_Drive_End_IR021_0_209.mat':
                    key = list(data.keys())[-3]
                else:
                    key = list(data.keys())[-2]

            if folder_path == './dataset/matfiles/1HP/':
                key = list(data.keys())[3]

            if folder_path == './dataset/matfiles/2HP/':
                if file == 'normal_2_99.mat':
                    key = list(data.keys())[4]
                else:
                    key = list(data.keys())[3]
            data_columns.append(key)
        print(data_columns)

        # 读取MAT文件
        for index in range(len(file_names)):
            data = loadmat(f'{folder_path}{file_names[index]}')
            dataList = data[data_columns[index]].reshape(-1)
            # data_csv[columns_name[index]] = dataList[77312:103424]
            data_csv[columns_name[index]] = dataList[:102400] #102912
        print('cwru数据总量：', data_csv.shape)
        data_csv.set_index('de_normal', inplace=True)
        print(data_csv)

    if dm == 'jn':
        # 江南数据
        file_names = ['ib600.mat', 'n600.mat', 'ob600.mat', 'tb600.mat']
        data_columns = ['ib600', 'n600', 'ob600', 'tb600']
        columns_name = ['ib600', 'n600', 'ob600', 'tb600']
        # 读取MAT文件
        for index in range(len(file_names)):
            data = loadmat(f'./dataset/jiangnan/{file_names[index]}')
            dataList = data[data_columns[index]].reshape(-1)
            data_csv[columns_name[index]] = dataList[:102400]
        print('jn数据总量：', data_csv.shape)
        data_csv.set_index('n600', inplace=True)



    if dm == 'xj':
        #  西交数据
        file_names = ['B_0.mat', 'B_1.mat', 'IR_0.mat', 'IR_1.mat', 'Normal.mat', 'OR_0.mat', 'OR_1.mat']
        data_columns = ['B_0', 'B_1', 'IR_0', 'IR_1', 'Normal', 'OR_0', 'OR_1']
        columns_name = ['B_0', 'B_1', 'IR_0', 'IR_1', 'Normal', 'OR_0', 'OR_1']
        # file_names = [ 'IR_0.mat', 'IR_1.mat', 'Normal.mat', 'OR_0.mat', 'OR_1.mat']
        # data_columns = ['IR_0', 'IR_1', 'Normal', 'OR_0', 'OR_1']
        # columns_name = [ 'IR_0', 'IR_1', 'Normal', 'OR_0', 'OR_1']
        # 读取MAT文件
        for index in range(len(file_names)):
            data = loadmat(f'./dataset/mat_xijiao/s/{file_names[index]}')
            dataList = data[data_columns[index]].reshape(-1)
            data_csv[columns_name[index]] = dataList[:102912]
        print('xj数据总量：', data_csv.shape)
        data_csv.set_index('B_0', inplace=True)

    if dm == 'su':
        # 东南
        file_names = ['ball_20.mat', 'health.mat', 'inner_20.mat', 'ourter_20.mat',]
        data_columns = ['ball_20',  'health_20', 'inner_20', 'ourter_20',]
        columns_name = ['ball_20',  'health_20', 'inner_20', 'ourter_20',]
        # 读取MAT文件
        for index in range(len(file_names)):
            data = loadmat(f'./dataset/su/{file_names[index]}')
            dataList = data[data_columns[index]].reshape(-1)
            data_csv[columns_name[index]] = dataList[:102912]
        print('su数据总量：', data_csv.shape)
        data_csv.set_index('ball_20', inplace=True)

    if dm == 'PU':
        # pu
        file_names = ['N15_M07_F10_K001_20.mat', 'N15_M07_F10_KA01_20.mat', 'N15_M07_F10_KA03_20.mat',
                      'N15_M07_F10_KA04_20.mat','N15_M07_F10_KA05_20.mat',
                      'N15_M07_F10_KA07_20.mat','N15_M07_F10_KA15_20.mat', 'N15_M07_F10_KA16_20.mat',
                      'N15_M07_F10_KI01_20.mat',
                      'N15_M07_F10_KI03_20.mat', 'N15_M07_F10_KI04_20.mat', 'N15_M07_F10_KI07_20.mat',
                      'N15_M07_F10_KI16_20.mat','N15_M07_F10_KI18_20.mat']

        data_columns = ['N15_M07_F10_K001', 'N15_M07_F10_KA01', 'N15_M07_F10_KA03', 'N15_M07_F10_KA04',
                        'N15_M07_F10_KA05',
                        'N15_M07_F10_KA07', 'N15_M07_F10_KA15', 'N15_M07_F10_KA16', 'N15_M07_F10_KI01',
                        'N15_M07_F10_KI03', 'N15_M07_F10_KI04', 'N15_M07_F10_KI07', 'N15_M07_F10_KI16',
                        'N15_M07_F10_KI18']

        columns_name = ['N15_M07_F10_K001', 'N15_M07_F10_KA01', 'N15_M07_F10_KA03', 'N15_M07_F10_KA04',
                        'N15_M07_F10_KA05',
                        'N15_M07_F10_KA07', 'N15_M07_F10_KA15', 'N15_M07_F10_KA16', 'N15_M07_F10_KI01',
                        'N15_M07_F10_KI03', 'N15_M07_F10_KI04', 'N15_M07_F10_KI07', 'N15_M07_F10_KI16',
                        'N15_M07_F10_KI18']
        #
        # file_names = ['N15_M07_F10_K001_20.mat', 'N15_M07_F10_KA01_20.mat', 'N15_M07_F10_KA03_20.mat',
        #               'N15_M07_F10_KA04_20.mat','N15_M07_F10_KA05_20.mat',
        #               'N15_M07_F10_KA07_20.mat','N15_M07_F10_KA15_20.mat', 'N15_M07_F10_KA16_20.mat',
        #               'N15_M07_F10_KI01_20.mat',
        #               'N15_M07_F10_KI03_20.mat', 'N15_M07_F10_KI04_20.mat',
        #               'N15_M07_F10_KI16_20.mat','N15_M07_F10_KI18_20.mat']
        #
        # data_columns = ['N15_M07_F10_K001', 'N15_M07_F10_KA01', 'N15_M07_F10_KA03', 'N15_M07_F10_KA04',
        #                 'N15_M07_F10_KA05',
        #                 'N15_M07_F10_KA07', 'N15_M07_F10_KA15', 'N15_M07_F10_KA16', 'N15_M07_F10_KI01',
        #                 'N15_M07_F10_KI03', 'N15_M07_F10_KI04',  'N15_M07_F10_KI16',
        #                 'N15_M07_F10_KI18']
        #
        # columns_name = ['N15_M07_F10_K001', 'N15_M07_F10_KA01', 'N15_M07_F10_KA03', 'N15_M07_F10_KA04',
        #                 'N15_M07_F10_KA05',
        #                 'N15_M07_F10_KA07', 'N15_M07_F10_KA15', 'N15_M07_F10_KA16', 'N15_M07_F10_KI01',
        #                 'N15_M07_F10_KI03', 'N15_M07_F10_KI04',  'N15_M07_F10_KI16',
        #                 'N15_M07_F10_KI18']

        # 读取MAT文件
        for index in range(len(file_names)):
            data = loadmat(f'./dataset/PU/{file_names[index]}')
            dataList = data[data_columns[index]].reshape(-1)
            data_csv[columns_name[index]] = dataList[:245760]
        print('pu数据总量：', data_csv.shape)
        data_csv.set_index('N15_M07_F10_K001', inplace=True)

    if dm == 'ottawa':
        file_names = ['H-A-1.mat', 'I-A-1.mat', 'O-A-1.mat', ]     # 升速
        # file_names = ['H-B-1.mat', 'I-B-1.mat', 'O-B-1.mat']     # 减速
        # file_names = ['H-C-1.mat', 'I-C-1.mat', 'O-C-1.mat']
        data_columns = ['Channel_1', 'Channel_1', 'Channel_1']
        columns_name = ['norml', 'ir', 'or ']
        # 读取MAT文件
        for index in range(len(file_names)):
            data = loadmat(f'./dataset/Ottawa/{file_names[index]}')
            dataList = data[data_columns[index]].reshape(-1)
            # data_csv[columns_name[index]] = dataList[:]
            data_csv[columns_name[index]] = dataList[:1228800]
        print('ottwa数据总量：', data_csv.shape)
        data_csv.set_index('norml', inplace=True)

    # print(data_csv)
    data_csv.to_csv(data_file_csv)


# 切割划分方法: 参考论文 《时频图结合深度神经网络的轴承智能故障诊断研究》

def split_data_with_overlap(data, time_steps, lable, overlap_ratio=0.5):
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
        temp_data = data[start_idx:end_idx].tolist()
        temp_data.append(lable)  # 对应哪一类
        data_list.append(temp_data)
    Clasiffy_dataFrame = pd.DataFrame(data_list, columns=Clasiffy_dataFrame.columns)
    return Clasiffy_dataFrame

# 归一化数据
def normalize(data):
    ''' (0,1)归一化
        参数:一维时间序列数据
    '''
    s = (data-min(data)) / (max(data)-min(data))
    return s


def make_data_labels(dataframe):
    '''
        参数 dataframe: 数据框
        返回 x_data: 数据集     torch.tensor
            y_label: 对应标签值  torch.tensor
    '''
    # 信号值
    x_data = dataframe.iloc[:,0:-1]
    # 标签值
    y_label = dataframe.iloc[:,-1]
    x_data = torch.tensor(x_data.values).float()
    y_label = torch.tensor(y_label.values.astype('int64')) # 指定了这些张量的数据类型为64位整数，通常用于分类任务的类别标签
    return x_data, y_label

# 数据集的制作
def make_datasets(data_file,dm='cwru', time_steps = 1024, split_rate = [0.8,0.2,0.0],shuffle=True, overlap_ratio = 0.5, normal = False,noise = False, snr = 0):
    '''
        参数:
        data_file_csv: 故障分类的数据集,csv格式
        label_list: 故障分类标签
        split_rate: 训练集、验证集、测试集划分比例
        time_steps: 时间步长
        overlap_ratio: 重叠率

        返回:
        train_set: 训练集数据
        val_set: 验证集数据
        test_set: 测试集数据
    '''

    data_file_csv = data_file + '/dataset.csv'
    read_data(data_file_csv,dm)
    # 1.读取数据
    origin_data = pd.read_csv(data_file_csv)
    print('orgin',origin_data.shape)
    # 2.分割样本点
    # 用于存储生成的数据# 10个样本集合
    dataframes = []
    Train_set = []
    Val_set = []
    Test_set = []
    # 记录类别标签
    label = 0
    # 使用iteritems()方法遍历每一列
    for column_name, column_data in origin_data.items():
        # 对数据集的每一维进行归一化
        if normal == True:
            column_data = normalize(column_data)
        if noise == True:
        # 加入噪声
            column_data = wgn(column_data, snr)
        # 划分样本点
        split_data = split_data_with_overlap(column_data, time_steps, label, overlap_ratio)
        # print(label)
        label += 1 # 类别标签递增
        samples_data = split_data
        # 打乱顺序
        if shuffle == True:
            samples_data = samples_data.sample(frac=1).reset_index(drop=True)
            print(shuffle)
        # dataframes.append(split_data)
        # 3.分割训练集、验证集、测试集
        sample_len = len(samples_data)  # 每一类样本数量
        train_len = int(sample_len * split_rate[0])  # 向下取整
        val_len = int(sample_len * split_rate[1])
        train_set = samples_data.iloc[0:train_len, :]
        val_set = samples_data.iloc[train_len:train_len + val_len, :]
        test_set = samples_data.iloc[train_len + val_len:, :]
        # print(column_name)
        # print(len(test_set))
        Train_set.append(train_set)
        Val_set.append(val_set)
        Test_set.append(test_set)

        # 不平衡实验
        # if label == 0:
        #     Train_set.append(train_set)
        #
        # else:
        #     Train_set.append(train_set[:70])
        # Val_set.append(val_set)
        # Test_set.append(test_set)
        # label += 1  # 类别标签递增
    # print('....',Train_set)
    # print((len(Val_set)))
    # print(len(test_set))
    Train_set = pd.concat(Train_set, ignore_index=True)
    Val_set   = pd.concat(Val_set, ignore_index=True)
    Test_set  = pd.concat(Test_set, ignore_index=True)
    # print(Test_set[:5])

    print('训练集数量：{};验证集数量：{}；测试集数量：{}'.format(len(Train_set), len(Val_set), len(Test_set)))


    # 制作标签
    train_xdata, train_ylabel = make_data_labels(Train_set)
    val_xdata, val_ylabel = make_data_labels(Val_set)
    test_xdata, test_ylabel = make_data_labels(Test_set)
    # 保存数据
    dump(train_xdata, '{}/trainX'.format(data_file))
    dump(train_ylabel, '{}/trainY'.format(data_file))
    print('train已保存')
    dump(val_xdata, '{}/valX'.format(data_file))
    dump(val_ylabel, '{}/valY'.format(data_file))
    print('val已保存')

    dump(test_xdata, '{}/testX'.format(data_file))
    dump(test_ylabel, '{}/testY'.format(data_file))
    print('test已保存')

    print('数据 形状：')
    print(train_xdata.size(), train_ylabel.size())
    print(val_xdata.size(), val_ylabel.size())
    print(test_xdata.size(), test_ylabel.size())


def gen_data(i, j, time_steps = 1024, shuffle=False,split_rate=[0.7,0.2,0.1],overlap_ratio = 0.5,normal = False):
    print('noraml:', normal)
    root = ['CWRU', 'PU', 'su', 'xj','ottawa']
    data_name =['/standard/', '/noise6', '/noise4','/noise2', '/noise0',
                '/noise-2','/noise-4', '/noise-6', '/few','/D_D'] # -6:7
    file_path = './data/' + root[i] + data_name[j]
    SNR = [6,4,2,0,-2,-4,-6]
    k = j
    if j >= 7:
        k = 7
    snr = SNR[k-1]
    if j == 0 or j == 8 or j==9:
        noise = False
        print('noise:', noise)
    else:
        noise = True
        print('noise:{}\nsnr:{}'.format(noise, snr))
    make_datasets(file_path,dm=root[i],time_steps = time_steps, split_rate = split_rate, overlap_ratio = overlap_ratio, shuffle=shuffle,normal = normal,noise = noise, snr = snr)
    # print('noise:{},snr:{}'.format(noise,snr))




# 加载数据集
def dataloader(file_path,batch_size, workers=2,shuffle = True):
    # 训练集
    train_xdata  = load(file_path+'/trainX')
    train_ylabel = load(file_path+'/trainY')
    # 验证集
    val_xdata = load(file_path+'/valX')
    val_ylabel = load(file_path+'/valY')
    # # 测试集
    test_xdata = load(file_path+'/testX')
    test_ylabel = load(file_path+'/testY')

    # 加载数据
    train_loader = Data.DataLoader(dataset=Data.TensorDataset(train_xdata, train_ylabel),
                                   batch_size=batch_size, shuffle=shuffle, num_workers=workers, drop_last=False)
    val_loader = Data.DataLoader(dataset=Data.TensorDataset(val_xdata, val_ylabel),
                                 batch_size=batch_size, shuffle=shuffle, num_workers=workers, drop_last=False)
    test_loader = Data.DataLoader(dataset=Data.TensorDataset(test_xdata, test_ylabel),
                                  batch_size=batch_size, shuffle=shuffle, num_workers=workers, drop_last=False)
    return train_loader, val_loader, test_loader




if __name__ == '__main__':
    gen_data(0,0, time_steps=1024,shuffle=False, split_rate=[0.7,0.3,0.], overlap_ratio=0., normal=False)


