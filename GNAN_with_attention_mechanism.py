#Z. Zhang
#GNAN with attention mechanism
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler
import tqdm
import numpy as np
import scipy.io as scio
import torch.utils.data as Data

# 固定参数
ph = 0.035
mb_w = 1
b_w = 0.7
mb_h = 0

noiseSize = 15  # 噪声维度
para_size = 15  # 自由度
n_generator_feature = 256  # 生成器feature map数
n_discriminator_feature = 128  # 判别器feature map数
d_batch_size = 50
g_batch_size = 512
new_sample_size = 200

paraRange = [[0, 8], [0, 8.3], [0, 1], [0, 8], [0, 8.3], [0, 1], [0, 90],
      [8,18],[0, 8], [0,18], [0, 1], [0, 8], [0,18], [0, 1], [0, 90],[0, 1]]
useless = [1.927, 4.128, 1.000, 3.922, 6.858, 0.000, 341.573,
           5,1.927, 4.128, 1.000, 3.922, 6.858, 0.000, 341.573]
D_loss_list = []
G_loss_list = []

# 数据集导入
class TensorDataset(Data.Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __getitem__(self, index):
        return self.data_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)  

class AttentionLayer(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(AttentionLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc = nn.Linear(input_size, hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=2, batch_first=True)
        self.activation = nn.ReLU()

    def forward(self, x):
        print(x)
        seq_len = x.size(1)  
        print(seq_len)
        x = x.unsqueeze(1).repeat(1, seq_len, 1)  # (batch_size, seq_len, input_size)
        x = self.fc(x)
        attn_output, attn_output_weights = self.attention(x, x, x)
        output = attn_output[:, -1, :] 
        output = self.activation(output)
        return output

# 生成器网络
class NetGenerator(nn.Module):
    def __init__(self, input_size=noiseSize):
        super(NetGenerator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_size, n_generator_feature, bias=False),  # Adjust the input size
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(n_generator_feature, n_generator_feature * 2, bias=False),  # Adjust the input size
            nn.BatchNorm1d(n_generator_feature * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(n_generator_feature * 2, n_generator_feature * 4, bias=False),  # Adjust the input size
            nn.BatchNorm1d(n_generator_feature * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(n_generator_feature * 4, n_generator_feature * 2, bias=False),  # Adjust the input size
            nn.BatchNorm1d(n_generator_feature * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(n_generator_feature*2, para_size, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

    def initialize_weights(self):
        for i in range(len(self.main)):
            if isinstance(self.main[i], nn.Linear):
                nn.init.kaiming_uniform_(self.main[i].weight)

# 判别器网络
class NetDiscriminator(nn.Module):
    def __init__(self, input_size=para_size):
        super(NetDiscriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_size, n_discriminator_feature, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(n_discriminator_feature, n_discriminator_feature * 2, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(n_discriminator_feature * 2, n_discriminator_feature * 4, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(n_discriminator_feature * 4, n_discriminator_feature , bias=True),
            nn.ReLU(inplace=True),

            AttentionLayer(n_discriminator_feature, n_discriminator_feature),

            nn.Linear(n_discriminator_feature , 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1)

# 判别器训练
def D_train():
    for i, pe_parameter in tqdm.tqdm(enumerate(dataloader)):
        # parameter 是大小为1xpara_size的向量
        dataset_vector = pe_parameter[:, :para_size]
        dataset_vector = dataset_vector.cuda()

        dataset_labels = pe_parameter[:, para_size]
        dataset_labels = dataset_labels.cuda()

        optimizer_d.zero_grad()
        D_output = Discriminator(dataset_vector)
        D_loss = criterion(D_output, dataset_labels)
        D_loss.backward()
        optimizer_d.step()

    D_loss = D_loss.cpu()
    D_loss = D_loss.detach().numpy()
    D_loss_list.append(D_loss)

    print("[D_loss: %f]" % (D_loss.item()))

# 生成器训练
def G_train():
    noise = torch.rand(g_batch_size, noiseSize)  # (0,1)上的均匀分布
    noise = noise.cuda()
    true_labels = Variable(torch.ones(g_batch_size))
    true_labels = true_labels.cuda()

    optimizer_g.zero_grad()
    generated_vector = Generator(noise)
    generated_output = Discriminator(generated_vector)
    G_loss = criterion(generated_output, true_labels)
    G_loss.backward()
    optimizer_g.step()

    G_loss = G_loss.cpu()
    G_loss = G_loss.detach().numpy()
    G_loss_list.append(G_loss)

    print("[G loss: %f]" % (G_loss.item()))

def normal2init(para):  # 将在(0,1)映射回原始的范围
    para_Normal = [[] for _ in range(len(para))]
    for i in range(len(para[0])):
        min = paraRange[i][0]
        max = paraRange[i][1]
        if i != 2 and i != 5 and i != 10 and i != 13 and i != 15 :
            data = para[:, i] * (max - min) + min
        else:
            data = [1 if x > 0.5 else 0 for x in para[:, i]]

        for j in range(len(para)):
            para_Normal[j].append(data[j])
    return para_Normal

def init2normal(para):
    para_Normal = [[] for _ in range(len(para))]
    for i in range(len(para[0])):
        min = paraRange[i][0]
        max = paraRange[i][1]
        if i != 2 and i != 5 and i != 10 and i != 13 and i != 15 :
            data = (para[:,i] - min) / (max - min)
        else:
            data = [1 if x > 0.5 else 0 for x in para[:,i]]
        for j in range(len(para)):
            para_Normal[j].append(data[j])
    return para_Normal

def check(generated_vector):
    valid_list = []
    for i in range(len(generated_vector)):
        ml_2 = generated_vector[i][7]
        x_2 = [generated_vector[i][9], generated_vector[i][12]]
        a4 = (x_2[0] + b_w) < ml_2 and (x_2[1] + b_w) < ml_2 ;
        if a4==1:
            valid_list.append(generated_vector[i])
    return valid_list

if __name__ == '__main__':
    # 数据预处理
    data = scio.loadmat(r'E:\zzh\study\Master\work\2404\8-patch\loop4\data2\loop4_Dataset.mat')  # 换成改过的数据集路径
    initDataset = np.array(data['dataset'])
    pe_parameter = initDataset  # PE参数
    pe_parameter = init2normal(pe_parameter)  # 归一化
    pe_parameter = torch.tensor(pe_parameter, dtype=torch.float)
    pe_dataset = TensorDataset(pe_parameter)
    dataloader = torch.utils.data.DataLoader(pe_dataset, batch_size=d_batch_size, shuffle=True, num_workers=0,
                                             drop_last=False)
    print('数据加载完毕！')

    # 创建G和D
    Generator = NetGenerator()
    Discriminator = NetDiscriminator()
    # 优化器
    optimizer_d = torch.optim.Adam(Discriminator.parameters(), lr=1e-4, weight_decay=0.05)
    #optimizer_g = torch.optim.Adam(Generator.parameters(), lr=1e-3, weight_decay=0.05)
    optimizer_g = torch.optim.SGD(Generator.parameters(), lr=1e-5)

    scheduler_d = lr_scheduler.StepLR(optimizer_d, 200,  gamma=0.7)
    #scheduler_g = lr_scheduler.StepLR(optimizer_g, 100, gamma=0.9)
    criterion = torch.nn.BCELoss(reduction='sum')  # Loss function

    if torch.cuda.is_available() == True:
        Generator.cuda()
        Discriminator.cuda()
        criterion.cuda()

        '''
        checkpoint = torch.load("./state_dict.tar")
        Discriminator.load_state_dict(checkpoint['Discriminator.state_dict'])
        optimizer_d.load_state_dict(checkpoint['optimizer_d.state_dict'])
        Discriminator.eval()
        print("导入完成")
        '''

        start_time = time.time()
        # 训练D
        for i in range(500):  # 最大迭代次数
            D_train()
            scheduler_d.step()
            print('迭代次数：{}'.format(i))
            D_loss_list_t = np.array(D_loss_list)
            np.savetxt('D_loss.txt', D_loss_list_t, fmt='%.3f', delimiter=' ,', newline='\n')
        print("D训练结束")

        #保存模型
        state_dict = {"Discriminator.state_dict": Discriminator.state_dict(),
                      "optimizer_d.state_dict": optimizer_d.state_dict()}
        torch.save(state_dict, "./state_dict.tar")
        print("Discriminator保存成功!")

        # 训练G
        for i in range(1000):  # 最大迭代次数
            G_train()
            print('迭代次数：{}'.format(i))
        print("G训练结束")


        end_time = time.time()
        G_loss_list = np.array(G_loss_list)
        np.savetxt('G_loss.txt', G_loss_list, fmt='%.3f', delimiter=' ,', newline='\n')
        print("G_loss已保存")
        print("训练时长：",end_time-start_time)
        state_dict_g = {"Generator.state_dict": Generator.state_dict(),
                      "optimizer_g.state_dict": optimizer_g.state_dict()}
        torch.save(state_dict_g, "./state_dict_g.tar")
        print("Generator保存成功!")

        geometry_pass_sample = []
        sample_size = 0
        sample_pass_geometry = 0
        while (1):
            sample_size = sample_size + 1
            noise = torch.rand(g_batch_size, noiseSize)
            noise = noise.cuda()
            generated_vector = Generator(noise)

            generated_vector = generated_vector.cpu()
            generated_vector = generated_vector.detach().numpy()
            generated_vector = normal2init(generated_vector)

            # 符合几何限定条件
            generated_vector = [[float(item) for item in inner_list] for inner_list in generated_vector]
            match_list = check(generated_vector)

            for vector in match_list:
                geometry_pass_sample.append(vector)
                np.savetxt('geometry_pass.txt', geometry_pass_sample, fmt='%.3f', delimiter=' ,', newline='\n', )
            if len(geometry_pass_sample) >= new_sample_size:
                break

        geometryPassRate = len(geometry_pass_sample) / (sample_size * 256)

        np.savetxt('geometry_pass.txt', geometry_pass_sample, fmt='%.3f', delimiter=' ,', newline='\n', )
        print("保存完成")
        print("生成器的生成的样本总数为：", sample_size * 256)
        print("满足几何条件的样本数：", len(geometry_pass_sample))
        print("geometryPassRate = ", geometryPassRate)
