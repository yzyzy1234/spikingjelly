import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import numpy as np
from spikingjelly.clock_driven import neuron, encoding, functional
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


parser = argparse.ArgumentParser(description='spikingjelly LIF MNIST Training')  # 解析参数

parser.add_argument('--device', default='cuda:0', help='运行的设备，例如“cpu”或“cuda:0”\n Device, e.g., "cpu" or "cuda:0"')

parser.add_argument('--dataset-dir', default='./', help='保存MNIST数据集的位置，例如“./”\n Root directory for saving MNIST dataset, e.g., "./"')
parser.add_argument('--log-dir', default='./', help='保存tensorboard日志文件的位置，例如“./”\n Root directory for saving tensorboard logs, e.g., "./"')
parser.add_argument('--model-output-dir', default='./', help='模型保存路径，例如“./”\n Model directory for saving, e.g., "./"')

parser.add_argument('-b', '--batch-size', default=64, type=int, help='Batch 大小，例如“64”\n Batch size, e.g., "64"')
parser.add_argument('-T', '--timesteps', default=100, type=int, dest='T', help='仿真时长，例如“100”\n Simulating timesteps, e.g., "100"')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, metavar='LR', help='学习率，例如“1e-3”\n Learning rate, e.g., "1e-3": ', dest='lr')
parser.add_argument('--tau', default=2.0, type=float, help='LIF神经元的时间常数tau，例如“100.0”\n Membrane time constant, tau, for LIF neurons, e.g., "100.0"')
parser.add_argument('-N', '--epoch', default=100, type=int, help='训练epoch，例如“100”\n Training epoch, e.g., "100"')


def main():
    '''
    :return: None

    * :ref:`API in English <lif_fc_mnist.main-en>`

    .. _lif_fc_mnist.main-cn:

    使用全连接-LIF的网络结构，进行MNIST识别。\n
    这个函数会初始化网络进行训练，并显示训练过程中在测试集的正确率。

    * :ref:`中文API <lif_fc_mnist.main-cn>`

    .. _lif_fc_mnist.main-en:

    The network with FC-LIF structure for classifying MNIST.\n
    This function initials the network, starts trainingand shows accuracy on test dataset.
    '''
    
    args = parser.parse_args()
    print("########## Configurations ##########")
    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))
    print("####################################")

    # 写回参数
    device = args.device
    dataset_dir = args.dataset_dir
    log_dir = args.log_dir
    model_output_dir = args.model_output_dir
    batch_size = args.batch_size 
    lr = args.lr
    T = args.T
    tau = args.tau
    train_epoch = args.epoch

    # 创建一个 SummaryWriter 对象，并将 TensorBoard 日志文件保存到 log_dir 目录中。之后，在训练过程中，我们可以使用 writer.add_xxx 方法向 TensorBoard 中添加不同类型的数据，例如添加训练集和测试集的准确率、损失函数的变化、梯度的直方图等。
    writer = SummaryWriter(log_dir)

    # 初始化数据加载器
    train_dataset = torchvision.datasets.MNIST(
        root=dataset_dir,
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )
    test_dataset = torchvision.datasets.MNIST(
        root=dataset_dir,
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )

    train_data_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    test_data_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )

    # 定义并初始化网络
    net = nn.Sequential(
        nn.Flatten(),   # 将二维图像压缩为一维张量
        nn.Linear(28 * 28, 10, bias=False),  # 定义了一个线性层，用于将输入的一维张量映射为输出的一维张量。其中，28 * 28 表示输入的张量大小，10
        # 表示输出的张量大小，bias=False 表示不使用偏置项。
        neuron.LIFNode(tau=tau)  # 定义了一个 LIF（Leaky Integrate-and-Fire）节点，即一个带有时钟、能够累加输入电流并在电压达到阈值时发放脉冲的神经元。tau 是 LIF
        # 节点的时间常数，用于控制节点的反应速度
    )
    net = net.to(device)
    # 使用Adam优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # 使用泊松编码器
    encoder = encoding.PoissonEncoder()  # 定义了一个 Poisson 编码器，用于将连续的输入信号转化为脉冲信号，以便输入到 LIF 节点中
    train_times = 0
    max_test_accuracy = 0  # 用于记录训练轮数和测试准确率的最大值。

    test_accs = []
    train_accs = []

    for epoch in range(train_epoch):
        print("Epoch {}:".format(epoch))  # format用于将占位符{}替换成实际的值
        print("Training...")
        train_correct_sum = 0
        train_sum = 0
        net.train()  # 模型就会被设置为训练模式。这意味着，对于包含梯度的操作（如反向传播）以及包含 dropout 或 batch normalization 等随机过程的操作，PyTorch
        # 会在这些操作中保留梯度信息，以便进行参数更新。
        for img, label in tqdm(train_data_loader):
            img = img.to(device)
            label = label.to(device)
            label_one_hot = F.one_hot(label, 10).float()  # 转换成one-hot编码后，每个样本的标签就变成了一个长度为10
            # 的向量。这样做的好处是，模型在训练时可以直接对这个向量进行预测，而不需要对原始标签值进行处理。同时，由于one-hot编码中每个标签的位置是互斥的，因此也可以更好地表示标签之间的关系。

            optimizer.zero_grad()  # 在进行每个batch的训练之前，一般会先调用zero_grad()函数将模型中所有参数的梯度清零，以避免在进行多次反向传播计算时，历史梯度对当前的梯度计算产生影响。

            # 运行T个时长，out_spikes_counter是shape=[batch_size, 10]的tensor
            # 记录整个仿真时长内，输出层的10个神经元的脉冲发放次数
            for t in range(T):
                if t == 0:
                    out_spikes_counter = net(encoder(img).float())
                else:
                    out_spikes_counter += net(encoder(img).float())

            # out_spikes_counter / T 得到输出层10个神经元在仿真时长内的脉冲发放频率
            out_spikes_counter_frequency = out_spikes_counter / T

            # 损失函数为输出层神经元的脉冲发放频率，与真实类别的MSE
            # 这样的损失函数会使，当类别i输入时，输出层中第i个神经元的脉冲发放频率趋近1，而其他神经元的脉冲发放频率趋近0
            loss = F.mse_loss(out_spikes_counter_frequency, label_one_hot)
            loss.backward()
            optimizer.step()
            # 优化一次参数后，需要重置网络的状态，因为SNN的神经元是有“记忆”的
            functional.reset_net(net)

            # 正确率的计算方法如下。认为输出层中脉冲发放频率最大的神经元的下标i是分类结果
            train_correct_sum += (out_spikes_counter_frequency.max(1)[1] == label.to(device)).float().sum().item()
            train_sum += label.numel()

            train_batch_accuracy = (out_spikes_counter_frequency.max(1)[1] == label.to(device)).float().mean().item()
            writer.add_scalar('train_batch_accuracy', train_batch_accuracy, train_times)
            train_accs.append(train_batch_accuracy)

            train_times += 1
        train_accuracy = train_correct_sum / train_sum

        print("Testing...")
        net.eval()
        with torch.no_grad():
            # 每遍历一次全部数据集，就在测试集上测试一次
            test_correct_sum = 0
            test_sum = 0
            for img, label in tqdm(test_data_loader):
                img = img.to(device)
                for t in range(T):
                    if t == 0:
                        out_spikes_counter = net(encoder(img).float())
                    else:
                        out_spikes_counter += net(encoder(img).float())

                test_correct_sum += (out_spikes_counter.max(1)[1] == label.to(device)).float().sum().item()
                test_sum += label.numel()
                functional.reset_net(net)
            test_accuracy = test_correct_sum / test_sum
            writer.add_scalar('test_accuracy', test_accuracy, epoch)
            test_accs.append(test_accuracy)
            max_test_accuracy = max(max_test_accuracy, test_accuracy)
        print("Epoch {}: train_acc = {}, test_acc={}, max_test_acc={}, train_times={}".format(epoch, train_accuracy, test_accuracy, max_test_accuracy, train_times))
        print()
    
    # 保存模型
    torch.save(net, model_output_dir + "/lif_snn_mnist.ckpt")
    # 读取模型
    # net = torch.load(model_output_dir + "/lif_snn_mnist.ckpt")

    # 保存绘图用数据
    net.eval()
    # 注册钩子
    output_layer = net[-1] # 输出层
    output_layer.v_seq = []
    output_layer.s_seq = []
    def save_hook(m, x, y):
        m.v_seq.append(m.v.unsqueeze(0))
        m.s_seq.append(y.unsqueeze(0))

    output_layer.register_forward_hook(save_hook)


    with torch.no_grad():
        img, label = test_dataset[0]        
        img = img.to(device)
        for t in range(T):
            if t == 0:
                out_spikes_counter = net(encoder(img).float())
            else:
                out_spikes_counter += net(encoder(img).float())
        out_spikes_counter_frequency = (out_spikes_counter / T).cpu().numpy()
        print(f'Firing rate: {out_spikes_counter_frequency}')

        output_layer.v_seq = torch.cat(output_layer.v_seq)
        output_layer.s_seq = torch.cat(output_layer.s_seq)
        v_t_array = output_layer.v_seq.cpu().numpy().squeeze().T  # v_t_array[i][j]表示神经元i在j时刻的电压值
        np.save("v_t_array.npy",v_t_array)
        s_t_array = output_layer.s_seq.cpu().numpy().squeeze().T  # s_t_array[i][j]表示神经元i在j时刻释放的脉冲，为0或1
        np.save("s_t_array.npy",s_t_array)

    train_accs = np.array(train_accs)
    np.save('train_accs.npy', train_accs)
    test_accs = np.array(test_accs)
    np.save('test_accs.npy', test_accs)


if __name__ == '__main__':
    main()
