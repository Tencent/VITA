import numpy as np
import torch
import json
import math


# 用于特征归一化的全局倒谱均值和方差归一化 (CMVN) 类
# 均值 (mean) 和逆标准差 (istd) 是预先计算好的统计量
# norm_var 标志位指示是否要对-方差进行归一化


class GlobalCMVN(torch.nn.Module):
    def __init__(self, mean: torch.Tensor, istd: torch.Tensor, norm_var: bool = True):
        """
        初始化 GlobalCMVN 模块。

        Args:
            mean (torch.Tensor): 均值统计量张量。
            istd (torch.Tensor): 逆标准差张量，其值为 1.0 / std。
            norm_var (bool): 是否执行方差归一化。默认为 True。
        """
        super().__init__()
        # 确保均值和逆标准差的形状相同
        assert mean.shape == istd.shape
        self.norm_var = norm_var
        # 使用 register_buffer 将 mean 和 istd 注册为模块的缓冲区。
        # 缓冲区是模型状态的一部分（会随模型一起保存和加载，并移动到不同设备），
        # 但它们不像模型参数（Parameter）那样在训练过程中被更新。
        # 这对于存储固定的统计数据非常理想。
        self.register_buffer("mean", mean)
        self.register_buffer("istd", istd)


    def forward(self, x: torch.Tensor):
        """
        对输入张量应用 CMVN。

        Args:
            x (torch.Tensor): 输入的特征张量，形状为 (batch, max_len, feat_dim)。

        Returns:
            (torch.Tensor): 经过归一化处理的特征张量。
        """
        # 第一步：减去均值（中心化）
        x = x - self.mean
        # 第二步：如果启用了方差归一化，则乘以逆标准差（缩放）
        if self.norm_var:
            x = x * self.istd
        return x




def load_cmvn_json(json_cmvn_file):
    """
    从 JSON 格式的文件中加载并计算 CMVN 统计量。
    JSON 文件通常包含特征的累加和、平方和以及总帧数。

    Args:
        json_cmvn_file (str): CMVN JSON 文件的路径。

    Returns:
        numpy.ndarray: 一个形状为 (2, feat_dim) 的 NumPy 数组，
                       第一行是均值，第二行是逆标准差。
    """
    # 打开并加载 JSON 文件
    with open(json_cmvn_file) as f:
        cmvn_json = json.load(f)

    # 从 JSON 对象中提取统计数据
    avg = cmvn_json["mean_stat"]  # 特征的累加和 (sum)
    var = cmvn_json["var_stat"]   # 特征的平方和 (sum of squares)
    count = cmvn_json["frame_num"] # 总帧数

    # 遍历每个特征维度来计算均值和逆标准差
    for i in range(len(avg)):
        # 计算均值: E[X] = sum(X) / N
        avg[i] /= count
        # 计算方差: Var(X) = E[X^2] - (E[X])^2
        var[i] = var[i] / count - avg[i] * avg[i]
        # 为防止除以零，对非常小的方差值设置一个下限（epsilon）
        if var[i] < 1.0e-20:
            var[i] = 1.0e-20
        # 计算逆标准差: 1 / sqrt(Var(X))
        var[i] = 1.0 / math.sqrt(var[i])
    # 将均值和逆标准差组合成一个 NumPy 数组
    cmvn = np.array([avg, var])
    return cmvn




def load_cmvn_kaldi(kaldi_cmvn_file):
    """
    从 Kaldi 格式的文本文件中加载并计算 CMVN 统计量。
    该文件格式通常是一个包含统计矩阵的文本文件。

    Args:
        kaldi_cmvn_file (str): Kaldi CMVN 文件的路径。

    Returns:
        numpy.ndarray: 一个形状为 (2, feat_dim) 的 NumPy 数组，
                       第一行是均值，第二行是逆标准差。
    """
    avg = []
    var = []
    with open(kaldi_cmvn_file, "r") as file:
        # Kaldi 的二进制文件以 '\0B' 开头，此函数不支持二进制格式
        if file.read(2) == "\0B":
            # 如果是二进制文件，则记录错误并退出
            logging.error(
                "kaldi cmvn binary file is not supported, please "
            )
            sys.exit(1)
        # 将文件指针移回文件开头
        file.seek(0)
        # 读取整个文件并按空格分割成一个列表
        arr = file.read().split()
        # 校验文件格式是否符合 Kaldi 文本矩阵的典型结构
        assert arr[0] == "["
        assert arr[-2] == "0"
        assert arr[-1] == "]"
        # 推断特征维度
        feat_dim = int((len(arr) - 2 - 2) / 2)
        # 解析特征的累加和
        for i in range(1, feat_dim + 1):
            avg.append(float(arr[i]))
        # 解析总帧数
        count = float(arr[feat_dim + 1])
        # 解析特征的平方和
        for i in range(feat_dim + 2, 2 * feat_dim + 2):
            var.append(float(arr[i]))

    # 这部分计算逻辑与 load_cmvn_json 中的完全相同
    for i in range(len(avg)):
        avg[i] /= count
        var[i] = var[i] / count - avg[i] * avg[i]
        if var[i] < 1.0e-20:
            var[i] = 1.0e-20
        var[i] = 1.0 / math.sqrt(var[i])
    cmvn = np.array([avg, var])
    return cmvn




def load_cmvn(filename, is_json):
    """
    一个包装函数，根据文件类型（JSON 或 Kaldi）加载 CMVN 文件。

    Args:
        filename (str): CMVN 文件的路径。
        is_json (bool): 如果文件是 JSON 格式，则为 True；否则为 False（假定为 Kaldi 格式）。

    Returns:
        tuple: 一个包含两个元素的元组 (mean, istd)，
               分别是均值和逆标准差的 NumPy 数组。
    """
    if is_json:
        # 如果是 JSON 文件，调用相应的加载函数
        file = load_cmvn_json(filename)
    else:
        # 否则，调用 Kaldi 文件的加载函数
        file = load_cmvn_kaldi(filename)
    # 返回解包后的均值（第一行）和逆标准差（第二行）
    return file[0], file[1]
