import torch
import torch.nn as nn


class Scaler(nn.Module):
    def __init__(self, lowdim_dict, eps=1e-8):
        """
        初始化Scaler类，计算给定字典中每个数据集的均值和标准差

        参数:
        lowdim_dict (dict): 包含每个键及其维度的字典。
        eps (float): 防止除零的小值，默认为1e-8。
        """
        super(Scaler, self).__init__()
        self.lowdim_dict = lowdim_dict
        self.eps = eps
        self.mean_dict = nn.ParameterDict({
            key: nn.Parameter(torch.zeros(value), requires_grad=False) for key, value in lowdim_dict.items()
        })
        self.std_dict = nn.ParameterDict({
            key: nn.Parameter(torch.ones(value), requires_grad=False) for key, value in lowdim_dict.items()
        })

    def fit(self, data_dict):
        """
        计算并存储给定字典中每个数据集的均值和标准差

        参数:
        data_dict (dict): 字典，其中键是数据集名称，值是对应的数据集（torch.Tensor）
        """
        for key, data in data_dict.items():
            if key in self.lowdim_dict:
                if data.dim() > 1:  # 确保数据是多维的
                    mean = data.mean(dim=0)
                    std = data.std(dim=0)
                    std = std.clamp(min=self.eps)  # 防止标准差为零

                    self.mean_dict[key].data = mean
                    self.std_dict[key].data = std

                    if torch.all(std == self.eps):
                        print(f"警告: {key} 字段的标准差为0，归一化后的值将为0。")
                    else:
                        print(f"Fitted {key}:")
                        print(f"  Global Mean: {mean.mean().item():.4f}, Global Std: {std.mean().item():.4f}")
                else:
                    # 如果是1D数据（如单帧数据），打印警告并跳过
                    print(f"警告: {key} 字段的数据是1D，跳过均值和标准差计算。")
            else:
                print(f"Key {key} 不在 lowdim_dict 中，跳过。")

    def normalize(self, data_dict):
        """
        对给定字典中的数据进行标准化

        参数:
        data_dict (dict): 字典，其中键是数据集名称，值是对应的数据集（torch.Tensor）

        返回:
        dict: 标准化后的数据字典
        """
        normalized_data_dict = {}
        for key, data in data_dict.items():
            if key in self.lowdim_dict:
                mean = self.mean_dict[key]
                std = self.std_dict[key]
                if torch.all(std == self.eps):
                    normalized_data = torch.zeros_like(data)
                    print(f"{key} 字段已被标准化为0，因为其标准差等于eps。")
                else:
                    normalized_data = (data - mean) / std
                normalized_data_dict[key] = normalized_data
            else:
                normalized_data_dict[key] = data
        return normalized_data_dict

    def denormalize(self, data_dict):
        """
        对给定字典中的标准化数据进行逆标准化

        参数:
        data_dict (dict): 字典，其中键是数据集名称，值是对应的标准化数据（torch.Tensor）

        返回:
        dict: 逆标准化后的数据字典
        """
        denormalized_data_dict = {}
        for key, data in data_dict.items():
            if key in self.lowdim_dict:
                mean = self.mean_dict[key]
                std = self.std_dict[key]
                denormalized_data = data * std + mean
                denormalized_data_dict[key] = denormalized_data
            else:
                denormalized_data_dict[key] = data
        return denormalized_data_dict

    def save(self, filepath: str):
        """
        Scaler의 파라미터를 파일로 저장합니다.

        매개변수:
        filepath (str): 저장 경로
        """
        torch.save(self.state_dict(), filepath)
        print(f"Scaler 파라미터가 {filepath}에 저장되었습니다.")

    def load(self, filepath: str):
        """
        파일에서 Scaler의 파라미터를 로드합니다.

        매개변수:
        filepath (str): 로드 경로
        """
        self.load_state_dict(torch.load(filepath, map_location='cpu'))
        print(f"Scaler 파라미터가 {filepath}에서 로드되었습니다.")
