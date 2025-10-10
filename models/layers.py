# utils/layers.py
import torch
import torch.nn as nn
import warnings
from typing import Optional, Union

class ResidualWrapper(nn.Module): 
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.module(x)

class ComplexProjector(nn.Module):
    """param:
        input_dim (int):  
        output_dim (int):  
        expansion_factor (int, optional): Defaults to 4.
        mid_depth (int, optional): Defaults to 2.
        use_residual (bool, optional):  Defaults to True.
        use_attention (bool, optional):  Defaults to False.
        attention_heads (int, optional):  Defaults to 2.
        dropout_rate (float, optional): Defaults to 0.1.
        activation (str, optional): ['gelu', 'silu', 'mish']. Defaults to 'gelu'.
        norm_type (str, optional):['layernorm', 'batchnorm']. Defaults to 'layernorm'.
        use_output_act (bool, optional): Defaults to True.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        expansion_factor: int = 4,
        mid_depth: int = 2,
        use_residual: bool = True,
        activation: str = "gelu",
        norm_type: str = "layernorm", 
    ):
        super().__init__()
        
        # 参数校验
        self._validate_arguments(expansion_factor, mid_depth)
        
        # 计算扩展维度
        expanded_dim = input_dim * expansion_factor
        
        # 激活函数选择
        self.activation = self._get_activation(activation)
        
        # 构建模块列表
        modules = nn.ModuleList()
         
        modules.extend([
            nn.Linear(input_dim, expanded_dim),
            self.activation(),
            self._get_norm_layer(norm_type, expanded_dim)
        ])
         
        for _ in range(mid_depth): 
             
            block = [
                nn.Linear(expanded_dim, expanded_dim),
                self.activation(), 
                self._get_norm_layer(norm_type, expanded_dim)
            ]
            
            # 添加残差连接
            if use_residual:
                block = [ResidualWrapper(nn.Sequential(*block))]
                
            modules.extend(block)
         
        modules.append(nn.Linear(expanded_dim, output_dim)) 
            
        self.layers = nn.Sequential(*modules)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        参数:
            x (torch.Tensor): 输入张量，形状为 (..., input_dim)
            
        返回:
            torch.Tensor: 输出张量，形状为 (..., output_dim)
        """ 
        
        # with torch.cuda.amp.autocast(dtype=torch.float32): 
        #     output = self.layers(x)
        # return output
        return self.layers(x)
    
    
    def _validate_arguments(
        self,
        expansion_factor: int,
        mid_depth: int
    ):
        """参数校验"""
        if expansion_factor < 1:
            raise ValueError("expansion_factor 必须大于等于1")
        if mid_depth < 0:
            raise ValueError("mid_depth 不能为负数") 
            
    def _get_activation(self, activation: str) -> nn.Module:
        """获取激活函数类"""
        act_dict = {
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "mish": nn.Mish
        }
        activation = activation.lower()
        if activation not in act_dict:
            raise ValueError(f"不支持的激活函数: {activation}，可选 {list(act_dict.keys())}")
        return act_dict[activation]
    
    def _get_norm_layer(
        self, 
        norm_type: str, 
        dim: int
    ) -> Union[nn.LayerNorm, nn.BatchNorm1d]:
        """获取归一化层"""
        norm_type = norm_type.lower()
        if norm_type == "layernorm":
            return nn.LayerNorm(dim)
        elif norm_type == "batchnorm":
            return nn.BatchNorm1d(dim)
        else:
            raise ValueError(f"不支持的归一化类型: {norm_type}")
            
    def get_parameter_count(self) -> int:
        """返回可训练参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
