import torch
import torch.nn as nn
import torch.nn.functional as F
from model_utils import RGAT, DiffLoss, MultiModelRGAT

class MoEGAT(nn.Module):
    def __init__(self, args, num_relation, multi_model=False):
        super(MoEGAT, self).__init__()
        self.num_experts = args.num_experts
        self.multi_model = multi_model
        self.args = args
        if multi_model:
            self.experts = nn.ModuleList([MultiModelRGAT(args, args.hidden_dim, args.hidden_dim, dropout=args.dropout, num_relation=num_relation) for _ in range(args.num_experts)])
        else:
            self.experts = nn.ModuleList([RGAT(args, args.hidden_dim, args.hidden_dim, dropout=args.dropout, num_relation=num_relation) for _ in range(args.num_experts)])
        self.gate = nn.Linear(args.hidden_dim, args.num_experts)

    def forward(self, x, adj, y=None):
        gate_values = F.softmax(self.gate(x), dim=-1)  # gate_values维度[batch_size, sequence_length, num_experts]

        if self.args.moe_routing_method == 'hard':
            # 硬路由：根据 gate_values 选择最大值对应的专家
            hard_gate_indices = torch.argmax(gate_values, dim=-1, keepdim=True)  # hard_gate_indices维度[batch_size, sequence_length, 1]
            hard_gate_values = torch.zeros_like(gate_values).scatter_(-1, hard_gate_indices, 1.0)  # one-hot 硬选择
            gate_values = hard_gate_values

        if self.multi_model:
            expert_outputs = torch.stack([expert(x, y, adj) for expert in self.experts], dim=1)  # expert_outputs维度[batch_size, num_experts, sequence_length, hidden_dim]
        else:
            expert_outputs = torch.stack([expert(x, adj) for expert in self.experts], dim=1)  # expert_outputs维度[batch_size, num_experts, sequence_length, hidden_dim]
        moe_output = torch.einsum('bnez,ben->bez', expert_outputs, gate_values)  # moe_output维度[batch_size, sequence_length, hidden_dim]
        # 使用moe负载均衡
        if self.args.use_moe_lb_loss:
            load_balance_loss = self.compute_load_balance_loss(gate_values)
            return moe_output, load_balance_loss
        
        # 不使用moe负载均衡
        return moe_output

    def compute_load_balance_loss(self, gate_values):
        # 计算每个专家的负载，即每个专家被选择的频率
        experts_load = torch.mean(gate_values, dim=[0, 1])  # experts_load维度[num_experts]
        # 计算负载平衡损失，使用熵损失来衡量负载的均匀性
        load_balance_loss = torch.sum(experts_load * torch.log(experts_load + 1e-10))
        # 将负负载平衡损失变为非负数
        load_balance_loss = -load_balance_loss
        return load_balance_loss
