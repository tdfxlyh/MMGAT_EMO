from model_utils import RGAT, DiffLoss, MultiModelRGAT, PositionalEncoding
import torch
import torch.nn as nn
import torch.nn.functional as F
from CPC import Cross_CPC, CPC
from MoE import MoEGAT

# 加moe的方式二

class MMGATs(nn.Module):

    def __init__(self, args, num_class):
        super().__init__()
        self.args = args
        
        self.fc1 = nn.Linear(args.emb_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.audio_emb_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.visual_emb_dim, args.hidden_dim)

        # 定义四个数组管理MoE模块
        SpkGAT_MoE = []
        DisGAT_MoE = []
        MultiSpkGAT_MoE = []
        MultiDisGAT_MoE = []
        for _ in range(args.gnn_layers):
            SpkGAT_MoE.append(MoEGAT(args, num_relation=6))
            DisGAT_MoE.append(MoEGAT(args, num_relation=18))
            MultiSpkGAT_MoE.append(MoEGAT(args, num_relation=6, multi_model=True))
            MultiDisGAT_MoE.append(MoEGAT(args, num_relation=18, multi_model=True))
        self.SpkGAT_MoE = nn.ModuleList(SpkGAT_MoE)
        self.DisGAT_MoE = nn.ModuleList(DisGAT_MoE)
        self.MultiSpkGAT_MoE = nn.ModuleList(MultiSpkGAT_MoE)
        self.MultiDisGAT_MoE = nn.ModuleList(MultiDisGAT_MoE)


        self.affine1 = nn.Parameter(torch.empty(size=(args.hidden_dim, args.hidden_dim)))
        nn.init.xavier_uniform_(self.affine1.data, gain=1.414)
        self.affine2 = nn.Parameter(torch.empty(size=(args.hidden_dim, args.hidden_dim)))
        nn.init.xavier_uniform_(self.affine2.data, gain=1.414)

        # 下面是 文本与音频，文本与视频，【模态之间】的
        self.affine7 = nn.Parameter(torch.empty(size=(args.hidden_dim, args.hidden_dim)))
        nn.init.xavier_uniform_(self.affine7.data, gain=1.414)
        self.affine8 = nn.Parameter(torch.empty(size=(args.hidden_dim, args.hidden_dim)))
        nn.init.xavier_uniform_(self.affine8.data, gain=1.414)
        self.affine9 = nn.Parameter(torch.empty(size=(args.hidden_dim, args.hidden_dim)))
        nn.init.xavier_uniform_(self.affine9.data, gain=1.414)
        self.affine10 = nn.Parameter(torch.empty(size=(args.hidden_dim, args.hidden_dim)))
        nn.init.xavier_uniform_(self.affine10.data, gain=1.414)



        self.diff_loss = DiffLoss(args)
        if self.args.use_nce_loss:
            self.nce_loss = CPC(args.hidden_dim, args.hidden_dim, args.hidden_dim, 2, n_prediction_steps=5) if self.args.nce_single_or_cross == 1 else Cross_CPC(args.hidden_dim, args.hidden_dim, args.hidden_dim, 2, n_prediction_steps=5)
        self.beta = 0.3

        in_dim = args.hidden_dim * 6 + args.emb_dim + args.audio_emb_dim +args.visual_emb_dim    


        # output mlp layers
        layers = [nn.Linear(in_dim, args.hidden_dim), nn.ReLU()]
        for _ in range(args.mlp_layers - 1):
            layers += [nn.Linear(args.hidden_dim, args.hidden_dim), nn.ReLU()]
        layers += [nn.Linear(args.hidden_dim, num_class)]

        self.out_mlp = nn.Sequential(*layers)

        self.drop = nn.Dropout(args.dropout)

       

    def forward(self, utterance_features, audio_features, visual_features, semantic_adj, structure_adj):
        '''
        :param tutterance_features: (B, N, emb_dim)
        :param xx_adj: (B, N, N)
        :return:
        '''
        batch_size = utterance_features.size(0)

        H0 = [None, None, None, None, None] 
        H0[0] = F.relu(self.fc1(utterance_features))   # (B, N, hidden_dim)
        H0[1] = F.relu(self.fc2(audio_features))       # (B, N, hidden_dim)
        H0[2] = F.relu(self.fc3(visual_features))      # (B, N, hidden_dim)
        H = [H0]

        diff_loss = 0           # diff loss
        nce_loss = 0            # nce loss
        moe_lb_loss = 0         # moe loss


        # 单模态cpc:nce loss：文本
        # 多模态cpc:nce loss结合：文本+音频 文本+视频 （不添加音频+视频，防止冲突）
        if self.args.use_nce_loss:
            nce_loss += self.nce_loss(H0[0]) if self.args.nce_single_or_cross == 1 else self.nce_loss(H0[0], H0[1]) + self.nce_loss(H0[0], H0[2])
    
        
        for l in range(self.args.gnn_layers):
            # 加moe loss
            if self.args.use_moe_lb_loss:
                if l == 0:
                    H1_semantic, H1_semantic_moe_loss = self.SpkGAT_MoE[l](H[l][0], semantic_adj)
                    H1_structure, H1_structure_moe_loss = self.DisGAT_MoE[l](H[l][0], structure_adj)

                    H1_Multi_sem_text_audio, H1_Multi_sem_text_audio_moe_loss = self.MultiSpkGAT_MoE[l](H[l][0], semantic_adj, H[l][1])
                    H1_Multi_sem_text_visual, H1_Multi_sem_text_visual_moe_loss = self.MultiSpkGAT_MoE[l](H[l][0], semantic_adj, H[l][2])

                    H1_Multi_stu_text_audio, H1_Multi_stu_text_audio_moe_loss = self.MultiDisGAT_MoE[l](H[l][0], structure_adj, H[l][1])
                    H1_Multi_stu_text_visual, H1_Multi_stu_text_visual_moe_loss = self.MultiDisGAT_MoE[l](H[l][0], structure_adj, H[l][2])


                else:
                    H1_semantic, H1_semantic_moe_loss = self.SpkGAT_MoE[l](H[2*l-1][0], semantic_adj)
                    H1_structure, H1_structure_moe_loss = self.DisGAT_MoE[l](H[2*l][0], structure_adj)
                   
                    multi_audio_index, multi_visual_index = 1, 2
                    H1_Multi_sem_text_audio, H1_Multi_sem_text_audio_moe_loss = self.MultiSpkGAT_MoE[l](H[2*l-1][0], semantic_adj, H[2*l-1][multi_audio_index])
                    H1_Multi_sem_text_visual, H1_Multi_sem_text_visual_moe_loss = self.MultiSpkGAT_MoE[l](H[2*l-1][0], semantic_adj, H[2*l-1][multi_visual_index])

                    H1_Multi_stu_text_audio, H1_Multi_stu_text_audio_moe_loss = self.MultiDisGAT_MoE[l](H[2*l][0], structure_adj, H[2*l][multi_audio_index])
                    H1_Multi_stu_text_visual, H1_Multi_stu_text_visual_moe_loss = self.MultiDisGAT_MoE[l](H[2*l][0], structure_adj, H[2*l][multi_visual_index])


                 # moe loss
                moe_lb_loss = moe_lb_loss + H1_semantic_moe_loss + H1_structure_moe_loss + H1_Multi_sem_text_audio_moe_loss + H1_Multi_sem_text_visual_moe_loss + H1_Multi_stu_text_audio_moe_loss + H1_Multi_stu_text_visual_moe_loss
            else: # 不加moe loss
                if l == 0:
                    H1_semantic = self.SpkGAT_MoE[l](H[l][0], semantic_adj)
                    H1_structure = self.DisGAT_MoE[l](H[l][0], structure_adj)

                    H1_Multi_sem_text_audio = self.MultiSpkGAT_MoE[l](H[l][0], semantic_adj, H[l][1])
                    H1_Multi_sem_text_visual = self.MultiSpkGAT_MoE[l](H[l][0], semantic_adj, H[l][2])

                    H1_Multi_stu_text_audio = self.MultiDisGAT_MoE[l](H[l][0], structure_adj, H[l][1])
                    H1_Multi_stu_text_visual = self.MultiDisGAT_MoE[l](H[l][0], structure_adj, H[l][2])


                else:
                    H1_semantic = self.SpkGAT_MoE[l](H[2*l-1][0], semantic_adj)
                    H1_structure = self.DisGAT_MoE[l](H[2*l][0], structure_adj)

                    multi_audio_index, multi_visual_index = 1, 2
                    H1_Multi_sem_text_audio = self.MultiSpkGAT_MoE[l](H[2*l-1][0], semantic_adj, H[2*l-1][multi_audio_index])
                    H1_Multi_sem_text_visual = self.MultiSpkGAT_MoE[l](H[2*l-1][0], semantic_adj, H[2*l-1][multi_audio_index])

                    H1_Multi_stu_text_audio = self.MultiDisGAT_MoE[l](H[2*l][0], structure_adj, H[2*l][multi_audio_index])
                    H1_Multi_stu_text_visual = self.MultiDisGAT_MoE[l](H[2*l][0], structure_adj, H[2*l][multi_audio_index])



            # -------------------------------------------------------------------------------------------------loss 部分 start
            if self.args.use_diff_loss:
                diff_loss = diff_loss + self.diff_loss(H1_semantic, H1_structure) + \
                            self.diff_loss(H1_Multi_sem_text_audio, H1_Multi_stu_text_audio) + \
                                self.diff_loss(H1_Multi_sem_text_visual, H1_Multi_stu_text_visual)
            # -------------------------------------------------------------------------------------------------loss 部分 end


            # BiAffine 
            # 文本内部
            A1 = F.softmax(torch.bmm(torch.matmul(H1_semantic, self.affine1), torch.transpose(H1_structure, 1, 2)), dim=-1)
            A2 = F.softmax(torch.bmm(torch.matmul(H1_structure, self.affine2), torch.transpose(H1_semantic, 1, 2)), dim=-1)

            # 文本音频
            A7 = F.softmax(torch.bmm(torch.matmul(H1_Multi_sem_text_audio, self.affine7), torch.transpose(H1_Multi_stu_text_audio, 1, 2)), dim=-1)
            A8 = F.softmax(torch.bmm(torch.matmul(H1_Multi_stu_text_audio, self.affine8), torch.transpose(H1_Multi_sem_text_audio, 1, 2)), dim=-1)
            
            # 文本视频
            A9 = F.softmax(torch.bmm(torch.matmul(H1_Multi_sem_text_visual, self.affine9), torch.transpose(H1_Multi_stu_text_visual, 1, 2)), dim=-1)
            A10= F.softmax(torch.bmm(torch.matmul(H1_Multi_stu_text_visual, self.affine10), torch.transpose(H1_Multi_sem_text_visual, 1, 2)), dim=-1)



            H1_semantic_new = torch.bmm(A1, H1_structure)
            H1_structure_new = torch.bmm(A2, H1_semantic)

            H1_multi_audio_semantic_new = torch.bmm(A7, H1_Multi_stu_text_audio)
            H1_multi_audio_structure_new = torch.bmm(A8, H1_Multi_sem_text_audio)

            H1_multi_visual_semantic_new = torch.bmm(A9, H1_Multi_stu_text_visual)
            H1_multi_visual_structure_new = torch.bmm(A10, H1_Multi_sem_text_visual)

            H1_semantic_out = self.drop(H1_semantic_new) if l < self.args.gnn_layers - 1 else H1_semantic_new
            H1_structure_out = self.drop(H1_structure_new) if l <self.args.gnn_layers - 1 else H1_structure_new

            H1_multi_audio_semantic_out = self.drop(H1_multi_audio_semantic_new) if l < self.args.gnn_layers - 1 else H1_multi_audio_semantic_new
            H1_multi_audio_structure_out = self.drop(H1_multi_audio_structure_new) if l <self.args.gnn_layers - 1 else H1_multi_audio_structure_new

            H1_multi_visual_semantic_out = self.drop(H1_multi_visual_semantic_new) if l < self.args.gnn_layers - 1 else H1_multi_visual_semantic_new
            H1_multi_visual_structure_out = self.drop(H1_multi_visual_structure_new) if l <self.args.gnn_layers - 1 else H1_multi_visual_structure_new

            H.append([H1_semantic_out, H1_multi_audio_semantic_out, H1_multi_visual_semantic_out])
            H.append([H1_structure_out, H1_multi_audio_structure_out, H1_multi_visual_structure_out])


        H.append([utterance_features, audio_features, visual_features]) 

        H = torch.cat([H[-3][0],H[-3][1],H[-3][2],  H[-2][0],H[-2][1],H[-2][2], H[-1][0],H[-1][1],H[-1][2]], dim = 2) #(B, N, 6*hidden_dim+emb_dim+audio_emb_dim+visual_emb_dim)  只需要把最后一层的输出 和 原始特征 拼在一起就行

        logits = self.out_mlp(H)

        return logits, self.beta * (diff_loss/self.args.gnn_layers), nce_loss, self.beta * (moe_lb_loss/self.args.gnn_layers)







