import torch
import torch.nn as nn

class Cross_CPC(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, context_dim, num_layers, n_prediction_steps=2, min_start_steps=3):
        super(Cross_CPC, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.num_layers = num_layers
        self.n_prediction_steps = n_prediction_steps
        self.min_start_steps = min_start_steps
        self.softmax  = nn.Softmax()
        self.lsoftmax = nn.LogSoftmax(dim=1)    # 防止后面警告，手动指定维度
        
        # Autoregressive LSTM network for text
        self.text_ar_lstm = nn.LSTM(embedding_dim, context_dim, num_layers, batch_first=True)
        
        # Autoregressive LSTM network for video or audio
        self.other_ar_lstm = nn.LSTM(embedding_dim, context_dim, num_layers, batch_first=True)
        
        # Predictor network for text
        self.text_predictors = nn.ModuleList([
            nn.Linear(context_dim, embedding_dim) for _ in range(n_prediction_steps)
        ])
        
        # Predictor network for video or audio
        self.other_predictors = nn.ModuleList([
            nn.Linear(context_dim, embedding_dim) for _ in range(n_prediction_steps)
        ])
    
    """
    text_forward_seq取了[0:t_samples]这t_samples+1样本
    text_encode_samples取了[t_samples+1:t_samples+self.n_prediction_steps]这些样本作为真实未来结果
    lstm利用text_forward_seq进行得到text_video_context
    再用text_predictors预测得到text_pred
    pred再和encode_samples进行nce计算
    """
    def forward(self, text_vq, other_vq):
        batch_dim, time_length, _ = text_vq.shape # [batch_dim, time_length, embedding_dim] e.g.[80, 10, 256]     # 32 23 300
        # 左闭右开
        # 从[3, 8)中随便一个作为开始，然后预测后2位，所以forward_seq最少都有4位（从0开始），
        t_samples = (torch.randint(time_length - self.n_prediction_steps - self.min_start_steps, size=(1,)) + self.min_start_steps).long() # randomly pick time stamps
        # losses = list()
        nce = 0 # average over timestep and batch
        text_encode_samples = torch.empty((self.n_prediction_steps,batch_dim,self.embedding_dim), device = text_vq.device).double() # e.g. size 5*80*256    # 5 32 300
        other_encode_samples = torch.empty((self.n_prediction_steps,batch_dim,self.embedding_dim), device = other_vq.device).double() # e.g. size 5*80*256
        for i in range(0, self.n_prediction_steps):# 左闭右开，预测步重合一步
            text_encode_samples[i-1] = text_vq[:,t_samples+i,:].reshape(batch_dim,self.embedding_dim) # z_tk e.g. size 80*256
            other_encode_samples[i-1] = other_vq[:,t_samples+i,:].reshape(batch_dim,self.embedding_dim) # z_tk e.g. size 80*256
        text_forward_seq = text_vq[:,:t_samples+1,:] # e.g. size 80*t_samples*256
        other_forward_seq = other_vq[:,:t_samples+1,:] # e.g. size 80*t_samples*256
        # Autoregressive LSTM for text
        text_hidden = (torch.zeros(self.num_layers, batch_dim, self.hidden_dim, device = text_vq.device).float(),
                  torch.zeros(self.num_layers, batch_dim, self.hidden_dim, device = text_vq.device).float())
        text_context, text_hidden = self.text_ar_lstm(text_forward_seq, text_hidden)           # 32 5 300   2 32 300
        
        # Autoregressive LSTM for video or audio
        other_hidden = (torch.zeros(self.num_layers, batch_dim, self.hidden_dim, device = other_vq.device).float(),
                  torch.zeros(self.num_layers, batch_dim, self.hidden_dim, device = other_vq.device).float())
        other_context, other_hidden = self.other_ar_lstm(other_forward_seq, other_hidden)
        
        text_context = text_context[:,t_samples,:].reshape(batch_dim,self.context_dim) # c_t e.g. size 80*512
        other_context = other_context[:,t_samples,:].reshape(batch_dim,self.context_dim) # c_t e.g. size 80*512
        
        text_pred = torch.empty((self.n_prediction_steps,batch_dim,self.embedding_dim), device = text_vq.device).double() # e.g. size 5*80*256
        other_pred = torch.empty((self.n_prediction_steps,batch_dim,self.embedding_dim), device = other_vq.device).double() # e.g. size 5*80*256
        for i in range(0, self.n_prediction_steps):
            text_linear = self.text_predictors[i]
            text_pred[i] = text_linear(text_context) #e.g. size 80*512 -> 80*256
            other_linear = self.other_predictors[i]
            other_pred[i] = other_linear(other_context) #e.g. size 80*512 -> 80*256
        for i in range(0, self.n_prediction_steps):
            total1 = torch.mm(text_encode_samples[i], torch.transpose(other_pred[i],0,1)) # e.g. size 80*80
            total2 = torch.mm(other_encode_samples[i], torch.transpose(text_pred[i],0,1)) # e.g. size 80*80
            total3 = torch.mm(text_encode_samples[i], torch.transpose(text_pred[i],0,1)) # e.g. size 80*80
            total4 = torch.mm(other_encode_samples[i], torch.transpose(other_pred[i],0,1)) # e.g. size 80*80
            w1 = 1.0
            w2 = 1.0
            w3 = 0.1
            w4 = 0.1
            nce += w1 * torch.sum(torch.diag(self.lsoftmax(total1))) # nce is a tensor
            nce += w2 * torch.sum(torch.diag(self.lsoftmax(total2))) # nce is a tensor
            nce += w3 * torch.sum(torch.diag(self.lsoftmax(total3))) # nce is a tensor
            nce += w4 * torch.sum(torch.diag(self.lsoftmax(total4))) # nce is a tensor
            
        nce /= -1.*batch_dim*self.n_prediction_steps
        return 0.05 * nce


class CPC(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, context_dim, num_layers, n_prediction_steps=2, min_start_steps=3):
        super(CPC, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.num_layers = num_layers
        self.n_prediction_steps = n_prediction_steps
        self.min_start_steps = min_start_steps
        self.softmax  = nn.Softmax()
        self.lsoftmax = nn.LogSoftmax(dim=1)    
        
        # Autoregressive LSTM network
        self.ar_lstm = nn.LSTM(embedding_dim, context_dim, num_layers, batch_first=True)
        
        # Predictor network
        self.predictors = nn.ModuleList([
            nn.Linear(context_dim, embedding_dim) for _ in range(n_prediction_steps)
        ])

    def forward(self, input_vq):
        batch_dim, time_length, _ = input_vq.shape 
        t_samples = (torch.randint(time_length - self.n_prediction_steps - self.min_start_steps, size=(1,)) + self.min_start_steps).long() # randomly pick time stamps
        nce = 0 
        encode_samples = torch.empty((self.n_prediction_steps,batch_dim,self.embedding_dim), device = input_vq.device).double()
        for i in range(0, self.n_prediction_steps):
            encode_samples[i-1] = input_vq[:,t_samples+i,:].reshape(batch_dim,self.embedding_dim) 
        forward_seq = input_vq[:,:t_samples+1,:] 
        hidden = (torch.zeros(self.num_layers, batch_dim, self.hidden_dim, device = input_vq.device).float(),
                  torch.zeros(self.num_layers, batch_dim, self.hidden_dim, device = input_vq.device).float())
        context, hidden = self.ar_lstm(forward_seq, hidden)          
        context = context[:,t_samples,:].reshape(batch_dim,self.context_dim) 
        
        pred = torch.empty((self.n_prediction_steps,batch_dim,self.embedding_dim), device = input_vq.device).double() 
        for i in range(0, self.n_prediction_steps):
            linear = self.predictors[i]
            pred[i] = linear(context)
        for i in range(0, self.n_prediction_steps):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i],0,1)) 
            w = 1.0
            nce += w * torch.sum(torch.diag(self.lsoftmax(total)))
            
        nce /= -1.*batch_dim*self.n_prediction_steps
        return 0.3 * nce
