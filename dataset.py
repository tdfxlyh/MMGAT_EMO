from turtle import pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import random

import pickle
def read_pickle(filename):
    try:
        with open(filename,'rb') as f:
            data = pickle.load(f)
    except:
        with open(filename,'rb') as f:
            data = pickle.load(f,encoding='latin1')
    return data

class MyDataset(Dataset):

    def __init__(self, dataset_name = 'IEMOCAP', split = 'train', args = None):
        self.args = args
        self.dataset_name = dataset_name

        if dataset_name == 'IEMOCAP':
            self.videoSpeakers, self.videoLabels, _, \
            self.links, self.relations, self.videoSentence, self.trainVid, self.testVid, self.validVid = read_pickle('./data/IEMOCAP/IEMOCAP_Link_Relation_Features.pkl')

            roberta_feature = read_pickle('./data/IEMOCAP/TextFeatures.pkl')
            self.videoAudio = read_pickle('./data/IEMOCAP/AudioFeatures.pkl')
            self.videoVisual = read_pickle('./data/IEMOCAP/VisualFeatures.pkl')

            self.utterance_feature = roberta_feature
       
        elif dataset_name == 'MELD':
            self.videoSpeakers, self.videoLabels,  _, \
            self.links, self.relations, self.videoSentence, self.trainVid, self.testVid, self.validVid = read_pickle('./data/MELD/MELD_Link_Relation_Features.pkl')

            roberta_feature = read_pickle('./data/MELD/TextFeatures.pkl')
            self.videoAudio = read_pickle('./data/MELD/AudioFeatures.pkl')
            self.videoVisual = read_pickle('./data/MELD/VisualFeatures.pkl')

            self.utterance_feature = roberta_feature

        self.data = self.read(split)
        self.len = len(self.data)

    def read(self, split):
        # process dialogue
        if split=='train':
            dialog_ids = self.trainVid
        elif split=='dev':
            dialog_ids = self.validVid
        elif split=='test':
            dialog_ids = self.testVid

        dialogs = []
        for dialog_id in dialog_ids:
            utterances = self.videoSentence[dialog_id]
            labels = self.videoLabels[dialog_id]
            if self.dataset_name == 'IEMOCAP':
                speakers = self.videoSpeakers[dialog_id]
            elif self.dataset_name == 'MELD':
                speakers = [speaker.index(1) for speaker in self.videoSpeakers[dialog_id]]

            utterance_features = [item.tolist() for item in self.utterance_feature[dialog_id]]
            audio_features = [item.tolist() for item in self.videoAudio[dialog_id]]
            visual_features = [item.tolist() for item in self.videoVisual[dialog_id]]
            utterance_links = self.links[dialog_id]
            utterance_relations = self.relations[dialog_id]

            dialogs.append({
                'id':dialog_id,
                'utterances': utterances,
                'labels': labels,
                'speakers': speakers,
                'utterance_features': utterance_features,
                'audio_features': audio_features,
                'visual_features': visual_features,
                'utterance_links': utterance_links,
                'utterance_relations': utterance_relations
            })

        random.shuffle(dialogs) # 打乱对话
        return dialogs

    def __getitem__(self, index):  # 获取一个样本/ 对话
        return torch.FloatTensor(self.data[index]['utterance_features']), \
               torch.LongTensor(self.data[index]['labels']),\
               self.data[index]['speakers'], \
               self.data[index]['utterance_links'],\
               self.data[index]['utterance_relations'],\
               len(self.data[index]['labels']), \
               self.data[index]['utterances'],\
               self.data[index]['id'],\
               torch.FloatTensor(self.data[index]['audio_features']),\
               torch.FloatTensor(self.data[index]['visual_features'])

    def __len__(self):
        return self.len
    
    def get_semantic_adj(self, speakers, max_dialog_len):
        semantic_adj = []
        for speaker in speakers:  # 遍历每个对话 对应的说话人列表（非去重）
            s = torch.zeros(max_dialog_len, max_dialog_len, dtype = torch.long) # （N,N） 0 表示填充部分 没有语义关系
            for i in range(len(speaker)): # 每个utterance 的说话人 和 其他 utterance 的说话人 是否相同
                for j in range(len(speaker)):
                    if speaker[i] == speaker[j]:
                        if i==j:
                            s[i,j] = 1  # 对角线  self
                        elif i < j:
                            s[i,j] = 2   # self-future
                        else:
                            s[i,j] =3    # self-past
                    else:
                        if i<j:
                            s[i,j] = 4   # inter-future
                        elif i>j:
                            s[i,j] = 5   # inter-past

            semantic_adj.append(s)
        
        return torch.stack(semantic_adj)


    def get_structure_adj(self, links, relations, lengths, max_dialog_len):
        '''
        map_relations = {'Comment': 0, 'Contrast': 1, 'Correction': 2, 'Question-answer_pair': 3, 'QAP': 3, 'Parallel': 4, 'Acknowledgement': 5,
                     'Elaboration': 6, 'Clarification_question': 7, 'Conditional': 8, 'Continuation': 9, 'Result': 10, 'Explanation': 11,
                     'Q-Elab': 12, 'Alternation': 13, 'Narration': 14, 'Background': 15}

        '''
        structure_adj = []
        for link,relation,length in zip(links,relations,lengths):  
            s = torch.zeros(max_dialog_len, max_dialog_len, dtype = torch.long) # （N,N） 0 表示填充部分 或 没有关系
            assert len(link)==len(relation)
            for index, (i,j) in enumerate(link):
                s[i,j] = relation[index] + 1
                s[j,i] = s[i,j]   # 变成对称矩阵

            for i in range(length):  # 填充对角线
                s[i,i] = 17

            structure_adj.append(s)
        
        return torch.stack(structure_adj)



    def collate_fn(self, data):  # data 是一个batch 的对话    获取一批样本/对话 并填充
        '''
        :param data:
            utterance_features, labels, speakers, utterance_links, utterance_relations, length, texts, id, audio_features, visual_features
        :return:
            text_features: (B, N, D) padded
    
            labels: (B, N) padded
            adj: (B, N, N) adj[:,i,:] means the direct predecessors of node i
            s_mask: (B, N, N) s_mask[:,i,:] means the speaker informations for predecessors of node i, where 1 denotes the same speaker, 0 denotes the different speaker
            lengths: (B, )
            utterances:  not a tensor
        '''
        max_dialog_len = max([d[5] for d in data]) # batch 中 对话的最大长度 N

        utterance_features = pad_sequence([d[0] for d in data], batch_first = True) # (B, N, D)
        audio_features     = pad_sequence([d[8] for d in data], batch_first = True) 
        visual_features    = pad_sequence([d[9] for d in data], batch_first = True) 

        labels = pad_sequence([d[1] for d in data], batch_first = True, padding_value = -1) # (B, N ) label 填充值为 -1

        semantic_adj = self.get_semantic_adj([d[2] for d in data], max_dialog_len)

        structure_adj = self.get_structure_adj([d[3] for d in data], [d[4] for d in data], [d[5] for d in data],max_dialog_len)

        lengths = torch.LongTensor([d[5] for d in data]) # batch 每个对话的长度

        speakers = pad_sequence([torch.LongTensor(d[2]) for d in data], batch_first = True, padding_value = -1) # (B, N) speaker 填充值为 -1
        utterances = [d[6] for d in data]  # batch 中每个对话对应的 utterance 文本
        ids = [d[7] for d in data]  # batch 中每个对话对应的id

        return utterance_features, audio_features, visual_features, labels, semantic_adj, structure_adj, lengths, speakers, utterances, ids
