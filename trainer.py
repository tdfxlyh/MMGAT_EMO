import numpy as np
import torch

from sklearn.metrics import f1_score, accuracy_score

import json


def train_or_eval_model(model, loss_function, dataloader, device, args, optimizer=None, train=False, print_loss=True):
    losses, preds, labels = [], [], []

    assert not train or optimizer != None
    if train:  
        model.train()
    else: 
        model.eval()

    for index, data in enumerate(dataloader): 
        if train:
            optimizer.zero_grad()

        utterance_features, audio_features, visual_features, label, semantic_adj, structure_adj, lengths, speakers, utterances, ids = data


        utterance_features = utterance_features.to(device)
        audio_features = audio_features.to(device)
        visual_features = visual_features.to(device)
        label = label.to(device)  # (B,N)
        semantic_adj = semantic_adj.to(device)
        structure_adj = structure_adj.to(device)

        log_prob, diff_loss, nce_loss, lb_loss = model(utterance_features, audio_features, visual_features, semantic_adj, structure_adj) # (B, N, C)
        loss = loss_function(log_prob.permute(0,2,1), label) * 3

        if print_loss and index%5==0:
            print("loss:", get_item_value(loss), "diff_loss:",get_item_value(diff_loss),"nce_loss:", get_item_value(nce_loss),"lb_loss",get_item_value(lb_loss), "total_loss:", get_item_value(loss + diff_loss + nce_loss + lb_loss))

        loss = loss + diff_loss + nce_loss + lb_loss
        label = label.cpu().numpy().tolist()
        pred = torch.argmax(log_prob, dim = 2).cpu().numpy().tolist() # (B,N)
        preds += pred
        labels += label
        losses.append(loss.item())

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

    if preds != []:
        new_preds = []
        new_labels = []
        for i,label in enumerate(labels): # 遍历每个对话
            for j,l in enumerate(label): # 遍历每个utterance
                if l != -1: # 去除填充标签 （IEMOCAP内部utterance 也有填充）
                    new_labels.append(l)
                    new_preds.append(preds[i][j])
    else:
        return float('nan'), float('nan'), [], [], float('nan'), [], [], [], [], []

    avg_loss = round(np.sum(losses) / len(losses), 4)
    avg_accuracy = round(accuracy_score(new_labels, new_preds) * 100, 2)

    if args.dataset_name in ['IEMOCAP', 'MELD', 'EmoryNLP_small', 'EmoryNLP_big']:
        avg_fscore = round(f1_score(new_labels, new_preds, average='weighted') * 100, 2)
    elif args.dataset_name == 'DailyDialog':
        avg_fscore = round(f1_score(new_labels, new_preds, average='micro', labels=[0,2,3,4,5,6]) * 100, 2) #1 is neutral

    return avg_loss, avg_accuracy, labels, preds, avg_fscore


def get_item_value(val):
    if type(val)!=int and type(val)!=float:
        return val.item()
    return val