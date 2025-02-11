from copy import deepcopy
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import pairwise_distances
from sklearn.metrics import r2_score,mean_absolute_error
from scipy.stats import pearsonr

def tanimoto_distance(x, y):

    dot_product = np.dot(x, y)
    x_norm = np.dot(x, x)
    y_norm = np.dot(y, y)
    similarity = dot_product / (x_norm + y_norm - dot_product)
    return 1 - similarity if (x_norm + y_norm - dot_product) != 0 else 1.0

def model_delta_pred(tgt_x,tgt_y,base_x,base_y,model,simi=False,dist_type='euclidean',topk=20,tgt_simi_desc=None,base_simi_desc=None,ret_metrics=False,ret_train_test=False):
    cv = LeaveOneOut()
    base_model = deepcopy(model)
    delta_model = deepcopy(model)
    if dist_type == 'tanimoto':
        dist_type = tanimoto_distance
    all_test_p = []
    all_test_y = []
    train_test_data_index = []
    if not simi:
        base_model.fit(base_x,base_y)
        for train_idx,test_idx in cv.split(tgt_x):
            train_x,test_x = tgt_x[train_idx],tgt_x[test_idx]
            train_y,test_y = tgt_y[train_idx],tgt_y[test_idx]
            train_p = base_model.predict(train_x)
            train_d = train_y - train_p 
            delta_model.fit(train_x,train_d)
            all_test_p.append(base_model.predict(test_x)+delta_model.predict(test_x))
            all_test_y.append(test_y)
            train_test_data_index.append([train_idx,test_idx])
    else:
        
        for train_idx,test_idx in cv.split(tgt_x):
            train_x,test_x = tgt_x[train_idx],tgt_x[test_idx]
            train_y,test_y = tgt_y[train_idx],tgt_y[test_idx]

            if tgt_simi_desc is None:
                dist_pair = pairwise_distances(train_x,base_x,metric=dist_type).sum(axis=0)
            else:
                train_simi_desc = tgt_simi_desc[train_idx]
                dist_pair = pairwise_distances(train_simi_desc,base_simi_desc,metric=dist_type).sum(axis=0)
            if isinstance(topk,int):
                simi_data_idx = np.argsort(dist_pair)[:topk] 
            else:
                std_dist_pair = (dist_pair-dist_pair.min())/(dist_pair.max()-dist_pair.min())
                simi_data_idx = np.where(std_dist_pair<topk)[0]  
                
            if len(simi_data_idx) <= 1:
                if ret_metrics:
                    return [None] * 5
                else:
                    return [None] * 2
            base_model.fit(base_x[simi_data_idx],base_y[simi_data_idx])
            train_p = base_model.predict(train_x)
            train_d = train_y - train_p
            delta_model.fit(train_x,train_d)
            all_test_p.append(base_model.predict(test_x)+delta_model.predict(test_x))
            all_test_y.append(test_y)
            
            train_test_data_index.append([train_idx,test_idx,simi_data_idx])
    all_test_p = np.concatenate(all_test_p,axis=0)
    all_test_y = np.concatenate(all_test_y,axis=0)
    if ret_metrics:
        r2 = r2_score(all_test_y,all_test_p)
        mae = mean_absolute_error(all_test_y,all_test_p)
        prsr = pearsonr(all_test_y,all_test_p)[0]
        if not ret_train_test:
            return all_test_y,all_test_p,r2,mae,prsr
        else:
            return all_test_y,all_test_p,r2,mae,prsr,train_test_data_index
    else:
        if not ret_train_test:

            return all_test_y,all_test_p
        else:
            return all_test_y,all_test_p,train_test_data_index
    
def model_delta_pred_virt(model,base_x,base_y,delta_x,delta_y,tgt_x,
                          dist_type='cosine',topk=20,tgt_simi_desc=None,base_simi_desc=None):
    dist_type = dist_type.lower()
    base_model = deepcopy(model)
    delta_model = deepcopy(model)

    simi_data_idx_data_pt_map = {}
    for data_idx in range(len(tgt_x)):
        sel_tgt_x = tgt_x[data_idx].reshape(1,-1)
        simi_data_idx = None
        if tgt_simi_desc is None:
            dist_pair = pairwise_distances(sel_tgt_x,base_x,metric=dist_type).sum(axis=0)
        else:
            sel_tgt_simi_desc = tgt_simi_desc[data_idx].reshape(1,-1)
            dist_pair = pairwise_distances(sel_tgt_simi_desc,base_simi_desc,metric=dist_type).sum(axis=0)
        if isinstance(topk,int):
            simi_data_idx = np.argsort(dist_pair)[:topk]  
        else:
            std_dist_pair = (dist_pair-dist_pair.min())/(dist_pair.max()-dist_pair.min())
            simi_data_idx = np.where(std_dist_pair<topk)[0]  
            if len(simi_data_idx) <= 1:
                simi_data_idx = np.argsort(dist_pair)[:50]

        if simi_data_idx is None or len(simi_data_idx) <= 1:
            return None
        simi_data_idx = tuple(sorted(simi_data_idx.tolist()))
        if not simi_data_idx in simi_data_idx_data_pt_map:
            simi_data_idx_data_pt_map[simi_data_idx] = [data_idx]
        else:
            simi_data_idx_data_pt_map[simi_data_idx].append(data_idx)
    print(f'[INFO] There are {len(simi_data_idx_data_pt_map)} time(s) different delta prediciton')

    tgt_p = -np.ones(len(tgt_x))*9999  ## initialize
    for simi_data_idx in simi_data_idx_data_pt_map:
        simi_data_x = base_x[list(simi_data_idx)]
        simi_data_y = base_y[list(simi_data_idx)]
        sel_tgt_x = tgt_x[simi_data_idx_data_pt_map[simi_data_idx]]
        base_model.fit(simi_data_x,simi_data_y)
        delta_p = base_model.predict(delta_x)
        delta_d = delta_y - delta_p
        delta_model.fit(delta_x,delta_d)
        sel_tgt_p = base_model.predict(sel_tgt_x) + delta_model.predict(sel_tgt_x)
        tgt_p[simi_data_idx_data_pt_map[simi_data_idx]] = sel_tgt_p
    
    return tgt_p