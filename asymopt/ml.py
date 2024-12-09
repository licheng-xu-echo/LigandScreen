from copy import deepcopy
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import pairwise_distances
from sklearn.metrics import r2_score,mean_absolute_error
from scipy.stats import pearsonr
def model_delta_pred(tgt_x,tgt_y,base_x,base_y,model,simi=False,dist_type='euclidean',topk=20,tgt_simi_desc=None,base_simi_desc=None,ret_metrics=False):
    cv = LeaveOneOut()
    base_model = deepcopy(model)
    delta_model = deepcopy(model)
    
    all_test_p = []
    all_test_y = []
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
    all_test_p = np.concatenate(all_test_p,axis=0)
    all_test_y = np.concatenate(all_test_y,axis=0)
    if ret_metrics:
        r2 = r2_score(all_test_y,all_test_p)
        mae = mean_absolute_error(all_test_y,all_test_p)
        prsr = pearsonr(all_test_y,all_test_p)[0]
        return all_test_y,all_test_p,r2,mae,prsr
    else:
        return all_test_y,all_test_p