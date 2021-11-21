#encode:utf-8
import numpy as np


def ACE_TDR_Cal(arr_result, rate=0.01):
    ace_list = []
    tdr_list = [0,]
    arr_result = np.array(arr_result)
    total = len(arr_result)
    for thres in arr_result[:,1]:
        TP = TN = FP = FN = 0
        for l,sc in arr_result:
            if sc > thres and l == 1:
                TP = TP + 1.
            elif sc <= thres and l== 0:
                TN = TN + 1.
            elif sc < thres and l==1:
                FN = FN + 1.
            else:
                FP = FP + 1.
        Ferrlive = FP / (FP + TN+1e-7)
        Ferrfake = FN / (FN + TP+1e-7)
        FDR = FP / (FP+TN+1e-7)
        TDR = TP / (TP+FN+1e-7)
        if FDR < rate:
            tdr_list.append(TDR)
        ace_list.append((Ferrlive+Ferrfake)/2.)
    return min(ace_list),max(tdr_list)