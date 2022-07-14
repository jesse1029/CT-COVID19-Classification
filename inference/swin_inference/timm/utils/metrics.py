""" Eval metrics and related

Hacked together by / Copyright 2020 Ross Wightman
"""

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]

def perf_measure(y_actual, y_pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)): 
        if y_actual[i] == y_pred[i] == 1:
           TP += 1
        if y_pred[i] == 1 and y_actual[i] != y_pred[i]:
           FP += 1
        if y_actual[i] == y_pred[i] == 0:
           TN += 1
        if y_pred[i] == 0 and y_actual[i] != y_pred[i]:
           FN += 1

    return(TN, FP, FN, TP)

def cal_f1(output, target):
    
    maxk = min(max((1,)), output.size()[1])
    # batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    
    pred_numpy = pred.detach().cpu().numpy()
    target_numpy = target.detach().cpu().numpy()
    
    # print(f'pred:{pred_numpy[0]}')
    # print(f'ture:{target_numpy}')
    
    tn, fp, fn, tp = perf_measure(target_numpy, pred_numpy[0])
    
    return tn, fp, fn, tp, target_numpy, pred_numpy[0]



# confusion_matrix([1, 1, 1, 1], [1, 1, 1, 1]).ravel()

# pred = [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0]
# true = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

# tn, fp, fn, tp = perf_measure(true, pred)
# print(f'tn:{tn}, fp:{fp}, fn:{fn}, tp:{tp}')



# pred = [0, 0, 0]
# true = [0, 0, 0]

# # tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
# tn, fp, fn, tp = perf_measure(true, pred)
# print(f'tn:{tn}, fp:{fp}, fn:{fn}, tp:{tp}')
# if (fn + tp) == 0:
#     pos_acc = 999
# else:
#     pos_acc = tp / (fn + tp)
# if (tn + fp) == 0:
#     neg_acc =  999
# else:
#     neg_acc = fp / (tn + fp)
# print(f'pos_acc:{pos_acc}, neg_acc:{neg_acc}')


# pred = [1, 1, 1]
# true = [1, 1, 1]

# tn, fp, fn, tp = perf_measure(true, pred)
# print(f'tn:{tn}, fp:{fp}, fn:{fn}, tp:{tp}')
# if (fn + tp) == 0:
#     pos_acc = 999
# else:
#     pos_acc = tp / (fn + tp)
# if (tn + fp) == 0:
#     neg_acc =  999
# else:
#     neg_acc = fp / (tn + fp)
# print(f'pos_acc:{pos_acc}, neg_acc:{neg_acc}')




