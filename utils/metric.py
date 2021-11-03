import numpy as np 


class ConfusionMatrix(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class**2)
        hist = hist.reshape(n_class, n_class)
        
        return hist

    def update(self, label_trues, label_preds):
        if not isinstance(label_preds, list):
            self.confusion_matrix += self._fast_hist(label_trues.flatten(), label_preds.flatten(), self.n_classes)
        else:
            for lt, lp in zip(label_trues, label_preds):
                tmp = self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)
                self.confusion_matrix += tmp

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - IoU
            - mean IoU
            - dice
            - mean dice
            - IoU based on frequency
        """
        hist = self.confusion_matrix
        # accuracy is recall/sensitivity for each class, predicted TP / all real positives
        acc = np.nan_to_num(np.diag(hist) / hist.sum(axis=1))
        mean_acc = np.mean(np.nan_to_num(acc))
        
        intersect = np.diag(hist)
        union = hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        iou = intersect / union
        mean_iou = np.mean(np.nan_to_num(iou))
        
        dice = 2 * intersect / (hist.sum(axis=1) + hist.sum(axis=0))
        mean_dice = np.mean(np.nan_to_num(dice))

        freq = hist.sum(axis=1) / hist.sum() # freq of each target
        freq_iou = (freq * iou).sum()

        recall = np.nan_to_num(np.diag(hist) / hist.sum(axis=1))
        precision = np.nan_to_num(np.diag(hist) / hist.sum(axis=0))

        return {'Acc': acc,
                'mAcc': mean_acc,
                'IoU': iou, 
                'mIoU': mean_iou, 
                'Dice': dice,
                'mDice': mean_dice,
                'fIoU': freq_iou,
                'Recall': recall,
                'Precision': precision
                }

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
