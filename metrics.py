
import numpy as np

def extract_metrics_from_confusion_matrix(matrix):
    FP = matrix.sum(axis=0) - np.diag(matrix) 
    FN = matrix.sum(axis=1) - np.diag(matrix)
    TP = np.diag(matrix)
    TN = matrix.sum() - (FP + FN + TP)
    
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)
    # Overall accuracy for each class
    ACC = (TP+TN)/(TP+FP+FN+TN)

    results = {
        "False Positive": FP,
        "False Negative": FN,
        "True Positive": TP,
        "True Negative": TN,
        "true positive rate": TPR,
        "true negative rate": TNR,
        "positive predictive value": PPV,
        "Negative predictive value": NPV,
        "false positive rate": FPR,
        "False negative rate": FNR,
        "False discovery rate": FDR,
        "accuracy": ACC
    }

    for k in results:
        results[k] = list(np.nan_to_num(results[k]))

    return results