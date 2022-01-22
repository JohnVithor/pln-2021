
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

    FP = np.append(FP, [FP.sum()])
    FN = np.append(FN, [FN.sum()])
    TP = np.append(TP, [TP.sum()])
    TN = np.append(TN, [TN.sum()])

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # False discovery rate
    FDR = FP/(TP+FP)
    # Overall accuracy for each class
    ACC = (TP+TN)/(TP+FP+FN+TN)
    # Overall accuracy for each class
    F1_SCORE = TP/(TP+((FP+FN)/2))

    results = {
        "False Positive": FP,
        "False Negative": FN,
        "True Positive": TP,
        "True Negative": TN,
        "True Positive rate": TPR,
        "True Negative rate": TNR,
        "False Positive rate": FPR,
        "False Negative rate": FNR,
        "Positive Predictive value": PPV,
        "Negative Predictive value": NPV,
        "False Discovery rate": FDR,
        "Accuracy": ACC,
        "F1 Score": F1_SCORE
    }

    for k in results:
        results[k] = list(np.nan_to_num(results[k]))

    return results