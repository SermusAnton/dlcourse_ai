def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, accuracy, f1 - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    #     Накладываем маску правды на предсказание
    tp = len(prediction[ground_truth])
    # Считаем из предсказания все положительные ответы (будет сумма угаданных и нет) 
    tp_fp = (prediction==True).sum()
    # tp_fp = tp+fp 
    fp = tp_fp - tp
    # Точность - насколько предсказание полезно
    precision = tp/(tp + fp)
    # Считаем из правды все положительные ответы (будет общее число значений, которые гадали) 
    tp_fn = (ground_truth==True).sum()
    # tp_fn = tp+fn
    fn = tp_fn - tp
    # Охват предсказания - насколько полно предсказание
    recall = tp/(tp + fn)

    tn =  (ground_truth==False).sum()
    #   Чувствительность (истенный положительный показатель) -  измеряет долю фактических положительных результатов, которые правильно определены как таковые (например, процент больных людей, которые правильно определены как имеющие состояние). 
    #     accuracy  = (tp + tn) / (tp + tn + fp + fn)
    accuracy  = (tp + tn) / (tp + tn + fp + fn)
    #  Специфичность (истенный отрицательны показатель) - измеряет долю фактических отрицательных значений, которые правильно определены как таковые (например, процент здоровых людей, которые правильно определены как не имеющие состояния).
    # true_negaive_rate = tn/(tn+fp)    
    # F1 в двоичной классификации оценка меры точноти и охвата
    f1 = 2*precision*recall/(precision+recall)
    
    return accuracy, precision, recall, f1


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    return 0
