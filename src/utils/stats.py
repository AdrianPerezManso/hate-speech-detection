import pandas as pd

from sklearn import metrics

def get_stats_for_data(classifier, X, y, e_time=0, make_report=False, multilabel=False, mllabels=[]):
    score = classifier.score(X, y)
    print("L2-sag")
    print("Accuracy:", score)

    labels = ['0','1']

    y_pred = classifier.predict(X)

    print('Accuracy:', metrics.accuracy_score(y, y_pred))
    print('Precision:', metrics.average_precision_score(y, y_pred, average='weighted'))
    print('Recall:', metrics.recall_score(y, y_pred,average='weighted'))
    print('F1 Score:', metrics.f1_score(y, y_pred, average='weighted')) 
    print('Hamming loss: ',metrics.hamming_loss(y, y_pred))
    print('Jaccard score: ',metrics.jaccard_score(y, y_pred, average='weighted'))
    print('Execution time: ', e_time)
    if(make_report):
        print(metrics.classification_report(y, y_pred, target_names=labels))
    
    labels = [0,1]
    if(multilabel):
        for i, label in enumerate(mllabels):
            print(label)
            cm = metrics.confusion_matrix(y[:,i], y_pred[:,i], labels=labels)
            df_cm = pd.DataFrame(cm, index=labels, columns=labels)
            #print(df_cm)
            df_cm_percentage = df_cm.copy()
            for i in df_cm_percentage:
                df_cm_percentage[i]/=df_cm_percentage[i].sum()
            print(df_cm_percentage)
    else:
        cm = metrics.confusion_matrix(y, y_pred, labels=labels)
        df_cm = pd.DataFrame(cm, index=labels, columns=labels)
        print(df_cm)
        df_cm_percentage = df_cm.copy()
        for i in df_cm_percentage:
            df_cm_percentage[i]/=df_cm_percentage[i].sum()
        print(df_cm_percentage)
    