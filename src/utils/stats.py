import pandas as pd

from sklearn import metrics

def get_stats_for_data(classifier, X, y):
    score = classifier.score(X, y)
    print("L2-sag")
    print("Accuracy:", score)

    labels = [0,1]

    y_pred = classifier.predict(X)
    cm = metrics.confusion_matrix(y, y_pred, labels=labels)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)

    df_cm_percentage = df_cm.copy()
    for i in df_cm_percentage:
        df_cm_percentage[i]/=df_cm_percentage[i].sum()

    print(df_cm_percentage)
    print('Accuracy:', metrics.accuracy_score(y, y_pred))
    print('F1 Score:', metrics.f1_score(y, y_pred, average='weighted'))
    print('Precision:', metrics.average_precision_score([int(yt) for yt in y], [int (yp) for yp in y_pred], average='weighted'))
    print('Recall:', metrics.recall_score(y, y_pred,average='weighted'))
    print(metrics.classification_report(y, y_pred, target_names=labels))
    print('-----------------------------')