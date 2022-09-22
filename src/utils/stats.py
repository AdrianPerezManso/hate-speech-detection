import pandas as pd
from sklearn import metrics
from configs import config, logconfig
import logging

"""
Statistics for validation datasets
"""

def get_stats_for_data(classifier, X, y, e_time=0, make_report=False, multilabel=False, mllabels=[]):
    labels = config.STATS_LABELS

    y_pred = classifier.predict(X)
    logging.info(logconfig.LOG_STATS_TRAIN_PREDICTION)

    logging.info(logconfig.LOG_STATS_METRIC_ACCURACY.format(result=metrics.accuracy_score(y, y_pred)))
    logging.info(logconfig.LOG_STATS_METRIC_PRECISION.format(result=metrics.average_precision_score(y, y_pred, average='weighted')))
    logging.info(logconfig.LOG_STATS_METRIC_RECALL.format(result=metrics.recall_score(y, y_pred,average='weighted')))
    logging.info(logconfig.LOG_STATS_METRIC_F1.format(result=metrics.f1_score(y, y_pred, average='weighted')) )
    logging.info(logconfig.LOG_STATS_METRIC_HAMMING.format(result=metrics.hamming_loss(y, y_pred)))
    logging.info(logconfig.LOG_STATS_METRIC_JACCARD.format(result=metrics.jaccard_score(y, y_pred, average='weighted')))
    logging.info(logconfig.LOG_STATS_METRIC_EXEC_TIME.format(result=e_time))

    if(make_report):
        logging.info('\n' + metrics.classification_report(y, y_pred, target_names=labels))
    
    labels = [int(l) for l in labels]
    if(multilabel):
        for i, label in enumerate(mllabels):
            logging.info(label)
            cm = metrics.confusion_matrix(y[:,i], y_pred[:,i], labels=labels)
            df_cm = pd.DataFrame(cm, index=labels, columns=labels)
            df_cm_percentage = df_cm.copy()
            for i in df_cm_percentage:
                df_cm_percentage[i]/=df_cm_percentage[i].sum()
            logging.info(df_cm_percentage)
    else:
        cm = metrics.confusion_matrix(y, y_pred, labels=labels)
        df_cm = pd.DataFrame(cm, index=labels, columns=labels)
        logging.info(df_cm)
        df_cm_percentage = df_cm.copy()
        for i in df_cm_percentage:
            df_cm_percentage[i]/=df_cm_percentage[i].sum()
        logging.info(df_cm_percentage)
    