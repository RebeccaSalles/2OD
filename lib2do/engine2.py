from trucmodels import ae, cblof, cof, copod, deepsvdd, ecod, featuringbagging, hbos, iforest, knn, loda, lof, mcd, \
    mogaal, ocsvm, pca, sod, sogaal, vae, deepsvdd_ae
from pyspark.sql import SparkSession

import numpy as np
import warnings

from datetime import datetime

from numpy.lib.stride_tricks import sliding_window_view
import numpy as np

from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score, precision_score, \
    confusion_matrix

import time

from pyod.utils.utility import score_to_label


def execute(param, X_VALUES):
    if param == "hbos":
        res = hbos(X_VALUES)
    elif param == "ae":
        res = ae(X_VALUES)
    elif param == "iforest":
        res = iforest(X_VALUES)
    elif param == "ocsvm":
        res = ocsvm(X_VALUES)
    elif param == "cblof":
        res = cblof(X_VALUES)
    elif param == "copod":
        res = copod(X_VALUES)
    elif param == "ecod":
        res = ecod(X_VALUES)
    elif param == "featuringbagging":
        res = featuringbagging(X_VALUES)
    elif param == "knn":
        res = knn(X_VALUES)
    elif param == "loda":
        res = loda(X_VALUES)
    elif param == "lof":
        res = lof(X_VALUES)
    elif param == "mcd":
        res = mcd(X_VALUES)
    elif param == "pca":
        res = pca(X_VALUES)
    elif param == "cof":
        res = cof(X_VALUES)
    elif param == "sod":
        res = sod(X_VALUES)
    elif param == "sogaal":
        res = sogaal(X_VALUES)
    elif param == "mogaal":
        res = mogaal(X_VALUES)
    elif param == "deepsvdd":
        res = deepsvdd(X_VALUES)
    elif param == "deepsvdd_ae":
        res = deepsvdd_ae(X_VALUES)
    elif param == "vae":
        res = vae(X_VALUES)
    return res


def getPrecisionRecallFscoreSupport(Y_ANOMALY, y_train_pred):
    try:
        precision, recall, f1, support = precision_recall_fscore_support(Y_ANOMALY, y_train_pred, average='binary')
    except:
        precision = None
        recall = None
        f1 = None
        support = None
    return precision, recall, f1, support


def getConfusionMatrix(Y_ANOMALY, y_train_pred):
    try:
        tn, fp, fn, tp = confusion_matrix(Y_ANOMALY, y_train_pred).ravel()
    except:
        tn = None
        fp = None
        fn = None
        tp = None
    return tn, fp, fn, tp


# def analyseAnomalies(Y_ANOMALY, y_train_scores):
#    aucroc = roc_auc_score(y_true=Y_ANOMALY, y_score=y_train_scores)
#    aucpr = average_precision_score(y_true=Y_ANOMALY, y_score=y_train_scores, pos_label=1)
#    prn = 0
#    return aucroc, aucpr, prn


def analyseAnomalies(Y_ANOMALY, y_train_scores):
    try:
        aucroc = roc_auc_score(y_true=Y_ANOMALY, y_score=y_train_scores)
    except:
        aucroc = None

    try:
        aucpr = average_precision_score(y_true=Y_ANOMALY, y_score=y_train_scores, pos_label=1)
    except:
        aucpr = None

    return aucroc, aucpr


def payloadResult(global_time, windows_time, fit_time, aggr_time, acc_fit_time, acc_pred_time, acc_df_time):
    return {
        "global_time": global_time,
        "windows_time": windows_time,
        "fit_time": fit_time,
        "aggr_time": aggr_time,
        "acc_fit_time": acc_fit_time,
        "acc_pred_time": acc_pred_time,
        "acc_df_time": acc_df_time,
        "precision": None,
        "diff": None,
        "recall": None,
        "f1": None,
        "support": None,
        "aucroc": None,
        "aucpr": None,
        "prn": None,
        "aucroc2": None,
        "aucpr2": None,
        "prn2": None,
        "tn": None,
        "fp": None,
        "fn": None,
        "tp": None
    }


def computeRes(global_time, windows_time, fit_time, aggr_time, acc_fit_time, acc_pred_time, acc_df_time, Y_ANOMALY,
               y_pred, train_scores_norm_by, train_scores_norm_2_by):
    res = payloadResult(global_time, windows_time, fit_time, aggr_time, acc_fit_time, acc_pred_time, acc_df_time)
    res["precision"], res["recall"], res["f1"], res["support"] = getPrecisionRecallFscoreSupport(Y_ANOMALY,
                                                                                                 y_pred)
    res["aucroc"], res["aucpr"] = analyseAnomalies(Y_ANOMALY, train_scores_norm_by)
    res["aucroc2"], res["aucpr2"] = analyseAnomalies(Y_ANOMALY, train_scores_norm_2_by)

    res["diff"] = (Y_ANOMALY != y_pred).sum()
    res["tn"], res["fp"], res["fn"], res["tp"] = getConfusionMatrix(Y_ANOMALY, y_pred)
    return res


def computeLocalRes(Y_ANOMALY, y_pred, train_scores_norm_by, train_scores_norm_2_by):
    precision, recall, f1, support = getPrecisionRecallFscoreSupport(Y_ANOMALY, y_pred)
    aucroc, aucpr = analyseAnomalies(Y_ANOMALY, train_scores_norm_by)
    aucroc2, aucpr2 = analyseAnomalies(Y_ANOMALY, train_scores_norm_2_by)

    diff = (Y_ANOMALY != y_pred).sum()

    tn, fp, fn, tp = getConfusionMatrix(Y_ANOMALY, y_pred)
    return precision, recall, f1, support, aucroc, aucpr, aucroc2, aucpr2, diff, tn, fp, fn, tp


def computeAggregation2(train_pred):
    y_pred = score_to_label(train_pred, outliers_fraction=0.1)
    return y_pred


def computeLocalAggregation(train_scores, train_scores2, train_pred,
                            type):
    warnings.simplefilter("ignore", category=RuntimeWarning)

    if type == "avg":
        train_scores_norm_by = np.nanmean(train_scores)
        train_scores_norm_2_by = np.nanmean(train_scores2)
        train_pred_norm_by = np.nanmean(train_pred)
    elif type == "max":
        train_scores_norm_by = np.nanmax(train_scores, axis=1)
        train_scores_norm_2_by = np.nanmax(train_scores2, axis=1)
        train_pred_norm_by = np.nanmax(train_pred, axis=1)
    else:
        train_scores_norm_by = np.nanmedian(train_scores, axis=1)
        train_scores_norm_2_by = np.nanmedian(train_scores2, axis=1)
        train_pred_norm_by = np.nanmedian(train_pred, axis=1)

    y_pred = score_to_label(train_pred_norm_by, outliers_fraction=0.1)
    return train_scores_norm_by, train_scores_norm_2_by, train_pred_norm_by, y_pred


def executeXP1(param, X_VALUES, Y_ANOMALY):
    tig = time.time()
    acc_fit_time, acc_pred_time, acc_df_time, y_pred, train_scores_norm_by, train_scores_norm_2_by = execute(param,
                                                                                                             X_VALUES)
    tfg = time.time()
    global_time = tfg - tig
    windows_time = 0
    aggr_time = 0

    print("\texecuting:", param, "\ttime: ", global_time, " seconds")
    res = computeRes(global_time, windows_time, global_time, aggr_time, acc_fit_time, acc_pred_time, acc_df_time,
                     Y_ANOMALY, y_pred, train_scores_norm_by,
                     train_scores_norm_2_by)
    return res


def rolling_window(a, window, step):
    shape = a.shape[:-1] + ((a.shape[-1] - window + 1) // step, window)
    strides = (a.strides[0] * step,) + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def executeXP2(param, X_VALUES, Y_ANOMALY, node):
    now = datetime.now()
    month = now.strftime("%m")
    year = now.strftime("%Y")
    day = now.strftime("%d")
    hour = now.strftime("%H")
    minutes = now.strftime("%M")

    window_size = int(node)
    step = int(window_size / 2)
    tig = time.time()
    tiw = time.time()
    shape = (window_size, X_VALUES.shape[1])
    X_train_online = sliding_window_view(X_VALUES, shape)[::step, :]
    Y_ANOMALY_by_step = sliding_window_view(Y_ANOMALY, window_size)[::step, :]

    tfw = time.time()
    windows_time = tfw - tiw

    acc_fit_time = 0
    acc_pred_time = 0
    acc_df_time = 0

    a_train_scores = np.zeros([X_VALUES.shape[0]], dtype=float)
    a_train_scores2 = np.zeros([X_VALUES.shape[0]], dtype=float)
    a_train_pred = np.zeros([X_VALUES.shape[0]], dtype=float)

    m_train_scores = np.zeros([X_VALUES.shape[0]], dtype=float)
    m_train_scores2 = np.zeros([X_VALUES.shape[0]], dtype=float)
    m_train_pred = np.zeros([X_VALUES.shape[0]], dtype=float)

    counter = np.zeros([X_VALUES.shape[0]], dtype=float)
    tif = time.time()
    resultsOfSubwindow = []
    #    sparkDF = sc.sparkContext.parallelize(X_train_online, X_train_online.shape[0])
    #    result = (sparkDF.mapPartitionsWithIndex(
    #        lambda index, iterator: computeXP4(index, iterator, param))
    #              .collect())
    #    print("result: ", result.shape)
    #    for v in result:
    i = 0
    for subWindow in X_train_online:
        fit_time, time_pred, time_df, y_train_pred, y_train_scores, y_train_scores2 = execute(param, subWindow[0])

        res = {
            "position": i,
            "param": param,
            "window_size": window_size,
            "node": node,
            "fit_time": fit_time,
            "time_pred": time_pred,
            "time_df": time_df
        }

        (res["precision"], res["recall"], res["f1"],
         res["support"], res["aucroc"], res["aucpr"],
         res["aucroc2"], res["aucpr2"], res["diff"],
         res["tn"], res["fp"], res["fn"], res["tp"]) = computeLocalRes(
            Y_ANOMALY_by_step[i], y_train_pred, y_train_scores,
            y_train_scores2)

        startSub = i * step
        endSub = X_VALUES.shape[0] - y_train_pred.shape[0] - startSub
        startZeroSub = np.zeros(startSub, dtype=float)
        endZeroSub = np.zeros(endSub, dtype=float)
        #        startZeroSub[:] = np.nan
        #        endZeroSub[:] = np.nan

        local_to_global_score = np.concatenate((startZeroSub, y_train_scores, endZeroSub),
                                               axis=None)
        a_train_scores = a_train_scores + local_to_global_score
        m_train_scores = np.maximum(m_train_scores, local_to_global_score)

        local_to_global_score2 = np.concatenate((startZeroSub, y_train_scores2, endZeroSub),
                                                axis=None)
        a_train_scores2 = a_train_scores2 + local_to_global_score2
        m_train_scores2 = np.maximum(m_train_scores2, local_to_global_score2)

        local_to_global_pred = np.concatenate((startZeroSub, y_train_pred, endZeroSub),
                                              axis=None)
        a_train_pred = a_train_pred + local_to_global_pred
        m_train_pred = np.maximum(m_train_pred, local_to_global_pred)

        local_to_global_one = np.concatenate((startZeroSub, np.ones(y_train_scores.shape), endZeroSub),
                                             axis=None)
        counter = counter + local_to_global_one

        resultsOfSubwindow.append(res)
        #        g_y_train_pred.append(y_train_pred)
        #        g_y_train_scores.append(y_train_scores)
        #        g_y_train_scores2.append(y_train_scores2)

        acc_fit_time += fit_time
        acc_pred_time += time_pred
        acc_df_time += time_df
        i = i + 1
    tff = time.time()
    fit_time = tff - tif
    tia = time.time()
    a_train_pred = a_train_pred / counter
    a_train_scores2 = a_train_scores2 / counter
    a_train_scores = a_train_scores / counter

    a_y_pred = computeAggregation2(a_train_pred)
    m_y_pred = computeAggregation2(m_train_pred)

    tfa = time.time()
    aggr_time = tfa - tia

    tfg = time.time()
    global_time = tfg - tig
    print("\texecuting:", param, "\ttime: ", global_time, " seconds", "\tg_time: ", acc_fit_time, " seconds",
          "\twindows_time: ", windows_time, " seconds")

    res = {}
    res["avg"] = computeRes(global_time, windows_time, fit_time, aggr_time, acc_fit_time, acc_pred_time,
                            acc_df_time,
                            Y_ANOMALY, a_y_pred, a_train_scores,
                            a_train_scores2)
    res["max"] = computeRes(global_time, windows_time, fit_time, aggr_time, acc_fit_time, acc_pred_time,
                            acc_df_time,
                            Y_ANOMALY, m_y_pred, m_train_scores,
                            m_train_scores2)

    return res, resultsOfSubwindow


def executeXP3(param, X_VALUES, Y_ANOMALY, window_size, node):
    now = datetime.now()
    month = now.strftime("%m")
    year = now.strftime("%Y")
    day = now.strftime("%d")
    hour = now.strftime("%H")
    minutes = now.strftime("%M")

    sc = (SparkSession.builder
          .appName("run-xp3" + year + "_" + month + "_" + day + "-" + hour + "_" + minutes)
          .config("spark.executor.memory", "64g")
          .config("spark.driver.memory", "64g")
          .config("spark.executor.memoryOverhead", "64g")
          .config("spark.driver.memoryOverhead", "64g")
          #          .master("local")
          .getOrCreate())

    window_size = int(window_size)
    step = int(window_size / 2)
    tig = time.time()
    tiw = time.time()
    shape = (window_size, X_VALUES.shape[1])
    X_train_online = sliding_window_view(X_VALUES, shape)[::step, :]
    Y_ANOMALY_by_step = sliding_window_view(Y_ANOMALY, window_size)[::step, :]

    tfw = time.time()
    windows_time = tfw - tiw

    acc_fit_time = 0
    acc_pred_time = 0
    acc_df_time = 0

    a_train_scores = np.zeros([X_VALUES.shape[0]], dtype=float)
    a_train_scores2 = np.zeros([X_VALUES.shape[0]], dtype=float)
    a_train_pred = np.zeros([X_VALUES.shape[0]], dtype=float)

    m_train_scores = np.zeros([X_VALUES.shape[0]], dtype=float)
    m_train_scores2 = np.zeros([X_VALUES.shape[0]], dtype=float)
    m_train_pred = np.zeros([X_VALUES.shape[0]], dtype=float)

    counter = np.zeros([X_VALUES.shape[0]], dtype=float)
    tif = time.time()
    resultsOfSubwindow = []
    i = 0
    for subWindow in X_train_online:
        subWindow = subWindow[0]
        sub_window_size = int(subWindow.shape[0] / node)
        sub_step = int(sub_window_size / 2)
        sub_shape = (sub_window_size, subWindow.shape[1])

        sub_a_train_scores = np.zeros([subWindow.shape[0]], dtype=float)
        sub_a_train_scores2 = np.zeros([subWindow.shape[0]], dtype=float)
        sub_a_train_pred = np.zeros([subWindow.shape[0]], dtype=float)

        sub_m_train_scores = np.zeros([subWindow.shape[0]], dtype=float)
        sub_m_train_scores2 = np.zeros([subWindow.shape[0]], dtype=float)
        sub_m_train_pred = np.zeros([subWindow.shape[0]], dtype=float)

        sub_to_global_one = np.zeros([subWindow.shape[0]], dtype=float)

        fit_time = 0
        time_pred = 0
        time_df = 0

        sub_X_train_online = sliding_window_view(subWindow, sub_shape)[::sub_step, :]
        sparkDF = sc.sparkContext.parallelize(sub_X_train_online, sub_X_train_online.shape[0])
        result = (sparkDF.mapPartitionsWithIndex(
            lambda index, iterator: computeXP4(index, iterator, param))
                  .collect())
        for v in result:
            j = v[0]
            sub_sub_fit_time = v[1]
            sub_sub_time_pred = v[2]
            sub_sub_time_df = v[3]
            sub_sub_y_train_scores = v[4]
            sub_sub_y_train_scores2 = v[5]
            sub_sub_y_train_pred = v[6]

            startSubSub = j * sub_step
            endSubSub = subWindow.shape[0] - sub_sub_y_train_pred.shape[0] - startSubSub
            startZeroSubSub = np.zeros(startSubSub, dtype=float)
            endZeroSubSub = np.zeros(endSubSub, dtype=float)

            local_sub_sub_y_train_scores = np.concatenate((startZeroSubSub, sub_sub_y_train_scores, endZeroSubSub),
                                                          axis=None)
            sub_a_train_scores = sub_a_train_scores + local_sub_sub_y_train_scores
            sub_m_train_scores = np.maximum(sub_m_train_scores, local_sub_sub_y_train_scores)

            local_sub_sub_y_train_scores2 = np.concatenate((startZeroSubSub, sub_sub_y_train_scores2, endZeroSubSub),
                                                           axis=None)
            sub_a_train_scores2 = sub_a_train_scores2 + local_sub_sub_y_train_scores2
            sub_m_train_scores2 = np.maximum(sub_m_train_scores2, local_sub_sub_y_train_scores2)

            local_sub_sub_y_train_pred = np.concatenate((startZeroSubSub, sub_sub_y_train_pred, endZeroSubSub),
                                                        axis=None)
            sub_a_train_pred = sub_a_train_pred + local_sub_sub_y_train_pred
            sub_m_train_pred = np.maximum(sub_m_train_pred, local_sub_sub_y_train_pred)

            local_sub_sub_to_global_one = np.concatenate(
                (startZeroSubSub, np.ones(sub_sub_y_train_pred.shape), endZeroSubSub),
                axis=None)
            sub_to_global_one = sub_to_global_one + local_sub_sub_to_global_one
            fit_time = fit_time + sub_sub_fit_time
            time_pred = time_pred + sub_sub_time_pred
            time_df = time_df + sub_sub_time_df

        sub_a_train_pred = sub_a_train_pred / sub_to_global_one
        sub_a_train_scores = sub_a_train_scores / sub_to_global_one
        sub_a_train_scores2 = sub_a_train_scores2 / sub_to_global_one

        res = {
            "position": i,
            "param": param,
            "window_size": window_size,
            "node": node,
            "fit_time": fit_time,
            "time_pred": time_pred,
            "time_df": time_df
        }

        (res["precision"], res["recall"], res["f1"],
         res["support"], res["aucroc"], res["aucpr"],
         res["aucroc2"], res["aucpr2"], res["diff"],
         res["tn"], res["fp"], res["fn"], res["tp"]) = computeLocalRes(
            Y_ANOMALY_by_step[i], sub_a_train_pred, sub_a_train_scores,
            sub_a_train_scores2)

        startSub = i * step
        endSub = X_VALUES.shape[0] - sub_a_train_pred.shape[0] - startSub
        startZeroSub = np.zeros(startSub, dtype=float)
        endZeroSub = np.zeros(endSub, dtype=float)
        #        startZeroSub[:] = np.nan
        #        endZeroSub[:] = np.nan

        a_local_to_global_score = np.concatenate((startZeroSub, sub_a_train_scores, endZeroSub),
                                                 axis=None)
        a_train_scores = a_train_scores + a_local_to_global_score
        m_local_to_global_score = np.concatenate((startZeroSub, sub_m_train_scores, endZeroSub),
                                                 axis=None)
        m_train_scores = np.maximum(m_train_scores, m_local_to_global_score)

        a_local_to_global_score2 = np.concatenate((startZeroSub, sub_a_train_scores2, endZeroSub),
                                                  axis=None)
        m_local_to_global_score2 = np.concatenate((startZeroSub, sub_m_train_scores2, endZeroSub),
                                                  axis=None)
        a_train_scores2 = a_train_scores2 + a_local_to_global_score2
        m_train_scores2 = np.maximum(m_train_scores2, m_local_to_global_score2)

        a_local_to_global_pred = np.concatenate((startZeroSub, sub_a_train_pred, endZeroSub),
                                                axis=None)
        m_local_to_global_pred = np.concatenate((startZeroSub, sub_m_train_pred, endZeroSub),
                                                axis=None)
        a_train_pred = a_train_pred + a_local_to_global_pred
        m_train_pred = np.maximum(m_train_pred, m_local_to_global_pred)

        local_to_global_one = np.concatenate((startZeroSub, np.ones(sub_a_train_scores.shape), endZeroSub),
                                             axis=None)
        counter = counter + local_to_global_one

        resultsOfSubwindow.append(res)

        acc_fit_time += fit_time
        acc_pred_time += time_pred
        acc_df_time += time_df
        i = i + 1
    a_train_pred = a_train_pred / counter
    a_train_scores2 = a_train_scores2 / counter
    a_train_scores = a_train_scores / counter

    tff = time.time()
    fit_time = tff - tif
    tia = time.time()

    a_y_pred = computeAggregation2(a_train_pred)
    m_y_pred = computeAggregation2(m_train_pred)

    tfa = time.time()
    aggr_time = tfa - tia

    tfg = time.time()
    global_time = tfg - tig
    print("\texecuting:", param, "\ttime: ", global_time, " seconds", "\tg_time: ", acc_fit_time, " seconds",
          "\twindows_time: ", windows_time, " seconds")

    res = {}
    res["avg"] = computeRes(global_time, windows_time, fit_time, aggr_time, acc_fit_time, acc_pred_time,
                            acc_df_time,
                            Y_ANOMALY, a_y_pred, a_train_scores,
                            a_train_scores2)
    res["max"] = computeRes(global_time, windows_time, fit_time, aggr_time, acc_fit_time, acc_pred_time,
                            acc_df_time,
                            Y_ANOMALY, m_y_pred, m_train_scores,
                            m_train_scores2)

    return res, resultsOfSubwindow


def executeXP3a(param, X_VALUES, Y_ANOMALY, node):
    now = datetime.now()
    month = now.strftime("%m")
    year = now.strftime("%Y")
    day = now.strftime("%d")
    hour = now.strftime("%H")
    minutes = now.strftime("%M")

    window_size = int(node)
    step = int(window_size / 2)
    tig = time.time()
    tiw = time.time()
    shape = (window_size, X_VALUES.shape[1])
    X_train_online = sliding_window_view(X_VALUES, shape)[::step, :]
    Y_ANOMALY_by_step = sliding_window_view(Y_ANOMALY, window_size)[::step, :]

    tfw = time.time()
    windows_time = tfw - tiw

    acc_fit_time = 0
    acc_pred_time = 0
    acc_df_time = 0

    a_train_scores = np.zeros([X_VALUES.shape[0]], dtype=float)
    a_train_scores2 = np.zeros([X_VALUES.shape[0]], dtype=float)
    a_train_pred = np.zeros([X_VALUES.shape[0]], dtype=float)

    m_train_scores = np.zeros([X_VALUES.shape[0]], dtype=float)
    m_train_scores2 = np.zeros([X_VALUES.shape[0]], dtype=float)
    m_train_pred = np.zeros([X_VALUES.shape[0]], dtype=float)

    counter = np.zeros([X_VALUES.shape[0]], dtype=float)
    tif = time.time()
    resultsOfSubwindow = []
    #    sparkDF = sc.sparkContext.parallelize(X_train_online, X_train_online.shape[0])
    #    result = (sparkDF.mapPartitionsWithIndex(
    #        lambda index, iterator: computeXP4(index, iterator, param))
    #              .collect())
    #    print("result: ", result.shape)
    #    for v in result:
    i = 0
    for subWindow in X_train_online:
        fit_time, time_pred, time_df, y_train_pred, y_train_scores, y_train_scores2 = execute(param, subWindow[0])

        res = {
            "position": i,
            "param": param,
            "window_size": window_size,
            "node": node,
            "fit_time": fit_time,
            "time_pred": time_pred,
            "time_df": time_df
        }

        (res["precision"], res["recall"], res["f1"],
         res["support"], res["aucroc"], res["aucpr"],
         res["aucroc2"], res["aucpr2"], res["diff"],
         res["tn"], res["fp"], res["fn"], res["tp"]) = computeLocalRes(
            Y_ANOMALY_by_step[i], y_train_pred, y_train_scores,
            y_train_scores2)

        startSub = i * step
        endSub = X_VALUES.shape[0] - y_train_pred.shape[0] - startSub
        startZeroSub = np.zeros(startSub, dtype=float)
        endZeroSub = np.zeros(endSub, dtype=float)
        #        startZeroSub[:] = np.nan
        #        endZeroSub[:] = np.nan

        local_to_global_score = np.concatenate((startZeroSub, y_train_scores, endZeroSub),
                                               axis=None)
        a_train_scores = a_train_scores + local_to_global_score
        m_train_scores = np.maximum(m_train_scores, local_to_global_score)

        local_to_global_score2 = np.concatenate((startZeroSub, y_train_scores2, endZeroSub),
                                                axis=None)
        a_train_scores2 = a_train_scores2 + local_to_global_score2
        m_train_scores2 = np.maximum(m_train_scores2, local_to_global_score2)

        local_to_global_pred = np.concatenate((startZeroSub, y_train_pred, endZeroSub),
                                              axis=None)
        a_train_pred = a_train_pred + local_to_global_pred
        m_train_pred = np.maximum(m_train_pred, local_to_global_pred)

        local_to_global_one = np.concatenate((startZeroSub, np.ones(y_train_scores.shape), endZeroSub),
                                             axis=None)
        counter = counter + local_to_global_one

        resultsOfSubwindow.append(res)
        #        g_y_train_pred.append(y_train_pred)
        #        g_y_train_scores.append(y_train_scores)
        #        g_y_train_scores2.append(y_train_scores2)

        acc_fit_time += fit_time
        acc_pred_time += time_pred
        acc_df_time += time_df
        i = i + 1
    tff = time.time()
    fit_time = tff - tif
    tia = time.time()

    a_y_pred = computeAggregation2(a_train_pred)
    m_y_pred = computeAggregation2(m_train_pred)

    tfa = time.time()
    aggr_time = tfa - tia

    tfg = time.time()
    global_time = tfg - tig
    print("\texecuting:", param, "\ttime: ", global_time, " seconds", "\tg_time: ", acc_fit_time, " seconds",
          "\twindows_time: ", windows_time, " seconds")

    res = {}
    res["avg"] = computeRes(global_time, windows_time, fit_time, aggr_time, acc_fit_time, acc_pred_time,
                            acc_df_time,
                            Y_ANOMALY, a_y_pred, a_train_scores,
                            a_train_scores2)
    res["max"] = computeRes(global_time, windows_time, fit_time, aggr_time, acc_fit_time, acc_pred_time,
                            acc_df_time,
                            Y_ANOMALY, m_y_pred, m_train_scores,
                            m_train_scores2)

    return res, resultsOfSubwindow


def executeXP3a(param, X_VALUES, Y_ANOMALY, node):
    now = datetime.now()
    month = now.strftime("%m")
    year = now.strftime("%Y")
    day = now.strftime("%d")
    hour = now.strftime("%H")
    minutes = now.strftime("%M")

    window_size = int(X_VALUES.shape[0] / node)
    step = int(window_size / 2)
    tig = time.time()
    tiw = time.time()
    shape = (window_size, X_VALUES.shape[1])
    X_train_online = sliding_window_view(X_VALUES, shape)[::step, :]

    Y_ANOMALY_by_step = sliding_window_view(Y_ANOMALY, window_size)[::step, :]
    sc = (SparkSession.builder
          .appName("run-xp2" + year + "_" + month + "_" + day + "-" + hour + "_" + minutes)
          .config("spark.executor.memory", "64g")
          .config("spark.driver.memory", "64g")
          .config("spark.executor.memoryOverhead", "64g")
          .config("spark.driver.memoryOverhead", "64g")
          #          .master("local")
          .getOrCreate())

    tfw = time.time()
    windows_time = tfw - tiw

    acc_fit_time = 0
    acc_pred_time = 0
    acc_df_time = 0

    a_train_scores = np.zeros([X_VALUES.shape[0]], dtype=float)
    a_train_scores2 = np.zeros([X_VALUES.shape[0]], dtype=float)
    a_train_pred = np.zeros([X_VALUES.shape[0]], dtype=float)

    m_train_scores = np.zeros([X_VALUES.shape[0]], dtype=float)
    m_train_scores2 = np.zeros([X_VALUES.shape[0]], dtype=float)
    m_train_pred = np.zeros([X_VALUES.shape[0]], dtype=float)

    counter = np.zeros([X_VALUES.shape[0]], dtype=float)
    tif = time.time()
    resultsOfSubwindow = []
    #    g_y_train_pred = []
    #    g_y_train_scores = []
    #    g_y_train_scores2 = []
    #    print("window_size: ", window_size)
    #    print("node: ", node)
    #    print("X_VALUES: ", X_VALUES.shape)
    #    print("X_train_online: ", X_train_online.shape)
    sparkDF = sc.sparkContext.parallelize(X_train_online, X_train_online.shape[0])
    result = (sparkDF.mapPartitionsWithIndex(
        lambda index, iterator: computeXP4(index, iterator, param))
              .collect())
    #    print("result: ", result.shape)
    for v in result:
        i = v[0]
        fit_time = v[1]
        time_pred = v[2]
        time_df = v[3]
        y_train_scores = v[4]
        y_train_scores2 = v[5]
        y_train_pred = v[6]

        res = {
            "position": i,
            "param": param,
            "window_size": window_size,
            "node": node,
            "fit_time": fit_time,
            "time_pred": time_pred,
            "time_df": time_df
        }

        (res["precision"], res["recall"], res["f1"],
         res["support"], res["aucroc"], res["aucpr"],
         res["aucroc2"], res["aucpr2"], res["diff"],
         res["tn"], res["fp"], res["fn"], res["tp"]) = computeLocalRes(
            Y_ANOMALY_by_step[i], y_train_pred, y_train_scores,
            y_train_scores2)

        startSub = i * step
        endSub = X_VALUES.shape[0] - y_train_pred.shape[0] - startSub
        startZeroSub = np.zeros(startSub, dtype=float)
        endZeroSub = np.zeros(endSub, dtype=float)
        #        startZeroSub[:] = np.nan
        #        endZeroSub[:] = np.nan

        local_to_global_score = np.concatenate((startZeroSub, y_train_scores, endZeroSub),
                                               axis=None)
        a_train_scores = a_train_scores + local_to_global_score
        m_train_scores = np.maximum(m_train_scores, local_to_global_score)

        local_to_global_score2 = np.concatenate((startZeroSub, y_train_scores2, endZeroSub),
                                                axis=None)
        a_train_scores2 = a_train_scores2 + local_to_global_score2
        m_train_scores2 = np.maximum(m_train_scores2, local_to_global_score2)

        local_to_global_pred = np.concatenate((startZeroSub, y_train_pred, endZeroSub),
                                              axis=None)
        a_train_pred = a_train_pred + local_to_global_pred
        m_train_pred = np.maximum(m_train_pred, local_to_global_pred)

        local_to_global_one = np.concatenate((startZeroSub, np.ones(y_train_scores.shape), endZeroSub),
                                             axis=None)
        counter = counter + local_to_global_one

        resultsOfSubwindow.append(res)
        #        g_y_train_pred.append(y_train_pred)
        #        g_y_train_scores.append(y_train_scores)
        #        g_y_train_scores2.append(y_train_scores2)

        acc_fit_time += fit_time
        acc_pred_time += time_pred
        acc_df_time += time_df
    tff = time.time()
    fit_time = tff - tif
    tia = time.time()

    a_y_pred = computeAggregation2(a_train_pred)
    m_y_pred = computeAggregation2(m_train_pred)

    tfa = time.time()
    aggr_time = tfa - tia

    tfg = time.time()
    global_time = tfg - tig
    print("\texecuting:", param, "\ttime: ", global_time, " seconds", "\tg_time: ", acc_fit_time, " seconds",
          "\twindows_time: ", windows_time, " seconds")

    res = {}
    res["avg"] = computeRes(global_time, windows_time, fit_time, aggr_time, acc_fit_time, acc_pred_time,
                            acc_df_time,
                            Y_ANOMALY, a_y_pred, a_train_scores,
                            a_train_scores2)
    res["max"] = computeRes(global_time, windows_time, fit_time, aggr_time, acc_fit_time, acc_pred_time,
                            acc_df_time,
                            Y_ANOMALY, m_y_pred, m_train_scores,
                            m_train_scores2)

    return res, resultsOfSubwindow


def executeXP4(param, X_VALUES, Y_ANOMALY, window_size, step, ss_windows_size, ss_step):
    now = datetime.now()
    month = now.strftime("%m")
    year = now.strftime("%Y")
    day = now.strftime("%d")
    hour = now.strftime("%H")
    minutes = now.strftime("%M")
    window_size = int(window_size)
    step = int(step)
    tig = time.time()
    sc = (SparkSession.builder.appName("run-xp3" + year + "_" + month + "_" + day + "-" + hour + "_" + minutes)
          .getOrCreate())

    tiw = time.time()
    shape = (window_size, X_VALUES.shape[1])
    X_train_online = sliding_window_view(X_VALUES, shape)[::step, :]

    Y_ANOMALY_by_step = sliding_window_view(Y_ANOMALY, window_size)[::step, :]

    tfw = time.time()
    windows_time = tfw - tiw

    acc_fit_time = 0
    acc_pred_time = 0
    acc_df_time = 0

    train_scores = np.zeros([X_VALUES.shape[0], X_train_online.shape[0] * ss_windows_size])
    train_scores2 = np.zeros([X_VALUES.shape[0], X_train_online.shape[0] * ss_windows_size])
    train_pred = np.zeros([X_VALUES.shape[0], X_train_online.shape[0] * ss_windows_size])
    tif = time.time()
    resultsOfSubwindow = []
    g_y_train_pred = []
    g_y_train_scores = []
    g_y_train_scores2 = []
    for i in range(X_train_online.shape[0]):
        start = i * step
        end = X_VALUES.shape[0] - start - window_size
        startZero = np.zeros(start)
        endZero = np.zeros(end)
        startZero[:] = np.nan
        endZero[:] = np.nan

        d = X_train_online[i][0]

        sub_windows_size = int(window_size / ss_windows_size)
        sub_shape = (sub_windows_size, d.shape[1])
        sub_X_train_online = sliding_window_view(d, sub_shape)[::ss_step, :]

        local_Y_ANOMALY_by_step = sliding_window_view(Y_ANOMALY_by_step[i], sub_shape)[::ss_step, :]

        sparkDF = sc.sparkContext.parallelize(sub_X_train_online, sub_X_train_online.shape[0])
        result = (sparkDF.mapPartitionsWithIndex(
            lambda index, iterator: computeXP4(index, iterator, param))
                  .collect())
        for v in result:
            j = v[0]
            fit_time = v[1]
            time_pred = v[2]
            time_df = v[3]
            y_train_scores = v[4]
            y_train_scores2 = v[5]
            y_train_pred = v[6]

            res = {
                "position": i,
                "position2": j,
                "param": param,
                "window_size": window_size,
                "step": step,
                "ss_windows_size": ss_windows_size,
                "ss_step": ss_step,
                "fit_time": fit_time,
                "time_pred": time_pred,
                "time_df": time_df
            }

            (res["precision"], res["recall"], res["f1"],
             res["support"], res["aucroc"], res["aucpr"],
             res["aucroc2"], res["aucpr2"], res["diff"],
             res["tn"], res["fp"], res["fn"], res["tp"]) = computeLocalRes(
                local_Y_ANOMALY_by_step[i], y_train_pred, y_train_scores,
                y_train_scores2)

            startSub = j * step
            endSub = d.shape[0] - startSub - sub_windows_size
            startZeroSub = np.zeros(startSub)
            endZeroSub = np.zeros(endSub)
            startZeroSub[:] = np.nan
            endZeroSub[:] = np.nan

            train_scores[:, i] = np.concatenate((startZero, startZeroSub, y_train_scores, endZeroSub, endZero),
                                                axis=None)
            train_scores2[:, i] = np.concatenate((startZero, startZeroSub, y_train_scores2, endZeroSub, endZero),
                                                 axis=None)
            train_pred[:, i] = np.concatenate((startZero, startZeroSub, y_train_pred, endZeroSub, endZero),
                                              axis=None)
            resultsOfSubwindow.append(res)
            g_y_train_pred.append(y_train_pred)
            g_y_train_scores.append(y_train_scores)
            g_y_train_scores2.append(y_train_scores2)

            acc_fit_time += fit_time
            acc_pred_time += time_pred
            acc_df_time += time_df

    tff = time.time()
    fit_time = tff - tif
    tia = time.time()

    a_train_scores_norm_by, a_train_scores_norm_2_by, a_train_pred_norm_by, a_y_pred = computeAggregation(
        train_scores,
        train_scores2,
        train_pred,
        "avg")

    m_train_scores_norm_by, m_train_scores_norm_2_by, m_train_pred_norm_by, m_y_pred = computeAggregation(
        train_scores,
        train_scores2,
        train_pred,
        "max")

    md_train_scores_norm_by, md_train_scores_norm_2_by, md_train_pred_norm_by, md_y_pred = computeAggregation(
        train_scores,
        train_scores2,
        train_pred, type)
    tfa = time.time()
    aggr_time = tfa - tia

    tfg = time.time()
    global_time = tfg - tig
    print("\texecuting:", param, "\ttime: ", global_time, " seconds", "\tg_time: ", acc_fit_time, " seconds",
          "\twindows_time: ", windows_time, " seconds")

    res = {}
    res["avg"] = computeRes(global_time, windows_time, fit_time, aggr_time, acc_fit_time, acc_pred_time,
                            acc_df_time,
                            Y_ANOMALY, a_y_pred, a_train_scores_norm_by,
                            a_train_scores_norm_2_by)
    res["max"] = computeRes(global_time, windows_time, fit_time, aggr_time, acc_fit_time, acc_pred_time,
                            acc_df_time,
                            Y_ANOMALY, m_y_pred, m_train_scores_norm_by,
                            m_train_scores_norm_2_by)
    res["med"] = computeRes(global_time, windows_time, fit_time, aggr_time, acc_fit_time, acc_pred_time,
                            acc_df_time,
                            Y_ANOMALY, md_y_pred, md_train_scores_norm_by,
                            md_train_scores_norm_2_by)

    return res, resultsOfSubwindow, g_y_train_pred, g_y_train_scores, g_y_train_scores2


def computeXP4(index, iterator, param):
    res = next(iterator)
    acc_fit_time, time_pred, time_df, y_train_pred, y_train_scores, y_train_scores2 = execute(param, res[0])
    return [[index, acc_fit_time, time_pred, time_df, y_train_pred, y_train_scores, y_train_scores2]]


def computeSlice(x, nodeG, node):
    n = x.shape[0]
    ws = int(n / node + n / node / 2)
    if node == 1:
        ws = n
    step = int(ws / 2)
    v = sliding_window_view(x, ws, axis=0)[::step, :]
    #    print("ws = ", ws, step, node, nodeG, v.shape[0])
    if (v.shape[0] > nodeG):
        return computeSlice(x, nodeG, node - 1)
    else:
        return ws, step


def executeXP5(model, X_VALUES, Y_ANOMALY, nbElements, node):
    tig = time.time()
    now = datetime.now()
    month = now.strftime("%m")
    year = now.strftime("%Y")
    day = now.strftime("%d")
    hour = now.strftime("%H")
    minutes = now.strftime("%M")

    sc = (SparkSession.builder
          .appName("run-xp" + year + "_" + month + "_" + day + "-" + hour + "_" + minutes)
          .config("spark.executor.memory", "64g")
          .config("spark.driver.memory", "64g")
          .config("spark.executor.memoryOverhead", "64g")
          .config("spark.driver.memoryOverhead", "64g")
            .config("spark.rpc.message.maxSize", "2000")
#          .master("local")
          .getOrCreate())

    tiw = time.time()
    window_size, step = computeSlice(X_VALUES, node, node)
    shape = (window_size, X_VALUES.shape[1])

    X_train_online = sliding_window_view(X_VALUES, shape)[::step, :]
    Y_ANOMALY_by_step = sliding_window_view(Y_ANOMALY, window_size)[::step, :]
    tfw = time.time()
    a_train_pred = np.zeros([X_VALUES.shape[0]])
    a_train_scores = np.zeros([X_VALUES.shape[0]])
    a_train_scores2 = np.zeros([X_VALUES.shape[0]])

    m_train_pred = np.zeros([X_VALUES.shape[0]])
    m_train_scores = np.zeros([X_VALUES.shape[0]])
    m_train_scores2 = np.zeros([X_VALUES.shape[0]])

    counter = np.zeros([X_VALUES.shape[0]])

    tif = time.time()
    sparkDF = sc.sparkContext.parallelize(X_train_online, X_train_online.shape[0])
    result = (sparkDF.mapPartitionsWithIndex(lambda index, iterator: computeXP4(index, iterator, model)).collect())
    tff = time.time()
    acc_fit_time = 0
    acc_pred_time = 0
    acc_df_time = 0

    tia = time.time()
    for v in result:
        i = v[0]
        fit_time = v[1]
        time_pred = v[2]
        time_df = v[3]

        acc_fit_time = acc_fit_time + fit_time
        acc_pred_time = acc_pred_time + time_pred
        acc_df_time = acc_df_time + time_df

        y_train_scores = v[4]
        y_train_scores2 = v[5]
        y_train_pred = v[6]
        res = {
            "position": i,
            "param": model,
            "window_size": window_size,
            "step": step,
            "fit_time": fit_time,
            "time_pred": time_pred,
            "time_df": time_df
        }

        (res["precision"], res["recall"], res["f1"],
         res["support"], res["aucroc"], res["aucpr"],
         res["aucroc2"], res["aucpr2"], res["diff"],
         res["tn"], res["fp"], res["fn"], res["tp"]) = computeLocalRes(Y_ANOMALY_by_step[i], y_train_pred,
                                                                       y_train_scores, y_train_scores2)
        startSub = i * step
        endSub = X_VALUES.shape[0] - y_train_pred.shape[0] - startSub
        startZeroSub = np.zeros(startSub, dtype=float)
        endZeroSub = np.zeros(endSub, dtype=float)

        local_to_global_score = np.concatenate((startZeroSub, y_train_scores, endZeroSub),
                                               axis=None)
        a_train_scores = a_train_scores + local_to_global_score
        m_train_scores = np.maximum(m_train_scores, local_to_global_score)

        local_to_global_score2 = np.concatenate((startZeroSub, y_train_scores2, endZeroSub),
                                                axis=None)
        a_train_scores2 = a_train_scores2 + local_to_global_score2
        m_train_scores2 = np.maximum(m_train_scores2, local_to_global_score2)

        local_to_global_pred = np.concatenate((startZeroSub, y_train_pred, endZeroSub),
                                              axis=None)
        a_train_pred = a_train_pred + local_to_global_pred
        m_train_pred = np.maximum(m_train_pred, local_to_global_pred)

        local_to_global_one = np.concatenate((startZeroSub, np.ones(y_train_scores.shape), endZeroSub),
                                             axis=None)
        counter = counter + local_to_global_one
    counter = np.where(counter==0, 1, counter)

    a_train_pred = a_train_pred / counter
    a_train_scores2 = a_train_scores2 / counter
    a_train_scores = a_train_scores / counter

    a_train_pred = computeAggregation2(a_train_pred)
    m_train_pred = computeAggregation2(m_train_pred)
    tfa = time.time()

    res = {}
    tfg = time.time()
    global_time = tfg - tig
    window_size_time = tfw - tiw
    fit_time = tff - tif
    aggr_time = tfa - tia
    res["avg"] = computeRes(global_time, window_size_time, fit_time, aggr_time, acc_fit_time, acc_pred_time,
                            acc_df_time,
                            Y_ANOMALY, a_train_pred, a_train_scores,
                            a_train_scores2)

    res["max"] = computeRes(global_time, window_size_time, fit_time, aggr_time, acc_fit_time, acc_pred_time,
                            acc_df_time,
                            Y_ANOMALY, m_train_pred, m_train_scores,
                            m_train_scores2)

    return res, a_train_pred, a_train_scores, a_train_scores2, m_train_pred, m_train_scores, m_train_scores2
