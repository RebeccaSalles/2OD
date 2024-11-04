import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import lib2do
import argparse
import tensorflow as tf
import time
import socket
from numpy.lib.stride_tricks import sliding_window_view
import numpy as np

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

parser = argparse.ArgumentParser()

parser.add_argument("--threads",
                    help="Number of threads : 1,2,5,10,20,30,40,50,60,70,80,160,320",
                    nargs='+',
                    type=int,
                    default=[1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 160, 320])

parser.add_argument("--nb",
                    help="Window size: 300000, 500000, 1000000, 3000000,5000000, 10000000",
                    nargs='+',
                    type=int,
                    default=[300000, 500000, 1000000, 3000000, 5000000, 10000000])

parser.add_argument("--headers",
                    help="Columns of csv : global_time, precision, recall, f1, aucroc, aucpr, diff, tn, fp, fn, tp",
                    nargs='+',
                    default=["global_time", "split", "windows_time", "fit_time", "aggr_time", "acc_fit_time",
                             "acc_pred_time",
                             "acc_df_time", "precision", "recall", "f1", "aucroc", "aucpr",
                             "diff", "tn", "fp", "fn", "tp"])

parser.add_argument("--models",
                    help="Models to run: hbos, iforest, ocsvm, cblof, copod, ecod, featuringbagging, knn, loda, lof, mcd, pca, cof, sod, sogaal, mogaal, deepsvdd,deepsvdd_ae,vae,ae",
                    nargs='+',
                    default=["copod", "deepsvdd", "ecod", "pca", "hbos", "iforest", "knn", "loda", "mcd", "cblof"])

parser.add_argument("--n", help="Number of runs", type=int, default=1)
parser.add_argument("--limit", help="limit", type=int, default=-1)
parser.add_argument("--result_folder", help="Result folder name", type=str, default="./results/")
parser.add_argument("--result", help="Result file name", type=str, default="results.csv")
parser.add_argument("--result2", help="Global Result file name", type=str, default="global-results.csv")
parser.add_argument("--source", help="Source file name", type=str,
                    default="data/GutenTAG/cbf-channels-single-of-2/test.csv")

args = parser.parse_args()
source_filename = args.source

res_filename = args.result
res_filename2 = args.result2
result_folder = args.result_folder

limit = args.limit

headers = args.headers
listOfModels = args.models
n = args.n

nbElements = args.nb
threads = args.threads

GutenTAG_1 = pd.read_csv(source_filename)
data_x = GutenTAG_1.drop(["timestamp", "is_anomaly"], axis=1)
data_y = GutenTAG_1.is_anomaly
X_VALUES = data_x.to_numpy()
Y_ANOMALY = data_y.to_numpy()

lib2do.writeCsvHeader2(result_folder + "/" + res_filename, headers)
lib2do.writeCsvHeader2(result_folder + "/" + res_filename2, headers)

header = False
xp = 3

for model in listOfModels:
    for i in range(n):
        for nbElement in nbElements:
            for thread in threads:
                tig = time.time()
                if X_VALUES.shape[0] < nbElement:
                    nbElement = X_VALUES.shape[0]
                a_train_pred = np.zeros([Y_ANOMALY.shape[0]])
                a_train_scores = np.zeros([Y_ANOMALY.shape[0]])
                a_train_scores2 = np.zeros([Y_ANOMALY.shape[0]])

                m_train_pred = np.zeros([Y_ANOMALY.shape[0]])
                m_train_scores = np.zeros([Y_ANOMALY.shape[0]])
                m_train_scores2 = np.zeros([Y_ANOMALY.shape[0]])

                counter = np.zeros([Y_ANOMALY.shape[0]])

                shape = (nbElement, X_VALUES.shape[1])
                step = int(nbElement / 2)

                SUB_X_VALUES = sliding_window_view(X_VALUES, shape)[::step, :]

                shape = (nbElement,)
                SUB_Y_ANOMALY = sliding_window_view(Y_ANOMALY, shape)[::step, :]
                tags = {
                    "host": socket.gethostname(),
                    "run": str(i + 1),
                    "n": str(n),
                    "xp": str(xp),
                    "model": model,
                    "windows": nbElement,
                    "thread": thread,
                    "file": res_filename
                }

                j = 0
                if limit == -1:
                    limit = len(SUB_X_VALUES)
                for X_VALUE in SUB_X_VALUES:
                    if j >= limit:
                        break
                    X_VALUE = X_VALUE[0]
                    print("run: ", i + 1, " of ", n, " xp: ", xp, " model: ", model, " thread :", thread, " windows: ",
                          nbElement, " position: ", j + 1, " of ", limit)
                    res, local_a_train_pred, local_a_train_scores, local_a_train_scores2, local_m_train_pred, local_m_train_scores, local_m_train_scores2 = lib2do.executeXP5(
                        model, X_VALUE, SUB_Y_ANOMALY[j], nbElement, thread)

                    fn = result_folder + "/" + res_filename
                    lib2do.writeCsvLine2(fn, i, j, xp, model, "avg", thread, nbElement, res["avg"], headers)
                    lib2do.writeCsvLine2(fn, i, j, xp, model, "max", thread, nbElement, res["max"], headers)

                    startSub = j * step
                    endSub = Y_ANOMALY.shape[0] - local_a_train_scores2.shape[0] - startSub
                    startZeroSub = np.zeros(startSub, dtype=float)
                    endZeroSub = np.zeros(endSub, dtype=float)

                    a_local_to_global_score = np.concatenate((startZeroSub, local_a_train_scores, endZeroSub),
                                                             axis=None)

                    a_train_scores = a_train_scores + a_local_to_global_score

                    m_local_to_global_score = np.concatenate((startZeroSub, local_m_train_scores, endZeroSub),
                                                             axis=None)
                    m_train_scores = np.maximum(m_train_scores, m_local_to_global_score)

                    a_local_to_global_score2 = np.concatenate((startZeroSub, local_a_train_scores2, endZeroSub),
                                                              axis=None)
                    a_train_scores2 = a_train_scores2 + a_local_to_global_score2

                    m_local_to_global_score2 = np.concatenate((startZeroSub, local_m_train_scores2, endZeroSub),
                                                              axis=None)
                    m_train_scores2 = np.maximum(m_train_scores2, m_local_to_global_score2)

                    a_local_to_global_pred = np.concatenate((startZeroSub, local_a_train_pred, endZeroSub),
                                                            axis=None)
                    a_train_pred = a_train_pred + a_local_to_global_pred

                    m_local_to_global_pred = np.concatenate((startZeroSub, local_m_train_pred, endZeroSub),
                                                            axis=None)
                    m_train_pred = np.maximum(m_train_pred, m_local_to_global_pred)

                    local_to_global_one = np.concatenate((startZeroSub, np.ones(local_a_train_pred.shape), endZeroSub),
                                                         axis=None)
                    counter = counter + local_to_global_one

                    j = j + 1

                a_train_scores = a_train_scores / counter
                a_train_scores2 = a_train_scores2 / counter
                a_train_pred = a_train_pred / counter

                a_train_scores2 = lib2do.computeAggregation2(a_train_scores2)
                a_train_scores = lib2do.computeAggregation2(a_train_scores)
                a_train_pred = lib2do.computeAggregation2(a_train_pred)

                tfg = time.time()
                global_time = tfg - tig

                g_res = {}
                g_res["avg"] = lib2do.computeRes(global_time, 0, 0, 0, 0, 0,
                                               0, Y_ANOMALY, a_train_pred, a_train_scores, a_train_scores2)

                g_res["max"] = lib2do.computeRes(global_time, 0, 0, 0, 0, 0,
                                               0, Y_ANOMALY, m_train_pred, m_train_scores, m_train_scores2)

                fn = result_folder + "/" + res_filename2
                lib2do.writeCsvLine2(fn, i, j, xp, model, "avg", thread, nbElement, g_res["avg"], headers)
                lib2do.writeCsvLine2(fn, i, j, xp, model, "max", thread, nbElement, g_res["max"], headers)

                time.sleep(15)
