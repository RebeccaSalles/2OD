from . import engine2 as engine2
from . import csvresults as csvresults
from . import influx as influx


def executeXP1(model, X_VALUES, Y_ANOMALY):
    return engine2.executeXP1(model, X_VALUES, Y_ANOMALY)


def executeXP2(model, X_VALUES, Y_ANOMALY, windows_size):
    return engine2.executeXP2(model, X_VALUES, Y_ANOMALY, windows_size)


def executeXP3(model, X_VALUES, Y_ANOMALY, windows_size, node):
    return engine2.executeXP3(model, X_VALUES, Y_ANOMALY, windows_size, node)


def executeXP4(model, X_VALUES, Y_ANOMALY, windows_size, step, ss_windows_size, ss_step):
    return engine2.executeXP4(model, X_VALUES, Y_ANOMALY, windows_size, step, ss_windows_size, ss_step)


def executeXP5(model, X_VALUES, Y_ANOMALY, nbElements, node):
    return engine2.executeXP5(model, X_VALUES, Y_ANOMALY, nbElements, node)


def writeCsvLine(filename, i, xp, model, type, nodes, nbElements, res, headers):
    return csvresults.writeCsvLine(filename, i, xp, model, type, nodes, nbElements, res, headers)


def writeCsvHeader(filename, headers):
    return csvresults.writeCsvHeader(filename, headers)

def writeCsvHeader2(filename, headers):
    return csvresults.writeCsvHeader2(filename, headers)


def writeInit(tags):
    return influx.writeInit(tags)


def writeStart(tags):
    return influx.writeStart(tags)


def writeStop(tags):
    return influx.writeStop(tags)


def writeCsvLineOfSubWindows(filename, i, xp, model, node, nbElements, resultsOfSubwindow, header):
    csvresults.writeCsvLineOfSubWindows(filename, i, xp, model, node, nbElements, resultsOfSubwindow, header)


def writeListOfValues(filename, g_y_train_scores):
    csvresults.writeListOfValues(filename, g_y_train_scores)


def initCsv(filename):
    csvresults.initCsv(filename)


def writeCsvLine2(filename, i,j, xp, model, t, nodes, nbElements, res, headers):
    return csvresults.writeCsvLine2(filename, i,j, xp, model, t, nodes, nbElements, res, headers)


def computeAggregation2(train_pred):
    return engine2.computeAggregation2(train_pred)

def computeRes(global_time, windows_time, fit_time, aggr_time, acc_fit_time, acc_pred_time, acc_df_time, Y_ANOMALY,
               y_pred, train_scores_norm_by, train_scores_norm_2_by):
    return engine2.computeRes(global_time, windows_time, fit_time, aggr_time, acc_fit_time, acc_pred_time, acc_df_time, Y_ANOMALY,
                              y_pred, train_scores_norm_by, train_scores_norm_2_by)