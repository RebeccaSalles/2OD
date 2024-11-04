from datetime import datetime
import numpy as np

def writeCsvHeader(filename, headers):
    f = open(filename, "w")
    f.write("log_time;")
    f.write("xp;")
    f.write("run;")
    f.write("method;")
    f.write("type;")
    f.write("node;")
    f.write("nbElements;")
    for k in headers:
        f.write(k + ";")
    f.write("\r\n")
    f.close()

def writeCsvHeader2(filename, headers):
    f = open(filename, "w")
    f.write("log_time;")
    f.write("xp;")
    f.write("run;")
    f.write("position;")
    f.write("method;")
    f.write("type;")
    f.write("node;")
    f.write("nbElements;")
    for k in headers:
        f.write(k + ";")
    f.write("\r\n")
    f.close()


def writeCsvLine(filename, i, xp, model, type, nodes,nbElements, res, headers):
    f = open(filename, "a")
    now = datetime.now()  # current date and time
    f.write(now.strftime("%m/%d/%Y, %H:%M:%S") + ";")
    f.write(str(xp) + ";")
    f.write(str(i) + ";")
    f.write(model + ";")
    f.write(type + ";")
    f.write(str(nodes) + ";")
    f.write(str(nbElements) + ";")
    for key in headers:
        if key in res and res[key] is not None:
            f.write(str(res[key]) + ";")
        else:
            f.write("NA;")
    f.write("\r\n")
    f.close()

def writeCsvLine2(filename, i,j, xp, model, t, nodes,nbElements, res, headers):
    f = open(filename, "a")
    now = datetime.now()  # current date and time
    f.write(now.strftime("%m/%d/%Y, %H:%M:%S") + ";")
    f.write(str(xp) + ";")
    f.write(str(i) + ";")
    f.write(str(j) + ";")
    f.write(model + ";")
    f.write(t + ";")
    f.write(str(nodes) + ";")
    f.write(str(nbElements) + ";")
    for key in headers:
        if key in res and res[key] is not None:
            f.write(str(res[key]) + ";")
        else:
            f.write("NA;")
    f.write("\r\n")
    f.close()


def writeCsvLineWithoutHeader(filename, res):
    if res:
        f = open(filename, "w")
        now = datetime.now()  # current date and time
        f.write(now.strftime("%m/%d/%Y, %H:%M:%S") + ";")
        for key in res.keys():
            if key !="max" or  key !="avg"  or key !="med" :
                f.write(str(res[key]) + ";")
        f.write("\r\n")
        f.close()


def writeCsvLineOfSubWindows(filename, i, xp, model, node, nbElements, resultsOfSubwindow,header):
    if resultsOfSubwindow:
        f = open(filename, "a")
        now = datetime.now()  # current date and time
        if not header:
            header = resultsOfSubwindow[0]
            f.write("log_time;")
            f.write("i;")
            f.write("xp;")
            f.write("model;")
            f.write("node;")
            f.write("nbElements;")
            for key in header.keys():
                f.write(str(key) + ";")
            f.write("\r\n")

        for res in resultsOfSubwindow:
            f.write(now.strftime("%m/%d/%Y, %H:%M:%S") + ";")
            f.write(str(i)+";")
            f.write(str(xp)+";")
            f.write(str(model)+";")
            f.write(str(node)+";")
            f.write(str(nbElements)+";")
            for key in res.keys():
                f.write(str(res[key]) + ";")
            f.write("\r\n")
        f.close()


def writeListOfValues(filename, d):
    np.savetxt(filename, d, delimiter=";")


def initCsv(filename):
    f = open(filename, "w")
    f.write("")
    f.close()