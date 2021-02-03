import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
from warnings import warn

def cm_stat(cnf_mat):
    r'''
    This function prints the confusion matrix & calculates statistical sizes.
    :param cnf_mat: sklearn.metrics import confusion_matrix
    :return: overall_accuracy, cohens_kappa
    '''

    cnt = 0  # total count all values
    diag = list()  # diagonal elements
    kap = list()
    for i in range(cnf_mat.shape[0]):
        for j in range(cnf_mat.shape[1]):
            cnt += cnf_mat[i, j]  # count all
            if i == j:
                diag.append(cnf_mat[i, j])
                kap.append([sum(cnf_mat[i, :]), sum(cnf_mat[:, j])])  # sum row and column

    pe = 0  # total
    for diag_elem in kap:
        elem = 1
        for rate in diag_elem:
            elem *= rate / cnt
        pe += elem

    # Overall_accuracy: (TP + TN) / (len(all))
    overall_accuracy = sum(diag) / cnt

    # Cohen's kappa: (o_a - pe) / (1 - pe)
    # pe: propability both rate the same (0&0 + 1&1 + 2&2)
    cohen_kappa = (overall_accuracy - pe) / (1 - pe)
    print("\nConfusion-matrix:\n" + str(cnf_mat))
    print("Overall Accuracy: " + str(int(overall_accuracy * 100)) + "%")
    print("Cohen's kappa: %.2f\n" % cohen_kappa)
    return overall_accuracy, cohen_kappa


def plt_train_test_acc(data, epoch, path=False, title="Accuracy", label1="train", label2="valid"):
    """This function saves the test/train accuracy in a plt figure"""
    # data: data[0] = train_acc; data[1] = test_acc
    fig, ax = plt.subplots()
    train = ax.plot(data[0], label=label1)
    test = ax.plot(data[1], label=label2)
    ax.set_title(title)
    y_min = 0
    y_max = 1.05
    y_intervall = 0.05
    ax.axis([0, epoch, y_min, y_max])
    ax.set_yticks(np.arange(y_min, y_max + y_intervall, y_intervall))
    ax.set_xlabel("epochs")
    ax.set_ylabel("accuracy")
    ax.text(1, 0.1, "Best validation: %.2f [%%]" % (max(data[1]) * 100))
    leg = ax.legend(loc='lower right')
    if path is False:
        file = title + '.png'
    else:
        file = path + "/" + title + '.pdf'
    plt.savefig(file, bbox='tight')
    plt.close(fig)
    pass


def plt_table(data, col_labels=('Parameter', 'Value'), title="Hyperparameter", path=False, extension=".pdf"):
    """This function saves the hyperparamter in a plt table"""
    fig, ax = plt.subplots()
    ax.axis('off')
    ax.axis('tight')
    ax.table(cellText=data, colLabels=col_labels, loc='center')
    ax.set_title(title)
    fig.tight_layout()
    if path is False:
        file = title + extension
    else:
        file = path + "/" + title + extension
    plt.savefig(file, bbox='tight')
    plt.close(fig)
    pass


def make_res_folder(path):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    if path == "Default" or path == "default":
        # use default path
        path = "/home/pflab/Desktop/Roessl/Pointnet++/Pytorch/Pointnet2_PyTorch-master/pointnet2/result"
        print("====== Default path is used ======")
        if os.path.exists(path=path):
            date_str = datetime.today().strftime("%Y_%m_%d")
            path += '/' + date_str
            # check if folder for today already exists
            if os.path.exists(path=path) is False:
                os.mkdir(path=path)
                print("made Directory ", date_str)
            # make directory for the current training
            len_dir = os.listdir(path=path)
            path += "/" + str(len(len_dir))
            os.mkdir(path)
            result = path
        else:
            # should not happen
            warn("Warning in default path: ", path)
            result = False
    else:
        # check offered path
        if os.path.exists(path=path):
            result = path
        else:
            result = False
    # os.chdir(BASE_DIR)
    return result


def save_hyper(path, data):
    """This function saves the hyperparamters in the offered folder"""
    if os.path.exists(path=path):
        plt_table(data=data, path=path)
        res = True
    else:
        warn("Offered path is not available")
        res = False
    return res


def save_epoch(storage, path, filename="Result"):
    """This function checks the path and saves the accuracy, cohens kappa, confusion matrix and the report each epoch"""
    if os.path.exists(path=path):
        res = True
        data = "=" * 53 + "\n"
        if type(storage) == list:
            data += "\n".join(map(str, storage))
        else:
            data += storage
        file = path + "/" + filename + ".txt"
        with open(file, 'a') as f:
            f.write(data)
            f.close()
        sleep(0.1)
    else:
        res = False
        print("Path doesnt exists: ", path)
    return res

def save_epoch_val(actual, prediction, accuracy, epoch, path, filename="Validation_accuracy"):
    """This function checks the path and saves the validation of the test-dataset"""
    if os.path.exists(path=path):
        res = True
        data = "# Epoch: " + str(epoch) + "\nActual: " + ", ".join(map(str, actual)) + "\nPrediction: " +\
               ", ".join(map(str, prediction))+ "\nAccuracy: " + str(accuracy) + "\n"
        file = path + "/" + filename + ".txt"
        with open(file, 'a') as f:
            f.write(data)
            f.close()
        sleep(0.1)
    else:
        res = False
        print("Path doesnt exists: ", path)
    return res

def eval_arr(arr):
    """This function calulates statistical sizes for a list of numbers and returns them as a string"""
    if len(arr) > 0:
        results = "Max : " + str(max(arr)) + "\nMean: " + str(np.mean(arr)) + "\nMedian: " + str(np.median(arr)) + \
                  "\nMin: " + str(min(arr)) + "\nStandard_deviation: " + str(np.std(arr))
    else:
        results = ""
    return results


def save_results(data, path, parts=4, filename="Summary"):
    if parts > 0 and data.shape[0] > parts:
        start = 0
        l_data = data.shape[0]
        res = ["**** PARTIAL STATS ****\n"]
        for x in range(1, parts + 1):
            stop = int(x / parts * l_data)
            temp_str = "\n=== " + str(int(start / l_data * 100)) + "% until " + str(int(stop / l_data * 100)) + "% ===\n"
            temp_str += "Overall-accuracy-test:\n"
            temp_str += eval_arr(data[start:stop, 0])
            temp_str += "\nCohen's kappa:\n"
            temp_str += eval_arr(data[start:stop, 1])
            res.append(temp_str)
            start = stop
        res.append("\n\n**** OVERALL STATS ****\nACCURACY:")
        res.append(eval_arr(data[:, 0]))
        res.append("Cohen's kappa")
        res.append(eval_arr(data[:, 1]))
        save_epoch(res, path=path, filename=filename)
        train = data[:, 2]
        test = data[:, 0]
        plt_train_test_acc(data=[train, test], epoch=data.shape[0], path=path)
    else:
        warn("Variable part is invalid")
    pass
