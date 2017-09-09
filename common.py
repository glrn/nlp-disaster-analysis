import collections
import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot
import time
import pandas
import os
from pandas.tools.plotting import table

Accuracy = collections.namedtuple('Accuracy', 'acc ppv npv')
DATE_FORMAT_STRING = '%Y-%m-%d %H:%M:%S'

def timeit(func):
    def wrapper(*args, **kwds):
        start   = time.time()
        print('Measure times for function: {} ({})'.format(func.__name__, datetime.datetime.now().strftime(DATE_FORMAT_STRING)))
        ret = func(*args, **kwds)
        end     = time.time()
        print('Total running time of {} in seconds: {}'.format(func.__name__, int(end - start)))
        return ret
    return wrapper

def compute_accuracy(prediction, real, corpus=None, debug=False):
    if len(prediction) != len(real):
        raise ValueError('prediction {} and real {} length should by equal'.format(len(prediction), len(real)))
    correct = 0
    true_positive   = 0
    true_negative   = 0
    num_of_pred_pos = len([x for x in prediction if x == 1])
    num_of_pred_neg = len([x for x in prediction if x == 0])
    num_of_real_pos = len([x for x in real if x == 1])
    num_of_real_neg = len([x for x in real if x == 0])

    for i in xrange(len(prediction)):
        p = int(prediction[i])
        r = int(real[i])
        if p == r:
            correct += 1
            if p == 0:
                true_negative += 1
            else:
                true_positive += 1

        if debug and corpus is not None and p != r:
            # print false-positives and false-negatives
            print "Real: %s, Prediction: %s" % (r, p)
            print "Tweet is: %s" % corpus[i]
            print

    npv   = float(true_negative) / num_of_pred_neg if num_of_pred_neg else None
    ppv   = float(true_positive) / num_of_pred_pos if num_of_pred_pos else None

    #specificity = float(true_negative) / num_of_real_neg
    return Accuracy(float(correct) / len(prediction), ppv, npv)

def plot(xs, ys, colors, x_label, y_label, title, func_labels=None, x_scale=None, legend_location=None, save=None):
    f, ax = matplotlib.pyplot.subplots(1)
    plots = []
    for i, params in enumerate(zip(xs, ys, colors)):
        x, y, color = params
        if func_labels is not None:
            plots.append(ax.plot(x, y, color, label=func_labels[i]))
        else:
            ax.plot(x, y, color)
    ax.set_ylim(ymin=max(0, min([x for y in ys for x in y]) * 0.9), ymax=min(1, max([x for y in ys for x in y]) * 1.1))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if x_scale is not None:
        ax.set_xscale(x_scale)
    if plots:
        legend_location = legend_location if legend_location is not None else 'best'
        ax.legend([plot for plot, in plots], [plot.get_label() for plot, in plots], loc=legend_location)
    f.suptitle(title, fontsize=14, fontweight='bold')
    if save is not None:
        print('Saving graph into: {}'.format(save))
        try:
            os.makedirs(os.path.dirname(save))
        except:
            pass
        matplotlib.pyplot.savefig(save)
    else:
        matplotlib.pyplot.show()
    matplotlib.pyplot.close(f)

def plot_table(title, cells, column_names, row_names, save=None):
    f, ax = matplotlib.pyplot.subplots(1)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)
    f.suptitle(title, fontsize=14, fontweight='bold')

    df = pandas.DataFrame(cells)
    df.columns  = column_names
    df.index    = row_names
    tab = table(
        ax          = ax,
        data        = df,
        colLabels   = column_names,
        rowLabels   = row_names,
        loc         = 'upper right',
    )

    tab.auto_set_font_size(False)
    tab.set_fontsize(12)

    print(df)

    if save is not None:
        print('Saving table into: {}'.format(save))
        try:
            os.makedirs(os.path.dirname(save))
        except:
            pass
        matplotlib.pyplot.savefig(save)
    else:
        matplotlib.pyplot.show()
    matplotlib.pyplot.close(f)


def max_specific_accuracy(accuracies):
    max_acc = 0
    max_idx = 0
    for i, acc in enumerate(accuracies):
        if acc > max_acc:
            max_acc = acc
            max_idx = i
    return max_idx, max_acc

def max_accuracy(accuracies):
    max_acc_idx, max_acc = max_specific_accuracy([acc.acc for acc in accuracies])
    max_ppv_idx, max_ppv = max_specific_accuracy([acc.ppv for acc in accuracies])
    max_npv_idx, max_npv = max_specific_accuracy([acc.npv for acc in accuracies])
    return max_acc_idx, max_ppv_idx, max_npv_idx

def best_feature_names(named_features, name, bools):
    return [f.__name__ for f, b in zip(named_features[name], bools) if b]