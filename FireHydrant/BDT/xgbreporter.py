#!/usr/bin/env python
"""make some postprocessing plots/reports by the predictions and true labels

usage: python xgbreporter.py -d <OUTPUTDIR> [-t <TRAINDATA> -k <KEY>]

An `xgbreport.txt will be dumped under <OUTPUTDIR>, including info below:

* prediction distribution
* roc curve
* auc score
* accuracy score
* classification report

** prediction for full dataset (if TRAINDATA and KEY given)

"""

import argparse
import os
from os.path import join

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import roc_curve, roc_auc_score
np.seterr(divide='ignore', invalid='ignore', over='ignore')

parser = argparse.ArgumentParser(description="Postprocessing output of xgbtrainer")
parser.add_argument("--dir", '-d', type = str, required = True, help = "training output")
parser.add_argument("--train", '-t', type = str, required = False, default = None, help = "training data")
parser.add_argument('--key', '-k', type=str, required=False, default=None, help='key for dataframe')
args = parser.parse_args()
assert (os.path.isdir(args.dir) and os.path.isfile(join(args.dir, 'xgbpredictions.h5')))



def make_prediction_distribution(trainoutdir):
    print('Making xgb prediction distributions in {}'.format(trainoutdir))
    from coffea import hist

    datasource = join(trainoutdir, 'xgbpredictions.h5')
    traindf = pd.read_hdf(datasource, 'train')
    testdf = pd.read_hdf(datasource, 'test')

    dataset_axis = hist.Cat('dataset', 'train/test')
    label_axis = hist.Cat('label', 'S/B')
    bdt_axis = hist.Bin('score', 'BDT score', 50, -10, 10)


    default = hist.Hist("norm. counts", dataset_axis, label_axis, bdt_axis)
    default.fill(dataset='test', label='signal (test)', score=testdf.query('y==1')['default'].values)
    default.fill(dataset='test', label='background (test)', score=testdf.query('y==0')['default'].values)
    default.fill(dataset='train', label='signal (train)', score=traindf.query('y==1')['default'].values)
    default.fill(dataset='train', label='background (train)', score=traindf.query('y==0')['default'].values)

    optimized = hist.Hist("norm. counts", dataset_axis, label_axis, bdt_axis)
    optimized.fill(dataset='test', label='signal (test)', score=testdf.query('y==1')['optimized'].values)
    optimized.fill(dataset='test', label='background (test)', score=testdf.query('y==0')['optimized'].values)
    optimized.fill(dataset='train', label='signal (train)', score=traindf.query('y==1')['optimized'].values)
    optimized.fill(dataset='train', label='background (train)', score=traindf.query('y==0')['optimized'].values)

    data_err_opts = {
        'linestyle':'none',
        'marker': '.',
        'markersize': 10.,
        'elinewidth': 1,
        'emarker': '_',
        'markeredgecolor': 'k'
    }
    fill_opts = {
        'edgecolor': (0,0,0,0.3),
        'alpha': 0.8
    }

    fig, ax = plt.subplots(figsize=(8,6))
    hist.plot1d(default.project('dataset', 'test'), overlay='label', ax=ax, density=True, clear=False, error_opts=data_err_opts)
    hist.plot1d(default.project('dataset', 'train'), overlay='label', ax=ax, line_opts=None, clear=False, density=True, fill_opts=fill_opts)
    ax.legend()
    ax.autoscale(axis='y', tight=True)
    ax.set_ylim(0, None);
    ax.set_title('default BDT response', x=0.0, ha="left")
    ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
    ax.set_ylabel(ax.get_ylabel(), y=1.0, ha="right")
    plt.savefig(join(trainoutdir, "prediction_dist_default.pdf"), bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(8,6))
    hist.plot1d(optimized.project('dataset', 'test'), overlay='label', ax=ax, density=True, clear=False, error_opts=data_err_opts)
    hist.plot1d(optimized.project('dataset', 'train'), overlay='label', ax=ax, line_opts=None, clear=False, density=True, fill_opts=fill_opts)
    ax.legend()
    ax.autoscale(axis='y', tight=True)
    ax.set_ylim(0, None);
    ax.set_title('optimized BDT response', x=0.0, ha="left")
    ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
    ax.set_ylabel(ax.get_ylabel(), y=1.0, ha="right")
    plt.savefig(join(trainoutdir, "prediction_dist_optimized.pdf"), bbox_inches='tight')
    plt.close()


def make_full_prediction_distribution(trainoutdir, trainingdata, key):
    print(f"Make predictions on full training dataset: {trainingdata} and model in {trainoutdir}")
    from coffea import hist
    import xgboost as xgb

    ## full dataset
    df = pd.read_hdf(trainingdata, key)
    featurecols = [x for x in df.columns if x != 'label']
    dfull = xgb.DMatrix(df[featurecols], label=df['label'])

    ## default and optimized models
    xgbm_default = xgb.Booster({"nthread": 16})
    xgbm_default.load_model(join(trainoutdir, "model_default/model.bin"))
    xgbm_optimized = xgb.Booster({"nthread": 16})
    xgbm_optimized.load_model(join(trainoutdir, "model_optimized/model.bin"))

    ## predictions
    preds_default = xgbm_default.predict(dfull)
    preds_optimized = xgbm_optimized.predict(dfull)

    ## making plots
    label_axis = hist.Cat('label', 'S/B')
    bdt_axis = hist.Bin('score', 'BDT score', 50, -10, 10)

    default = hist.Hist("norm. counts", label_axis, bdt_axis)
    default.fill(label='signal', score=preds_default[df['label'].values.astype(bool)])
    default.fill(label='background', score=preds_default[~df['label'].values.astype(bool)])
    optimized = hist.Hist("norm. counts", label_axis, bdt_axis)
    optimized.fill(label='signal', score=preds_optimized[df['label'].values.astype(bool)])
    optimized.fill(label='background', score=preds_optimized[~df['label'].values.astype(bool)])

    fig, ax = plt.subplots(figsize=(8, 6))
    hist.plot1d(default, overlay='label', ax=ax, density=True)
    ax.set_ylim(0, None);
    ax.set_title('default BDT response on full dataset', x=0.0, ha="left")
    ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
    ax.set_ylabel(ax.get_ylabel(), y=1.0, ha="right")
    plt.savefig(join(trainoutdir, "prediction_fulldist_default.pdf"), bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    hist.plot1d(optimized, overlay='label', ax=ax, density=True)
    ax.set_ylim(0, None);
    ax.set_title('optimized BDT response on full dataset', x=0.0, ha="left")
    ax.set_xlabel(ax.get_xlabel(), x=1.0, ha="right")
    ax.set_ylabel(ax.get_ylabel(), y=1.0, ha="right")
    plt.savefig(join(trainoutdir, "prediction_fulldist_optimized.pdf"), bbox_inches='tight')
    plt.close()



class RocPlot():
    def __init__(self, logscale=False, xlabel=None, ylabel=None,
                 xlim=None, ylim=None, rlim=None, height_ratios=[2, 1],
                 percentage=False, grid=False, ncol=1):
        self.gs = gridspec.GridSpec(2, 1, height_ratios=height_ratios)
        self.axis = plt.subplot(self.gs[0])
        self.axr = plt.subplot(self.gs[1])
        self.gs.update(wspace=0.025, hspace=0.075)
        plt.setp(self.axis.get_xticklabels(), visible=False)

        if xlim is None: xlim = (0, 1)
        self.xlim_ = xlim
        self.ylim_ = ylim

        if xlabel is None: xlabel = 'True positive rate'
        if ylabel is None: ylabel = 'False positive rate'

        if percentage:
            xlabel += " [%]"
            ylabel += " [%]"
        self.logscale_ = logscale
        self.percentage_ = percentage
        self.scale_ = 1 + 99 * percentage

        self.axis.set_ylabel(ylabel)
        self.axr.set_xlabel(xlabel)
        self.axr.set_ylabel("Ratio")
        self.axr.set_xlabel(self.axr.get_xlabel(), x=1.0, ha="right")
        self.axis.set_ylabel(self.axis.get_ylabel(), y=1.0, ha="right")

        self.axis.grid(grid, which='both', ls=':')
        self.axr.grid(grid, ls=':')

        self.axis.set_xlim([x * self.scale_ for x in xlim])
        self.axr.set_xlim([x * self.scale_ for x in xlim])
        if not ylim is None:
            self.axis.set_ylim([y * self.scale_ for y in ylim])
        if not rlim is None:
            self.axr.set_ylim(rlim)

        self.auc_ = []
        self.ncol_ = ncol  # legend columns

    def plot(self, y_true, y_score, **kwargs):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        self.auc_.append(roc_auc_score(y_true, y_score))

        if not hasattr(self, 'fpr_ref'):
            self.fpr_ref = fpr
            self.tpr_ref = tpr

        sel = tpr >= self.xlim_[0]
        if self.logscale_:
            self.axis.semilogy(tpr[sel] * self.scale_,
                               fpr[sel] * self.scale_,
                               **kwargs)
        else:
            self.axis.plot(tpr[sel] * self.scale_,
                           fpr[sel] * self.scale_,
                           **kwargs)

        ratios = fpr / np.interp(tpr, self.tpr_ref, self.fpr_ref)
        self.axr.plot(tpr[sel]*self.scale_, ratios[sel], **kwargs)

        self.axis.legend(loc="upper left", ncol=self.ncol_)
        if self.percentage_:
            self.axis.get_yaxis().set_major_formatter(PercentFormatter(decimals=1, symbol=None))


def make_roc_curve(trainoutdir):
    print('Making roc curve in {}'.format(trainoutdir))

    datasource = join(trainoutdir, 'xgbpredictions.h5')
    testdf = pd.read_hdf(datasource, 'test')

    plt.figure(figsize=(8, 6))
    roc = RocPlot(xlim=(0.6, 1), ylim=(1e-5, 1), height_ratios=[4, 1],
                 logscale=True, grid=True, percentage=True,
                 ncol=2, rlim=(0.95, 1.05))
    for m in ['default', 'optimized']:
        roc.plot(testdf['y'].values, testdf[m].values, label=m)

    plt.savefig(join(trainoutdir, 'roccurve.pdf'), bbox_inches='tight')
    plt.close()

    print("Extracing working points ...")
    rocstats = {
        'workingpoints': ['tight', 'medium', 'loose'],
        'targetfpr': [1e-4, 1e-3, 1e-2],
        'fakepositiverate_default': [],
        'truepositiverate_default': [],
        'threshold_default': [],
        'fakepositiverate_optimized': [],
        'truepositiverate_optimized': [],
        'threshold_optimized': [],
    }

    for m in ['default', 'optimized']:
        fpr, tpr, thresholds = roc_curve(testdf['y'].values, testdf[m].values)
        rocstats['fakepositiverate_' + m] = [fpr[fpr > t][0] for t in rocstats['targetfpr']]
        rocstats['truepositiverate_' + m] = [tpr[fpr > t][0] for t in rocstats['targetfpr']]
        rocstats['threshold_' + m] = [thresholds[fpr > t][0] for t in rocstats['targetfpr']]

    pd.DataFrame(rocstats).to_csv(join(trainoutdir, 'rocworkingpoints.csv'))


def make_text_report(trainoutdir):

    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

    print("Dumping `accuracy_score`, `roc_auc_score`, and `classification_report` as text files")
    datasource = join(trainoutdir, 'xgbpredictions.h5')
    testdf = pd.read_hdf(datasource, 'test')

    with open(join(trainoutdir, 'xgbtestreport.txt'), 'w') as outf:
        for m in ['default', 'optimized']:
            outf.write('**** {} ****\n'.format(m))

            outf.write('accuracy score: {}\n'.format(accuracy_score(testdf['y'].values, testdf[m].values.astype(bool))))
            outf.write('roc auc score: {}\n'.format(roc_auc_score(testdf['y'].values, testdf[m].values)))
            outf.write('classification report:\n')
            outf.write(classification_report(testdf['y'].values, testdf[m].values.astype(bool), digits=4))
            outf.write('\n\n')



if __name__ == "__main__":

    import time
    starttime = time.time()

    make_prediction_distribution(args.dir)
    make_roc_curve(args.dir)
    make_text_report(args.dir)

    if args.train and args.key:
        assert (os.path.exists(args.train))
        make_full_prediction_distribution(args.dir, args.train, args.key)

    print("---> Took {} s".format(time.time()-starttime))