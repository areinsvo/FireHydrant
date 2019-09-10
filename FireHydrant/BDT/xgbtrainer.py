#!/usr/bin/env python
"""Run BDT training and optimization

usage: python xgbtrainer.py -d <data.h5> -k <key> -c <config.yml>
"""

import argparse
import os
from os.path import join

import yaml

parser = argparse.ArgumentParser(description="BDT training with XGBoost")
parser.add_argument("--data", "-d", type=str, help="data file as pandas DataFrame stored in hdf5")
parser.add_argument("--key", '-k', type=str, default='df', help='key name of dataframe in hdf5')
parser.add_argument("--config", "-c", type=str, help="config file in yaml")
parser.add_argument("--outdir", "-o", type=str, default=None, help='specify output directory')
args = parser.parse_args()
assert os.path.exists(args.data)
assert os.path.exists(args.config)
if args.outdir:
    if not os.path.isdir(args.outdir):
        print("Output directory --> {} set, but do not exist.".format(args.outdir))
        print("I am goint to create one for you.")
        os.makedirs(args.outdir)


def setup_output():
    import time

    timestr = time.strftime("%y%m%d")
    res = join(os.getenv("FH_BASE"), "FireHydrant/BDT/xgbgarage/{}".format(timestr))
    if not os.path.exists(res):
        res = join(res, "0")
        os.makedirs(res)
        return res
    else:
        versions = os.listdir(res)
        vermax = max([int(d) for d in versions])
        versionnext = str(vermax + 1)
        res = join(res, versionnext)
        os.makedirs(res)
        return res



if __name__ == "__main__":

    from collections import defaultdict

    import numpy as np
    import pandas as pd
    import xgboost as xgb
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from xgboptimizer import XgboClassifier

    config = yaml.load(open(args.config), Loader=yaml.Loader)
    print("## xgbtrainer configuration ##")
    print(yaml.dump(config, default_flow_style=None))
    print('++++++++++++++++++++++++++++++')

    training_fraction = config.pop('training_fraction')
    hyperparam_setting = config.pop('hyperparam_setting')
    classifier_setting = config.pop('classifier_setting')
    optimizer_setting = config.pop('optimizer_setting')

    ## load training data
    df = pd.read_hdf(args.data, args.key)
    featurecols = [c for c in df.columns if c != 'label']
    X_train, X_test, y_train, y_test = train_test_split(df[featurecols],
                                                        df['label'],
                                                        random_state=np.random.randint(2019),
                                                        test_size=1 - training_fraction)
    ## entries from class with more entries are discarded.
    #  This is because classifier performance is usually bottlenecked by the
    #  size of the dataset with fewer entries. Having one class with extra
    #  statistics usually just adds computing time.
    n_perclass = min(y_train.value_counts())
    selectedidx = np.concatenate([
        y_train[y_train==0].head(n_perclass).index.values,
        y_train[y_train==1].head(n_perclass).index.values,
    ])
    X_train = X_train.loc[selectedidx]
    y_train = y_train.loc[selectedidx]
    xgtrain = xgb.DMatrix(X_train, label=y_train)
    xgtest = xgb.DMatrix(X_test, label=y_test)


    ## setup output
    if args.outdir:
        outputdir = args.outdir
    else:
        outputdir = setup_output()
    print("Training outputs:", outputdir)


    ## build classifier and run optimization
    classifier = XgboClassifier(outputdir, hyperparam_setting, **classifier_setting)
    classifier.optimize(xgtrain, **optimizer_setting)


    ## saving model and predictions for post report
    predictions = defaultdict(dict)
    for m in ['default', 'optimized']:
        classifier.fit(xgtrain, model=m)
        classifier.save_model(featurecols, model=m)
        predictions['test'][m] = classifier.predict(xgtest, model=m)
        predictions['train'][m] = classifier.predict(xgtrain, model=m)

        ## feature importance
        plt.figure(figsize=(8, 6))
        xgb.plot_importance(classifier.models_[m], height=0.8)
        plt.savefig(join(outputdir, f'xgbfeatureimportance_{m}.pdf'), bbox_inches='tight')
        plt.close()

    predictions['test']['y'] = y_test
    predictions['train']['y'] = y_train

    predictionfile = join(outputdir, 'xgbpredictions.h5')
    pd.DataFrame(predictions['test']).to_hdf(predictionfile, key='test')
    pd.DataFrame(predictions['train']).to_hdf(predictionfile, key='train')
