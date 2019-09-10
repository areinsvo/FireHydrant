#!/usr/bin/env python
"""classes to enforce optimization sequences
"""
import os
import warnings
from os.path import join

import numpy as np
import pandas as pd
import xgboost as xgb
from bayes_opt import BayesianOptimization
from xgb2tmva import convert_model
from xgbcallbacks import callback_overtraining, early_stop


class XgboFitter:
    def __init__(self, outdir, hyperparamsetting,
                randomstate=2019,
                maxrounds=3000,
                minrounds=3,
                earlystoprounds=30,
                nthread=16,
                doregression=False,
                useeffrms=True,
                usegpu=False
                ):
        assert (hyperparamsetting and isinstance(hyperparamsetting, dict))

        self.outdir_ = outdir                        # results caching
        self.hyperparamdefault_ = {k: v['default'] for k, v in hyperparamsetting.items()}  # default hyperparameter setting for xgboost
        self.hyperparamranges_ = {k: tuple(v['range']) for k, v in hyperparamsetting.items()} # hyperparameter ranges to be optimized
        self.hyperparamloguniform_ = \
            [k for k in hyperparamsetting if hyperparamsetting[k]['loguniform']] # hyperparameter names whose value will be sampled in a log-uniform way
        self.randomstate_ = randomstate
        self.maxrounds_ = maxrounds
        self.minrounds_ = minrounds
        self.earlystoprounds_ = earlystoprounds
        self.doregression_ = doregression
        self.useeffrms_ = useeffrms

        self.params_ = {
            'silent': 1,
            'verbose_eval': 0,
            'nthread': nthread,
            'objective': 'reg:linear',
        }
        self.cvcolumns_ = [] # sequence matters
        self.cvresults_ = [] # holding result of each cross validation
        self.cviter_ = 0     # number of cross validation performed

        if usegpu: # enable GPU acceleration
            self.params_.update({"tree_method": "gpu_hist",})

        ## setting cvresults subdir
        if not os.path.exists(join(self.outdir_, 'cvresults')):
            os.makedirs(join(self.outdir_, 'cvresults'))

        if doregression: # regression task
            self.hyperparamdefault_['base_score'] = 1
            if useeffrms:
                self.cvcolumns_ = [
                    "train-effrms-mean",
                    "train-effrms-std",
                    "test-effrms-mean",
                    "test-effrms-std"
                    ]
            else:
                self.cvcolumns_ = [
                    "train-rmse-mean",
                    "train-rmse-std",
                    "test-rmse-mean",
                    "test-rmse-std"
                    ]
        else: # classification task
            self.cvcolumns_ = [
                "train-auc-mean",
                "train-auc-std",
                "test-auc-mean",
                "test-auc-std"
            ]

            self.params_.update({
                'objective': 'binary:logitraw',
                'eval_metric': 'auc',
            })

        self.earlystophistory_ = []
        self.models_ = {}
        self.callbackstatus_ = []
        self.trieddefault_ = False

        ## optimizer
        self.optimizer_ = BayesianOptimization(self.evaluate_xgb,
                                               self.hyperparamranges_,
                                               self.randomstate_)

        ## if trained before, adjust random state and load history
        summaryfile = join(self.outdir_, 'summary.csv')
        if os.path.isfile(summaryfile):
            _df = pd.read_csv(summaryfile)
            self.randomstate_ += len(_df)
            self._load_data(summaryfile)


    def _load_data(self, summaryfile):

        df = pd.read_csv(summaryfile)
        print("Found results of {} optimization rounds in output directory, loading...".format(len(df)))

        self.earlystophistory_.extend(list(df.n_estimators.values))
        self.callbackstatus_.extend(list(df.callback.values))
        self.trieddefault_=True

        ## load cross validation results
        for i in range(len(df)):
            cvfile = join(self.outdir_, 'cvresults/{0:04d}.csv'.format(i))
            self.cvresults_.append(pd.read_csv(cvfile))
        self.cviter_ = len(df)

        ## load the optimization results so far into the Bayesian optimization object
        eval_col = self.cvcolumns_[2]
        df['target'] = -df[eval_col] if self.doregression_ else df[eval_col]
        # idx_max, val_max = 0, 0
        # if self.doregression_:
        #     idx_max = df[eval_col].idxmin()
        #     val_max = -df[eval_col].min()
        #     df['target'] = -df[eval_col]
        # else:
        #     idx_max = df[eval_col].idxmax()
        #     val_max = df[eval_col].max()
        #     df['target'] = df[eval_col]

        for idx in df.index:
            value = df.loc[idx, eval_col]
            if self.doregression_: value = -value

            params = df.loc[idx, list(self.hyperparamranges_)].to_dict()
            self.optimizer_.register(params, value)


    def evaluate_xgb(self, **hyperparameters):

        for k in hyperparameters:
            if k in self.hyperparamloguniform_:
                hyperparameters[k] = 10 ** hyperparameters[k]

        self.params_.update(hyperparameters)
        self.params_ = guardxgbparams(self.params_)

        best_test_eval_metric = -9999999.0
        if self.optimizer_.res:
            self.summary.to_csv(join(self.outdir_, 'summary.csv'))
            best_test_eval_metric = max([d['target'] for d in self.optimizer_.res])

        feval = None # evaluation function
        callback_status = {'status': 0}

        if self.doregression_ and self.useeffrms_:
            callbacks = [early_stop(self.earlystoprounds_, start_round=self.minrounds_, eval_idx=-2),]
            feval = evaleffrms
        else:
            callbacks = [
                early_stop(self.earlystoprounds_, start_round=self.minrounds_,),
                callback_overtraining(best_test_eval_metric, callback_status),
            ]

        cv_result = xgb.cv(self.params_, self.xgtrain_,
                           num_boost_round=self.maxrounds_,
                           nfold=self.nfold_,
                           seed=self.randomstate_,
                           callbacks=callbacks,
                           verbose_eval=50,
                           feval=feval)
        cv_result.to_csv(join(self.outdir_, 'cvresults/{0:04d}.csv'.format(self.cviter_)))

        self.cviter_ += 1
        self.earlystophistory_.append(len(cv_result))
        self.cvresults_.append(cv_result)
        self.callbackstatus_.append(callback_status['status'])

        if self.doregression_:
            return -cv_result[self.cvcolumns_[2]].values[-1]
        else:
            return cv_result[self.cvcolumns_[2]].values[-1]


    def optimize(self, xgtrain, init_points=3, n_iter=3, nfold=5, acq='ei'):

        self.nfold_ = nfold
        self.xgtrain_ = xgtrain

        if not self.trieddefault_:
            self.optimizer_.probe(params=list(self.hyperparamdefault_.values()), lazy=False)
            self.trieddefault_ = True

        ## NOTE
        # The following block is mostly equivalent to
        #   self.optimizer_.maximize(init_points=init_points, n_iter=n_iter, acq=acq)
        # but saving summary file after each hyperparameter point probed,
        # in case program stopped, and we want to reload and continue next time.
        self.optimizer_.maximize(init_points=init_points, n_iter=0, acq=acq)

        for i in range(n_iter):
            self.optimizer_.maximize(init_points=0, n_iter=1, acq=acq)
            self.summary.to_csv(join(self.outdir_, 'summary.csv'))
        self.summary.to_csv(join(self.outdir_, 'summary.csv'))


    def fit(self, xgtrain, model='optimized'):

        params = self.params_

        if model == 'default':
            params.update(self.hyperparamdefault_)
            params['n_estimators'] = self.earlystophistory_[0]

        if model == 'optimized':
            idxmax = np.argmax([d['target'] for d in self.optimizer_.res])
            params.update(guardxgbparams(self.optimizer_.res[idxmax]['params']))
            for k in params:
                if k in self.hyperparamloguniform_:
                    params[k] = 10 ** params[k]
            params['n_estimators'] = self.earlystophistory_[idxmax]

        self.models_[model] = xgb.train(params, xgtrain, params['n_estimators'], verbose_eval=50)


    def predict(self, xgtest, model='optimized'):
        return self.models_[model].predict(xgtest)


    @property
    def summary(self):

        res = [dict(d) for d in self.optimizer_.res] # [{'target': float, 'params': dict}, ]
        for d in res:
            d['params'] = guardxgbparams(d['params'])

        data = {}
        for name in self.cvcolumns_:
            data[name] = [r[name].values[-1] for r in self.cvresults_]
        for hp in self.hyperparamranges_:
            data[hp] = [r['params'][hp] for r in res]
        data['n_estimators'] = self.earlystophistory_
        data['callback'] = self.callbackstatus_

        return pd.DataFrame(data)


    def save_model(self, feature_names, model='optimized'):

        modeldir = join(self.outdir_, 'model_' + model)
        print("saving {} model --> {}".format(model, modeldir))

        if not os.path.exists(modeldir):
            os.makedirs(modeldir)

        self.models_[model].dump_model(join(modeldir, 'dump.raw.txt')) # dump text
        self.models_[model].save_model(join(modeldir, 'model.bin')) # save binary

        tmvafile = join(modeldir, 'weights.xml')
        try:
            convert_model(self.models_[model].get_dump(),
                          input_variables=[(n, 'F') for n in feature_names],
                          output_xml=tmvafile)
            os.system("xmllint --format {0} > {0}.tmp".format(tmvafile))
            os.system("mv {0} {0}.bak".format(tmvafile))
            os.system("mv {0}.tmp {0}".format(tmvafile))
            os.system("gzip -f {0}".format(tmvafile))
            os.system("mv {0}.bak {0}".format(tmvafile))
        except:
            warnings.warn("\n".join([
                "Warning:",
                "Saving model<{}> in TMVA XML format failed.".format(model),
                "Don't worry now, you can still convert xgboost model later."
                ]))


class XgboClassifier(XgboFitter):
    def __init__(self, outdir, hyperparamsetting, **kwargs):
        super().__init__(outdir, hyperparamsetting, **kwargs, doregression=False)


class XgboRegressor(XgboFitter):
    def __init__(self, outdir, hyperparamsetting, **kwargs):
        super().__init__(outdir, hyperparamsetting, **kwargs, doregression=True)

def evaleffrms(preds, dtrain, c=0.683):
    """Effective RMS evaluation function for xgboost

    :param preds: prediction value
    :type preds: numpy.array
    :param dtrain: training dmatrix
    :type dtrain: xgboost.DMatrix
    :param c: percentage to evaluate, defaults to 0.683
    :type c: float, optional
    :return: tuple of metrics -- (str, float)
    :rtype: tuple
    """
    labels = dtrain.get_label()
    # return a pair metric_name, result. The metric name must not contain a colon (:) or a space
    # since preds are margin(before logistic transformation, cutoff at 0)
    x = np.sort(preds / labels, kind="mergesort")
    m = int(c * len(x)) + 1
    effrms = np.min(x[m:] - x[:-m]) / 2.0
    return ("effrms", effrms)


def guardxgbparams(params):

    res = dict(params)
    res['colsample_bytree'] = max(min(res["colsample_bytree"], 1), 0)
    res["max_depth"] = int(res["max_depth"])
    res["gamma"] = max(res["gamma"], 0)
    res["reg_lambda"] = max(res["reg_lambda"], 0)

    return res
