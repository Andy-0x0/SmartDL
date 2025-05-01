import warnings
import random
import logging
from collections import OrderedDict
import copy
from tqdm import tqdm

import ray
from joblib import cpu_count
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.feature_selection import RFECV, RFE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline

from feature_engine.selection import SelectBySingleFeaturePerformance, SelectByInformationValue

import lightgbm as lgb
import xgboost as xgb
import catboost as cab


# Helper function for RFECV
@ray.remote
def _RFECV_select_vote(features, labels, model, normalize="z-score", score='f1', cv=5, n_jobs=1):
    warnings.simplefilter("ignore", FutureWarning)
    # Set the dynamic normalizer cooperate along with the core estimator
    use_pipeline = False
    if normalize == "z-score":
        scalar = StandardScaler()
        estimator = Pipeline([
            ('scaler', scalar),
            ('model', model)
        ])
        use_pipeline = True
    elif normalize == "min-max":
        scalar = MinMaxScaler()
        estimator = Pipeline([
            ('scaler', scalar),
            ('model', model)
        ])
        use_pipeline = True
    elif normalize == "" or normalize is None:
        warnings.warn("Invalid normalization method, treated as None")
        estimator = model
    else:
        estimator = model

    # Set the feature importance getter for different estimators
    if use_pipeline:
        if isinstance(model, (LogisticRegression, LinearSVC)):
            getter = lambda est: abs(est.named_steps['model'].coef_)
            n_jobs_cv = n_jobs
        elif isinstance(model, cab.CatBoostClassifier):
            getter = lambda est: est.named_steps['model'].feature_importances_
            n_jobs_cv = n_jobs
        else:
            getter = lambda est: est.named_steps['model'].feature_importances_
            estimator.set_params(model__n_jobs=n_jobs)
            n_jobs_cv = None
    else:
        if isinstance(model, (LogisticRegression, LinearSVC)):
            getter = lambda est: abs(est.coef_)
            n_jobs_cv = n_jobs
        elif isinstance(model, cab.CatBoostClassifier):
            getter = lambda est: est.feature_importances_
            n_jobs_cv = n_jobs
        else:
            getter = lambda est: est.feature_importances_
            estimator.set_params(n_jobs=n_jobs)
            n_jobs_cv = None

    # Set the RFECV / REF object based on the combination of param cv and tops
    try:
        assert cv is None or cv >= 2
    except Exception as e:
        raise ValueError(
            "You have to specify a cv [2, inf) or None if you want to automatically determine the selected feature size"
        )
    else:
        selector = RFECV(estimator=estimator, step=1, cv=cv, scoring=score, importance_getter=getter, n_jobs=n_jobs_cv)
        selector.fit(features, labels)

    return selector.support_


# Helper function for RFE
@ray.remote
def _RFE_select_rank(features, labels, model, normalize="z-score", tops=5, n_jobs=1):
    warnings.simplefilter("ignore", FutureWarning)
    # Set the dynamic normalizer cooperate along with the core estimator
    use_pipeline = False
    if normalize == "z-score":
        scalar = StandardScaler()
        estimator = Pipeline([
            ('scaler', scalar),
            ('model', model)
        ])
        use_pipeline = True
    elif normalize == "min-max":
        scalar = MinMaxScaler()
        estimator = Pipeline([
            ('scaler', scalar),
            ('model', model)
        ])
        use_pipeline = True
    elif normalize == "" or normalize is None:
        warnings.warn("Invalid normalization method, treated as None")
        estimator = model
    else:
        estimator = model

    # Set the feature importance getter for different estimators
    if use_pipeline:
        if isinstance(model, (LogisticRegression, LinearSVC)):
            getter = lambda est: abs(est.named_steps['model'].coef_)
        elif isinstance(model, cab.CatBoostClassifier):
            getter = lambda est: est.named_steps['model'].feature_importances_
        else:
            getter = lambda est: est.named_steps['model'].feature_importances_
            estimator.set_params(model__n_jobs=n_jobs)
    else:
        if isinstance(model, (LogisticRegression, LinearSVC)):
            getter = lambda est: abs(est.coef_)
        elif isinstance(model, cab.CatBoostClassifier):
            getter = lambda est: est.feature_importances_
        else:
            getter = lambda est: est.feature_importances_
            estimator.set_params(n_jobs=n_jobs)

    # Set the RFECV / REF object based on the combination of param cv and tops
    try:
        assert tops is None or tops >= 1
    except Exception as e:
        raise ValueError(
            "You have to specify a tops [1, num_of_features) or None if you want to select half of the features"
        )
    else:
        selector = RFE(estimator=estimator, n_features_to_select=tops, step=1, importance_getter=getter)
        selector.fit(features, labels)

    return selector.ranking_


# Helper function for SingleSelection
@ray.remote
def _single_select_vote(features, labels, model, threshold, normalize="z-score", score='roc_auc', cv=5, n_jobs=1):
    warnings.simplefilter("ignore", FutureWarning)
    # Set the dynamic normalizer cooperate along with the core estimator
    use_pipeline = False
    if normalize == "z-score":
        scalar = StandardScaler()
        estimator = Pipeline([
            ('scaler', scalar),
            ('model', model)
        ])
        use_pipeline = True
    elif normalize == "min-max":
        scalar = MinMaxScaler()
        estimator = Pipeline([
            ('scaler', scalar),
            ('model', model)
        ])
        use_pipeline = True
    elif normalize == "" or normalize is None:
        warnings.warn("Invalid normalization method, treated as None")
        estimator = model
    else:
        estimator = model

    # Set the feature importance getter for different estimators
    if use_pipeline:
        if isinstance(model, (xgb.XGBClassifier, lgb.LGBMClassifier)):
            estimator.set_params(model__n_jobs=n_jobs)
    else:
        if isinstance(model, (XGBClassifier, LightGBMClassifier)):
            estimator.set_params(n_jobs=n_jobs)

    # Set the RFECV / REF object based on the combination of param cv and tops
    try:
        assert cv is None or cv >= 2
    except Exception as e:
        raise ValueError(
            "You have to specify a cv [2, inf) or None if you want to automatically determine the selected feature size"
        )
    else:
        selector = SelectBySingleFeaturePerformance(
            estimator=estimator,
            scoring=score,
            threshold=threshold,
            cv=cv
        )
        selector.fit(features, labels)

    return selector.get_support(indices=False)


# Temp
@ray.remote
def _single_select_score(features, labels, model, threshold, normalize="z-score", score='roc_auc', cv=5, n_jobs=1):
    warnings.simplefilter("ignore", FutureWarning)
    # Set the dynamic normalizer cooperate along with the core estimator
    use_pipeline = False
    if normalize == "z-score":
        scalar = StandardScaler()
        estimator = Pipeline([
            ('scaler', scalar),
            ('model', model)
        ])
        use_pipeline = True
    elif normalize == "min-max":
        scalar = MinMaxScaler()
        estimator = Pipeline([
            ('scaler', scalar),
            ('model', model)
        ])
        use_pipeline = True
    elif normalize == "" or normalize is None:
        warnings.warn("Invalid normalization method, treated as None")
        estimator = model
    else:
        estimator = model

    # Set the feature importance getter for different estimators
    if use_pipeline:
        if isinstance(model, (xgb.XGBClassifier, lgb.LGBMClassifier)):
            estimator.set_params(model__n_jobs=n_jobs)
    else:
        if isinstance(model, (XGBClassifier, LightGBMClassifier)):
            estimator.set_params(n_jobs=n_jobs)

    # Set the RFECV / REF object based on the combination of param cv and tops
    try:
        assert cv is None or cv >= 2
    except Exception as e:
        raise ValueError(
            "You have to specify a cv [2, inf) or None if you want to automatically determine the selected feature size"
        )
    else:
        selector = SelectBySingleFeaturePerformance(
            estimator=estimator,
            scoring=score,
            threshold=threshold,
            cv=cv
        )
        selector.fit(features, labels)

    return list(selector.feature_performance_.values())


# FeatureEngineer for Classification
class ClassificationFeatureEngineer:
    def __init__(self, features, labels, random_state=42):
        warnings.simplefilter("ignore", FutureWarning)

        self.features = ClassificationFeatureEngineer._list2pandas(features)
        self.labels = ClassificationFeatureEngineer._list2pandas(labels)
        self.seed = random_state
        self._set_seed(random_state)

        self.model_lookup = {
            'LR': LogisticRegression(
                n_jobs=4,
                max_iter=50000,
                random_state=self.seed
            ),
            'SVM': LinearSVC(
                penalty='l1',
                max_iter=50000,
                random_state=self.seed,
            ),
            'RF': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                n_jobs=-1,
                random_state=self.seed
            ),
            'ADA': AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=5, random_state=self.seed),
                n_estimators=100,
                random_state=self.seed
            ),
            'LGB': lgb.LGBMClassifier(
                boosting_type='gbdt',
                n_estimators=100,
                learning_rate=0.1,
                max_depth=10,
                num_leaves=31,
                verbose=-1,
                n_jobs=-1,
                random_state=self.seed
            ),
            'XGB': xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                n_jobs=-1,
                random_state=self.seed,
            ),
            'CAB': cab.CatBoostClassifier(
                iterations=1000,
                learning_rate=0.05,
                depth=6,
                verbose=False,
                thread_count=1,
                random_seed=self.seed
            )
        }

        self.name_lookup = {
            "LR": "Logistic Regression",
            "SVM": "Support Vector Machine",
            "RF": "Random Forest",
            "ADA": "AdaBoost",
            "XGB": "Xgboost",
            "LGB": "LightGBM",
            "CAB": "CatBoost",
        }

        self.parallel_lookup = {
            "LR":   {"n_jobs": 2, "cpus": 2},
            "SVM":  {"n_jobs": 2, "cpus": 2},
            "RF":   {"n_jobs": 8, "cpus": 8},
            "ADA":  {"n_jobs": 8, "cpus": 8},
            "XGB":  {"n_jobs": 4, "cpus": 4},
            "LGB":  {"n_jobs": 4, "cpus": 4},
            "CAB":  {"n_jobs": 8, "cpus": 8}
        }

    @staticmethod
    def _list2pandas(unk):
        if isinstance(unk, (pd.DataFrame, pd.Series)):
            return unk

        if isinstance(unk, list):
            unk = np.array(unk)

        if isinstance(unk, np.ndarray):
            if unk.ndim == 1:
                return pd.Series(unk)
            else:
                return pd.DataFrame(unk)

    def _set_seed(
        self,
        seed:int = 42
    ) -> None:
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def lasso_select(
        self,
        start:int = -3,
        end:int = 3,
        tops:int = 5,
        viz:bool = True,
        save:str | None = None,
        log=False
    ) -> pd.Series:
        regularizers = np.logspace(start=end, stop=start, num=100, base=10)
        tracker = pd.DataFrame()

        for regularizer in regularizers:
            model = LogisticRegression(C=regularizer, penalty='l1', solver='saga')
            model.fit(self.features, self.labels)
            coefficient = model.coef_

            if coefficient.shape[0] == 1:
                coefficient = coefficient.flatten()
                coefficient = pd.DataFrame(coefficient.reshape(1, -1), columns=self.features.columns)
                if tracker.empty:
                    tracker = coefficient
                else:
                    tracker = pd.concat([tracker, coefficient], ignore_index=True, axis=0)
            else:
                coefficient = coefficient.sum(axis=0)
                coefficient = coefficient.flatten()
                coefficient = pd.DataFrame(coefficient.reshape(1, -1), columns=self.features.columns)
                if tracker.empty:
                    tracker = coefficient
                else:
                    tracker = pd.concat([tracker, coefficient], ignore_index=True, axis=0)

        counter = tracker.copy()
        counter[counter != 0] = 1
        counter = counter.sum(axis=1)

        top_features = []
        for idx in range(len(counter) - 1, -1, -1):
            if counter.iloc[idx] >= tops:
                top_features = tracker.iloc[idx, :]
                top_features = top_features.loc[top_features != 0]
                top_features = top_features.sort_values(ascending=False, key=lambda x: abs(x))
                break
            else:
                continue

        # Printing the top features
        if log:
            print(' [Logistic Regression] Feature Selection '.center(54, '='))
            for rank in range(0, len(top_features)):
                if rank + 1 > tops:
                    break
                else:
                    print(f'{rank + 1}. {str(top_features.index[rank])}    weight: {top_features.iloc[rank]:>6.3f}')

        # Visualize the top features
        if viz or save:
            plt.figure(figsize=(16, 9))

            for feature in range(0, tracker.shape[1]):
                if tracker.columns[feature] in top_features.index:
                    tracker.iloc[:, feature].plot(
                        kind='line',
                        label=str(tracker.columns[feature])
                    )
                else:
                    tracker.iloc[:, feature].plot(
                        kind='line',
                        alpha=0.5,
                        lw=0.8,
                        color='gray',
                        label='_nolegend_'
                    )

            plt.axhline(0, color='black', linestyle='--', linewidth=0.8)

            plt.legend(loc='upper right')
            plt.title("Logistic Regression Feature Selection")
            plt.xlabel("Regularizer's Significance")
            plt.ylabel("Features' Significance")

            if viz:
                plt.show()
                plt.close()

            if save:
                try:
                    plt.savefig(Path(save))
                except:
                    raise ValueError('Invalid save path')

        return pd.Series(top_features, name='Tops')

    def RFECV_select(
        self,
        models:str | list[str] = ('LR', 'LGB', 'XGB', 'RF', 'SVM'),
        cv:int = 10,
        score:any = 'f1',
        normalize:str = "z-score",
        standard:str | float | int = "strict",
        verbose:bool = True,
        parallel:bool = True,
    ) -> tuple[list, pd.DataFrame]:
        """
        Determine the most helpful features based on an ensemble of different models

        :param models:      A string representing the ingredients of the ensemble
        :param cv:          An integer representing the cross-validation folds
        :param score:       A string representation of the objective indicator of the selection
        :param normalize:   A string representation of the normalization method
        :param verbose:     A boolean value to decide whether to do the logging or not
        :param parallel:    A boolean value to decide whether to parallelize each model's selection or not
        :return:
        """

        # Making the logging into blue
        def highlight(text: str) -> str:
            return "\033[94m" + str(text) + '\033[0m'

        def sub_highlight(text: str) -> str:
            return "\033[93m" + str(text) + '\033[0m'

        # Uniform the type of param models
        if isinstance(models, str):
            models = [models]

        if not parallel:
            votes = np.zeros((self.features.shape[-1], len(models)))
            with tqdm(total=len(models), leave=True, disable=not verbose) as pbar:
                pbar.set_description(f"[Init]")
                for idx, model in enumerate(models):
                    future_vote = _RFECV_select_vote.remote(
                        self.features,
                        self.labels,
                        self.model_lookup[model],
                        normalize,
                        score,
                        cv,
                        -1
                    )
                    votes[:, idx] = ray.get(future_vote)

                    pbar.update(1)
                    pbar.set_description(f"[{self.name_lookup[models[idx]]} - Done]")
        else:
            ray.init(num_cpus=cpu_count(), ignore_reinit_error=True, log_to_driver=False)

            model_instances = [self.model_lookup[m] for m in models]
            model_configs = [self.parallel_lookup[m] for m in models]
            future2idx = {}
            futures = []
            for idx, (config, model) in enumerate(zip(model_configs, model_instances)):
                future = _RFECV_select_vote.options(num_cpus=config["cpus"]).remote(
                    self.features,
                    self.labels,
                    model,
                    normalize,
                    score,
                    cv,
                    config["n_jobs"]
                )
                futures.append(future)
                future2idx[future] = idx

            votes = [None] * len(futures)
            with tqdm(total=len(models), leave=True, disable=not verbose) as pbar:
                pbar.set_description(f"[Init]")

                while futures:
                    ready, futures = ray.wait(futures, num_returns=1)
                    for future in ready:
                        vote = ray.get(future)
                        idx = future2idx[future]
                        votes[idx] = vote

                        pbar.update(1)
                        pbar.set_description(f"[{self.name_lookup[models[idx]]} - Done]")

            votes = np.stack(votes, axis=1)
            ray.shutdown()

        feature_names = self.features.columns.to_numpy()
        votes = pd.DataFrame(votes, index=feature_names, columns=models).astype(bool)
        hits = votes.sum(axis=1)

        if isinstance(standard, str) and standard == "all":
            selected = votes.loc[hits == len(models)].index.to_list()
        elif isinstance(standard, str) and standard == "strict":
            selected = votes.loc[hits > (len(models) // 2)].index.to_list()
        elif isinstance(standard, str) and standard == "lose":
            selected = votes.loc[hits >= (len(models) // 2)].index.to_list()
        elif isinstance(standard, float):
            if standard <= 0 or standard > 1:
                raise ValueError("Invalid standard parameter: should be in (0, 1]")
            selected = votes.loc[hits >= int(len(models) * standard)].index.to_list()
        elif isinstance(standard, int):
            if standard <= 0 or standard > len(models):
                raise ValueError("Invalid standard parameter: should be in (0, num_models]")
            selected = votes.loc[hits >= standard].index.to_list()
        else:
            selected = votes.loc[hits > (len(models) // 2)].index.to_list()

        if verbose:
            max_space = max([len(name) + 2 for name in self.name_lookup.values()])

            print("The selected features are: ", end="")
            for idx, fea in enumerate(selected):
                if idx != len(selected) - 1:
                    print(f"{highlight(fea)}, ", end="")
                else:
                    print(f"{highlight(fea)}", end="")
            print()

            for abb in models:
                print(f"\t{self.name_lookup[abb] + ':':<{max_space}}", end="")
                picked = feature_names[votes.loc[:, abb].to_list()]
                for idx, pic in enumerate(picked):
                    if idx != len(picked) - 1:
                        print(f"{sub_highlight(pic)}, ", end="")
                    else:
                        print(f"{sub_highlight(pic)}", end="")
                print()

        votes = votes.reset_index().rename(columns={'index': 'features'})

        return selected, votes

    def RFE_select(
        self,
        models: str | list[str] = ('LR', 'LGB', 'XGB', 'RF', 'SVM'),
        tops: int | str = 5,
        normalize: str = "z-score",
        verbose: bool = True,
        parallel: bool = True,
    ) -> tuple[list, pd.DataFrame]:
        """
        Determine the most helpful features based on an ensemble of different models

        :param models:      A string representing the ingredients of the ensemble
        :param tops:        Manually decide how many features to select
        :param normalize:   A string representation of the normalization method
        :param verbose:     A boolean value to decide whether to do the logging or not
        :param parallel:    A boolean value to decide whether to parallelize each model's selection or not
        :return:
        """

        # Making the logging into blue
        def highlight(text: str) -> str:
            return "\033[94m" + str(text) + '\033[0m'

        def sub_highlight(text: str) -> str:
            return "\033[93m" + str(text) + '\033[0m'

        # Uniform the type of param models
        if isinstance(models, str):
            models = [models]

        if not parallel:
            ranks = np.zeros((self.features.shape[-1], len(models)))
            with tqdm(total=len(models), leave=True, disable=not verbose) as pbar:
                pbar.set_description(f"[Init]")

                for idx, model in enumerate(models):
                    future_vote = _RFE_select_rank.remote(
                        self.features,
                        self.labels,
                        self.model_lookup[model],
                        normalize,
                        tops,
                        -1
                    )
                    ranks[:, idx] = ray.get(future_vote)

                    pbar.update(1)
                    pbar.set_description(f"[{self.name_lookup[models[idx]]} - Done]")
        else:
            ray.init(num_cpus=cpu_count(), ignore_reinit_error=True, log_to_driver=False)

            model_instances = [self.model_lookup[m] for m in models]
            model_configs = [self.parallel_lookup[m] for m in models]
            future2idx = {}
            futures = []
            for idx, (config, model) in enumerate(zip(model_configs, model_instances)):
                future = _RFE_select_rank.options(num_cpus=config["cpus"]).remote(
                    self.features,
                    self.labels,
                    model,
                    normalize,
                    tops,
                    config["n_jobs"]
                )
                futures.append(future)
                future2idx[future] = idx

            ranks = [None] * len(futures)
            with tqdm(total=len(models), leave=True, disable=not verbose) as pbar:
                pbar.set_description(f"[Init]")

                while futures:
                    ready, futures = ray.wait(futures, num_returns=1)
                    for future in ready:
                        rank = ray.get(future)
                        idx = future2idx[future]
                        ranks[idx] = rank

                        pbar.update(1)
                        pbar.set_description(f"[{self.name_lookup[models[idx]]} - Done]")

            ranks = np.stack(ranks, axis=1)
            ray.shutdown()

        feature_names = self.features.columns.to_numpy()
        ranks = pd.DataFrame(ranks, index=feature_names, columns=models)

        ranks = ranks.rank(method='min', ascending=True)
        ranks.loc[:, 'ranks'] = ranks.sum(axis=1)
        ranks = ranks.sort_values(axis=0, by='ranks', kind='stable', ascending=True)
        real_tops = sorted(list(ranks.loc[:, 'ranks'].to_list()))[tops - 1]
        selected = ranks.loc[ranks.loc[:, 'ranks'] <= real_tops].index.to_list()

        if verbose:
            def list2str(content_list):
                return_str = ""
                for i, c in enumerate(content_list):
                    return_str += f"{c}, " if i != len(content_list) - 1 else f"{c}"
                return return_str

            def display_block(title, col, color_funct):
                nonlocal ranks
                ranks_copy = copy.deepcopy(ranks).sort_values(axis=0, by=col, kind='stable', ascending=True)
                local_tops = sorted(list(ranks_copy.loc[:, col].to_list()))[tops - 1]
                ranks_copy = ranks_copy.loc[ranks_copy.loc[:, col] <= local_tops]

                ref_list = ranks_copy.loc[:, col].to_list()
                idx_list = ranks_copy.index.to_list()

                title_len = len(title)
                print(title, end="")

                prev_rank = 1
                logging_dict = OrderedDict()
                collector = []
                for idx, fea in enumerate(idx_list):
                    if ref_list[idx] == ref_list[max(0, idx - 1)]:
                        collector.append(fea)
                    else:
                        post_rank = prev_rank + len(collector)
                        logging_dict[f"[{prev_rank}-{post_rank - 1}]" if prev_rank + 1 < post_rank else f"[{prev_rank}]"] = collector
                        prev_rank = post_rank
                        collector = [fea]
                if collector:
                    post_rank = prev_rank + len(collector)
                    logging_dict[f"[{prev_rank}-{post_rank - 1}]" if prev_rank + 1 < post_rank else f"[{prev_rank}]"] = collector

                max_tag_len = max(list(map(len, list(logging_dict.keys()))))
                for idx, (key, value) in enumerate(logging_dict.items()):
                    if idx == 0:
                        print(color_funct(f"{key.ljust(max_tag_len)} {list2str(value)}"))
                    else:
                        print(" " * title_len + color_funct(f"{key.ljust(max_tag_len)} {list2str(value)}"))

            max_space = max([len(name) + 2 for name in self.name_lookup.values()])
            display_block("The selected features are: ", 'ranks', highlight)
            for abb in models:
                display_block(f"    {self.name_lookup[abb] + ':':<{max_space}}", abb, sub_highlight)

        ranks = ranks.reset_index().rename(columns={'index': 'features'})

        return selected, ranks

    def single_performance_select(
        self,
        models: str | list[str] = ('LR', 'LGB', 'XGB', 'RF', 'SVM'),
        cv: int = 10,
        threshold: float = 0.7,
        score: any = 'f1',
        normalize: str = "z-score",
        standard: str | float | int = "strict",
        verbose: bool = True,
        parallel: bool = True,
    ) -> tuple[list, pd.DataFrame]:
        """
        Determine the most helpful features based on an ensemble of different models

        :param models:      A string representing the ingredients of the ensemble
        :param cv:          An integer representing the cross-validation folds
        :param threshold:   A float representing the threshold for the scoring
        :param score:       A string representation of the objective indicator of the selection
        :param normalize:   A string representation of the normalization method
        :param verbose:     A boolean value to decide whether to do the logging or not
        :param parallel:    A boolean value to decide whether to parallelize each model's selection or not
        :return:
        """

        # Making the logging into blue
        def highlight(text: str) -> str:
            return "\033[94m" + str(text) + '\033[0m'

        def sub_highlight(text: str) -> str:
            return "\033[93m" + str(text) + '\033[0m'

        # Uniform the type of param models
        if isinstance(models, str):
            models = [models]

        if not parallel:
            votes = np.zeros((self.features.shape[-1], len(models)))
            with tqdm(total=len(models), leave=True, disable=not verbose) as pbar:
                pbar.set_description(f"[Init]")
                for idx, model in enumerate(models):
                    future_vote = _single_select_vote.remote(
                        self.features,
                        self.labels,
                        self.model_lookup[model],
                        threshold,
                        normalize,
                        score,
                        cv,
                        -1
                    )
                    votes[:, idx] = ray.get(future_vote)

                    pbar.update(1)
                    pbar.set_description(f"[{self.name_lookup[models[idx]]} - Done]")
        else:
            ray.init(num_cpus=cpu_count(), ignore_reinit_error=True, log_to_driver=False)

            model_instances = [self.model_lookup[m] for m in models]
            model_configs = [self.parallel_lookup[m] for m in models]
            future2idx = {}
            futures = []
            for idx, (config, model) in enumerate(zip(model_configs, model_instances)):
                future = _single_select_vote.remote(
                    self.features,
                    self.labels,
                    model,
                    threshold,
                    normalize,
                    score,
                    cv,
                    -1
                )
                futures.append(future)
                future2idx[future] = idx

            votes = [None] * len(futures)
            with tqdm(total=len(models), leave=True, disable=not verbose) as pbar:
                pbar.set_description(f"[Init]")

                while futures:
                    ready, futures = ray.wait(futures, num_returns=1)
                    for future in ready:
                        vote = ray.get(future)
                        idx = future2idx[future]
                        votes[idx] = vote

                        pbar.update(1)
                        pbar.set_description(f"[{self.name_lookup[models[idx]]} - Done]")

            votes = np.stack(votes, axis=1)
            ray.shutdown()

        feature_names = self.features.columns.to_numpy()
        votes = pd.DataFrame(votes, index=feature_names, columns=models).astype(bool)
        hits = votes.sum(axis=1)

        if isinstance(standard, str) and standard == "all":
            selected = votes.loc[hits == len(models)].index.to_list()
        elif isinstance(standard, str) and standard == "strict":
            selected = votes.loc[hits > (len(models) // 2)].index.to_list()
        elif isinstance(standard, str) and standard == "lose":
            selected = votes.loc[hits >= (len(models) // 2)].index.to_list()
        elif isinstance(standard, float):
            if standard <= 0 or standard > 1:
                raise ValueError("Invalid standard parameter: should be in (0, 1]")
            selected = votes.loc[hits >= int(len(models) * standard)].index.to_list()
        elif isinstance(standard, int):
            if standard <= 0 or standard > len(models):
                raise ValueError("Invalid standard parameter: should be in (0, num_models]")
            selected = votes.loc[hits >= standard].index.to_list()
        else:
            selected = votes.loc[hits > (len(models) // 2)].index.to_list()

        if verbose:
            max_space = max([len(name) + 2 for name in self.name_lookup.values()])

            print("The selected features are: ", end="")
            for idx, fea in enumerate(selected):
                if idx != len(selected) - 1:
                    print(f"{highlight(fea)}, ", end="")
                else:
                    print(f"{highlight(fea)}", end="")
            print()

            for abb in models:
                print(f"\t{self.name_lookup[abb] + ':':<{max_space}}", end="")
                picked = feature_names[votes.loc[:, abb].to_list()]
                for idx, pic in enumerate(picked):
                    if idx != len(picked) - 1:
                        print(f"{sub_highlight(pic)}, ", end="")
                    else:
                        print(f"{sub_highlight(pic)}", end="")
                print()

        votes = votes.reset_index().rename(columns={'index': 'features'})

        return selected, votes

    def PCA_select(
        self,
        tops:int = 5,
        viz:bool = True,
        save: str | None = None
    ) -> tuple[pd.DataFrame, pd.Series]:
        '''
        Perform PCA with visualization.

        :param tops:    How many top features to look into for specific weights
        :param viz:     To visualize the results or not
        :param save:    The path to save the PCA result plot
        :return:        Dataframe(specific weights for top features) | Series(variances of all PCs)
        '''
        feature_names = self.features.columns
        centered_features = self.features - self.features.mean(axis=0)
        feature_num = len(self.features.columns)

        # Get the pca with assigned tops & Get all pca with var
        target_pca = PCA(n_components=tops)
        target_pca.fit(centered_features)

        var_pca = PCA(n_components=feature_num)
        var_pca.fit(centered_features)
        variance_sr = pd.Series(var_pca.explained_variance_ratio_, name='pca_vars', index=[f'PC{i}' for i in range(1, feature_num + 1)])
        variance_diff = np.abs(np.diff(variance_sr.to_list())).tolist()
        variance_diff.insert(0, variance_sr.iloc[0])

        # Calculate the Meaningful cutoffs among all PCs
        def elbow(data):
            data = np.array(data)
            points = np.array(list(enumerate(data)))
            start_point = points[0]
            end_point = points[-1]

            def distance(point, start, end):
                return np.abs(np.cross(end - start, point - start) / np.linalg.norm(end - start))

            distances = [distance(point, start_point, end_point) for point in points]
            elbow_index = np.argmax(distances)

            return elbow_index
        cutoff_index = elbow(variance_diff) + 1

        weight_df = pd.DataFrame(target_pca.components_.T, index=feature_names, columns=[f"Top{rank + 1}" for rank in range(tops)])
        PC_idx = [f'IPC{i}' for i in range(1, cutoff_index + 1)] + [f'PC{i}' for i in range(cutoff_index + 1, feature_num + 1)]
        variance_sr.index = PC_idx

        # Plotting
        if viz or save:
            # Global Plotting Configuration
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9))

            # Specific Importance for Given Tops
            ax1.matshow(target_pca.components_, cmap='hot')
            ax1.set_yticks([rank for rank in range(tops)])
            ax1.set_yticklabels([f'PC{rank + 1}' for rank in range(tops)])
            ax1.set_xticks(np.arange(len(feature_names)))
            ax1.set_xticklabels(list(feature_names), rotation=60, ha='left')
            ax1.set_title('PCA Feature Selection')
            ax1.set_xlabel("Component Weights")
            ax1.set_ylabel("Principle Component")

            # Plot all Variance ratios of all PCs and the Differences Trends
            ax2.bar(np.arange(1, cutoff_index + 1), variance_sr.to_list()[: cutoff_index], color='blue', alpha=0.9)
            ax2.bar(np.arange(cutoff_index + 1, feature_num + 1), variance_sr.to_list()[cutoff_index:], color='gray', alpha=0.5)
            ax2_twin = ax2.twinx()
            ax2_twin.plot(np.arange(1, feature_num + 1), variance_diff, color='red', marker='o', linestyle='-', alpha=0.9)
            ax2.set_xticks(np.arange(1, len(feature_names) + 1))
            ax2.set_xticklabels(variance_sr.index, rotation=60, ha='left')
            ax2_twin.grid()
            ax2_twin.set_ylabel('Variance Diff')
            ax2.set_title('Variance of PCs')
            ax2.set_xlabel('PC Indexes')
            ax2.set_ylabel('Variance(%)')

            # Save the Plot
            if save:
                plt.savefig(save)

            # Visualize the Plot
            if viz:
                plt.show()

            plt.close()

        return weight_df, variance_sr

    def single_score_select(
        self,
        models: str | list[str] = ('LR', 'LGB', 'XGB', 'RF', 'SVM'),
        cv: int = 10,
        threshold: float = 0.7,
        score: any = 'f1',
        normalize: str = "z-score",
        parallel: bool = True,
        verbose: bool = True,
    ) -> tuple[list, pd.DataFrame]:
        """
        Determine the most helpful features based on an ensemble of different models

        :param models:      A string representing the ingredients of the ensemble
        :param cv:          An integer representing the cross-validation folds
        :param threshold:   A float representing the threshold for the scoring
        :param score:       A string representation of the objective indicator of the selection
        :param normalize:   A string representation of the normalization method
        :param verbose:     A boolean value to decide whether to do the logging or not
        :param parallel:    A boolean value to decide whether to parallelize each model's selection or not
        :return:
        """
        # Uniform the type of param models
        if isinstance(models, str):
            models = [models]

        if not parallel:
            scores = np.zeros((self.features.shape[-1], len(models)))
            with tqdm(total=len(models), leave=True, disable=not verbose) as pbar:
                pbar.set_description(f"[Init]")
                for idx, model in enumerate(models):
                    future_vote = _single_select_score.remote(
                        self.features,
                        self.labels,
                        self.model_lookup[model],
                        threshold,
                        normalize,
                        score,
                        cv,
                        -1
                    )
                    scores[:, idx] = ray.get(future_vote)

                    pbar.update(1)
                    pbar.set_description(f"[{self.name_lookup[models[idx]]} - Done]")
        else:
            ray.init(num_cpus=cpu_count(), ignore_reinit_error=True, log_to_driver=False)

            model_instances = [self.model_lookup[m] for m in models]
            model_configs = [self.parallel_lookup[m] for m in models]
            future2idx = {}
            futures = []
            for idx, (config, model) in enumerate(zip(model_configs, model_instances)):
                future = _single_select_score.remote(
                    self.features,
                    self.labels,
                    model,
                    threshold,
                    normalize,
                    score,
                    cv,
                    -1
                )
                futures.append(future)
                future2idx[future] = idx

            scores = [None] * len(futures)
            with tqdm(total=len(models), leave=True, disable=not verbose) as pbar:
                pbar.set_description(f"[Init]")

                while futures:
                    ready, futures = ray.wait(futures, num_returns=1)
                    for future in ready:
                        sc = ray.get(future)
                        idx = future2idx[future]
                        scores[idx] = sc

                        pbar.update(1)
                        pbar.set_description(f"[{self.name_lookup[models[idx]]} - Done]")

            scores = np.stack(scores, axis=1)
            ray.shutdown()

        feature_names = self.features.columns.to_numpy()
        scores = pd.DataFrame(scores, index=feature_names, columns=models)

        return scores


