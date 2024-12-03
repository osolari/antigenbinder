from collections import defaultdict
from typing import Dict

import matplotlib.axes
import numpy
import pandas
import seaborn
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_recall_curve,
    PrecisionRecallDisplay,
    average_precision_score,
    roc_curve,
    auc,
    RocCurveDisplay,
)
from sklearn.model_selection import GridSearchCV

from src.utils import LogReg

COLORS = {
    "HiTIDE": "crimson",
    "NCI": "steelblue",
    "TESLA": "forestgreen",
    "ALL": "goldenrod",
}


class ImmunoRank:

    def __init__(
        self,
        model_name: str,
        metric: str = "balanced_accuracy",
        random_state: int = 1,
        *args,
        **kwargs,
    ):

        self.model_name = model_name
        if model_name == "logistic regression":
            self.model_class = LogReg

        self.metric = metric
        if self.metric == "balanced_accuracy":
            self.scoring = "balanced_accuracy"
            self.metric = balanced_accuracy_score
        elif self.metric == "average_precision":
            self.scoring = "average_precision"
            self.metric = average_precision_score
        else:
            raise ValueError(f"Invalid metric {self.metric}")

        self.classifier = None
        self.random_state = random_state

    def fit_predict_logreg(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        parameters: Dict = None,
        opt_params: Dict = None,
    ):

        if parameters is None:
            parameters = {"C": numpy.logspace(-5, 5, 100)}
        elif not isinstance(parameters, dict):
            raise TypeError(f"parameters must be a dictionary")

        if opt_params is None:
            lr = self.model_class(
                random_state=self.random_state, class_weight="balanced"
            )
            clf = GridSearchCV(lr, parameters, scoring=self.scoring)
            clf.fit(X_train, y_train)

            cv_res = pandas.DataFrame(clf.cv_results_)
            opt_params = cv_res.iloc[cv_res["mean_test_score"].argmax()]["params"]
        else:
            cv_res = None

        self.classifier = self.model_class(**opt_params, class_weight="balanced")
        self.classifier.fit(X_train, y_train)

        if hasattr(self.classifier, "compute_ci"):
            self.classifier.compute_ci(X_train, confidence=0.95)
        pproba = self.classifier.predict_proba(X_test)[:, 1]

        pred_df = pandas.DataFrame(zip(y_test, pproba)).rename(
            columns={0: "label", 1: "pproba"}
        )
        pred_df.index = X_test.index

        pred_df = pred_df.sort_values("pproba", ascending=False)
        pred_df["positives"] = pred_df["label"].cumsum()
        pred_df["rank"] = pred_df["pproba"].values.argsort()[::-1]

        return pred_df, cv_res

    @staticmethod
    def plot_cv_results(
        ax: matplotlib.axes.Axes,
        cv_res: pandas.DataFrame,
        pred_df: pandas.DataFrame,
        metric: str,
    ):

        if metric == "balanced_accuracy":
            metric_fn = balanced_accuracy_score
        elif metric == "average_precision":
            metric_fn = average_precision_score
        else:
            raise ValueError(f"Invalid metric {metric}")

        computed_metrics = dict()
        computed_metrics["test_metric"] = metric_fn(
            pred_df["label"], pred_df["pproba"] > 0.5
        )
        computed_metrics["test_metrics"] = {
            name: metric_fn(group["label"], group["pproba"] > 0.5)
            for name, group in pred_df.groupby("dataset")
        }

        cv_opt = cv_res.iloc[cv_res["mean_test_score"].argmax()]

        seaborn.scatterplot(
            cv_res, x="param_C", y="mean_test_score", ax=ax, c="k", s=10
        )
        seaborn.lineplot(
            cv_res, x="param_C", y="mean_test_score", ax=ax, c="k", alpha=0.6, lw=2
        )

        ax.axvline(cv_opt["param_C"], ls="--", c="crimson")
        ax.text(
            cv_opt["param_C"],
            max(
                cv_opt["mean_test_score"],
                max(computed_metrics["test_metrics"].values()),
            )
            + 0.01,
            f"{cv_opt['param_C']:.3f}",
            horizontalalignment="center",
            verticalalignment="top",
            c="crimson",
        )

        ax.axhline(cv_opt["mean_test_score"], ls="--", c="crimson")
        ax.text(
            cv_res["param_C"].min(),
            cv_opt["mean_test_score"],
            f"{cv_opt['mean_test_score']:.3f}",
            horizontalalignment="right",
            verticalalignment="center",
            c="crimson",
        )

        if computed_metrics is not None:
            ax.scatter(
                cv_opt["param_C"],
                computed_metrics["test_metric"],
                s=50,
                label=f"Test Set ({computed_metrics['test_metric']:.2f})",
                c=COLORS["ALL"],
            )

            _ = [
                ax.scatter(
                    cv_opt["param_C"],
                    v,
                    s=50,
                    label=f"{k} ({v:.2f})",
                    marker="x",
                    c=COLORS[k],
                )
                for i, (k, v) in enumerate(computed_metrics["test_metrics"].items())
            ]

        ax.set_xscale("log")
        ax.set_xlim([cv_res["param_C"].min(), cv_res["param_C"].max()])
        ax.set_ylim(
            [
                min(
                    cv_opt["mean_test_score"],
                    min(computed_metrics["test_metrics"].values()),
                )
                - 0.01,
                max(
                    cv_opt["mean_test_score"],
                    max(computed_metrics["test_metrics"].values()),
                )
                + 0.01,
            ]
        )
        ax.set_ylabel(metric)
        ax.set_xlabel("Regularization Parameter")
        ax.legend()

        return ax

    @staticmethod
    def compute_rank_dataset(pred_df: pandas.DataFrame):
        """

        :param pred_df:
        :return:
        """

        if not all(
            [
                c in pred_df.columns
                for c in ["label", "pproba", "positives", "rank", "dataset", "run_id"]
            ]
        ):
            raise ValueError(f"Invalid pred_df.")

        dataset_rank_dict = defaultdict(dict)

        for name, group in pred_df.groupby(["dataset", "run_id"]):
            group["rank"] = group["pproba"].rank(ascending=False)
            group = group.sort_values("rank")
            group["positives"] = group["label"].cumsum()

            dataset_rank_dict[name[0]][name[1]] = group[["rank", "label"]].set_index(
                "rank"
            )

        dataset_rank = {
            k: pandas.concat(v, axis=1).fillna(0).sum(axis=1).cumsum()
            for k, v in dataset_rank_dict.items()
        }
        dataset_rank["ALL"] = (
            pandas.concat(
                [
                    pandas.concat(v, axis=1).fillna(0).sum(axis=1).rename(k)
                    for k, v in dataset_rank_dict.items()
                ],
                axis=1,
            )
            .sum(axis=1)
            .cumsum()
        )

        return dataset_rank

    @staticmethod
    def plot_positives_rank_dataset(ax, pred_df_info: pandas.DataFrame):

        rank_dataset = ImmunoRank.compute_rank_dataset(pred_df_info)

        for k, v in rank_dataset.items():
            data = v.reset_index().rename(columns={0: "positives"})
            seaborn.scatterplot(
                data,
                x="rank",
                y="positives",
                ax=ax,
                s=10,
                label="Test Set" if k == "ALL" else k,
                color=COLORS[k],
            )
            seaborn.lineplot(
                data, x="rank", y="positives", ax=ax, alpha=0.6, color=COLORS[k]
            )

        LR_muller = {
            "NCI": [(20, 27), (50, 35), (100, 38)]
        }  # , 'TESLA':[(20,12),(50, 20),(100, 26)], 'HiTIDE':[(20,12),(50, 23),(100, 26)]}
        for k, v in LR_muller.items():
            for vi in v:
                ax.scatter(vi[0], vi[1], c=COLORS[k])

        ax.set_xscale("log")

        return ax, rank_dataset

    @staticmethod
    def plot_pr_curves(ax, pred_df):
        """

        :param ax:
        :param pred_df:
        :return:
        """

        precision, recall, thresholds = precision_recall_curve(
            pred_df["label"],
            pred_df["pproba"],
        )

        average_precision = average_precision_score(pred_df["label"], pred_df["pproba"])
        display = PrecisionRecallDisplay(
            precision=precision,
            recall=recall,
            average_precision=average_precision,
            estimator_name="Test Set",
        )
        display.plot(ax, color=COLORS["ALL"])

        for name, g in pred_df.groupby("dataset"):

            precision, recall, thresholds = precision_recall_curve(
                g["label"],
                g["pproba"],
            )
            average_precision = average_precision_score(g["label"], g["pproba"])

            display = PrecisionRecallDisplay(
                precision=precision,
                recall=recall,
                average_precision=average_precision,
                estimator_name=name,
            )

            display.plot(ax, color=COLORS[name])

        return ax

    @staticmethod
    def plot_roc_curves(ax, pred_df):

        fpr, tpr, thresholds = roc_curve(pred_df["label"], pred_df["pproba"])
        roc_auc = auc(fpr, tpr)
        display = RocCurveDisplay(
            fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name="Test Set"
        )
        display.plot(ax, color=COLORS["ALL"])

        for name, g in pred_df.groupby("dataset"):
            fpr, tpr, thresholds = roc_curve(
                g["label"],
                g["pproba"],
            )

            roc_auc = auc(fpr, tpr)
            display = RocCurveDisplay(
                fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=name
            )
            display.plot(ax, color=COLORS[name])

        return ax
