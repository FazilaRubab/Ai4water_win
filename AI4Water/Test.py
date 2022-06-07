import site
site.addsitedir("AI4WaterLatest\\AI4Water")


import os
import math
from typing import Union

import numpy as np
import tensorflow as tf
from skopt.plots import plot_objective
from SeqMetrics import RegressionMetrics

from ai4water.functional import Model
from ai4water.datasets import busan_beach
from ai4water.utils.utils import jsonize, dateandtime_now
from ai4water.hyperopt import HyperOpt, Categorical, Real, Integer

data = busan_beach()

SEP = os.sep
PREFIX = f"hpo_{dateandtime_now()}"
ITER = 0

# sphinx_gallery_thumbnail_number = 2


def objective_fn(
        prefix=None,
        **suggestions)->float:
    """This function must build, train and evaluate the ML model.
    The output of this function will be minimized by optimization algorithm.
    """
    suggestions = jsonize(suggestions)
    global ITER

    # build model
    model = Model(model={"RandomForestRegressor": suggestions},
                  prefix=prefix or PREFIX,
                  train_fraction=1.0,
                  split_random=True,
                  verbosity=0,
                  )

    # train model
    model.fit(data=data)

    # evaluate model
    t, p = model.predict(data='validation', return_true=True, process_results=False)
    val_score = RegressionMetrics(t, p).r2_score()

    if not math.isfinite(val_score):
        val_score = 1.0

    # since the optimization algorithm solves minimization algorithm
    # we have to subtract r2_score from 1.0
    # if our validation metric is something like mse or rmse,
    # then we don't need to subtract it from 1.0
    val_score = 1.0 - val_score

    ITER += 1

    print(f"{ITER} {val_score}")

    return val_score


num_samples=10
space = [
# maximum number of trees that can be built
Integer(low=100, high=5000, name='iterations', num_samples=num_samples),
# Used for reducing the gradient step.
Real(low=0.0001, high=0.5, prior='log', name='learning_rate', num_samples=num_samples),
# Coefficient at the L2 regularization term of the cost function.
Real(low=0.5, high=5.0, name='l2_leaf_reg', num_samples=num_samples),
# arger the value, the smaller the model size.
Real(low=0.1, high=10, name='model_size_reg', num_samples=num_samples),
# percentage of features to use at each split selection, when features are selected over again at random.
Real(low=0.1, high=0.95, name='rsm', num_samples=num_samples),
# number of splits for numerical features
Integer(low=32, high=1032, name='border_count', num_samples=num_samples),
# The quantization mode for numerical features.  The quantization mode for numerical features.
Categorical(categories=['Median', 'Uniform', 'UniformAndQuantiles',
                        'MaxLogSum', 'MinEntropy', 'GreedyLogSum'], name='feature_border_type')
]


x0 = [200, 0.01, 1.0, 1.0, 0.2, 64, "Uniform"]

# Now instantiate the HyperOpt class and call .fit on it
# algorithm can be either ``random``, ``grid``, ``bayes``, ``tpe``, ``bayes_rf``
#

optimizer = HyperOpt(
    algorithm="bayes",
    objective_fn=objective_fn,
    param_space=space,
    x0=x0,
    num_iterations=15,
    process_results=False,
    opt_path=f"results{SEP}{PREFIX}",
    verbosity=0,
)

results = optimizer.fit()

optimizer._plot_convergence(save=False)
