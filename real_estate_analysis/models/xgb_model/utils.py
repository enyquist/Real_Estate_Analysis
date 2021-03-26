import xgboost as xgb
from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer


def xgb_cv(max_depth, min_child_weight, eta, subsample, colsample_bytree, gamma, lambda_, alpha, dtrain):
    """
    Function to perform cross validation of an XGBoost model
    :param max_depth: Maximum depth of a tree
    :param min_child_weight: Minimum sum of instance weight (hessian) needed in a child
    :param eta: Step size shrinkage used in update to prevents overfitting
    :param subsample: Subsample ratio of the training instances
    :param colsample_bytree: subsample ratio of columns when constructing each tree
    :param gamma: Minimum loss reduction required to make a further partition on a leaf node of the tree
    :param lambda_: L2 regularization term on weights
    :param alpha: L1 regularization term on weights
    :param dtrain: Training matrix of type xgb.DMatrix
    :return: Mean Squared Error of XGBoost CV
    """
    params = {
        'max_depth': int(max_depth),
        'eta': eta,
        'verbosity': 2,
        'objective': 'reg:squarederror',
        'booster': 'gbtree',
        'tree_method': 'gpu_hist',
        'gamma': gamma,
        'min_child_weight': min_child_weight,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'reg_alpha': alpha,
        'reg_lambda': lambda_,
        'random_state': 42
    }

    cv_result = xgb.cv(
        params=params,
        dtrain=dtrain,
        num_boost_round=999,
        seed=42,
        nfold=5,
        metrics=['rmse', 'mae'],
        early_stopping_rounds=10
    )

    mse = cv_result['test-rmse-mean'].iloc[-1] ** 2

    return -1 * mse


def optimize_xgb(dtrain, pbounds, n_iter, init_points):
    """
    Performs Bayesian Optimization of an XGBoost Model
    :param dtrain: Training matrix of type xgb.DMatrix
    :param pbounds: Parameter Bounds for Optimization
    :param n_iter: Number of exploitation iterations
    :param init_points: Number of initialization points, to explore the Gaussian Process
    :return: Bayesian Optimized Object
    """

    def xgb_crossval(max_depth, min_child_weight, eta, subsample, colsample_bytree, gamma, lambda_, alpha):
        """
        Wrapper for xgb_cv
        :param max_depth: Maximum depth of a tree
        :param min_child_weight: Minimum sum of instance weight (hessian) needed in a child
        :param eta: Step size shrinkage used in update to prevents overfitting
        :param subsample: Subsample ratio of the training instances
        :param colsample_bytree: subsample ratio of columns when constructing each tree
        :param gamma: Minimum loss reduction required to make a further partition on a leaf node of the tree
        :param lambda_: L2 regularization term on weights
        :param alpha: L1 regularization term on weights
        :return: Mean Squared Error of XGBoost CV
        """
        return xgb_cv(max_depth, min_child_weight, eta, subsample, colsample_bytree, gamma, lambda_, alpha, dtrain)

    optimizer = BayesianOptimization(
        f=xgb_crossval,
        pbounds=pbounds,
        bounds_transformer=SequentialDomainReductionTransformer(),
        random_state=42,
        verbose=2
    )

    optimizer.maximize(n_iter=n_iter, init_points=init_points, acq='ei')

    return optimizer