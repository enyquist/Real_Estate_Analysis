from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from catboost import CatBoostRegressor
from sklearn import model_selection


def catboost_cv(learning_rate, l2_leaf_reg, bagging_temperature, depth, x_train, y_train):
    """
    Function to perform cross validation of an CatBoost model
    :param learning_rate:
    :param l2_leaf_reg:
    :param bagging_temperature:
    :param depth:
    :param x_train:
    :param y_train:
    :return:
    """

    params = {
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'iterations': 999,
        'learning_rate': learning_rate,
        'random_seed': 42,
        'l2_leaf_reg': l2_leaf_reg,
        'bagging_temperature': bagging_temperature,
        'depth': int(depth),
        'early_stopping_rounds': 10,
        'od_pval': 10e-2,
        'task_type': 'GPU',
        'logging_level': 'Silent',
        'boosting_type': 'Plain'
    }

    catboost = CatBoostRegressor(**params)

    cv_result = model_selection.cross_validate(estimator=catboost, X=x_train, y=y_train)

    mse = cv_result['test_score'] ** 2

    return -1.0 * mse


def optimize_catboost(x_train, y_train, pbounds, n_iter, init_points):
    """
    Performs Bayesian Optimization of an CatBoost Model
    :param x_train:
    :param y_train:
    :param pbounds:
    :param n_iter:
    :param init_points:
    :return:
    """

    def catboost_crossval(learning_rate, l2_leaf_reg, bagging_temperature, depth):
        """
        Wrapper for catboost_cv
        :param learning_rate:
        :param l2_leaf_reg:
        :param bagging_temperature:
        :param depth:
        :return:
        """
        return catboost_cv(learning_rate, l2_leaf_reg, bagging_temperature, depth, x_train, y_train)

    optimizer = BayesianOptimization(
        f=catboost_crossval,
        pbounds=pbounds,
        bounds_transformer=SequentialDomainReductionTransformer(),
        random_state=42,
        verbose=2
    )

    optimizer.maximize(n_iter=n_iter, init_points=init_points, acq='ei')

    return optimizer
