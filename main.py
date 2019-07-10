from pandas import read_csv
from santander import engineer_features, train_model


def main():
    lgb_param = {
        'learning_rate': 0.06,
        'num_leaves': 3,
        'metric': 'auc',
        'boost_from_average': 'false',
        'feature_fraction': 1.0,
        'max_depth': -1,
        'objective': 'binary',
        'verbosity': -10,
        'device_type': 'cpu',
        'num_threads': 4
    }
    train, test, submission = (
        read_csv('data/train.csv'),
        read_csv('data/test.csv'),
        read_csv('data/sample_submission.csv')
    )
    train, test = engineer_features(train, test)
    train_model(
        df_train=train,
        df_test=test,
        df_submission=submission,
        params=lgb_param,
        num_folds=5,
        plot=False,
        out_dir_data='data',
        out_dir_model='models',
        load_previous_models=True
    )
    return


if __name__ is '__main__':
    main()
