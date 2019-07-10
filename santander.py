import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import lightgbm as lgb
import statsmodels.api as sm

from sklearn.metrics import roc_auc_score


# Add value counts for each variable as new feature.
def engineer_features(df_train_in: pd.DataFrame, df_test_in: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):

    # Samples which have unique values are real the others are fake
    def separate_real_from_synthetic(in_df: pd.DataFrame, id_field: str) -> np.ndarray:
        np_test = in_df.drop([id_field], axis=1, inplace=False).to_numpy()
        unique_count = np.zeros_like(np_test)
        for feature in range(np_test.shape[1]):
            _, index_, count_ = np.unique(np_test[:, feature], return_counts=True, return_index=True)
            unique_count[index_[count_ == 1], feature] += 1
        return np.argwhere(np.sum(unique_count, axis=1) > 0)[:, 0]

    df_test_in['target'] = -1
    df_comb = pd.concat(
        objs=[df_train_in, df_test_in.loc[separate_real_from_synthetic(df_test_in, 'ID_code')]],
        axis=0,
        sort=True
    )
    num_vars = 0
    for col in df_comb.columns:
        num_vars += 1 if (col.split('_')[0] == 'var' and col.split('_')[-1] != 'FE') else 0
    for i in range(num_vars):
        col = 'var_'+str(i)
        cv = df_comb[col].value_counts()
        nm = col+'_FE'
        df_comb[nm] = df_comb[col].map(cv)
        df_test_in[nm] = df_test_in[col].map(cv)
        df_test_in[nm].fillna(0, inplace=True)
        v_dtype = 'uint8' if cv.max() <= 255 else 'uint16'
        df_comb[nm] = df_comb[nm].astype(v_dtype)
        df_test_in[nm] = df_test_in[nm].astype(v_dtype)
    return df_comb, df_test_in


def train_model(df_train: pd.DataFrame, df_test: pd.DataFrame, df_submission: pd.DataFrame, params: dict,
                num_folds: int = 5, plot: bool = False, out_dir_data: str = None, out_dir_model: str = None,
                load_previous_models: bool = False) -> pd.DataFrame:
    df_train = df_train[df_train['target'] != -1].sample(frac=1, random_state=42)
    train_len = len(df_train)
    test_len = len(df_test)
    num_vars = 0
    for col in df_train.columns:
        num_vars += 1 if (col.split('_')[0] == 'var' and col.split('_')[-1] != 'FE') else 0
    evals_result = {}

    # Save out-of-fold predictions
    all_oof = np.zeros((train_len, num_vars + 1))
    all_oof[:, 0] = np.ones(train_len)

    # Save test predictions
    all_preds = np.zeros((test_len, num_vars + 1))
    all_preds[:, 0] = np.ones(test_len)

    for j in range(num_vars):

        features = ['var_'+str(j), 'var_'+str(j)+'_FE']
        oof = np.zeros(train_len)
        preds = np.zeros(test_len)

        # Set up plot for model to output predictions/evaluations to for visualization.
        v_min, v_max, v_step = df_train['var_' + str(j)].min(), df_train['var_' + str(j)].max(), 50
        v_min_fe, v_max_fe = df_train['var_' + str(j) + '_FE'].min(), df_train['var_' + str(j) + '_FE'].max()
        v_step_fe = df_train['var_' + str(j) + '_FE'].nunique()
        w = (v_max - v_min) / v_step
        x = w * (np.arange(0, v_step) + 0.5) + v_min
        x2 = np.array([])
        for _ in range(v_step_fe):
            x2 = np.concatenate([x, x2])
        df = pd.DataFrame({'var_' + str(j): x2})
        df['var_' + str(j) + '_FE'] = v_min_fe + (v_max_fe - v_min_fe) / (v_step_fe - 1) * (df.index // v_step)
        df['pred'] = 0

        if plot:
            plt.figure(figsize=(16, 5))
            plt.subplot(1, 2, 2)
            sns.distplot(df_train[df_train['target'] == 0]['var_' + str(j)], label='t=0')
            sns.distplot(df_train[df_train['target'] == 1]['var_' + str(j)], label='t=1')
            plt.legend()
            plt.yticks([])
            plt.xlabel('Var_'+str(j))

        # Train model for each individual variable.
        cv_row_cnt = train_len // num_folds
        for k in range(num_folds):
            row_min, row_max = k * cv_row_cnt, (k + 1) * cv_row_cnt
            df_valid_cv = df_train.iloc[row_min: row_max]
            df_train_cv = df_train[~ df_train.index.isin(df_valid_cv.index)]
            var_model_fn = '{}/var_{}_{}.txt'.format(
                out_dir_model, str(j), str(k)
            ) if load_previous_models or out_dir_model else None
            if not load_previous_models:
                ds_train_cv = lgb.Dataset(df_train_cv[features], label=df_train_cv['target'])
                ds_valid_cv = lgb.Dataset(df_valid_cv[features], label=df_valid_cv['target'])
                model = lgb.train(
                    params=params,
                    train_set=ds_train_cv,
                    num_boost_round=1000,
                    valid_sets=[ds_train_cv, ds_valid_cv],
                    verbose_eval=False,
                    early_stopping_rounds=100,
                    evals_result=evals_result
                )
                x = evals_result['valid_1']['auc']
                best = x.index(max(x))
                if out_dir_model:
                    model.save_model(var_model_fn, num_iteration=best)
            else:
                model = lgb.Booster(model_file=var_model_fn)
                best = 0
            oof[row_min: row_max] = model.predict(df_valid_cv[features], num_iteration=best)
            preds += model.predict(df_test[features], num_iteration=best) / float(num_folds)
            if plot:
                # Output to df for visualization purposes.
                df['pred'] += model.predict(df[features], num_iteration=best) / float(num_folds)

        val_auc = roc_auc_score(df_train['target'], oof)
        print('VAR_'+str(j)+' with val_auc =', round(val_auc, num_folds))
        all_oof[:, j+1] = oof
        all_preds[:, j+1] = preds

        if plot:
            # Plot the predictions.
            x = df['pred'].values
            x = np.reshape(x, (v_step_fe, v_step))
            x = np.flip(x, axis=0)
            plt.subplot(1, 2, 1)
            sns.heatmap(x, cmap='RdBu_r', center=0.0)
            plt.title('VAR_'+str(j)+' Predictions with Magic', fontsize=16)
            plt.xticks(np.linspace(0, 49, 5), np.round(np.linspace(v_min, v_max, 5), 1))
            plt.xlabel('Var_'+str(j))
            s = min(v_max_fe - v_min_fe + 1, 20)
            plt.yticks(np.linspace(v_min_fe, v_max_fe, s) - 0.5, np.linspace(v_max_fe, v_min_fe, s).astype('int'))
            plt.ylabel('Count')
            plt.show()

    # Create ensemble model
    ensemble_fn = '{}/ensemble_logit.pickle'.format(out_dir_model) if load_previous_models or out_dir_model else None
    if not load_previous_models:
        logr = sm.Logit(df_train['target'], all_oof[:, :num_vars+1])
        logr = logr.fit(disp=0)
    else:
        logr = sm.load(ensemble_fn)
    logr.save(ensemble_fn, remove_data=True)

    ensemble_preds = logr.predict(all_oof[:, :num_vars+1])
    ensemble_auc = roc_auc_score(df_train['target'], ensemble_preds)
    print('##################')
    print('Combined Model Val_AUC=', round(ensemble_auc, 5))

    # Prepare output
    df_submission['target'] = logr.predict(all_preds[:, :num_vars + 1])

    df_submission_oof = df_train[['ID_code', 'target']].copy()
    df_submission_oof['predict'] = ensemble_preds
    df_submission_oof = df_submission_oof.reset_index().sort_values('index')

    if out_dir_data:
        submission_fn = '{}/submission.csv'.format(out_dir_data)
        df_submission.to_csv(submission_fn, index=False)
        print('Test predictions saved as {}'.format(submission_fn))

        submission_oof_fn = '{}/submission_oof.csv'.format(out_dir_data)
        df_submission_oof.to_csv(submission_oof_fn, index=False)
        print('OOF predictions saved as {}'.format(submission_oof_fn))

    if plot:
        # Histogram of predictions
        print('Histogram of test predictions displayed below:')
        b = plt.hist(df_submission['target'], bins=200)
        b.show()

    return df_submission
