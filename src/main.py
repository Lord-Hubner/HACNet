"""

Transforming table data into images and Training CNN simultaneously

Library: Hydra, Mlflow, PyTorch

"""

import time
import hydra
import warnings
import numpy as np
import pandas as pd
from datetime import timedelta
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split

from utils import utils
from manager.train_manager import Trainer
from manager.data_manager import DataManager
from manager.model_manager import ModelManager

warnings.simplefilter('ignore')


def get_cv_splits(cfg, path, X, y):
    fold_path = Path(path) / 'datasets' / cfg.dataset / 'fold_assignments.csv'
    if not fold_path.exists():
        cv = StratifiedKFold(n_splits=cfg.n_splits)
        yield from enumerate(cv.split(X, y), start=1)
        return

    folds = pd.read_csv(fold_path)
    if not {'sample_idx', 'fold'}.issubset(folds.columns):
        raise ValueError(f'{fold_path} must contain sample_idx and fold columns.')

    if folds['sample_idx'].max() >= len(X) or folds['sample_idx'].min() < 0:
        raise ValueError(f'{fold_path} contains sample_idx values outside the dataset range.')

    min_fold = int(folds['fold'].min())
    if min_fold < 1:
        raise ValueError(f'{fold_path} fold numbers must start at 1; found fold {min_fold}.')

    duplicate_assignments = folds.duplicated(subset=['sample_idx', 'fold'])
    if duplicate_assignments.any():
        duplicates = folds.loc[duplicate_assignments, ['sample_idx', 'fold']].head().to_dict('records')
        raise ValueError(f'{fold_path} contains duplicated fold assignments, examples: {duplicates}.')

    all_idx = np.arange(len(X))
    for fold in sorted(folds['fold'].unique()):
        train_idx = folds.loc[folds['fold'] == fold, 'sample_idx'].to_numpy(dtype=int)
        test_idx = np.setdiff1d(all_idx, train_idx, assume_unique=False)
        yield int(fold), (train_idx, test_idx)


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    path, writer = utils.set_mlflow_hydra(cfg)
    utils.set_seed(cfg.seed)
    utils.set_cudnn()

    data_manager = DataManager(cfg, path)
    X, y = data_manager.get_data()
    in_dim, out_dim = data_manager.get_info()

    for trial, (train_idx, test_idx) in get_cv_splits(cfg, path, X, y):
        # create mlflow run
        arti_path = utils.make_run(trial, cfg, writer)

        # split data
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        X_train, X_test = data_manager.preprocess(X_train, X_test)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train,
                                                              y_train,
                                                              test_size=cfg.val_size,
                                                              stratify=y_train,
                                                              random_state=cfg.seed)
        
        # train and test
        model_manager = ModelManager(cfg, in_dim, out_dim, )
        trainer = Trainer(model_manager, data_manager, cfg, writer, arti_path)
        inner_start = time.time()
        trainer.train(cfg, X_train, y_train, X_valid, y_valid)
        trainer.retrain(X_train, y_train, X_valid, y_valid)
        inner_elapsed = str(timedelta(seconds=(time.time() - inner_start)))
        train_acc, test_acc = trainer.test(X_train, y_train, X_test, y_test)

        # savings
        trainer.save()
        utils.save_result(train_acc,
                          test_acc,
                          writer,
                          inner_elapsed)


if __name__ == "__main__":
    main()
