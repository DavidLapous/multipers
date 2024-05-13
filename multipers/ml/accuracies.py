import pandas as pd
from warnings import warn
import numpy as np
from tqdm import tqdm
from os.path import exists


def accuracy_to_csv(
    X,
    Y,
    cl,
    k: float = 10,
    dataset: str = "",
    shuffle=True,
    verbose: bool = True,
    **more_columns,
):
    assert k > 0, "k is either the number of kfold > 1 or the test size > 0."
    if k > 1:
        k = int(k)
        from sklearn.model_selection import StratifiedKFold as KFold

        kfold = KFold(k, shuffle=shuffle).split(X, Y)
        accuracies = np.zeros(k)
        for i, (train_idx, test_idx) in enumerate(
            tqdm(kfold, total=k, desc="Computing kfold")
        ):
            xtrain = [X[i] for i in train_idx]
            ytrain = [Y[i] for i in train_idx]
            cl.fit(xtrain, ytrain)
            xtest = [X[i] for i in test_idx]
            ytest = [Y[i] for i in test_idx]
            accuracies[i] = cl.score(xtest, ytest)
            if verbose:
                print(f"step {i+1}, {dataset} : {accuracies[i]}", flush=True)
                try:
                    print("Best classification parameters : ", cl.best_params_)
                except:
                    None

        print(
            f"""Accuracy {dataset} : {np.mean(accuracies).round(decimals=3)}±{np.std(accuracies).round(decimals=3)}"""
        )
    elif k > 0:
        from sklearn.model_selection import train_test_split

        print("Computing accuracy, with train test split", flush=True)
        xtrain, xtest, ytrain, ytest = train_test_split(
            X, Y, shuffle=shuffle, test_size=k
        )
        print("Fitting...", end="", flush=True)
        cl.fit(xtrain, ytrain)
        print("Computing score...", end="", flush=True)
        accuracies = cl.score(xtest, ytest)
        try:
            print("Best classification parameters : ", cl.best_params_)
        except:
            None
        print("Done.")
        if verbose:
            print(f"Accuracy {dataset} : {accuracies} ")
    file_path: str = f"result_{dataset}.csv".replace("/", "_").replace(".off", "")
    columns: list[str] = ["dataset", "cv", "mean", "std"]
    if exists(file_path):
        df: pd.DataFrame = pd.read_csv(file_path)
    else:
        df: pd.DataFrame = pd.DataFrame(columns=columns)
    more_names = []
    more_values = []
    for key, value in more_columns.items():
        if key not in columns:
            more_names.append(key)
            more_values.append(value)
        else:
            warn(f"Duplicate key {key} ! with value {value}")
    new_line: pd.DataFrame = pd.DataFrame(
        [
            [
                dataset,
                k,
                np.mean(accuracies).round(decimals=3),
                np.std(accuracies).round(decimals=3),
            ]
            + more_values
        ],
        columns=columns + more_names,
    )
    print(new_line)
    df = pd.concat([df, new_line])
    df.to_csv(file_path, index=False)
