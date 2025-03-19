'''
Helper functions for performing the train/val/test split on a dataset.

The main functions, get_repetitions_split_v2, and print_df_summary, were created by Iulian Emil Tampu.
'''

import warnings
import pandas as pd

from sklearn.model_selection import (
    #train_test_split,
    StratifiedKFold,
)


def get_repetition_split_v2(
    #cfg: DictConfig, df, random_seed: int = 29122009, print_summary: bool = False
    cfg: dict, df, random_seed: int = 29122009, print_summary: bool = False
):
    """
    Utility that splits the slide_ids in the df using a per case_id split (subject wise-splitting).
    It applies label stratification is requested. TODO site stratification

    INPUT
        cfg : DictConfig
            Configuration dictionary
        df : pandas Dataframe.
            Dataframe with the case_id, slide_id, label and site (if requested) information.
        random_seed : int
            Seeds the random split

    OUTPUT
        df : pandas Dataframe
            Dataframe with each of the slide_id as training, val or test for each of the specified folds.
    """

    if print_summary:
        # print summary before start splitting
        print_df_summary(df)

    # get indexes in the dataset for each of the subjects
    unique_case_ids = list(pd.unique(df.case_id))
    case_id_to_index_map = {}  # maps where each slide for each case id are.
    case_id_to_label = {}
    for c_id in unique_case_ids:
        case_id_to_index_map[c_id] = df.index[df.case_id == c_id].tolist()
        case_id_to_label[c_id] = pd.unique(df.loc[df.case_id == c_id].label).tolist()[0]

    # get a df which has two columns: case_id and label
    df_for_split = pd.DataFrame(case_id_to_label.items(), columns=["case_id", "label"])

    # ################## work on splitting
    #if cfg.class_stratification:
    if cfg["class_stratification"]:
        # get test set case_id indexes and then the train and validation case_id indexes.
        # Using these, create a column for each fold and flag the slide_id as test, train aor validation.

        split_indexes = []  # saves the split indexes for each of the folds.

        # ################## TEST SET
        # The number of splits for the first split (test and train_val) is computed based on the fraction
        # of the test set
        
        #if cfg.test_fraction != 0:
        if cfg["test_fraction"] != 0:
            #n_splits = int(1 / cfg.test_fraction)
            n_splits = int(1 / cfg["test_fraction"])
            skf = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=random_seed,
            )

            train_val_ix, test_ix = next(
                skf.split(X=df_for_split.case_id, y=df_for_split.label)
            )
        else:
            # test fraction set to 0. All the samples used for training and validation
            test_ix = []
            train_val_ix = list(range(len(df_for_split)))

        # ################## TRAIN and VALIDATION
        df_train_val_for_split = df_for_split.loc[train_val_ix].reset_index()

        # Build nbr_splits considering that the fraction cfg.validation_fraction is wrt to the entire dataset.
        #if cfg.number_of_folds == 1:
        if cfg["number_of_folds"] == 1:
            #if cfg.validation_fraction is not None:
            if cfg["validation_fraction"] is not None:
                #n_splits = int(1 / (cfg.validation_fraction / (1 - cfg.test_fraction)))
                n_splits = int(1 / (cfg["validation_fraction"] / (1 - cfg["test_fraction"])))
            else:
                n_splits = 2
        else:
            #n_splits = cfg.number_of_folds
            n_splits = cfg["number_of_folds"]

        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_seed,
        )

        # get splits for all the folds
        for cv_f, (train_ix, val_ix) in enumerate(
            skf.split(X=df_train_val_for_split.case_id, y=df_train_val_for_split.label)
        ):
            # save indexes
            split_indexes.append(
                {
                    "test": list(df_for_split.loc[test_ix, "case_id"]),
                    "train": list(df_train_val_for_split.loc[train_ix, "case_id"]),
                    "validation": list(df_train_val_for_split.loc[val_ix, "case_id"]),
                }
            )

            #if cv_f == cfg.number_of_folds - 1:
            if cv_f == cfg["number_of_folds"] - 1:
                break
    else:
        raise ValueError("Not label-stratified split is not implemented.")

    # add folds columns to the dataframe and return
    splitted_df = df.copy()
    for cv_f, s in enumerate(split_indexes):
        # create column for this fold
        splitted_df[f"fold_{cv_f+1}"] = "NA"
        for split_name, case_ids in s.items():
            # check that there are elements for each of the classes
            classes = list(pd.unique(splitted_df.label))
            classes.sort()
            per_class_nbr_subjs = df_for_split.loc[df_for_split.case_id.isin(case_ids)]
            per_class_nbr_subjs = [
                len(
                    pd.unique(
                        per_class_nbr_subjs.loc[per_class_nbr_subjs.label == c].case_id
                    )
                )
                for c in classes
            ]

            # if any of the classes has nbr_subjs == 0, raise warning
            if any([i == 0 for i in per_class_nbr_subjs]):
                warnings.warn(
                    f"Some of the classes in {split_name} set have nbr_subjs == 0 (fold=={cv_f})."
                )
                print(
                    f"{[print(f'{classes[i]:42s}: {per_class_nbr_subjs[i]}') for i in range(len(classes))]}"
                )

            # get the indexes of the slide ids for these case_id
            slide_indexes = []
            [
                slide_indexes.extend(case_id_to_index_map[case_id])
                for case_id in case_ids
            ]

            # set flag the slide ids
            splitted_df.loc[slide_indexes, f"fold_{cv_f+1}"] = split_name

        #     print(f'{cv_f}: {split_name} -> {len(case_ids) / len(df_for_split) * 100:0.2f}% (subj: {len(case_ids)}, slides: {len(slide_indexes)})')
        # print('\n')
    return splitted_df


def print_df_summary(df):
    # print totals first
    print(f"Number of slides: {len(df)}")
    print(f"Number of unique case_ids (subjects): {len(pd.unique(df.case_id))}")
    if "site_id" in df.columns:
        print(f"Number of sites: {len(pd.unique(df.site_id))}")
    print(f"Number of unique classes/labels: {len(pd.unique(df.label))}")

    # break down on a class level
    if "site_id" in df.columns:
        aus = df.groupby(["label"]).agg(
            {
                "case_id": lambda x: len(pd.unique(x)),
                "slide_id": lambda x: len(x),
                "site_id": lambda x: len(pd.unique(x)),
            }
        )
    else:
        aus = df.groupby(["label"]).agg(
            {"case_id": lambda x: len(pd.unique(x)), "slide_id": lambda x: len(x)}
        )
    print(aus)



# CHECK DATA LEAKAGE
def check_data_leakage(train_df, test_df, valid_df=None):
    # Training set data leakage sanity check
    train_count, val_count, test_count = 0, 0, 0
    for i in train_df["subjetID"].tolist():
        if i in train_df["subjetID"].tolist(): train_count += 1
        if valid_df is not None:
            if i in valid_df["subjetID"].tolist(): val_count += 1
        if i in test_df["subjetID"].tolist(): test_count += 1
    print(f"Training leakage counts:\n\tTrain:{train_count}\n\tValidation:{val_count}\n\tTest:{test_count}")

    # Validation set data leakage sanity check
    if valid_df is not None:
        train_count, val_count, test_count = 0, 0, 0
        for i in valid_df["subjetID"].tolist():
            if i in train_df["subjetID"].tolist(): train_count += 1
            if valid_df is not None:
                if i in valid_df["subjetID"].tolist(): val_count += 1
            if i in test_df["subjetID"].tolist(): test_count += 1
        print(f"Validation leakage counts:\n\tTrain:{train_count}\n\tValidation:{val_count}\n\tTest:{test_count}")

    # Test set data leakage sanity check
    train_count, val_count, test_count = 0, 0, 0
    for i in test_df["subjetID"].tolist():
        if i in train_df["subjetID"].tolist(): train_count += 1
        if valid_df is not None:
            if i in valid_df["subjetID"].tolist(): val_count += 1
        if i in test_df["subjetID"].tolist(): test_count += 1
    print(f"Test leakage counts:\n\tTrain:{train_count}\n\tValidation:{val_count}\n\tTest:{test_count}")