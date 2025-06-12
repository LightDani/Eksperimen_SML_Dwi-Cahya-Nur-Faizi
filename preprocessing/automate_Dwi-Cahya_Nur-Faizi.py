import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    # load the dataset
    dataset = pd.read_csv("../personality_raw.csv")

    # SPLIT DATASET INTO TRAINING AND TESTING SETS TO AVOID DATA LEAKAGE
    X = dataset.iloc[:, :-1]
    y = dataset["Personality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_dataset = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
    test_dataset = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)

    # handle missing values
    train_dataset.dropna(subset=["Personality"], inplace=True)
    test_dataset.dropna(subset=["Personality"], inplace=True)

    na_cols = train_dataset.columns[train_dataset.isna().any()]
    personality_groups = train_dataset.groupby("Personality")

    for col in na_cols:
        if train_dataset[col].dtype == "O":
            fill_values = personality_groups[col].agg(
                lambda x: x.mode().iloc[0] if not x.mode().empty else None
            )

        else:
            fill_values = personality_groups[col].median()

        train_dataset[col] = train_dataset[col].fillna(
            train_dataset["Personality"].map(fill_values)
        )

        test_dataset[col] = test_dataset[col].fillna(
            test_dataset["Personality"].map(fill_values)
        )

    # handle duplicate rows
    train_dataset.drop_duplicates(inplace=True)
    test_dataset.drop_duplicates(inplace=True)

    # encoding categorical variables
    train_dataset.loc[:, "Stage_fear"] = train_dataset["Stage_fear"].map(
        {"No": 0, "Yes": 1}
    )
    test_dataset.loc[:, "Stage_fear"] = test_dataset["Stage_fear"].map(
        {"No": 0, "Yes": 1}
    )

    train_dataset.loc[:, "Drained_after_socializing"] = train_dataset[
        "Drained_after_socializing"
    ].map({"No": 0, "Yes": 1})
    test_dataset.loc[:, "Drained_after_socializing"] = test_dataset[
        "Drained_after_socializing"
    ].map({"No": 0, "Yes": 1})

    train_dataset.loc[:, "Personality"] = train_dataset["Personality"].map(
        {"Extrovert": 0, "Introvert": 1}
    )
    test_dataset.loc[:, "Personality"] = test_dataset["Personality"].map(
        {"Extrovert": 0, "Introvert": 1}
    )

    # binning the dataset
    train_dataset["has_lower_time_spent_alone"] = (
        train_dataset["Time_spent_Alone"] < 4
    ).astype(int)
    test_dataset["has_lower_time_spent_alone"] = (
        test_dataset["Time_spent_Alone"] < 4
    ).astype(int)

    train_dataset["has_lower_social_event_attendance"] = (
        train_dataset["Social_event_attendance"] < 4
    ).astype(int)
    test_dataset["has_lower_social_event_attendance"] = (
        test_dataset["Social_event_attendance"] < 4
    ).astype(int)

    train_dataset["has_lower_going_outside_count"] = (
        train_dataset["Going_outside"] < 3
    ).astype(int)
    test_dataset["has_lower_going_outside_count"] = (
        test_dataset["Going_outside"] < 3
    ).astype(int)

    train_dataset["has_lower_friends_circle"] = (
        train_dataset["Friends_circle_size"] < 6
    ).astype(int)
    test_dataset["has_lower_friends_circle"] = (
        test_dataset["Friends_circle_size"] < 6
    ).astype(int)

    train_dataset["has_lower_post_frequency"] = (
        train_dataset["Post_frequency"] < 3
    ).astype(int)
    test_dataset["has_lower_post_frequency"] = (
        test_dataset["Post_frequency"] < 3
    ).astype(int)

    # renaming and dropping original columns
    train_dataset.rename(
        columns={
            "Stage_fear": "has_stage_fear",
            "Drained_after_socializing": "is_drained_after_socializing",
        },
        inplace=True,
    )
    test_dataset.rename(
        columns={
            "Stage_fear": "has_stage_fear",
            "Drained_after_socializing": "is_drained_after_socializing",
        },
        inplace=True,
    )

    train_dataset.drop(
        columns=[
            "Time_spent_Alone",
            "Social_event_attendance",
            "Going_outside",
            "Friends_circle_size",
            "Post_frequency",
        ],
        inplace=True,
    )
    test_dataset.drop(
        columns=[
            "Time_spent_Alone",
            "Social_event_attendance",
            "Going_outside",
            "Friends_circle_size",
            "Post_frequency",
        ],
        inplace=True,
    )

    # save the processed dataset
    dataset.to_csv("personality_preprocessing/train.csv", index=False)
    dataset.to_csv("personality_preprocessing/test.csv", index=False)


if __name__ == "__main__":
    main()
