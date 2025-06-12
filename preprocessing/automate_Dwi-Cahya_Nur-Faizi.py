import pandas as pd


def main():
    # load the dataset
    dataset = pd.read_csv("../personality_raw.csv")

    # handle missing values
    dataset.dropna(subset="Personality", inplace=True)

    na_cols = dataset.columns[dataset.isna().any()]
    personality_groups = dataset.groupby("Personality")

    for col in na_cols:
        if dataset[col].dtype == "O":
            dataset[col] = dataset[col].fillna(
                dataset.groupby("Personality")[col].transform(
                    lambda x: x.mode().iloc[0]
                )
            )
        else:
            dataset[col] = dataset[col].fillna(
                dataset.groupby("Personality")[col].transform("median")
            )

    # handle duplicate rows
    dataset.drop_duplicates(inplace=True)

    # encoding categorical variables
    dataset.loc[:, "Stage_fear"] = dataset["Stage_fear"].map({"No": 0, "Yes": 1})
    dataset.loc[:, "Drained_after_socializing"] = dataset[
        "Drained_after_socializing"
    ].map({"No": 0, "Yes": 1})
    dataset.loc[:, "Personality"] = dataset["Personality"].map(
        {"Extrovert": 0, "Introvert": 1}
    )

    # binning the dataset
    dataset["has_lower_time_spent_alone"] = (dataset["Time_spent_Alone"] < 4).astype(
        int
    )
    dataset["has_lower_social_event_attendance"] = (
        dataset["Social_event_attendance"] < 4
    ).astype(int)
    dataset["has_lower_going_outside_count"] = (dataset["Going_outside"] < 3).astype(
        int
    )
    dataset["has_lower_friends_circle"] = (dataset["Friends_circle_size"] < 6).astype(
        int
    )
    dataset["has_lower_post_frequency"] = (dataset["Post_frequency"] < 3).astype(int)

    # renaming and dropping original columns
    dataset.rename(
        columns={
            "Stage_fear": "has_stage_fear",
            "Drained_after_socializing": "is_drained_after_socializing",
        },
        inplace=True,
    )

    dataset.drop(
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
    dataset.to_csv("personality_preprocessing.csv", index=False)


if __name__ == "__main__":
    main()
