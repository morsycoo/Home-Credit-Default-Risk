from lightgbm import LGBMClassifier


def build_lightgbm():

    model = LGBMClassifier(
        objective="binary",
        n_estimators=645,
        learning_rate=0.027875757009700217,
        max_depth=7,
        num_leaves=46,
        min_child_samples=92,
        subsample=0.7601914784130286,
        colsample_bytree=0.6343484712463058,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    return model