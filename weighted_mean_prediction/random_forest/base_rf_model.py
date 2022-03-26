import os

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from root import ROOT_DIR
from weighted_mean_prediction.data_setup import get_encoded_split_data
from weighted_mean_prediction.model_storage import save_model, load_model
from weighted_mean_prediction.regression_performance import plot_rf_feature_importances


def train_base_model(X_train: pd.DataFrame, y_train: pd.DataFrame,
                     file_path: str) -> RandomForestRegressor:
    model = RandomForestRegressor(random_state=0)
    model.fit(X_train, y_train)
    save_model(model, file_path)
    return model


if __name__ == "__main__":
    model_dir = f"{ROOT_DIR}/weighted_mean_prediction/random_forest/models"
    model_path = os.path.join(model_dir, "base_rf_model.joblib")

    X_train, _, X_test, y_train, _, y_test = get_encoded_split_data()

    rf = load_model(model_path)
    rf = rf if rf is not None else train_base_model(X_train, y_train["weighted_mean"], model_path)

    train_acc = rf.score(X_train, y_train)
    test_acc = rf.score(X_test, y_test)
    print(train_acc, test_acc)

    for idx, importance in enumerate(rf.feature_importances_):
        print(f"{X_train.columns[idx]} : {importance}")

    plot_rf_feature_importances(rf.feature_importances_)

    """
    Mean Error: -0.0019248004486820023
    Mean Absolute Error: 0.46429385402014095
    Mean Squared Error: 0.4591522036073567
    Root Mean Squared Error: 0.677607706278018
    Mean Absolute Percentage Error: 0.1892614510198604
    Explained Variance Score: 0.830292217897689
    Mean Squared Log Error: 0.029144933658176
    Median Absolute Error: 0.32510296465748767
    R^2: 0.830290848530204
    
    [0.00786545 0.00646138 0.00831625 0.0070642  0.0065518  0.00824827
     0.00717376 0.00823789 0.00783744 0.01340055 0.01737048 0.01413202
     0.01706173 0.0164007  0.01496564 0.01276124 0.01389928 0.00991164
     0.00927144 0.00705245 0.00841061 0.00798325 0.00864004 0.00704123
     0.00912617 0.01327073 0.02230368 0.00796729 0.00712123 0.00695599
     0.00744542 0.00738225 0.00677473 0.00755621 0.00678942 0.006424
     0.52526618 0.11955797]

    """
