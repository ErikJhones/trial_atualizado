import numpy as np
import pandas as pd

from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor

from trials.networks.env import ReinforceTradingEnv
from trials.networks.feature_extractor import FEATURE_EXTRACTORS
from trials.networks.policy_network import PairSelectionActorCriticPolicy


import wandb
WANDB_ENABLED = wandb.run is not None

MODEL_PATH = "saved_model/best_model.zip"

import os
from loguru import logger


def load_data(path, file_name):
    df = pd.read_csv(
        path + file_name,
        encoding="gbk",
        header=[0, 1],
        thousands=",",
        index_col=0,
    )
    return df


def sub(file_name):
    end_point = file_name.index("_")
    dataset_type = file_name[:end_point]
    return dataset_type


def select_file_name(rolling_dataset_path, dataset_type):
    print(rolling_dataset_path)
    file_name = os.listdir(rolling_dataset_path)
    file_name = list(filter(lambda x: sub(x) == dataset_type, file_name))
    file_name.sort()
    return file_name


def build_dataset(train, valid, test, asset_number, feature_dim):
    """Build formation and trading for train, valid, and test"""
    logger.info(f"Start building dataset")
    asset_names = (
        train.columns.get_level_values(0).drop_duplicates().values.tolist()
    )
    logger.info(f"Assets: {asset_names}")
    train_size = train.shape[0]
    valid_size = valid.shape[0]
    test_size = test.shape[0]
    logger.info(
        f"Original dataset size: train {train_size} "
        f"| valid {valid_size} | test {test_size}"
    )
    assert test_size == valid_size
    trading_size = test_size
    formation_size = train_size - trading_size
    logger.info(
        f"Generate dataset size: trading {trading_size} "
        f"| formation {formation_size}"
    )

    # T x N x M
    train_value = train.values.astype(float).reshape(
        train_size, asset_number, feature_dim
    )
    valid_value = valid.values.astype(float).reshape(
        valid_size, asset_number, feature_dim
    )
    test_value = test.values.astype(float).reshape(
        test_size, asset_number, feature_dim
    )

    def log_price(data):
        data = np.transpose(data, (1, 0, 2))  # N x T x M
        return np.log(data[:, :, 1])

    def normalize(data):
        data = np.transpose(data, (1, 0, 2))  # N x T x M
        data[:, :, :2] = np.log(data[:, :, :2] / data[:, :1, :2])
        data[:, :, 2] = (
            data[:, :, 2] - np.mean(data[:, :, 2], axis=1, keepdims=True)
        ) / np.std(data[:, :, 2], axis=1, keepdims=True)
        return data

    train_formation = normalize(np.array(train_value[:formation_size]))
    train_formation_log_price = log_price(
        np.array(train_value[:formation_size])
    )
    train_trading = normalize(np.array(train_value[formation_size:]))
    train_trading_log_price = log_price(np.array(train_value[formation_size:]))
    train_formation_dates = train.index.values[:formation_size].tolist()
    train_trading_dates = train.index.values[formation_size:].tolist()

    valid_formation = normalize(np.array(train_value[trading_size:]))
    valid_formation_log_price = log_price(np.array(train_value[trading_size:]))
    valid_trading = normalize(np.array(valid_value))
    valid_trading_log_price = log_price(np.array(valid_value))
    valid_formation_dates = train.index.values[trading_size:].tolist()
    valid_trading_dates = valid.index.values.tolist()

    logger.info(
        f"{np.array(train_value[(trading_size * 2):]).shape}, "
        f"{np.array(valid_value).shape}"
    )

    test_formation_data = np.concatenate(
        [
            np.array(train_value[(trading_size * 2) :]),
            np.array(valid_value),
        ],
        axis=0,
    )
    test_formation = normalize(np.array(test_formation_data))
    test_formation_log_price = log_price(np.array(test_formation_data))
    test_trading = normalize(np.array(test_value))
    test_trading_log_price = log_price(np.array(test_value))
    test_formation_dates = (
        train.index.values[(trading_size * 2) :].tolist() + valid_trading_dates
    )
    test_trading_dates = test.index.values.tolist()

    return (
        asset_names,
        (
            train_formation,
            train_formation_log_price,
            train_trading,
            train_trading_log_price,
            train_formation_dates,
            train_trading_dates,
        ),
        (
            valid_formation,
            valid_formation_log_price,
            valid_trading,
            valid_trading_log_price,
            valid_formation_dates,
            valid_trading_dates,
        ),
        (
            test_formation,
            test_formation_log_price,
            test_trading,
            test_trading_log_price,
            test_formation_dates,
            test_trading_dates,
        ),
    )

def initialize_env(
    dataset,
    asset_names,
):
    env = ReinforceTradingEnv(
        name="inference",

        form_date=dataset[4],
        trad_date=dataset[5],

        asset_name=asset_names,

        form_asset_features=dataset[0],
        form_asset_log_prices=dataset[1],

        trad_asset_features=dataset[2],
        trad_asset_log_prices=dataset[3],

        feature_dim=3,

        serial_selection=True,
        asset_attention=False,

        trading_feature_extractor="lstm",
        trading_feature_extractor_feature_dim=3,
        trading_feature_extractor_num_layers=1,
        trading_feature_extractor_hidden_dim=64,
        trading_feature_extractor_num_heads=2,

        trading_train_steps=0,
        trading_num_process=1,
        trading_dropout=0.5,

        policy="simple_serial_selection",

        trading_learning_rate=1e-4,
        trading_log_dir="trading_log",

        trading_rl_gamma=1,
        trading_ent_coef=1e-4,

        seed=13,

        worker_model=None,
    )

    return Monitor(env)


def main():

    rolling_dataset_path = "trials/data/us500/"

    train_files = select_file_name(
        rolling_dataset_path,
        "train"
    )

    valid_files = select_file_name(
        rolling_dataset_path,
        "valid"
    )

    test_files = select_file_name(
        rolling_dataset_path,
        "test"
    )

    rolling_serial = 0

    df_train = load_data(
        rolling_dataset_path,
        train_files[rolling_serial]
    )

    df_valid = load_data(
        rolling_dataset_path,
        valid_files[rolling_serial]
    )

    df_test = load_data(
        rolling_dataset_path,
        test_files[rolling_serial]
    )

    (
        asset_names,
        train_dataset,
        valid_dataset,
        test_dataset,
    ) = build_dataset(
        df_train,
        df_valid,
        df_test,
        asset_number=30,
        feature_dim=3,
    )

    env = initialize_env(
        test_dataset,
        asset_names,
    )

    model = A2C.load(
        MODEL_PATH,
        env=env,
        custom_objects={
            "policy_class": PairSelectionActorCriticPolicy
        }
    )

    obs, _ = env.reset()

    done = False

    total_reward = 0

    while not done:

        action, state = model.predict(
            obs,
            deterministic=True
        )

        obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        total_reward += reward

        print("=" * 50)
        print("ACTION:", action)
        print("REWARD:", reward)
        print("INFO:", info)

    print("\nFINAL REWARD:", total_reward)


if __name__ == "__main__":
    main()
