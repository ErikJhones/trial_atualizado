import pandas as pd
from pathlib import Path
from tqdm import tqdm

import typer

def select_column(dataframe: pd.DataFrame):
    print("\nColunas disponíveis para seleção:")
    for i, col in enumerate(dataframe.columns):
        print(f"{i}: {col}")

    index = input("\nDigite os índices das colunas desejadas separados por vírgula sem espaço: \n")
    index = [int(i.strip()) for i in index.split(",")]
    selected_cols = [dataframe.columns[i] for i in index]
    dataframe = dataframe[selected_cols]
    print("\nColunas selecionadas:")
    print(selected_cols)

    return selected_cols

def aux_prepare_europe(dataframe_path: str,
                       start_period: str,
                       end_period: str,
                       columns: list[str] | None = None):

    dataframe = pd.read_csv(dataframe_path)
    dataframe["Date-Time"] = pd.to_datetime(dataframe["Date-Time"], utc=True, format='ISO8601')
    dataframe = dataframe.set_index("Date-Time")
    dataframe.index = pd.to_datetime(dataframe.index, utc=True)
    ticker = dataframe["#RIC"].iloc[-1]
    if columns is None:
        columns = select_column(dataframe)
        dataframe = dataframe[columns].loc["2025-01":"2025-02"].dropna()
        return dataframe, columns, ticker
    else:
        dataframe = dataframe[columns].loc["2025-01":"2025-02"].dropna()
        return dataframe, columns, ticker

def prepare_europe_trials(dataset_root_folder: str = "trials/data/EUROPA",
                          save_folder: str = "trials/data/europe",
                          tickers_name: list[str] | None = None,
                          start_period: str = "2025-01",
                          end_period: str = "2026-05",
                          size_train: float = 0.90):
    
    pasta = Path(save_folder)
    pasta.mkdir(parents=True, exist_ok=True)

    root_folder = Path(dataset_root_folder)
    csvs = list(root_folder.rglob("*.csv"))
    columns = None

    final_prepared_data = [[None], ["date"]]
    final_columns=["Unnamed: 0"]
    dataset_files = []

    for dataset_file_path in tqdm(csvs):
        if tickers_name is not None:
            there_is_ticker = False
            for ticker_name in tickers_name:
                if ticker_name.lower() in str(dataset_file_path).lower():
                    df, columns, ticker = aux_prepare_europe(str(dataset_file_path), start_period, end_period, columns)
                    there_is_ticker = True
                    # tickers_name.remove(ticker_name)
                    break 
        else:
            there_is_ticker = True
            df, columns, ticker = aux_prepare_europe(str(dataset_file_path), start_period, end_period, columns)    

        if there_is_ticker:
            final_prepared_data[0].extend(columns)
            final_prepared_data[1].extend([None]*len(columns))
            final_columns.extend([ticker]*len(columns))
            dataset_files.append(df)

    final_prepared_data.extend(pd.concat(dataset_files, axis=1).reset_index().dropna().values.tolist())
    final_prepared_data = pd.DataFrame(data=final_prepared_data,columns=final_columns)


    n = len(final_prepared_data)
    train_end = int(n * size_train)
    size_test = (1-size_train)/2
    size_val = size_test
    val_end = train_end + int(n * size_test)

    train_df = final_prepared_data.iloc[:train_end]
    val_df = pd.concat([train_df.head(2), final_prepared_data.iloc[train_end:val_end]], axis=0)
    test_df = pd.concat([train_df.head(2), final_prepared_data.iloc[val_end:]], axis=0)

    test_val_len_validator = min(len(test_df), len(val_df))
    test_df = test_df.iloc[:test_val_len_validator]
    val_df = val_df.iloc[:test_val_len_validator]

    test_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    train_df.reset_index(drop=True, inplace=True)

    train_df.to_csv(f"{save_folder}/train_rolling_1.csv", index=False)
    test_df.to_csv(f"{save_folder}/test_rolling_1.csv", index=False)
    val_df.to_csv(f"{save_folder}/valid_rolling_1.csv", index=False)
    

if __name__ == "__main__":
    typer.run(prepare_europe_trials)
