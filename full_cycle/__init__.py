import pandas as pd

from data_loader.webtext_data import load_data_pirates, load_data_king_arthur

if __name__ == "__main__":
    print("****")
    df1 = load_data_pirates().assign(target='chat_logs')
    df2 = load_data_king_arthur().assign(target='pirates')
    df = pd.concat([df1, df2])