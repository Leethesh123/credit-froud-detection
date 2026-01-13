from sklearn.preprocessing import StandardScaler

def preprocess(df):
    # Drop Time column
    df = df.drop(columns=["Time"])

    # Scale Amount
    scaler = StandardScaler()
    df["Amount"] = scaler.fit_transform(df[["Amount"]])

    # Split features and target
    X = df.drop(columns=["Class"]).values
    y = df["Class"].values

    return X, y
