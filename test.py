import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from training import predict
import torch.nn as nn
import pickle


model=nn.Sequential(
    nn.Linear(7, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
)
model.load_state_dict(
    torch.load("model.pt")
)
model.eval()
with open("cat_list.pkl", "rb") as f:
    cat_maps = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("feature_order.pkl", "rb") as f:
    feature_order = pickle.load(f)

df=pd.read_csv('financial_data/Test.csv')

X=df.drop(columns=['uniqueid', 'year', 'country','household_size', 'gender_of_respondent'])


numerical_cols=['age_of_respondent']

def encode_categories(df, cat_maps):
    df = df.copy()
    for col, mapping in cat_maps.items():
        df[col] = df[col].map(mapping)
        df[col] = df[col].fillna(0)  
    return df

X = encode_categories(X, cat_maps)

scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

X = X[feature_order]

X_test_tensor = torch.tensor(
    X.values,
    dtype=torch.float32
)

with torch.no_grad():
    logits = model(X_test_tensor)
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).long()


df["probability"] = probs.cpu().numpy()
df["prediction"] = preds.cpu().numpy()

df.to_csv("test_predictions.csv", index=False)
