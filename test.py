import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from training import predict
import torch.nn as nn


df=pd.read_csv('financial_data/Test.csv')

X=df.drop(columns=['uniqueid', 'year', 'country','household_size', 'gender_of_respondent'])


# Handle categorical variables
categorical_cols=['location_type', 'cellphone_access',  'education_level', 'relationship_with_head', 'marital_status', 'job_type']
numerical_cols=['age_of_respondent']

cat_list={}
for col in categorical_cols:
    uniq=X[col].unique()
    cat_list[col]={cat : idx for idx, cat in enumerate(uniq, start=1)}
    X[col]=X[col].map(cat_list[col])
    



X_tensor_cat= torch.tensor(X[categorical_cols].values, dtype=torch.float32)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[numerical_cols])

X_tensor_num=torch.tensor(X_scaled, dtype=torch.float32)
X_tensor_test = torch.cat([X_tensor_cat.float(),
                            X_tensor_num], dim=1)
print(X_tensor_test.shape)

test_dataset = TensorDataset(X_tensor_test)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False   # VERY IMPORTANT for test data
)

batch = next(iter(test_loader))
print(batch)

model=nn.Sequential(
    nn.Linear(7, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
)
loss_fn=nn.BCEWithLogitsLoss()

model.load_state_dict(torch.load("model.pt"))
model.eval()


all_preds = []

model.eval()
with torch.no_grad():
    for (inputs,) in test_loader:
        outputs = model(inputs)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())

df["prediction"] = all_preds
df.to_csv("test_predictions.csv", index=False)