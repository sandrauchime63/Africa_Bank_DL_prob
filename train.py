import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
from training import train_epoch, predict
import pickle
#######X_train → the input features
#######y_train → the target labels

df=pd.read_csv('financial_data/Train.csv')

#print(X_train.shape)
#print(df.columns)
df['target']=df['bank_account'].map({'Yes':1, 'No':0})
X=df.drop(columns=['uniqueid', 'year', 'country','bank_account','household_size', 'gender_of_respondent', 'target'])
y=df['target']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Handle categorical variables
categorical_cols=['location_type', 'cellphone_access',  'education_level', 'relationship_with_head', 'marital_status', 'job_type']
numerical_cols=['age_of_respondent']


cat_list={}
for col in categorical_cols:
    uniq=X_train[col].unique()
    cat_list[col]={cat : idx for idx, cat in enumerate(uniq, start=1)}
    X_train[col]=X_train[col].map(cat_list[col])
    X_val[col]=X_val[col].map(cat_list[col])


X_tensor_cat_train = torch.tensor(X_train[categorical_cols].values, dtype=torch.float32)
X_tensor_cat_val   = torch.tensor(X_val[categorical_cols].values, dtype=torch.float32)


### handling numerical columns###
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train[numerical_cols])
X_val_scaled = scaler.transform(X_val[numerical_cols])



X_tensor_num_train = torch.tensor(X_scaled, dtype=torch.float32)
X_tensor_num_val   = torch.tensor(X_val_scaled, dtype=torch.float32)

###### Combine categorical and numerical tensors
X_tensor_train = torch.cat([X_tensor_cat_train.float(),
                            X_tensor_num_train], dim=1)
X_tensor_val = torch.cat([X_tensor_cat_val.float(),
                          X_tensor_num_val], dim=1)


y_tensor_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
y_tensor_val   = torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_tensor_train, y_tensor_train)
val_dataset   = TensorDataset(X_tensor_val, y_tensor_val)
##define model, loss function, optimizer, dataloaders##
model=nn.Sequential(
    nn.Linear(7, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)


optimizer=torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn=nn.BCEWithLogitsLoss()


num_epochs = 10
train_losses=[]
val_losses=[]
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, optimizer, loss_fn, train_loader)
    val_loss, val_acc = predict(model, loss_fn, val_loader)  
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

with open("cat_list.pkl", "wb") as f:
    pickle.dump(cat_list, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
feature_order= categorical_cols + numerical_cols
with open("feature_order.pkl", "wb") as f:
    pickle.dump(feature_order, f)
torch.save(model.state_dict(), 'model.pt')