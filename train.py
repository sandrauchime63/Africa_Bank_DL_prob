import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss


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
new_train_list=[]
new_val_list=[]
for col in categorical_cols:
    uniq=X_train[col].unique()
    cat_list[col]={cat : idx for idx, cat in enumerate(uniq, start=1)}
    X_train[col]=X_train[col].map(cat_list[col])
    X_val[col]=X_val[col].map(cat_list[col])


X_tensor_cat_train = torch.tensor(X_train[categorical_cols].values, dtype=torch.long)
X_tensor_cat_val   = torch.tensor(X_val[categorical_cols].values, dtype=torch.long)


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


##define model, loss function, optimizer, dataloaders##
model=nn.Sequential(
    nn.Linear(7, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
    nn.Sigmoid()
)

train_loader=DataLoader(X_train, batch_size=32, shuffle=True)
val_loader=DataLoader(X_val, batch_size=32, shuffle=False)
optimizer=torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn=nn.BCEWithLogitsLoss()



def train_epoch(model, optimizer, loss_fn, data_loader, device="cpu"):
 
    training_loss = 0.0

    model.train()

    for inputs, targets in tqdm(data_loader, desc="Training", leave=False):
        optimizer.zero_grad()

        output = model(inputs)

        loss = loss_fn(output, targets)
        
        loss.backward()
        
        optimizer.step()

        training_loss += loss.data.item() * inputs.size(0)

    return training_loss / len(data_loader.dataset)






