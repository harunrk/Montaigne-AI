import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


train = pd.read_csv("train.csv")

train["Transported"].replace(False, 0, inplace=True)
train["Transported"].replace(True, 1, inplace=True)
train['Cabin'].fillna('Z/9999/Z', inplace=True)
train[["Cabin_deck","Cabin_number","Cabin_side"]] = train["Cabin"].str.split("/", expand=True)
train["Cabin_number"] = train["Cabin_number"].astype(int)
train['Group'] = train['PassengerId'].apply(lambda x: x.split('_')[0]).astype(int) 
# This code creates a new 'Group' column using the values in the 'PassengerId' column of a dataset.
# The generated 'Group' column divides the 'PassengerId' values using the underscore "_" character, takes the first part
# (which contains the group name), and then converts this value to an integer before saving it to the 'Group' column.

train['Group_size']=train['Group'].map(lambda x: pd.concat([train['Group']]).value_counts()[x])
# The goal is to calculate how many times each group name in the 'Group' column appears in the 'train' dataset,
# and then save these counts in the 'Group_size' column.

# Put Nan's back in (we will fill these later)
train.loc[train['Cabin_deck']=='Z', 'Cabin_deck']=np.nan
train.loc[train['Cabin_number']==9999, 'Cabin_number']=np.nan
train.loc[train['Cabin_side']=='Z', 'Cabin_side']=np.nan


# Replace NaN's with outliers for now (so we can split feature)
train['Name'] = train['Name'].fillna(value = 'Unknown Unknown')

# New feature - Surname
train['Surname']=train['Name'].str.split().str[-1]

# New feature - Family size
train['Family_size']=train['Surname'].map(lambda x: pd.concat([train['Surname']]).value_counts()[x])

# Put Nan's back in (we will fill these later)
train.loc[train['Surname']=='Unknown','Surname']=np.nan
train.loc[train['Family_size']>100,'Family_size']=np.nan

train['Solo']=(train['Group_size']==1).astype(int)

exp_feats=['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
train['Expenditure']=train[exp_feats].sum(axis=1)
train['No_spending']=(train['Expenditure']==0).astype(int)

# Drop name (we don't need it anymore)
train.drop('Name', axis=1, inplace=True)


# Decks A, B, C or T came from Europa
train.loc[(train['HomePlanet'].isna()) & (train['Cabin_deck'].isin(['A', 'B', 'C', 'T'])), 'HomePlanet']='Europa'

# Deck G came from Earth
train.loc[(train['HomePlanet'].isna()) & (train['Cabin_deck']=='G'), 'HomePlanet']='Earth'


# Individuals with the same surname come from the same planet...
nanPlanet = train[train["HomePlanet"].isna()]
train = train.dropna(subset=["HomePlanet"])

for index in nanPlanet.index:
    if nanPlanet.loc[index,"Surname"]==np.nan:
        continue
    else:
        for xindex in train.index:
                if train.loc[xindex,"Surname"]==np.nan:
                     continue
                elif nanPlanet.loc[index,"Surname"] == train.loc[xindex,"Surname"]:
                    nanPlanet.loc[index,"HomePlanet"] = train.loc[xindex,"HomePlanet"]
                    break
train = pd.concat([train, nanPlanet], axis=0)


# Fill missing Destination values with mode
train.loc[(train['Destination'].isna()), 'Destination']='TRAPPIST-1e'

# Most of the individuals who journeyed to TRAPPIST-1e hailed from Earth,
# hence deducing their origin from there is logical. Nonetheless, 
# as previously remembered, none of the occupants on Deck D had an Earthly origin,
# necessitating the need to screen out this subset.
nanPlanet = train[train["HomePlanet"].isna()]
train = train.dropna(subset=["HomePlanet"])

for index in nanPlanet.index:
    if nanPlanet.loc[index, "Destination"]=="TRAPPIST-1e":
          nanPlanet.loc[index, "HomePlanet"] = "Earth"
    elif nanPlanet.loc[index, "Cabin_deck"]=="D":
         nanPlanet.loc[index, "HomePlanet"] = train[train["Cabin_deck"]=="D"]["HomePlanet"].mode().loc[0]
    elif nanPlanet.loc[index, "Destination"]=="55 Cancri e":
         nanPlanet.loc[index, "HomePlanet"] = train[train["Destination"]=="55 Cancri e"]["HomePlanet"].mode().loc[0]

train = pd.concat([train, nanPlanet], axis=0)
del nanPlanet,index

# Since the majority of the groups consist of only 1 family,
# fill in the missing surnames in the groups based on the majority within the group.
nanSurname = train[train["Surname"].isna()]
train = train.dropna(subset=["Surname"])

for index in nanSurname.index:
    group_value = nanSurname.loc[index, "Group"]
    if not np.isnan(group_value):
        matching_surnames = train[train["Group"] == group_value]["Surname"]
        if not matching_surnames.empty:
            nanSurname.loc[index, "Surname"] = matching_surnames.mode()[0]

train = pd.concat([train, nanSurname], axis=0)
del nanSurname, matching_surnames, group_value



# Set Family_size to 0 for those whose surname is unknown.
for index in train[train["Surname"].isna()].index:
     train.loc[index, "Family_size"] = 0


# Cabin_deck is the same for individuals within the same group. 
nanDeck = train[train["Cabin_deck"].isna()]
train = train.dropna(subset=["Cabin_deck"])

for index in nanDeck.index:
    if pd.isna(nanDeck.loc[index, "Cabin_deck"]):
        for xindex in train.index:
            if train.loc[xindex, "Group"] == nanDeck.loc[index, "Group"]:
                nanDeck.loc[index, "Cabin_deck"] = train.loc[xindex, "Cabin_deck"]

train = pd.concat([train, nanDeck], axis=0)
del nanDeck, index, xindex         

# Cabin_number is the same for individuals within the same group.
nanNumber = train[train["Cabin_number"].isna()]
train = train.dropna(subset=["Cabin_number"])

for index in nanNumber.index:
    if pd.isna(nanNumber.loc[index, "Cabin_number"]):
        for xindex in train.index:
            if train.loc[xindex, "Group"] == nanNumber.loc[index, "Group"]:
                nanNumber.loc[index, "Cabin_number"] = train.loc[xindex, "Cabin_number"]

train = pd.concat([train, nanNumber], axis=0)
del nanNumber, index, xindex

# Cabin_Side is the same for individuals within the same family.
nanSide = train[train["Cabin_side"].isna()]
train = train.dropna(subset=["Cabin_side"])

for index in nanSide.index:
    for xindex in train.index:
        if train.loc[xindex, "Surname"] == nanSide.loc[index, "Surname"]:
            nanSide.loc[index, "Cabin_side"] = train.loc[xindex, "Cabin_side"]

train = pd.concat([train, nanSide], axis=0)
train.loc[train["Cabin_side"].isna(), "Cabin_side"] = "Z"       # Associate the remaining Cabin_Side values with an outlier value.
del nanSide, index, xindex

"""
Notes:

    Passengers from Mars are most likely in deck F.
    Passengers from Europa are (more or less) most likely in deck C if travelling solo and deck B otherwise.
    Passengers from Earth are (more or less) most likely in deck G.
"""

nanCabinDeck = train[train["Cabin_deck"].isna()]
train = train.dropna(subset=["Cabin_deck"])

nanCabinDeck.loc[nanCabinDeck["HomePlanet"] == "Mars", "Cabin_deck"] = "F"
nanCabinDeck.loc[nanCabinDeck["HomePlanet"] == "Earth", "Cabin_deck"] = "G"

for index in nanCabinDeck.index:
    if nanCabinDeck.loc[index, "HomePlanet"] == "Europa":
        if nanCabinDeck.loc[index, "Group_size"] < 2:
            nanCabinDeck.loc[index, "Cabin_deck"] = "B"
        else:
            nanCabinDeck.loc[index, "Cabin_deck"] = "C"

train = pd.concat([train, nanCabinDeck], axis=0)
del nanCabinDeck, index

# New features - training set
train['Cabin_region1']=(train['Cabin_number']<300).astype(int)   # one-hot encoding
train['Cabin_region2']=((train['Cabin_number']>=300) & (train['Cabin_number']<600)).astype(int)
train['Cabin_region3']=((train['Cabin_number']>=600) & (train['Cabin_number']<900)).astype(int)
train['Cabin_region4']=((train['Cabin_number']>=900) & (train['Cabin_number']<1200)).astype(int)
train['Cabin_region5']=((train['Cabin_number']>=1200) & (train['Cabin_number']<1500)).astype(int)
train['Cabin_region6']=((train['Cabin_number']>=1500) & (train['Cabin_number']<1800)).astype(int)
train['Cabin_region7']=(train['Cabin_number']>=1800).astype(int)

# Adjust the Age column based on the medians of HomePlanet, No_spending, Solo, and Cabin_Deck attributes.
nanAge=train.loc[train['Age'].isna(),'Age'].index
train.loc[train['Age'].isna(),'Age']=train.groupby(['HomePlanet','No_spending','Solo','Cabin_deck'])['Age'].transform(lambda x: x.fillna(x.median()))[nanAge]

# If any expenditure was made, then CryoSleep is not active...
cryoSleep = train[train["CryoSleep"].isna()]
train = train.dropna(subset=["CryoSleep"])

for index in cryoSleep.index:
    if cryoSleep.loc[index, "No_spending"]==0:
        cryoSleep.loc[index, "CryoSleep"] = False
    elif cryoSleep.loc[index, "No_spending"]==1:
        cryoSleep.loc[index, "CryoSleep"] = True

train = pd.concat([train, cryoSleep], axis=0)
del cryoSleep, index, nanAge

# Let's identify the missing Cabin_numbers...
for deck in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
    # Features and labels
    Xdeck=train.loc[~(train['Cabin_number'].isna()) & (train['Cabin_deck']==deck),'Group']
    Ydeck=train.loc[~(train['Cabin_number'].isna()) & (train['Cabin_deck']==deck),'Cabin_number']
    XtestDeck=train.loc[(train['Cabin_number'].isna()) & (train['Cabin_deck']==deck),'Group']

    if XtestDeck.shape[0] == 0:
        continue

    # Linear regression
    linearR=LinearRegression()
    linearR.fit(Xdeck.values.reshape(-1, 1), Ydeck)
    Ypred=linearR.predict(XtestDeck.values.reshape(-1, 1))
    
    # Fill missing values with predictions
    train.loc[(train['Cabin_number'].isna()) & (train['Cabin_deck']==deck),'Cabin_number']=Ypred.astype(int)
del deck, Xdeck, XtestDeck, Ydeck, Ypred, linearR

# VIP is a highly unbalanced binary feature so we will just impute the mode.
train.loc[train['VIP'].isna(),'VIP']=False

for index in train.index:
    if pd.isna(train.loc[index, "Family_size"]):
        train.loc[index, "Family_size"] = 0

train.to_csv("Train_düzenlenmiş_veri.csv", index=False)
