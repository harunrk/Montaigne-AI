import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

test = pd.read_csv("test.csv")

test['Cabin'].fillna('Z/9999/Z', inplace=True)
test[["Cabin_deck","Cabin_number","Cabin_side"]] = test["Cabin"].str.split("/", expand=True)
test["Cabin_number"] = test["Cabin_number"].astype(int)
test['Group'] = test['PassengerId'].apply(lambda x: x.split('_')[0]).astype(int) 
# This code creates a new 'Group' column using the values in the 'PassengerId' column of a dataset.
# The generated 'Group' column divides the 'PassengerId' values using the underscore "_" character, takes the first part
# (which contains the group name), and then converts this value to an integer before saving it to the 'Group' column.

test['Group_size']=test['Group'].map(lambda x: pd.concat([test['Group']]).value_counts()[x])
# The goal is to calculate how many times each group name in the 'Group' column appears in the 'test' dataset,
# and then save these counts in the 'Group_size' column.

# Put Nan's back in (we will fill these later)
test.loc[test['Cabin_deck']=='Z', 'Cabin_deck']=np.nan
test.loc[test['Cabin_number']==9999, 'Cabin_number']=np.nan
test.loc[test['Cabin_side']=='Z', 'Cabin_side']=np.nan


# Replace NaN's with outliers for now (so we can split feature)
test['Name'] = test['Name'].fillna(value = 'Unknown Unknown')

# New feature - Surname
test['Surname']=test['Name'].str.split().str[-1]

# New feature - Family size
test['Family_size']=test['Surname'].map(lambda x: pd.concat([test['Surname']]).value_counts()[x])

# Put Nan's back in (we will fill these later)
test.loc[test['Surname']=='Unknown','Surname']=np.nan
test.loc[test['Family_size']>100,'Family_size']=np.nan

test['Solo']=(test['Group_size']==1).astype(int)

exp_feats=['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
test['Expenditure']=test[exp_feats].sum(axis=1)
test['No_spending']=(test['Expenditure']==0).astype(int)

# Drop name (we don't need it anymore)
test.drop('Name', axis=1, inplace=True)


# Decks A, B, C or T came from Europa
test.loc[(test['HomePlanet'].isna()) & (test['Cabin_deck'].isin(['A', 'B', 'C', 'T'])), 'HomePlanet']='Europa'

# Deck G came from Earth
test.loc[(test['HomePlanet'].isna()) & (test['Cabin_deck']=='G'), 'HomePlanet']='Earth'


# Individuals with the same surname come from the same planet...
nanPlanet = test[test["HomePlanet"].isna()]
test = test.dropna(subset=["HomePlanet"])

for index in nanPlanet.index:
    if nanPlanet.loc[index,"Surname"]==np.nan:
        continue
    else:
        for xindex in test.index:
                if test.loc[xindex,"Surname"]==np.nan:
                     continue
                elif nanPlanet.loc[index,"Surname"] == test.loc[xindex,"Surname"]:
                    nanPlanet.loc[index,"HomePlanet"] = test.loc[xindex,"HomePlanet"]
                    break
test = pd.concat([test, nanPlanet], axis=0)


# Fill missing Destination values with mode
test.loc[(test['Destination'].isna()), 'Destination']='TRAPPIST-1e'

# Most of the individuals who journeyed to TRAPPIST-1e hailed from Earth,
# hence deducing their origin from there is logical. Nonetheless, 
# as previously remembered, none of the occupants on Deck D had an Earthly origin,
# necessitating the need to screen out this subset.
nanPlanet = test[test["HomePlanet"].isna()]
test = test.dropna(subset=["HomePlanet"])

for index in nanPlanet.index:
    if nanPlanet.loc[index, "Destination"]=="TRAPPIST-1e":
          nanPlanet.loc[index, "HomePlanet"] = "Earth"
    elif nanPlanet.loc[index, "Cabin_deck"]=="D":
         nanPlanet.loc[index, "HomePlanet"] = test[test["Cabin_deck"]=="D"]["HomePlanet"].mode().loc[0]
    elif nanPlanet.loc[index, "Destination"]=="55 Cancri e":
         nanPlanet.loc[index, "HomePlanet"] = test[test["Destination"]=="55 Cancri e"]["HomePlanet"].mode().loc[0]

test = pd.concat([test, nanPlanet], axis=0)
del nanPlanet,index

# Since the majority of the groups consist of only 1 family,
# fill in the missing surnames in the groups based on the majority within the group.
nanSurname = test[test["Surname"].isna()]
test = test.dropna(subset=["Surname"])

for index in nanSurname.index:
    group_value = nanSurname.loc[index, "Group"]
    if not np.isnan(group_value):
        matching_surnames = test[test["Group"] == group_value]["Surname"]
        if not matching_surnames.empty:
            nanSurname.loc[index, "Surname"] = matching_surnames.mode()[0]

test = pd.concat([test, nanSurname], axis=0)
del nanSurname, matching_surnames, group_value



# Set Family_size to 0 for those whose surname is unknown.
for index in test[test["Surname"].isna()].index:
     test.loc[index, "Family_size"] = 0


# Cabin_deck is the same for individuals within the same group. 
nanDeck = test[test["Cabin_deck"].isna()]
test = test.dropna(subset=["Cabin_deck"])

for index in nanDeck.index:
    if pd.isna(nanDeck.loc[index, "Cabin_deck"]):
        for xindex in test.index:
            if test.loc[xindex, "Group"] == nanDeck.loc[index, "Group"]:
                nanDeck.loc[index, "Cabin_deck"] = test.loc[xindex, "Cabin_deck"]

test = pd.concat([test, nanDeck], axis=0)
del nanDeck, index, xindex         

# Cabin_number is the same for individuals within the same group.
nanNumber = test[test["Cabin_number"].isna()]
test = test.dropna(subset=["Cabin_number"])

for index in nanNumber.index:
    if pd.isna(nanNumber.loc[index, "Cabin_number"]):
        for xindex in test.index:
            if test.loc[xindex, "Group"] == nanNumber.loc[index, "Group"]:
                nanNumber.loc[index, "Cabin_number"] = test.loc[xindex, "Cabin_number"]

test = pd.concat([test, nanNumber], axis=0)
del nanNumber, index, xindex

# Cabin_Side is the same for individuals within the same family.
nanSide = test[test["Cabin_side"].isna()]
test = test.dropna(subset=["Cabin_side"])

for index in nanSide.index:
    for xindex in test.index:
        if test.loc[xindex, "Surname"] == nanSide.loc[index, "Surname"]:
            nanSide.loc[index, "Cabin_side"] = test.loc[xindex, "Cabin_side"]

test = pd.concat([test, nanSide], axis=0)
test.loc[test["Cabin_side"].isna(), "Cabin_side"] = "Z"       # Associate the remaining Cabin_Side values with an outlier value.
del nanSide, index, xindex

"""
Notes:

    Passengers from Mars are most likely in deck F.
    Passengers from Europa are (more or less) most likely in deck C if travelling solo and deck B otherwise.
    Passengers from Earth are (more or less) most likely in deck G.
"""

nanCabinDeck = test[test["Cabin_deck"].isna()]
test = test.dropna(subset=["Cabin_deck"])

nanCabinDeck.loc[nanCabinDeck["HomePlanet"] == "Mars", "Cabin_deck"] = "F"
nanCabinDeck.loc[nanCabinDeck["HomePlanet"] == "Earth", "Cabin_deck"] = "G"

for index in nanCabinDeck.index:
    if nanCabinDeck.loc[index, "HomePlanet"] == "Europa":
        if nanCabinDeck.loc[index, "Group_size"] < 2:
            nanCabinDeck.loc[index, "Cabin_deck"] = "B"
        else:
            nanCabinDeck.loc[index, "Cabin_deck"] = "C"

test = pd.concat([test, nanCabinDeck], axis=0)
del nanCabinDeck, index

# New features - training set
test['Cabin_region1']=(test['Cabin_number']<300).astype(int)   # one-hot encoding
test['Cabin_region2']=((test['Cabin_number']>=300) & (test['Cabin_number']<600)).astype(int)
test['Cabin_region3']=((test['Cabin_number']>=600) & (test['Cabin_number']<900)).astype(int)
test['Cabin_region4']=((test['Cabin_number']>=900) & (test['Cabin_number']<1200)).astype(int)
test['Cabin_region5']=((test['Cabin_number']>=1200) & (test['Cabin_number']<1500)).astype(int)
test['Cabin_region6']=((test['Cabin_number']>=1500) & (test['Cabin_number']<1800)).astype(int)
test['Cabin_region7']=(test['Cabin_number']>=1800).astype(int)

# Adjust the Age column based on the medians of HomePlanet, No_spending, Solo, and Cabin_Deck attributes.
nanAge=test.loc[test['Age'].isna(),'Age'].index
test.loc[test['Age'].isna(),'Age']=test.groupby(['HomePlanet','No_spending','Solo','Cabin_deck'])['Age'].transform(lambda x: x.fillna(x.median()))[nanAge]

# If any expenditure was made, then CryoSleep is not active...
cryoSleep = test[test["CryoSleep"].isna()]
test = test.dropna(subset=["CryoSleep"])

for index in cryoSleep.index:
    if cryoSleep.loc[index, "No_spending"]==0:
        cryoSleep.loc[index, "CryoSleep"] = False
    elif cryoSleep.loc[index, "No_spending"]==1:
        cryoSleep.loc[index, "CryoSleep"] = True

test = pd.concat([test, cryoSleep], axis=0)
del cryoSleep, index, nanAge

# Let's identify the missing Cabin_numbers...
for deck in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
    # Features and labels
    Xdeck=test.loc[~(test['Cabin_number'].isna()) & (test['Cabin_deck']==deck),'Group']
    Ydeck=test.loc[~(test['Cabin_number'].isna()) & (test['Cabin_deck']==deck),'Cabin_number']
    XtestDeck=test.loc[(test['Cabin_number'].isna()) & (test['Cabin_deck']==deck),'Group']

    if XtestDeck.shape[0] == 0:
        continue

    # Linear regression
    linearR=LinearRegression()
    linearR.fit(Xdeck.values.reshape(-1, 1), Ydeck)
    Ypred=linearR.predict(XtestDeck.values.reshape(-1, 1))
    
    # Fill missing values with predictions
    test.loc[(test['Cabin_number'].isna()) & (test['Cabin_deck']==deck),'Cabin_number']=Ypred.astype(int)
del deck, Xdeck, XtestDeck, Ydeck, Ypred, linearR

# VIP is a highly unbalanced binary feature so we will just impute the mode.
test.loc[test['VIP'].isna(),'VIP']=False
for index in test.index:
    if test.loc[index, "Family_size"]==np.nan:
        test.loc[index, "Family_size"] = 0

test.to_csv("Test_düzenlenmiş_veri.csv", index=False)