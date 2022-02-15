
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
pd.options.mode.chained_assignment = None  # default='warn'

df1 = pd.read_csv(
    r"C:\Users\aircu\Downloads\Overwatch Data\Overwatch Data\01-23-2021\Raw Data\1) Oasis Raw.csv")
df2 = pd.read_csv(
    r"C:\Users\aircu\Downloads\Overwatch Data\Overwatch Data\01-23-2021\Fights Data\1) Oasis Fights.csv")


df3 = pd.read_csv(
    r"C:\Users\aircu\Downloads\Overwatch Data\Overwatch Data\01-23-2021\Raw Data\5) Ilios Raw.csv")
df4 = pd.read_csv(
    r"C:\Users\aircu\Downloads\Overwatch Data\Overwatch Data\01-23-2021\Fights Data\5) Ilios Fights.csv")

df5 = pd.read_csv(
    r"C:\Users\aircu\Downloads\Overwatch Data\Overwatch Data\01-25-2021\Raw Data\1) Nepal Raw.csv")
df6 = pd.read_csv(
    r"C:\Users\aircu\Downloads\Overwatch Data\Overwatch Data\01-25-2021\Fights Data\1) Nepal Fights.csv")

df7 = pd.read_csv(
    r"C:\Users\aircu\Downloads\Overwatch Data\Overwatch Data\02-09-2021\Raw Data\1) Lijiang Tower Raw.csv")
df8 = pd.read_csv(
    r"C:\Users\aircu\Downloads\Overwatch Data\Overwatch Data\02-09-2021\Fights Data\1) Lijiang Tower Fights.csv")


data_1 = [df1, df3, df5, df7]
data_2 = [df2, df4, df6, df8]
df1 = pd.concat(data_1)
df2 = pd.concat(data_2)



# "Healing Received stat recorded only 0s
data = df1.drop(columns=["Healing Received", "Ultimate Charge", "Player Closest to Reticle", "Position", "Ultimates Earned", "Ultimates Used", "Cooldown 1", "Cooldown 2"])

# This is to decrease the # of variables
# "Player" stat must be included to use "Player Clostest to Reticle"


# will contain a list of dataframes
start_dfs = []
end_dfs = []


for index, row in df2.iterrows():

    # this creates 2 dataframes, one with the just 12 rows of start-of-fight data and one with just 12 rows of end-of-fight data
    # they are then added to a list of dataframes

    start_df = data[((data["Map"] == row["Map"]) & (
        data["Section"] == row["Section"]) & (data["Timestamp"] == row["Start Timestamp"]))]
    start_df.loc[:, "Fight Win"] = row["Winner"]
    start_dfs.append(start_df)

    end_df = data[((data["Map"] == row["Map"]) & (data["Section"] == row["Section"]) & (
        (data["Timestamp"] == row["End Timestamp"])))]
    end_df.loc[:, "Fight Win"] = row["Winner"]
    end_dfs.append(end_df)


# creates single dataframes
df_start = pd.concat(start_dfs)
df_end = pd.concat(end_dfs)


# set start of index back to 0
df_start = df_start.reset_index()
df_end = df_end.reset_index()


df_final = df_end

for i in df_end.index:
    if df_end['Fight Win'][i] == df_end["Team"][i]:
        df_final.loc[i, ('Fight Win')] = True
    else:
        df_final.loc[i, ('Fight Win')] = False

    if df_end['Team'][i] == "Team 1":
        df_final.loc[i, ('Team')] = 1
    else:
        df_final.loc[i, ('Team')] = 2

    # subtracts the values from the start and end dataframes
    df_final.loc[i, ("Hero Damage Dealt")] = df_end["Hero Damage Dealt"][i] - \
        df_start["Hero Damage Dealt"][i]
    df_final.loc[i, ("Barrier Damage Dealt")] = df_end["Barrier Damage Dealt"][i] - \
        df_start["Barrier Damage Dealt"][i]
    df_final.loc[i, ("Damage Blocked")] = df_end["Damage Blocked"][i] - \
        df_start["Damage Blocked"][i]
    df_final.loc[i, ("Damage Taken")] = df_end["Damage Taken"][i] - \
        df_start["Damage Taken"][i]
    df_final.loc[i, ("Deaths")] = df_end["Deaths"][i]-df_start["Deaths"][i]
    df_final.loc[i, ("Eliminations")] = df_end["Eliminations"][i] - \
        df_start["Eliminations"][i]
    df_final.loc[i, ("Final Blows")] = df_end["Final Blows"][i] - \
        df_start["Final Blows"][i]
    df_final.loc[i, ("Environmental Deaths")] = df_end["Environmental Deaths"][i] - \
        df_start["Environmental Deaths"][i]
    df_final.loc[i, ("Environmental Kills")] = df_end["Environmental Kills"][i] - \
        df_start["Environmental Kills"][i]
    df_final.loc[i, ("Healing Dealt")] = df_end["Healing Dealt"][i] - \
        df_start["Healing Dealt"][i]
    df_final.loc[i, ("Objective Kills")] = df_end["Objective Kills"][i] - \
        df_start["Objective Kills"][i]
    df_final.loc[i, ("Solo Kills")] = df_end["Solo Kills"][i] - \
        df_start["Solo Kills"][i]


# Separating by role

damage_list = ["Ashe", "Bastion", "Cassidy", "Doomfist", "Echo", "Genji", "Hanzo", "Junkrat", "Mei",
               "Pharah", "Reaper", "Soldier: 76", "Sombra", "Symmetra", "Torbjorn", "Tracer", "Widowmaker"]
df_damage = df_final[df_final["Hero"].isin(damage_list)]

tank_list = ["D.Va", "Orisa", "Reinhardt", "Roadhog",
             "Sigma", "Winston", "WreckingBall", "Zarya"]
df_tank = df_final[df_final["Hero"].isin(tank_list)]

support_list = ["Ana", "Baptiste", "Brigitte",
                "Lucio", "Mercy", "Moira", "Zenyatta"]
df_support = df_final[df_final["Hero"].isin(support_list)]


#######################################################
# start of analysis


# Plotting everything together

# fig, ax0 = plt.subplots()
# ax0.scatter(x=df_final["Hero Damage Dealt"],
#             y=df_final["Damage Taken"],
#             c=df_final["Fight Win"],
#             cmap=mcol.ListedColormap(["red", "blue"]))
# ax0.set_title("All Players- Damage Dealt vs Taken")
# ax0.set_xlabel("Hero Damage Dealt")
# ax0.set_ylabel("Damage Taken")


# Separating by team


team_1 = df_final[((df_final["Team"]==1))]
team_2 = df_final[((df_final["Team"]==2))]

fig, ax1 = plt.subplots()
ax1.scatter(x=team_1["Hero Damage Dealt"],
            y=team_1["Damage Taken"],
            c=team_1["Fight Win"],
            cmap=mcol.ListedColormap(["red", "blue"]))
ax1.set_title("Team 1- Damage Dealt vs Taken")
ax1.set_xlabel("Hero Damage Dealt")
ax1.set_ylabel("Damage Taken")


fig, ax2 = plt.subplots()
ax2.scatter(x=team_2["Hero Damage Dealt"],
            y=team_2["Damage Taken"],
            c=team_2["Fight Win"],
            cmap=mcol.ListedColormap(["red", "blue"]))
ax2.set_title("Team 2- Damage Dealt vs Taken")
ax2.set_xlabel("Hero Damage Dealt")
ax2.set_ylabel("Damage Taken")




fig, ax3 = plt.subplots()
ax3.scatter(x=df_damage["Hero Damage Dealt"],
            y=df_damage["Damage Taken"],
            c=df_damage["Fight Win"],
            cmap=mcol.ListedColormap(["red", "blue"]))
ax3.set_title("Damage Role- Damage Dealt vs Taken")
ax3.set_xlabel("Hero Damage Dealt")
ax3.set_ylabel("Damage Taken")


fig, ax4 = plt.subplots()
ax4.scatter(x=df_tank["Hero Damage Dealt"],
            y=df_tank["Damage Taken"],
            c=df_tank["Fight Win"],
            cmap=mcol.ListedColormap(["red", "blue"]))
ax4.set_title("Tank Role- Damage Dealt vs Taken")
ax4.set_xlabel("Hero Damage Dealt")
ax4.set_ylabel("Damage Taken")


fig, ax5 = plt.subplots()
ax5.scatter(x=df_support["Hero Damage Dealt"],
            y=df_support["Damage Taken"],
            c=df_support["Fight Win"],
            cmap=mcol.ListedColormap(["red", "blue"]))
ax5.set_title("Support Role- Damage Dealt vs Taken")
ax5.set_xlabel("Hero Damage Dealt")
ax5.set_ylabel("Damage Taken")


# Changing support stat

fig, ax5 = plt.subplots()
ax5.scatter(x=df_support["Healing Dealt"],
            y=df_support["Damage Taken"],
            c=df_support["Fight Win"],
            cmap=mcol.ListedColormap(["red", "blue"]))
ax5.set_title("Support Role- Healing Dealt vs Damage Taken")
ax5.set_xlabel("Healing Dealt")
ax5.set_ylabel("Damage Taken")




##########################


df_damage = df_damage.drop(columns=["Map", "Player", "Hero"])
df_tank = df_tank.drop(columns=["Map", "Player", "Hero"])
df_support = df_support.drop(columns=["Map", "Player", "Hero"])

# Damage Model
print("\n\nDamage Role Model\n")
X = df_damage.loc[:, ("Hero Damage Dealt", "Damage Taken", "Eliminations",
                      "Barrier Damage Dealt", "Objective Kills", "Solo Kills", "Damage Blocked")]
Y = df_damage.loc[:, ("Fight Win")]
Y = Y.astype('int')
standardizer = StandardScaler()
X = standardizer.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=0)
model = LogisticRegression()
model.fit(X_train, Y_train)
predictions = model.predict(X_test)
cm = confusion_matrix(Y_test, predictions)
TN, FP, FN, TP = confusion_matrix(Y_test, predictions).ravel()
print('True Positive(TP)  = ', TP)
print('False Positive(FP) = ', FP)
print('True Negative(TN)  = ', TN)
print('False Negative(FN) = ', FN)
accuracy = (TP+TN) / (TP+FP+TN+FN)
print('Accuracy of the binary classification = {:0.3f}'.format(accuracy))
precision = (TP) / (TP + FP)
print('Precision of the binary classification = {:0.3f}'.format(precision))
recall = (TP) / (TP + FN)
print('Recall of the binary classification = {:0.3f}'.format(recall))

# Damage Model (Specific)
print("\n\nDamage Role Model- Specified\n")
X = df_damage.loc[:, ("Hero Damage Dealt", "Deaths", "Eliminations", "Damage Taken", "Healing Dealt")]
Y = df_damage.loc[:, ("Fight Win")]
Y = Y.astype('int')
standardizer = StandardScaler()
X = standardizer.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=0)
model = LogisticRegression()
model.fit(X_train, Y_train)
predictions = model.predict(X_test)
cm = confusion_matrix(Y_test, predictions)
TN, FP, FN, TP = confusion_matrix(Y_test, predictions).ravel()
print('True Positive(TP)  = ', TP)
print('False Positive(FP) = ', FP)
print('True Negative(TN)  = ', TN)
print('False Negative(FN) = ', FN)
accuracy = (TP+TN) / (TP+FP+TN+FN)
print('Accuracy of the binary classification = {:0.3f}'.format(accuracy))
precision = (TP) / (TP + FP)
print('Precision of the binary classification = {:0.3f}'.format(precision))
recall = (TP) / (TP + FN)
print('Recall of the binary classification = {:0.3f}'.format(recall))


# Tank Model
print("\n\nTank Role Model\n")
X = df_tank.loc[:, ("Hero Damage Dealt", "Damage Taken", "Eliminations",
                    "Barrier Damage Dealt", "Objective Kills", "Solo Kills", "Damage Blocked")]
Y = df_tank.loc[:, ("Fight Win")]
Y = Y.astype('int')
standardizer = StandardScaler()
X = standardizer.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=0)
model = LogisticRegression()
model.fit(X_train, Y_train)
predictions = model.predict(X_test)
cm = confusion_matrix(Y_test, predictions)
TN, FP, FN, TP = confusion_matrix(Y_test, predictions).ravel()
print('True Positive(TP)  = ', TP)
print('False Positive(FP) = ', FP)
print('True Negative(TN)  = ', TN)
print('False Negative(FN) = ', FN)
accuracy = (TP+TN) / (TP+FP+TN+FN)
print('Accuracy of the binary classification = {:0.3f}'.format(accuracy))
precision = (TP) / (TP + FP)
print('Precision of the binary classification = {:0.3f}'.format(precision))
recall = (TP) / (TP + FN)
print('Recall of the binary classification = {:0.3f}'.format(recall))

# Tank Model (Specific)
print("\n\nTank Role Model- Specified\n")
X = df_tank.loc[:, ("Hero Damage Dealt", "Deaths", "Damage Taken", "Damage Blocked")]
Y = df_tank.loc[:, ("Fight Win")]
Y = Y.astype('int')
standardizer = StandardScaler()
X = standardizer.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=0)
model = LogisticRegression()
model.fit(X_train, Y_train)
predictions = model.predict(X_test)
cm = confusion_matrix(Y_test, predictions)
TN, FP, FN, TP = confusion_matrix(Y_test, predictions).ravel()
print('True Positive(TP)  = ', TP)
print('False Positive(FP) = ', FP)
print('True Negative(TN)  = ', TN)
print('False Negative(FN) = ', FN)
accuracy = (TP+TN) / (TP+FP+TN+FN)
print('Accuracy of the binary classification = {:0.3f}'.format(accuracy))
precision = (TP) / (TP + FP)
print('Precision of the binary classification = {:0.3f}'.format(precision))
recall = (TP) / (TP + FN)
print('Recall of the binary classification = {:0.3f}'.format(recall))


# Support Model
print("\n\nSupport Role Model\n")
X = df_support.loc[:, ("Hero Damage Dealt", "Damage Taken", "Eliminations",
                       "Barrier Damage Dealt", "Objective Kills", "Solo Kills", "Damage Blocked")]
Y = df_support.loc[:, ("Fight Win")]
Y = Y.astype('int')
standardizer = StandardScaler()
X = standardizer.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=0)
model = LogisticRegression()
model.fit(X_train, Y_train)
predictions = model.predict(X_test)
cm = confusion_matrix(Y_test, predictions)
TN, FP, FN, TP = confusion_matrix(Y_test, predictions).ravel()
print('True Positive(TP)  = ', TP)
print('False Positive(FP) = ', FP)
print('True Negative(TN)  = ', TN)
print('False Negative(FN) = ', FN)
accuracy = (TP+TN) / (TP+FP+TN+FN)
print('Accuracy of the binary classification = {:0.3f}'.format(accuracy))
precision = (TP) / (TP + FP)
print('Precision of the binary classification = {:0.3f}'.format(precision))
recall = (TP) / (TP + FN)
print('Recall of the binary classification = {:0.3f}'.format(recall))

# Support Model (Specific)
print("\n\nSupport Role Model- Specified\n")
X = df_support.loc[:, ("Damage Taken", "Healing Dealt", "Deaths", "Hero Damage Dealt")]
Y = df_support.loc[:, ("Fight Win")]
Y = Y.astype('int')
standardizer = StandardScaler()
X = standardizer.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=0)
model = LogisticRegression()
model.fit(X_train, Y_train)
predictions = model.predict(X_test)
cm = confusion_matrix(Y_test, predictions)
TN, FP, FN, TP = confusion_matrix(Y_test, predictions).ravel()
print('True Positive(TP)  = ', TP)
print('False Positive(FP) = ', FP)
print('True Negative(TN)  = ', TN)
print('False Negative(FN) = ', FN)
accuracy = (TP+TN) / (TP+FP+TN+FN)
print('Accuracy of the binary classification = {:0.3f}'.format(accuracy))
precision = (TP) / (TP + FP)
print('Precision of the binary classification = {:0.3f}'.format(precision))
recall = (TP) / (TP + FN)
print('Recall of the binary classification = {:0.3f}'.format(recall))


# All Players Model
print("\n\nAll Roles Model\n")
X = df_final.loc[:, ("Hero Damage Dealt", "Damage Taken", "Eliminations",
                     "Barrier Damage Dealt", "Objective Kills", "Solo Kills", "Damage Blocked")]
Y = df_final.loc[:, ("Fight Win")]
Y = Y.astype('int')
standardizer = StandardScaler()
X = standardizer.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=0)
model = LogisticRegression()
model.fit(X_train, Y_train)
predictions = model.predict(X_test)
cm = confusion_matrix(Y_test, predictions)
TN, FP, FN, TP = confusion_matrix(Y_test, predictions).ravel()
print('True Positive(TP)  = ', TP)
print('False Positive(FP) = ', FP)
print('True Negative(TN)  = ', TN)
print('False Negative(FN) = ', FN)
accuracy = (TP+TN) / (TP+FP+TN+FN)
print('Accuracy of the binary classification = {:0.3f}'.format(accuracy))
precision = (TP) / (TP + FP)
print('Precision of the binary classification = {:0.3f}'.format(precision))
recall = (TP) / (TP + FN)
print('Recall of the binary classification = {:0.3f}'.format(recall))









#################################################################
data = df1.drop(columns=["Healing Received",
                "Player Closest to Reticle", "Position"])


df_show = df_final.drop(columns=["index", "Map", "Player", "Barrier Damage Dealt", "Damage Blocked", "Damage Taken",
                                 "Deaths", "Eliminations", "Final Blows", "Environmental Deaths",
                                 "Environmental Kills", "Healing Dealt", "Objective Kills", "Solo Kills"])



"""

#left over code/plans

#use DATAFRAME.reset_index() to set index beginning at 0


need to mix in more csv files

to use the current code, the CSVs need to have the section values changed


precision and recall
fairness metrics



"""
