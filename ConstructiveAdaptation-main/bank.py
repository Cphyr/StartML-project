df = pd.read_excel("data/Bank_Personal_Loan_Modelling.xlsx", sheet_name="Data")
# ID_0
# Age_1
# Experience_2
# Income_3
# ZIP Code_4
# Family_5
# CCAvg_6
# Education_7
# Mortgage_8
# Personal Loan_9
# Securities Account_10
# CD Account_11
# Online_12
# CreditCard_13
x_0, y_0 = resample(df[df['CreditCard'] == 0.].iloc[:, :-1].values, df[df['CreditCard'] == 0.].iloc[:, -1].values, n_samples=10000, random_state=1234)
x_1, y_1 = resample(df[df['CreditCard'] == 1.].iloc[:, :-1].values, df[df['CreditCard'] == 1.].iloc[:, -1].values, n_samples=10000, random_state=4321)

x = np.concatenate((x_0, x_1))
y = np.concatenate((y_0-1, y_1))

x = preprocessing.StandardScaler().fit_transform(x)
improving_features = [2, 3, 6, 7, 8]
x_I = x[:, improving_features]
manipulated_features = [4, 5, 10, 11, 12]
x_M = x[:, manipulated_features]
unactionable_features = [0, 1, 9]
x = np.concatenate((x[:, improving_features], x[:, manipulated_features], x[:, unactionable_features]), axis=1)
N_I, N_M = len(improving_features), len(manipulated_features)
columns = list(df.columns)
# reorder columns to match the order of x
columns = [columns[i] for i in improving_features] + [columns[i] for i in manipulated_features] + [columns[i] for i in unactionable_features]
