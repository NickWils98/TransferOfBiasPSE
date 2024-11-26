import json

import pandas as pd
from dowhy import CausalModel

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error

# Use direct effect otherwise indirect effect
DIRECT=True
# Binary output or Numeric
BINARY = False
# remove the edge from sector to gender
REMOVE_EDGE = True
# Use external dataset for testing
EXTERNALDF = True

def extract_edges(edgesDict):
    # get all the edges in a string
    edges = "digraph {"
    for edge in edgesDict["edges"]:

        if REMOVE_EDGE:
            # all edges except Secotr -> Gender
            if edge["source"] == "Sector" and edge["target"] == "Gender":
                pass
            else:
                edges+= f"{edge["source"]} -> {edge["target"]};"
        else:
            edges += f"{edge["source"]} -> {edge["target"]};"

    edges += "}"
    return edges

def estimate_effect():
    # Load your dataset
    if BINARY:
        df = pd.read_csv('Generated60000_binary_fair.csv')
    else:
        df = pd.read_csv('Generated60000_fair.csv')
    # remove testing column before causal model
    df.drop('Income_Fair', axis=1, inplace=True)

    # get the edges
    with open('CausalDAG.json') as f:
        edgesJson = json.load(f)
    edges = extract_edges(edgesJson)

    # Define the causal model
    model = CausalModel(
        data=df,
        treatment="Gender",
        outcome="Income",
        graph=edges,
    )

    # Identify the total effect (Gender -> Income)
    identified_estimand_total = model.identify_effect(proceed_when_unidentifiable=True)
    estimate_total = model.estimate_effect(
        identified_estimand_total,
        method_name="backdoor.linear_regression"
    )
    print("Total Effect (Gender -> Income):", estimate_total.value)

    # Estimate the indirect effect through the mediator (Gender -> Sector -> Income)
    # Redefine the graph to isolate the indirect effect
    mediator_model = CausalModel(
        data=df,
        treatment="Gender",
        outcome="Sector",
        graph=edges
    )

    # Estimate effect of Gender on Sector
    identified_estimand_mediator = mediator_model.identify_effect(proceed_when_unidentifiable=True)
    estimate_gender_to_sector = mediator_model.estimate_effect(
        identified_estimand_mediator,
        method_name="backdoor.linear_regression"
    )

    # Estimate effect of Sector on Income
    sector_model = CausalModel(
        data=df,
        treatment="Sector",
        outcome="Income",
        graph=edges
    )
    identified_estimand_sector = sector_model.identify_effect(proceed_when_unidentifiable=True)
    estimate_sector_to_income = sector_model.estimate_effect(
        identified_estimand_sector,
        method_name="backdoor.linear_regression"
    )

    # Compute the indirect effect (Gender -> Sector -> Income)
    indirect_effect = estimate_gender_to_sector.value * estimate_sector_to_income.value
    print("Indirect Effect (Gender -> Sector -> Income):", indirect_effect)

    # The direct effect is the total effect minus the indirect effect
    direct_effect = estimate_total.value - indirect_effect
    print("Direct Effect (Gender -> Income, bypassing Sector):", direct_effect)
    return direct_effect, indirect_effect

def adjust_predictions_with_gender_effect(predictions, gender,sector, causal_effect):
    adjusted_predictions = predictions.copy()

    # Adjust predictions based on Gender
    for i in range(len(predictions)):
        # print(gender[0])

        if gender[i] == 1:  # Male (assuming 0 is Female, 1 is Male)
            if not DIRECT:
                if sector[i] == 1:  # Male (assuming 0 is Female, 1 is Male)
                    adjusted_predictions[i] -= causal_effect  # Remove the gender effect
            else:
                adjusted_predictions[i] -= causal_effect  # Remove the gender effect

    # Return the adjusted predictions
    return adjusted_predictions

if __name__ == '__main__':
    # get the PSE
    direct_effect, indirect_effect = estimate_effect()

    if DIRECT:
        gender_causal_effect = direct_effect
    else:
        gender_causal_effect = indirect_effect


    if BINARY:
        data = pd.read_csv('Generated60000_binary_fair.csv')

        if EXTERNALDF:
            df_fair = pd.read_csv('fairData_binary.csv')
            data.drop('Income_Fair', axis=1, inplace=True)
            data["Income_Fair"] = df_fair["Income"]
    else:
        data = pd.read_csv('Generated60000_fair.csv')
        if EXTERNALDF:
            df_fair = pd.read_csv('fairData.csv')
            data.drop('Income_Fair', axis=1, inplace=True)
            data["Income_Fair"] = df_fair["Income"]

    # Prepare the data for machine learning
    X = data[['Gender', 'Sector', 'Worked_Hours']]  # Features
    y = data[['Income', 'Income_Fair']]  # Target variable (Income), Income_Fair is  for testing

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Remove Income_Fair before training the model
    y_train.drop('Income_Fair', axis=1, inplace=True)

    y_test_fair = y_test.copy(deep=True)

    y_test.drop('Income_Fair', axis=1, inplace=True)
    y_test_fair.drop('Income', axis=1, inplace=True)

    # Initialize and train the RandomForest model
    if BINARY:
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

    rf_model.fit(X_train, y_train)

    # Get predictions and adjust them
    if BINARY:
        # Make predictions on the test set
        y_pred = rf_model.predict_proba(X_test)
        # adjust the probabilities
        adjusted_probabilities = adjust_predictions_with_gender_effect(y_pred[:, 1], X_test['Gender'].to_numpy(), X_test['Sector'].to_numpy(), gender_causal_effect)
        original_predictions = (y_pred[:, 1] >= 0.5).astype(int)  # Use 0.5 as a threshold for binary classification
        adjusted_predictions_fair = (adjusted_probabilities >= 0.5).astype(int)  # Use 0.5 as a threshold for binary classification
        metric = "Accuracy"



    else:
        y_pred = rf_model.predict(X_test)
        adjusted_predictions_fair = adjust_predictions_with_gender_effect(y_pred, X_test['Gender'].to_numpy(),
                                                                       X_test['Sector'].to_numpy(),
                                                                       gender_causal_effect)

        original_predictions = y_pred

        y_test_fair.to_numpy()
        y_test.to_numpy()
        metric = "Mean absolute error"

    # test the accuracy or MAE
    if BINARY:
        accuracy_adjusted = accuracy_score(y_test, original_predictions)
        accuracy_adjusted*=100
    else:
        accuracy_adjusted = mean_absolute_error(y_test, original_predictions)

    print(f"{metric} of original predictions on unfair test: {accuracy_adjusted:.2f}{"%" if BINARY else ""}")


    if BINARY:
        accuracy_adjusted = accuracy_score(y_test, adjusted_predictions_fair)
        accuracy_adjusted*=100
    else:
        accuracy_adjusted = mean_absolute_error(y_test, adjusted_predictions_fair)
    print(f"{metric} of altered predictions on unfair test: {accuracy_adjusted:.2f}{"%" if BINARY else ""}")


    if BINARY:
        accuracy_adjusted = accuracy_score(y_test_fair, original_predictions)
        accuracy_adjusted*=100
    else:
        accuracy_adjusted = mean_absolute_error(y_test_fair, original_predictions)
    print(f"{metric} of original predictions on fair test: {accuracy_adjusted:.2f}{"%" if BINARY else ""}")


    if BINARY:
        accuracy_adjusted = accuracy_score(y_test_fair, adjusted_predictions_fair)
        accuracy_adjusted*=100
    else:
        accuracy_adjusted = mean_absolute_error(y_test_fair, adjusted_predictions_fair)
    print(f"{metric} of altered predictions on fair test: {accuracy_adjusted:.2f}{"%" if BINARY else ""}")



