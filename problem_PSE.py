import numpy as np
import pandas as pd
from scipy.optimize import minimize
import statsmodels.api as sm
import matplotlib.pyplot as plt


def log_likelihood(params, X, y):
    intercept, beta_gender, beta_sector, beta_worked_hours = params
    predicted_y = (intercept +
                   beta_gender * X[:, 1] +  # Gender
                   beta_sector * X[:, 2] +  # Sector
                   beta_worked_hours * X[:, 3])  # Worked Hours
    residuals = y - predicted_y
    log_like = -0.5 * np.sum(residuals ** 2)
    return -log_like  # Negative log-likelihood for minimization


def constrained_log_likelihood(params, X, y):
    intercept, _, beta_sector, beta_worked_hours = params
    # Set gender effect to 0 (i.e., Gender -> Income direct effect is removed)
    predicted_y = (intercept +
                   0 * X[:, 1] +  # Constrain Gender effect to 0
                   beta_sector * X[:, 2] +  # Sector
                   beta_worked_hours * X[:, 3])  # Worked Hours
    residuals = y - predicted_y
    log_like = -0.5 * np.sum(residuals ** 2)
    return -log_like  # Negative log-likelihood for minimization

if __name__ == '__main__':

    # Step 1: Data Generation
    np.random.seed(6)

    # Parameters
    n = 1000  # Number of samples
    alpha = 0.5  # Gender -> Sector effect
    beta_W = 10  # Hours worked -> Income effect
    gender_effect_on_income = 1000  # Direct gender effect on income
    sector_effect_on_income = -100  # Sector effect on income
    worked_hours_effect = 10  # Effect of hours worked on income

    # Generate synthetic data
    S = np.random.binomial(1, 0.5, n)  # Gender: 1 = Male, 0 = Female
    W = np.random.randint(20, 60, n)  # Hours worked
    # Adjust sector assignment to make sector 1 (male-dominated) have more men and sector 0 (female-dominated) have more women.
    # We'll use a conditional assignment where men have a higher probability of being in sector 1.
    prob_sector_1 = 0.8  # Probability that a man is in sector 1 (male-dominated)
    prob_sector_0 = 0.2  # Probability that a woman is in sector 1 (female-dominated)

    M = np.where(S == 1,  # If male (S=1)
                 np.random.binomial(1, prob_sector_1, n),  # Higher chance to be in sector 1
                 np.random.binomial(1, prob_sector_0, n))  # Lower chance for women to be in sector 1

    Y = (gender_effect_on_income * S +
         sector_effect_on_income * M +
         beta_W * W +
         np.random.normal(0, 1, n))  # Income

    # Put the data into a pandas DataFrame
    data = pd.DataFrame({'Gender': S, 'Sector': M, 'Worked_Hours': W, 'Income': Y})
    # Step 2: Fit an unconstrained model (for comparison)
    X_unconstrained = sm.add_constant(data[['Gender', 'Sector', 'Worked_Hours']])
    model_unconstrained = sm.OLS(data['Income'], X_unconstrained).fit()

    print("Unconstrained model results:")
    print(model_unconstrained.summary())

    # Prepare the data for optimization
    X = np.column_stack((np.ones(n), data['Gender'], data['Sector'], data['Worked_Hours']))
    y = data['Income']

    # Step 3: Run unconstrained optimization (for reference)
    initial_params = np.array([0, 0, 0, 0])  # Initial guess for [intercept, beta_gender, beta_sector, beta_worked_hours]
    result_unconstrained = minimize(log_likelihood, initial_params, args=(X, y))

    # Print unconstrained optimization results
    print("\nUnconstrained optimization results:")
    print(f"Intercept: {result_unconstrained.x[0]:.4f}")
    print(f"Gender effect: {result_unconstrained.x[1]:.4f}")
    print(f"Sector effect: {result_unconstrained.x[2]:.4f}")
    print(f"Worked Hours effect: {result_unconstrained.x[3]:.4f}")

    # Step 4: Run constrained optimization (Gender effect = 0)
    result_constrained = minimize(constrained_log_likelihood, initial_params, args=(X, y))



    # Print constrained optimization results
    print("\nConstrained optimization results (Gender effect = 0):")
    print(f"Intercept: {result_constrained.x[0]:.4f}")
    print(f"Sector effect: {result_constrained.x[2]:.4f}")
    print(f"Worked Hours effect: {result_constrained.x[3]:.4f}")

    # Step 5: Compare income distribution before and after constraint
    data['Predicted_Income_Unconstrained'] = (result_unconstrained.x[0] +
                                              result_unconstrained.x[1] * data['Gender'] +
                                              result_unconstrained.x[2] * data['Sector'] +
                                              result_unconstrained.x[3] * data['Worked_Hours'])

    data['Predicted_Income_Constrained'] = (result_constrained.x[0] +
                                            result_constrained.x[2] * data['Sector'] +
                                            result_constrained.x[3] * data['Worked_Hours'])

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.hist(data['Income'], bins=30, alpha=0.5, label='Actual Income')
    plt.hist(data['Predicted_Income_Unconstrained'], bins=30, alpha=0.5, label='Predicted Income (Unconstrained)')
    plt.hist(data['Predicted_Income_Constrained'], bins=30, alpha=0.5, label='Predicted Income (Constrained)')
    plt.title('Income Distribution: Actual vs Predicted (Unconstrained and Constrained)')
    plt.xlabel('Income')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()


    # Difference between man and woman
    print("Difference between man and woman")
    # Step 1: Define the characteristics of the individual
    gender = 1  # Man
    sector = 1  # Male-dominated sector
    worked_hours = 40  # 40 hours per week

    # Step 2: Prediction using the unconstrained model
    predicted_income_unconstrainedman = (result_unconstrained.x[0] +  # Intercept
                                      result_unconstrained.x[1] * gender +  # Gender effect
                                      result_unconstrained.x[2] * sector +  # Sector effect
                                      result_unconstrained.x[3] * worked_hours)  # Hours worked effect

    print(f"Predicted income (Unconstrained model) MAN: {predicted_income_unconstrainedman:.2f} euros")

    # Step 3: Prediction using the constrained model (Gender effect is 0)
    predicted_income_constrainedman = (result_constrained.x[0] +  # Intercept
                                    0 * gender +  # Gender effect is constrained to 0
                                    result_constrained.x[2] * sector +  # Sector effect
                                    result_constrained.x[3] * worked_hours)  # Hours worked effect

    print(f"Predicted income (Constrained model) MAN: {predicted_income_constrainedman:.2f} euros")


    # Step 1: Define the characteristics of the individual
    gender = 0  # Woman
    sector = 1  # Male-dominated sector
    worked_hours = 40  # 40 hours per week

    # Step 2: Prediction using the unconstrained model
    predicted_income_unconstrainedwom = (result_unconstrained.x[0] +  # Intercept
                                      result_unconstrained.x[1] * gender +  # Gender effect
                                      result_unconstrained.x[2] * sector +  # Sector effect
                                      result_unconstrained.x[3] * worked_hours)  # Hours worked effect

    print(f"Predicted income (Unconstrained model) WOMAN: {predicted_income_unconstrainedwom:.2f} euros")

    # Step 3: Prediction using the constrained model (Gender effect is 0)
    predicted_income_constrainedwom = (result_constrained.x[0] +  # Intercept
                                    0 * gender +  # Gender effect is constrained to 0
                                    result_constrained.x[2] * sector +  # Sector effect
                                    result_constrained.x[3] * worked_hours)  # Hours worked effect

    print(f"Predicted income (Constrained model) WOMAN: {predicted_income_constrainedwom:.2f} euros")


    print("\n Average effect between man and woman:")
    x = (predicted_income_unconstrainedman+predicted_income_unconstrainedwom)/2
    print(f"average sector 1 unconstrained {x:.2f}")
    y=(predicted_income_constrainedman+predicted_income_constrainedwom)/2
    print(f"average sector 1 constrained {y:.2f}")

    print(f"diff = {(y-x):.2f}" )
    print("\n")


    # The same but for sector 0
    # Step 1: Define the characteristics of the individual
    gender = 1  # Man
    sector = 0  # Male-dominated sector
    worked_hours = 40  # 40 hours per week

    # Step 2: Prediction using the unconstrained model
    predicted_income_unconstrainedman = (result_unconstrained.x[0] +  # Intercept
                                      result_unconstrained.x[1] * gender +  # Gender effect
                                      result_unconstrained.x[2] * sector +  # Sector effect
                                      result_unconstrained.x[3] * worked_hours)  # Hours worked effect


    # Step 3: Prediction using the constrained model (Gender effect is 0)
    predicted_income_constrainedman = (result_constrained.x[0] +  # Intercept
                                    0 * gender +  # Gender effect is constrained to 0
                                    result_constrained.x[2] * sector +  # Sector effect
                                    result_constrained.x[3] * worked_hours)  # Hours worked effect

    # Step 1: Define the characteristics of the individual
    gender = 0  # Man
    sector = 0  # Male-dominated sector
    worked_hours = 40  # 40 hours per week

    # Step 2: Prediction using the unconstrained model
    predicted_income_unconstrainedwom = (result_unconstrained.x[0] +  # Intercept
                                      result_unconstrained.x[1] * gender +  # Gender effect
                                      result_unconstrained.x[2] * sector +  # Sector effect
                                      result_unconstrained.x[3] * worked_hours)  # Hours worked effect

    # Step 3: Prediction using the constrained model (Gender effect is 0)
    predicted_income_constrainedwom = (result_constrained.x[0] +  # Intercept
                                    0 * gender +  # Gender effect is constrained to 0
                                    result_constrained.x[2] * sector +  # Sector effect
                                    result_constrained.x[3] * worked_hours)  # Hours worked effect

    x = (predicted_income_unconstrainedman+predicted_income_unconstrainedwom)/2
    print(f"average sector 0 unconstrained {x:.2f}")
    y=(predicted_income_constrainedman+predicted_income_constrainedwom)/2
    print(f"average sector 0 constrained {y:.2f}")

    print(f"diff = {(y-x):.2f}" )
    print("\n")

    # All effects on the models
    print("All causal effects on the models")
    translation =  ["Intercept", "Gender", "sector", "Worked_hours"]
    for i in range(len(result_constrained.x)):
        print(f"{translation[i]} effect (Unconstrained model): {result_unconstrained.x[i]:.4f}")
        print(f"{translation[i]} effect (Constrained model): {result_constrained.x[i]:.4f}")
