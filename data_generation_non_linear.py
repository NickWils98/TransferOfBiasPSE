import pandas as pd
import numpy as np

# Enhance direct effect otherwise indirect effect
DIRECT = True
# High/Low income if True, Numeric income if False
BINARY = False

def generate_data(n):
    """Generate synthetic data with non-linearities for analysis."""
    np.random.seed(6)

    # Parameters
    if DIRECT:
        gender_effect_on_income = 500  # Direct gender effect on income
        sector_effect_on_income = -50  # Sector effect on income
    else:
        gender_effect_on_income = 50  # Direct gender effect on income
        sector_effect_on_income = -500  # Sector effect on income

    beta_W = 100  # Hours worked -> Income effect

    # Generate synthetic data
    S = np.random.binomial(1, 0.5, n)  # Gender: 1 = Male, 0 = Female
    W = np.random.randint(20, 60, n)  # Hours worked
    prob_sector_1 = 0.8  # Probability that a man is in sector 1
    prob_sector_0 = 0.2  # Probability that a woman is in sector 1

    # Sector assignment based on gender
    M = np.where(S == 1,
                 np.random.binomial(1, prob_sector_1, n),
                 np.random.binomial(1, prob_sector_0, n))

    # Adding noise
    noise = 100 * np.random.normal(size=n)

    # Non-linear effects
    hours_squared = 0.05 * (W ** 2)  # Non-linear term for hours worked
    interaction_effect = -0.3 * W * M  # Interaction between hours worked and sector
    log_effect = 200 * np.log1p(W)  # Logarithmic transformation of hours worked

    # Income calculation with non-linearities
    Y = (gender_effect_on_income * S +
         sector_effect_on_income * M +
         beta_W * W +
         hours_squared +
         interaction_effect +
         log_effect +
         noise)

    # Calculate the fair income
    if not DIRECT:
        Y_fair = (gender_effect_on_income * S +
                  sector_effect_on_income * M -
                  (prob_sector_1 - prob_sector_0) * M * S * sector_effect_on_income +
                  beta_W * W +
                  hours_squared +
                  interaction_effect +
                  log_effect +
                  noise)
    else:
        Y_fair = (sector_effect_on_income * M +
                  beta_W * W +
                  hours_squared +
                  interaction_effect +
                  log_effect +
                  noise)

    # Convert Income (Y) to binary: 1 for high income, 0 for low income
    income_threshold = np.median(Y)
    Y_binary_fair = np.where(Y_fair > income_threshold, 1, 0)
    Y_binary = np.where(Y > income_threshold, 1, 0)

    # Create DataFrame and save to CSV
    if BINARY:
        data = pd.DataFrame({'Gender': S, 'Sector': M, 'Worked_Hours': W, 'Income': Y_binary, 'Income_Fair': Y_binary_fair})
        data.to_csv(f'Generated{n}_binary_fair_non_linear.csv', index=False, encoding='utf-8')
    else:
        data = pd.DataFrame({'Gender': S, 'Sector': M, 'Worked_Hours': W, 'Income': Y, 'Income_Fair': Y_fair})
        data.to_csv(f'Generated{n}_fair_non_linear.csv', index=False, encoding='utf-8')

    return data

if __name__ == '__main__':
    n = 60000  # Number of samples
    data = generate_data(n)
