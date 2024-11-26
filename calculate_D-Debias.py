import pandas as pd
import numpy as np
import statsmodels.api as sm

# Use high/low income or real numbers
BINARY = False
# Remove the Fair column used for testing
REMOVE_FAIR = True


def debias_categorical_variable(prob_mat, D_v, lr=0.1, max_iterations=50):
    """
    Perform iterative debiasing on a categorical variable using a probability matrix.

    Parameters:
    - prob_mat: np.ndarray, the initial probability matrix for variable v.
    - D_v: np.ndarray, the original distribution (counts or probabilities) of the categorical variable.
    - lr: float, the learning rate.
    - max_iterations: int, the maximum number of iterations.

    Returns:
    - D_debiased: np.ndarray, the debiased distribution for the categorical variable.
    - prob_mat: np.ndarray, the adjusted probability matrix.
    """

    # Define the function to calculate discrete probability distribution (DPD)
    def DPD(distribution):
        normalized_distribution = distribution / (distribution.sum() + 1e-10)  # Add epsilon to avoid division by zero
        return normalized_distribution

    # Initialize the distributions and compute the original DPD
    dist_ori = DPD(D_v)  # Original distribution of income
    dist_deb = DPD(prob_mat.sum(axis=0))  # Sum across the first axis to get total probabilities for each category

    for iteration in range(max_iterations):
        # Ensure both distributions have the same shape
        if dist_ori.shape[0] != dist_deb.shape[0]:
            raise ValueError(f"Shape mismatch: dist_ori shape {dist_ori.shape} vs dist_deb shape {dist_deb.shape}")

        # Compute difference between original and debiased distributions
        diff = np.sum((dist_ori - dist_deb) / (dist_deb + 1e-10))  # Add epsilon to avoid division by zero

        # Compute the scaling factor for the probability matrix update
        scale_factor = 1 + lr * diff

        # Update probability matrix and avoid overflow
        prob_mat = np.clip(prob_mat * scale_factor, 1e-10, None)  # Prevent overflow by clipping

        # Recompute debiased distribution
        dist_deb = DPD(prob_mat.sum(axis=0))  # Update by summing across columns

        # Check stopping condition based on the change in distribution
        if np.abs(diff) < 1e-6:  # Set a small threshold for convergence
            break

    # Debiased distribution is the final computed distribution
    D_debiased = dist_deb

    return D_debiased, prob_mat


if __name__ == '__main__':
    # Load the dataset
    if BINARY:
        data = pd.read_csv("Generated60000_binary_fair.csv")
    else:
        data = pd.read_csv("Generated60000_fair.csv")
    if REMOVE_FAIR:
        data.drop('Income_Fair', axis=1, inplace=True)

    # Convert DataFrame to numpy array
    data_np = data.values


    # Step 1: Sector ~ Gender
    X1 = sm.add_constant(data['Gender'])  # Add constant for intercept
    model_sector = sm.OLS(data['Sector'], X1).fit()
    print("Sector ~ Gender")
    print(model_sector.summary())

    # Step 2: Income ~ Gender + Sector + Worked_Hours
    X2 = sm.add_constant(data[['Gender', 'Sector', 'Worked_Hours']])  # Add constant for intercept
    model_income = sm.OLS(data['Income'], X2).fit()
    print("\nIncome ~ Gender + Sector + Worked_Hours")
    print(model_income.summary())

    # Display fitted coefficients for interpretation
    print("\nFitted coefficients:")
    print("Sector model (Sector ~ Gender):")
    print(f"Intercept: {model_sector.params[0]:.2f}, Gender coefficient: {model_sector.params[1]:.2f}")
    print("\nIncome model (Income ~ Gender + Sector + Worked_Hours):")
    print(f"Intercept: {model_income.params[0]:.2f}, "
          f"Gender coefficient: {model_income.params['Gender']:.2f}, "
          f"Sector coefficient: {model_income.params['Sector']:.2f}, "
          f"Worked_Hours coefficient: {model_income.params['Worked_Hours']:.2f}")

    intercept = model_income.params['const']
    gender_coef = model_income.params['Gender']
    sector_coef = model_income.params['Sector']
    worked_hours_coef = model_income.params['Worked_Hours']

    # Define the new data point: Female (Gender = 0), Sector = 1, Worked_Hours = 20
    gender = 0  # Female
    sector = 1  # Sector 1
    worked_hours = 20
    gender_random_effect_mean = data['Gender'].mean()
    gender_random_effect_std = data['Gender'].std()
    random_effect_gender = np.random.normal(gender_random_effect_mean, gender_random_effect_std)

    # Calculate the predicted income
    predicted_income = (intercept +
                        gender_coef * gender +
                        sector_coef * sector +
                        worked_hours_coef * worked_hours)
    predicted_income2 = (intercept +
                         gender * random_effect_gender +
                         sector_coef * sector +
                         worked_hours_coef * worked_hours)
    print("Predicted Income for Female, Sector 1, Worked Hours 20:", predicted_income)
    print("Predicted Income with Random Gender Effect:", predicted_income2)

    new_data = pd.DataFrame({
        'Gender': data["Gender"],
        'Sector': data["Sector"],
        'Worked_Hours': data["Worked_Hours"],
    })
    new_data['Income'] = (
            intercept +
            (random_effect_gender) * gender_coef +  # Gender coefficient
            sector_coef * new_data['Sector'] +
            worked_hours_coef * new_data['Worked_Hours']
    )
    print("Original Data Sample:")
    print(data.head(10))
    print("\nNew Data Sample (Before Debiasing):")
    print(new_data.head(10))

    if BINARY:
        # Prepare inputs for debias_categorical_variable
        prob_mat = np.full((len(new_data), len(np.unique(data['Income']))),
                           1 / len(np.unique(data['Income'])))  # Uniform distribution
        D_v = np.bincount(data['Income'], minlength=len(np.unique(data['Income'])))  # Original counts

        # Perform debiasing
        D_debiased, updated_prob_mat = debias_categorical_variable(prob_mat, D_v)

        # Update new data Income based on debiased distribution
        new_data['Income'] = np.random.choice(np.arange(len(D_debiased)), p=D_debiased / D_debiased.sum(),
                                              size=len(new_data))
    else:
        meanDeb = new_data['Income'].mean()
        varDeb = new_data['Income'].std()
        meandata = data['Income'].mean()
        vardata = data['Income'].std()

        new_data["Income"] = vardata + (new_data['Income'] -meanDeb)/varDeb*vardata

    if BINARY:
        # Count the occurrences of Income = 1
        count_income_1_data = np.sum(data['Income'] == 1)
        print("Count of Income = 1 in original data:", count_income_1_data)

        # Count the occurrences of Income = 1 in the new data
        count_income_1_new_data = np.sum(new_data['Income'] == 1)
        print("Count of Income = 1 in new data:", count_income_1_new_data)
        new_data.to_csv(f'fairData_binary.csv', index=False, encoding='utf-8')

    else:
        # Count the occurrences of Income = 1
        mean_income_1_data = data['Income'].mean()
        print("Mean Income in original data:", mean_income_1_data)

        # Count the occurrences of Income = 1 in the new data
        mean_income_1_new_data = new_data['Income'].mean()
        print("Mean Income in new data:", mean_income_1_new_data)

        new_data.to_csv(f'fairData.csv', index=False, encoding='utf-8')
