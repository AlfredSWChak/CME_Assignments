import numpy as np
import pandas as pd
from scipy.stats import norm

def read_csv(filename, hasHeader) -> str:
    output = pd.read_csv(filename, header = hasHeader)
    return output

# import the dataset
raw_data = read_csv('../2_Bootstrap/DeBilt1900to2014.csv', 0)

# extract the annual averages
annualAverage = raw_data['Average']

# calculation for estimated beta by OLS regression
def getEstimatedBeta(input_matrix_X, input_vector_y):
    # calculate the inverse matrix (X′ * X)^{−1}
    df = input_matrix_X.T @ input_matrix_X
    df_inv = pd.DataFrame(np.linalg.inv(df.values),
                      index=df.columns,   # rows = original columns
                      columns=df.index) 
    
    # calculate estimated beta by (X′ * X)^{−1} * X′* y
    result_beta = df_inv @ input_matrix_X.T @ input_vector_y
    
    return result_beta

# estimate the regression by OLS regression
def runRegressionModel(input_y, input_x):
    # convert y to matrix y
    vector_y = pd.DataFrame(input_y)
    # convert y to matrix X
    matrix_X = pd.DataFrame({'intercept': 1, 'x': input_x})
    
    # calculate estimated beta by (X′ * X)^{−1} * X′* y
    estimate_beta = getEstimatedBeta(matrix_X, vector_y)

    # calculate estimated y by X * (estimated beta)
    estimate_y = matrix_X @ estimate_beta
    
    # calculate the residuals
    estimate_residuals = vector_y - estimate_y

    return matrix_X, estimate_beta, estimate_y, estimate_residuals

# calculation for standard error by OLS regression
def getStandardError(input_matrix, input_residuals):
    # get the shape of the matrix
    n, k = input_matrix.shape

    # estimate the error variance
    estimated_variance = (input_residuals.T @ input_residuals).iat[0, 0] / (n - k)
    
    df = input_matrix.T @ input_matrix
    df_inv = pd.DataFrame(np.linalg.inv(df.values),
                        index=df.columns,   # rows = original columns
                        columns=df.index) 

    # compute the variance-covariance matrix
    varCovMatrix = estimated_variance * df_inv

    # get the standard errors which are the diagonal element
    estimated_standard_error = np.sqrt(np.diag(varCovMatrix))
    
    return estimated_standard_error

# perform t-test using standard normal critical values
def t_test(input_dof, input_alpha, input_t):
    t_crit_upper = norm.ppf(1-input_alpha/2)
    t_crit_lower = norm.ppf(input_alpha)
    
    if t_crit_lower < input_t < t_crit_upper:
        print(f't ∈ [{t_crit_lower},{t_crit_upper}]: Do not reject the null.')
    else:
        print(f't ∉ [{t_crit_lower},{t_crit_upper}]: Reject the null.')

time_list = np.linspace(1, len(annualAverage), len(annualAverage))

matrix_X, estimate_beta, hat_y, hat_residuals = runRegressionModel(annualAverage, time_list)

# compute the HAC standard errors
estimate_SE = getStandardError(matrix_X, hat_residuals)

# compute t-statistic for β_0 under the null 
SE_beta_0 = estimate_SE[0]
t_statistic_beta_0 = estimate_beta.iat[0,0] / SE_beta_0

# compute t-statistic for β_1 under the null 
SE_beta_1 = estimate_SE[1]
t_statistic_beta_1 = estimate_beta.iat[1,0] / SE_beta_1

# significance level
alpha = 0.05

# perform a t-test using the annual averages for β_0
print(f't-test for β_0 = {t_statistic_beta_0}:')
t_test(alpha, t_statistic_beta_0)

# perform a t-test using the annual averages for β_1
print(f't-test for β_1 = {t_statistic_beta_1}:')
t_test(alpha, t_statistic_beta_1)