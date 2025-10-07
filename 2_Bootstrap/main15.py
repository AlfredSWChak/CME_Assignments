import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm

def read_csv(filename, hasHeader) -> str:
    output = pd.read_csv(filename, header = hasHeader)
    return output

# import the dataset
raw_data = read_csv('../2_Bootstrap/DeBilt1900to2014.csv', 0)

# extract the annual averages
annualAverage = raw_data['Average']
winterAverage = raw_data['Winter Av']
summerAverage = raw_data['Summer Av']

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

def getHACStandardError(input_matrix, input_y):
    n = len(input_y)
    maxlags = int(np.floor(4 * (n/100)**(2/9)))
    
    res_hac = sm.OLS(input_y, input_matrix).fit(cov_type='HAC', 
                                                cov_kwds={'maxlags': maxlags, 'use_correction': True})
    V_hat_HAC = res_hac.cov_params()  # robust Var(beta-hat)
    SEs = np.sqrt(np.diag(V_hat_HAC))
    return SEs

# perform t-test using standard normal critical values
def t_test(input_alpha, input_t):
    t_crit_upper = norm.ppf(1-input_alpha/2)
    t_crit_lower = norm.ppf(input_alpha)
    
    if t_crit_lower < input_t < t_crit_upper:
        print(f't ∈ [{t_crit_lower},{t_crit_upper}]: Do not reject the null.')
    else:
        print(f't ∉ [{t_crit_lower},{t_crit_upper}]: Reject the null.')
        
'''
Exercise 1a
'''

print('Exercise 1a: basic t-test')

time_list = np.linspace(1, len(annualAverage), len(annualAverage))

matrix_X, estimate_beta, hat_y, hat_residuals = runRegressionModel(annualAverage, time_list)

# compute the HAC standard errors
estimate_SE = getHACStandardError(matrix_X, annualAverage)

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

'''
Exercise 1b: perform the t-test using the different bootstrap methods
'''

print('Exercise 1b: t-test')

numberOfSimulations = 9999

# calculation for bootstrap p-value
def computeBootstrapPValue(input_t_list, observed_t):
    result_p = 0    

    for bootstrap_t in input_t_list:
        if abs(bootstrap_t) > abs(observed_t):
            result_p += 1
            
    result_p = round(result_p / len(input_t_list),3)

    return result_p

# Nonparametric residual bootstrap

def perform_iidBootstrap(input_numberOfSimulation, input_y, input_matrixX, input_beta):
    bootstrap_t_list = []
    
    # obtain residuals under the null
    input_beta.iloc[1,0] = 0
    hat_y_underNull = input_matrixX @ input_beta
    hat_residuals_underNull = np.array(input_y) - np.array(hat_y_underNull.squeeze())
    
    for b in range(0,input_numberOfSimulation):
        # simulate new_residuals as random variables from the empirical distribution
        simulate_residuals = np.random.choice(hat_residuals_underNull.squeeze(), len(hat_residuals_underNull), replace=True)
        mean_hat_residuals = np.average(hat_residuals_underNull)
        new_residuals = simulate_residuals - mean_hat_residuals

        # calculate new_y 
        new_y = np.array(hat_y_underNull.squeeze()) + new_residuals

        # re-estimate the model
        matrix_X, estimate_beta, hat_y, hat_residuals = runRegressionModel(new_y, time_list)

        # extract the estimate beta 1
        bootstrap_beta_1 = estimate_beta.iloc[1,0]

        # compute the standard error of beta 1
        bootstrap_estimate_SE = getHACStandardError(matrix_X, new_y)
        bootstrap_SE_beta_1 = bootstrap_estimate_SE[1]

        # compute the bootstrap t-statistic for β_1 under the null 
        bootstrap_t_statistic_beta_1 = bootstrap_beta_1 / bootstrap_SE_beta_1
        bootstrap_t_list.append(bootstrap_t_statistic_beta_1)
    
    return bootstrap_t_list

# obtain the sequence of simulated test statistic 
bootstrap_t_list = perform_iidBootstrap(numberOfSimulations, annualAverage, matrix_X, estimate_beta)
    
# compute the bootstrap p-value       
bootstrap_p = computeBootstrapPValue(bootstrap_t_list, t_statistic_beta_1)

# report the bootstrap p-value
print(f'i.i.d. bootstrap: The bootstrap p-value is {bootstrap_p}.') 

# Wild bootstrap

def perform_wildBootstrap(input_numberOfSimulation, input_y, input_matrixX, input_beta):
    bootstrap_t_list = []
    
    # obtain residuals under the null
    input_beta.iloc[1,0] = 0
    hat_y_underNull = input_matrixX @ input_beta
    hat_residuals_underNull = np.array(input_y) - np.array(hat_y_underNull.squeeze())
    
    for b in range(0,input_numberOfSimulation):
        # simulate new_residuals bootstrap errors as ∼ N(0,1)
        new_error = np.random.normal(loc=0, scale=1, size=len(hat_residuals_underNull))
        new_residuals = hat_residuals_underNull * new_error

        # calculate new_y 
        new_y = np.array(hat_y_underNull.squeeze()) + new_residuals

        # re-estimate the model
        matrix_X, estimate_beta, hat_y, hat_residuals = runRegressionModel(new_y, time_list)

        # extract the estimate beta 1
        bootstrap_beta_1 = estimate_beta.iloc[1,0]

        # compute the standard error of beta 1
        bootstrap_estimate_SE = getHACStandardError(matrix_X, new_y)
        bootstrap_SE_beta_1 = bootstrap_estimate_SE[1]

        # compute the bootstrap t-statistic for β_1 under the null 
        bootstrap_t_statistic_beta_1 = bootstrap_beta_1 / bootstrap_SE_beta_1
        bootstrap_t_list.append(bootstrap_t_statistic_beta_1)
    
    return bootstrap_t_list

# obtain the sequence of simulated test statistic 
bootstrap_t_list = perform_wildBootstrap(numberOfSimulations, annualAverage, matrix_X, estimate_beta)
    
# compute the bootstrap p-value       
bootstrap_p = computeBootstrapPValue(bootstrap_t_list, t_statistic_beta_1)

# report the bootstrap p-value
print(f'Wild bootstrap: The bootstrap p-value is {bootstrap_p}.') 

# Block bootstrap

# Sieve bootstrap

'''
Exercise 2a: perform the t-test using the winter and summer averages
'''

print('Exercise 2a: Winter Average')

matrix_X, estimate_beta, hat_y, hat_residuals = runRegressionModel(winterAverage, time_list)

# compute the HAC standard errors
estimate_SE = getHACStandardError(matrix_X, winterAverage)

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

# Nonparametric residual bootstrap

# obtain the sequence of simulated test statistic 
bootstrap_t_list = perform_iidBootstrap(numberOfSimulations, winterAverage, matrix_X, estimate_beta)
    
# compute the bootstrap p-value       
bootstrap_p = computeBootstrapPValue(bootstrap_t_list, t_statistic_beta_1)

# report the bootstrap p-value
print(f'i.i.d. bootstrap: The bootstrap p-value is {bootstrap_p}.') 

# Wild bootstrap

# obtain the sequence of simulated test statistic 
bootstrap_t_list = perform_wildBootstrap(numberOfSimulations, winterAverage, matrix_X, estimate_beta)
    
# compute the bootstrap p-value       
bootstrap_p = computeBootstrapPValue(bootstrap_t_list, t_statistic_beta_1)

# report the bootstrap p-value
print(f'Wild bootstrap: The bootstrap p-value is {bootstrap_p}.') 

print('Exercise 2a: Summer Average')

matrix_X, estimate_beta, hat_y, hat_residuals = runRegressionModel(summerAverage, time_list)

# compute the HAC standard errors
estimate_SE = getHACStandardError(matrix_X, summerAverage)

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

# Nonparametric residual bootstrap

# obtain the sequence of simulated test statistic 
bootstrap_t_list = perform_iidBootstrap(numberOfSimulations, summerAverage, matrix_X, estimate_beta)
    
# compute the bootstrap p-value       
bootstrap_p = computeBootstrapPValue(bootstrap_t_list, t_statistic_beta_1)

# report the bootstrap p-value
print(f'i.i.d. bootstrap: The bootstrap p-value is {bootstrap_p}.') 

# Wild bootstrap

# obtain the sequence of simulated test statistic 
bootstrap_t_list = perform_wildBootstrap(numberOfSimulations, summerAverage, matrix_X, estimate_beta)
    
# compute the bootstrap p-value       
bootstrap_p = computeBootstrapPValue(bootstrap_t_list, t_statistic_beta_1)

# report the bootstrap p-value
print(f'Wild bootstrap: The bootstrap p-value is {bootstrap_p}.') 

# Block bootstrap

# Sieve bootstrap

'''
Exercise 3a: construct the confidence intervals
'''

# Nonparametric residual bootstrap

# Wild bootstrap

# Block bootstrap

# Sieve bootstrap