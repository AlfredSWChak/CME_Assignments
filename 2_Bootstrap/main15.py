import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm

def read_csv(filename, hasHeader) -> str:
    output = pd.read_csv(filename, header = hasHeader)
    return output

# import the dataset
raw_data = read_csv('../2_Bootstrap/DeBilt1900to2014.csv', 0)

np.random.seed(66) # set seeds to be reproducible

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
    estimated_variance = (input_residuals.T @ input_residuals)/ (n - k)
    
    df = input_matrix.T @ input_matrix
    df_inv = pd.DataFrame(np.linalg.inv(df.values),
                        index=df.columns,   # rows = original columns
                        columns=df.index) 

    # compute the variance-covariance matrix
    varCovMatrix = estimated_variance * df_inv

    # get the standard errors which are the diagonal element
    estimated_standard_error = np.sqrt(np.diag(varCovMatrix))
    
    return estimated_standard_error

# calculation for heteroskedasticity standard errors
def getHCStandardError(input_matrix, input_residuals):
    df = input_matrix.T @ input_matrix
    df_inv = pd.DataFrame(np.linalg.inv(df.values),
                        index=df.columns,   # rows = original columns
                        columns=df.index) 
    
    # diagonal matrix with entry ε^2 t as its t’th diagonal element
    sigma = np.diag(input_residuals**2) 
    
    # compute the heteroskedasticity consistent covariance estimator
    hc_estimator = df_inv @ input_matrix.T @ sigma @ input_matrix @ df_inv

    # get the standard errors which are the diagonal element
    estimated_standard_error = np.sqrt(np.diag(hc_estimator))
    
    return estimated_standard_error

# calculation for HAC standard errors
def getHACStandardError(input_matrix, input_y):
    n = len(input_y)
    maxlags = int(np.floor(4 * (n/100)**(2/9)))
    
    res_hac = sm.OLS(input_y, input_matrix).fit(cov_type='HAC', 
                                                cov_kwds={'maxlags': maxlags, 'use_correction': True})
    V_hat_HAC = res_hac.cov_params()  
    SEs = np.sqrt(np.diag(V_hat_HAC))
    return SEs

# perform t-test using standard normal critical values
def t_test(input_alpha, input_t):
    t_crit_upper = round(norm.ppf(1-input_alpha/2),3)
    t_crit_lower = round(norm.ppf(input_alpha/2),3)
    
    if t_crit_lower < input_t < t_crit_upper:
        print(f't ∈ [{t_crit_lower},{t_crit_upper}]: Do not reject the null.')
    else:
        print(f't ∉ [{t_crit_lower},{t_crit_upper}]: Reject the null.')
        
'''
Exercise 1a
'''

print('Exercise 1a: t-test based on real data')

time_list = np.linspace(1, len(annualAverage), len(annualAverage))

# estimate the model using OLS regression
matrix_X_original, estimate_beta_original, hat_y, hat_residuals = runRegressionModel(annualAverage, time_list)

theta_n = estimate_beta_original.iat[1,0]

# compute the HAC standard errors
estimate_SE = getHACStandardError(matrix_X_original, annualAverage)

# compute t-statistic for β_1 under the null 
SE_beta_1_original = estimate_SE[1]
t_statistic_beta_1 = estimate_beta_original.iat[1,0] / SE_beta_1_original

# significance level
alpha = 0.05

# perform a t-test using the annual averages for β_1
print(f't-test for β_1 = {round(t_statistic_beta_1,3)}:', end = ' ')
t_test(alpha, t_statistic_beta_1)

print('')
'''
Exercise 1b: perform the t-test using the different bootstrap methods
'''
print('Exercise 1b: t-test using different bootstrap methods')

numberOfSimulations = 999

# calculation for bootstrap p-value
def computeBootstrapPValue(input_t_list, observed_t):
    result_p = 0    

    for bootstrap_t in input_t_list:
        if abs(bootstrap_t) > abs(observed_t):
            result_p += 1
            
    result_p = round(result_p / len(input_t_list),3)

    return result_p

# algorithm using Nonparametric residual bootstrap
def perform_iidBootstrap(input_numberOfSimulation, input_x, input_y, input_matrixX, input_beta, boolean_underTheNull):
    bootstrap_beta1_list = []
    bootstrap_t_list = []
    
    if boolean_underTheNull:
        input_beta.iloc[1,0] = 0
        
    hat_beta_1 = input_beta.iat[1,0]
    
    # obtain residuals under the null
    hat_y = input_matrixX @ input_beta
    hat_residuals = np.array(input_y) - np.array(hat_y.squeeze())
    
    for b in range(0,input_numberOfSimulation):
        # simulate new_residuals as random variables from the empirical distribution
        simulate_residuals = np.random.choice(hat_residuals, len(hat_residuals), replace=True)
        mean_hat_residuals = np.average(hat_residuals)
        new_residuals = simulate_residuals - mean_hat_residuals

        # calculate new_y 
        new_y = np.array(hat_y.squeeze()) + new_residuals

        # re-estimate the model
        matrix_X_star, estimate_beta_star, hat_y_star, hat_residuals_star = runRegressionModel(new_y, input_x)

        # compute the standard error of β_1
        bootstrap_estimate_SE = getStandardError(matrix_X_star, new_y)
        bootstrap_SE_beta_1 = bootstrap_estimate_SE[1]

        # extract the estimate β_1
        bootstrap_beta_1 = estimate_beta_star.iloc[1,0]
        bootstrap_beta1_list.append(bootstrap_beta_1 - hat_beta_1)
        
        # compute the bootstrap t-statistic for β_1 
        bootstrap_t_statistic_beta_1 = (bootstrap_beta_1 - hat_beta_1)/ bootstrap_SE_beta_1
        bootstrap_t_list.append(bootstrap_t_statistic_beta_1)
    
    return bootstrap_beta1_list, bootstrap_t_list

# algorithm using Wild bootstrap
def perform_wildBootstrap(errorDistribution, input_numberOfSimulation, input_x, input_y, input_matrixX, input_beta, boolean_underTheNull):
    bootstrap_beta1_list = []
    bootstrap_t_list = []
    
    # set β_1 = 0 if performing bootstrap under the null
    if boolean_underTheNull:
        input_beta.iloc[1,0] = 0
    
    # extract the β_1
    hat_beta_1 = input_beta.iat[1,0]
        
    # obtain residuals under the null
    hat_y = input_matrixX @ input_beta
    hat_residuals = np.array(input_y) - np.array(hat_y.squeeze())
    
    for b in range(0,input_numberOfSimulation):
        # simulate new_residuals bootstrap errors
        if errorDistribution == 'Standard Normal':
            new_error = np.random.normal(loc=0, scale=1, size=len(hat_residuals)) # Standard normal distribution
        elif errorDistribution == 'Rademacher':
            new_error = np.random.binomial(n=1, p=0.5, size=len(hat_residuals)) # Rademacher distribution
        new_residuals = hat_residuals * (2 * new_error - 1)

        # calculate new_y 
        new_y = np.array(hat_y.squeeze()) + new_residuals

        # re-estimate the model
        matrix_X_star, estimate_beta_star, hat_y_star, hat_residuals_star = runRegressionModel(new_y, input_x)

        # compute the standard error of β_1
        bootstrap_estimate_SE = getHCStandardError(matrix_X_star, hat_residuals_star.squeeze())
        bootstrap_SE_beta_1 = bootstrap_estimate_SE[1]
        
        # extract the estimate β_1
        bootstrap_beta_1 = estimate_beta_star.iloc[1,0]
        bootstrap_beta1_list.append(bootstrap_beta_1 - hat_beta_1)

        # compute the bootstrap t-statistic for β_1
        bootstrap_t_statistic_beta_1 = (bootstrap_beta_1 - hat_beta_1) / bootstrap_SE_beta_1
        bootstrap_t_list.append(bootstrap_t_statistic_beta_1)
    
    return bootstrap_beta1_list, bootstrap_t_list

# obtain the sequence of simulated test statistic using Nonparametric residual bootstrap
bootstrap_beta1_list_iid_annual, bootstrap_t_list_iid_annual = perform_iidBootstrap(numberOfSimulations, time_list, annualAverage, matrix_X_original, estimate_beta_original, True)
# compute the bootstrap p-value       
bootstrap_p = computeBootstrapPValue(bootstrap_t_list_iid_annual, t_statistic_beta_1)
print(f'i.i.d. bootstrap: The bootstrap p-value is {bootstrap_p}.') 

# obtain the sequence of simulated test statistic using Wild bootstrap
bootstrap_beta1_list_wild_annual, bootstrap_t_list_wild_annual = perform_wildBootstrap('Standard Normal', numberOfSimulations, time_list, annualAverage, matrix_X_original, estimate_beta_original, True)
# compute the bootstrap p-value       
bootstrap_p = computeBootstrapPValue(bootstrap_t_list_wild_annual, t_statistic_beta_1)
print(f'Wild bootstrap: The bootstrap p-value is {bootstrap_p}.') 

# Block bootstrap

# Sieve bootstrap

print('')
'''
Exercise 2a: perform the t-test using the winter and summer averages
'''
print('Exercise 2a: Winter Average')

# estimate the model using OLS regression
matrix_X, estimate_beta, hat_y, hat_residuals = runRegressionModel(winterAverage, time_list)

# report the estimated coefficients
print(f'Winter: β_0 = {round(estimate_beta.iat[0,0],3)} & β_1 = {round(estimate_beta.iat[1,0],3)}')

# compute the HAC standard errors
estimate_SE = getHACStandardError(matrix_X, winterAverage)

# compute t-statistic for β_1 under the null 
SE_beta_1 = estimate_SE[1]
t_statistic_beta_1 = estimate_beta.iat[1,0] / SE_beta_1

# significance level
alpha = 0.1

# perform a t-test using the annual averages for β_1
print(f't-test for β_1 = {round(t_statistic_beta_1,3)}:', end = ' ')
t_test(alpha, t_statistic_beta_1)

# obtain the sequence of simulated test statistic using Nonparametric residual bootstrap
bootstrap_beta1_list_iid_winter, bootstrap_t_list_iid_winter = perform_iidBootstrap(numberOfSimulations, time_list, winterAverage, matrix_X, estimate_beta, True)
# compute the bootstrap p-value       
bootstrap_p = computeBootstrapPValue(bootstrap_t_list_iid_winter, t_statistic_beta_1)
print(f'i.i.d. bootstrap: The bootstrap p-value is {bootstrap_p}.') 

# obtain the sequence of simulated test statistic using Wild bootstrap
bootstrap_beta1_list_wild_winter, bootstrap_t_list_wild_winter = perform_wildBootstrap('Standard Normal', numberOfSimulations, time_list, winterAverage, matrix_X, estimate_beta, True)
# compute the bootstrap p-value       
bootstrap_p = computeBootstrapPValue(bootstrap_t_list_wild_winter, t_statistic_beta_1)
print(f'Wild bootstrap: The bootstrap p-value is {bootstrap_p}.') 

# Block bootstrap

# Sieve bootstrap

print('')
print('Exercise 2a: Summer Average')

# estimate the model using OLS regression
matrix_X, estimate_beta, hat_y, hat_residuals = runRegressionModel(summerAverage, time_list)

# report the estimated coefficients
print(f'Summer: β_0 = {round(estimate_beta.iat[0,0],3)} & β_1 = {round(estimate_beta.iat[1,0],3)}')

# compute the HAC standard errors
estimate_SE = getHACStandardError(matrix_X, summerAverage)

# compute t-statistic for β_1 under the null 
SE_beta_1 = estimate_SE[1]
t_statistic_beta_1 = estimate_beta.iat[1,0] / SE_beta_1

# significance level
alpha = 0.05

# perform a t-test using the annual averages for β_1
print(f't-test for β_1 = {round(t_statistic_beta_1,3)}:', end = ' ')
t_test(alpha, t_statistic_beta_1)

# obtain the sequence of simulated test statistic using Nonparametric residual bootstrap
bootstrap_beta1_list_iid_summer, bootstrap_t_list_iid_summer = perform_iidBootstrap(numberOfSimulations, time_list, summerAverage, matrix_X, estimate_beta, True)
# compute the bootstrap p-value       
bootstrap_p = computeBootstrapPValue(bootstrap_t_list_iid_summer, t_statistic_beta_1)
print(f'i.i.d. bootstrap: The bootstrap p-value is {bootstrap_p}.') 

# obtain the sequence of simulated test statistic using Wild bootstrap
bootstrap_beta1_list_wild_summer, bootstrap_t_list_wild_summer = perform_wildBootstrap('Standard Normal', numberOfSimulations, time_list, summerAverage, matrix_X, estimate_beta, True)
# compute the bootstrap p-value       
bootstrap_p = computeBootstrapPValue(bootstrap_t_list_wild_summer, t_statistic_beta_1)
print(f'Wild bootstrap: The bootstrap p-value is {bootstrap_p}.') 

# Block bootstrap

# Sieve bootstrap

print('')
'''
Exercise 3a: construct the confidence intervals
'''
print('Exercise 3a: confidence intervals')

# calculation for quantile values
def getQuantile(c_1_quantile, c_2_quantile, resultList):
    c_1 = np.quantile(resultList, c_1_quantile)
    c_2 = np.quantile(resultList, c_2_quantile)
    
    return c_1, c_2

# construct the equal-tailed percentile interval
def getEqualTailedPercentileCI(input_alpha, input_bootstrap_list, input_theta):  
    c_1_quantile = input_alpha/2
    c_2_quantile = 1 - input_alpha/2
    
    cStar_1, cStar_2 = getQuantile(c_1_quantile, c_2_quantile, input_bootstrap_list)
    
    lowerBound = float(input_theta - cStar_2)
    upperBound = float(input_theta - cStar_1)
    
    result_CI = [lowerBound, upperBound]
    
    return result_CI

# construct the equal-tailed percentile-t interval
def getEqualTailedPercentileTCI(input_alpha, input_bootstrap_list, input_theta, input_theta_SE):  
    c_1_quantile = input_alpha/2
    c_2_quantile = 1 - input_alpha/2
    
    cStar_1, cStar_2 = getQuantile(c_1_quantile, c_2_quantile, input_bootstrap_list)
    
    lowerBound = float(input_theta - cStar_2 * input_theta_SE)
    upperBound = float(input_theta - cStar_1 * input_theta_SE)
    
    result_CI = [lowerBound, upperBound]
    
    return result_CI

# construct the symmetric percentile interval
def getSymmetricPercentileCI(input_alpha, input_bootstrap_list, input_theta):  
    c_1_quantile = input_alpha/2
    c_2_quantile = 1 - input_alpha/2
    
    cStar_1, cStar_2 = getQuantile(c_1_quantile, c_2_quantile, input_bootstrap_list)
    
    lowerBound = float(input_theta - cStar_2)
    upperBound = float(input_theta + cStar_2)
    
    result_CI = [lowerBound, upperBound]
    
    return result_CI

# construct the symmetric percentile-t interval
def getSymmetricPercentileTCI(input_alpha, input_bootstrap_list, input_theta, input_theta_SE):  
    c_1_quantile = input_alpha/2
    c_2_quantile = 1 - input_alpha/2
    
    cStar_1, cStar_2 = getQuantile(c_1_quantile, c_2_quantile, input_bootstrap_list)
    
    lowerBound = float(input_theta - cStar_2 * input_theta_SE)
    upperBound = float(input_theta + cStar_2 * input_theta_SE)
    
    result_CI = [lowerBound, upperBound]
    
    return result_CI

# obtain the sequence of simulated test statistic using iid Bootstrap (not under the null)
bootstrap_beta1_list_iid_annual, bootstrap_t_list_iid_annual = perform_iidBootstrap(numberOfSimulations, time_list, annualAverage, matrix_X_original, estimate_beta_original, False)

# obtain the sequence of simulated test statistic using Wild Bootstrap (not under the null)
bootstrap_beta1_list_wild_annual, bootstrap_t_list_wild_annual = perform_wildBootstrap('Standard Normal', numberOfSimulations, time_list, annualAverage, matrix_X_original, estimate_beta_original, False)

# Nonparametric residual bootstrap
print('i.i.d. bootstrap:')
etpi_iid = getEqualTailedPercentileCI(alpha, bootstrap_beta1_list_iid_annual, theta_n)
print(f'The equal-tailed percentile interval is {np.round(etpi_iid,3)}.')
etpti_iid = getEqualTailedPercentileTCI(alpha, bootstrap_t_list_iid_annual, theta_n ,SE_beta_1_original)
print(f'The equal-tailed percentile-t interval is {np.round(etpti_iid,3)}.')
spi_iid = getSymmetricPercentileCI(alpha, np.abs(bootstrap_beta1_list_iid_annual), theta_n)
print(f'The symmetric percentile interval is {np.round(spi_iid,3)}.')
spti_iid = getSymmetricPercentileTCI(alpha, np.abs(bootstrap_t_list_iid_annual), theta_n ,SE_beta_1_original)
print(f'The symmetric percentile-t interval is {np.round(etpti_iid,3)}.')

# Wild bootstrap
print('Wild bootstrap:')
etpi_wild = getEqualTailedPercentileCI(alpha, bootstrap_beta1_list_wild_annual, theta_n)
print(f'The equal-tailed percentile interval is {np.round(etpi_wild,3)}.')
etpti_wild = getEqualTailedPercentileTCI(alpha, bootstrap_t_list_wild_annual, theta_n ,SE_beta_1_original)
print(f'The equal-tailed percentile-t interval is {np.round(etpti_wild,3)}.')
spi_wild = getSymmetricPercentileCI(alpha, np.abs(bootstrap_beta1_list_wild_annual), theta_n)
print(f'The symmetric percentile interval is {np.round(spi_wild,3)}.')
spti_wild = getSymmetricPercentileTCI(alpha, np.abs(bootstrap_t_list_wild_annual), theta_n ,SE_beta_1_original)
print(f'The symmetric percentile-t interval is {np.round(spti_wild,3)}.')

# Block bootstrap

# Sieve bootstrap

print('')
'''
Exercise 4: the empirical coverages
'''
print('Exercise 4: the empirical coverages')

# calculation for the empirical coverage
def checkCoverage(input_ci_list, input_trueBeta):
    coverageCounter = 0
    
    for ci in input_ci_list:
        lowerBound = ci[0]
        upperBound = ci[1]
        
        if lowerBound <= input_trueBeta <= upperBound:
            coverageCounter += 1
            
    coverageCounter = coverageCounter / (len(input_ci_list))
    
    return coverageCounter

# given parameter
beta_0 = 1
true_beta_1 = 0.5
n = 500         # length of the time series
capitalN = 50   # number of simulation for Monte Carlo

new_time_list = np.linspace(1, n, n)

iid_etpt_ci_list = []
iid_sp_ci_list = []
wild_etpt_ci_list = []
wild_sp_ci_list = []

# obtain confidence intervals from each bootstrap
for i in range(0, capitalN):
    errorTerms_list = []
        
    # generate specific error terms which are depending on time
    for t in range(1, n+1):
        this_errorTerms = np.random.normal(loc=0, scale=np.sqrt(t), size=1)
        errorTerms_list.append(float(this_errorTerms[0]))
        
    generatedYList = []

    # generate data from model with specific error terms
    for t in range(1, n+1):
        this_y = beta_0 + true_beta_1 * t + float(errorTerms_list[t-1])
        generatedYList.append(this_y)
    
    # estimate the model using OLS regression
    matrix_X, estimate_beta, hat_y, hat_residuals = runRegressionModel(generatedYList, new_time_list)

    theta_n_newY = estimate_beta.iat[1,0]
    
    # compute the HAC standard errors
    estimate_SE = getHACStandardError(matrix_X, generatedYList)

    # compute standard error for β_1 
    this_SE_beta_1 = estimate_SE[1]
    
    # print(f'{i}: ',end='')
    
    # obtain the sequence of simulated test statistic using Nonparametric residual bootstrap
    bootstrap_beta1_list_iid_newY, bootstrap_t_list_iid_newY = perform_iidBootstrap(numberOfSimulations, new_time_list, generatedYList, matrix_X, estimate_beta, False)

    # obtain the Equal-Tailed Percentile-t and Symmetric Percentile bootstrap confidence interval
    etpti_iid = getEqualTailedPercentileTCI(alpha, bootstrap_t_list_iid_newY, theta_n_newY , this_SE_beta_1)
    iid_etpt_ci_list.append(etpti_iid)
    spi_iid = getSymmetricPercentileCI(alpha, np.abs(bootstrap_beta1_list_iid_newY), theta_n_newY)
    iid_sp_ci_list.append(spi_iid)
    # print(f'iid: {etpti_iid}', end=', ')
    
    # obtain the sequence of simulated test statistic using Wild bootstrap
    bootstrap_beta1_list_wild_newY, bootstrap_t_list_wild_newY = perform_wildBootstrap('Standard Normal', numberOfSimulations, new_time_list, generatedYList, matrix_X, estimate_beta, False)
    # obtain the Equal-Tailed Percentile-t and Symmetric Percentile bootstrap confidence interval
    etpti_wild = getEqualTailedPercentileTCI(alpha, bootstrap_t_list_wild_newY, theta_n_newY ,this_SE_beta_1)
    wild_etpt_ci_list.append(etpti_wild)
    spi_wild = getSymmetricPercentileCI(alpha, np.abs(bootstrap_beta1_list_wild_newY), theta_n_newY)
    wild_sp_ci_list.append(spi_wild)
    # print(f'Wild: {etpti_wild}')

# compute the empirical coverage of Equal-Tailed Percentile-t Interval using iid and Wild Bootstrap 
iid_etpt_Coverage = checkCoverage(iid_etpt_ci_list, true_beta_1)
wild_etpt_Coverage = checkCoverage(wild_etpt_ci_list, true_beta_1)
print(f'The empirical coverage of the equal-tailed percentile-t Interval using iid Bootstrap is {round(iid_etpt_Coverage, 3)}.')
print(f'The empirical coverage of the equal-tailed percentile-t Interval using Wild Bootstrap is {round(wild_etpt_Coverage, 3)}.')

# compute the empirical coverage of Symmetric Percentile Interval using iid and Wild Bootstrap
iid_sp_Coverage = checkCoverage(iid_sp_ci_list, true_beta_1)
wild_sp_Coverage = checkCoverage(wild_sp_ci_list, true_beta_1)
print(f'The empirical coverage of the symmetric percentile Interval using iid Bootstrap is {round(iid_sp_Coverage, 3)}.')
print(f'The empirical coverage of the symmetric percentile Interval using Wild Bootstrap is {round(wild_sp_Coverage, 3)}.')