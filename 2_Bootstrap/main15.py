import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import norm
import time

start_time = time.time()   # record start time

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
    varCovMatrix = estimated_variance.iat[0,0] * df_inv

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
    # set the maximum lags
    maxlags = int(np.floor(4 * (n/100)**(2/9)))
    
    # estimate the model and get the HAC residuals
    res_hac = sm.OLS(input_y, input_matrix).fit(cov_type='HAC', 
                                                cov_kwds={'maxlags': maxlags, 'use_correction': True})
    
    # compute the HAC variance-covariance matrix
    V_hat_HAC = res_hac.cov_params()  
    
    # get the standard errors which are the diagonal element
    estimated_standard_error = np.sqrt(np.diag(V_hat_HAC))
    
    return estimated_standard_error

# perform t-test using standard normal critical values
def t_test(input_alpha, input_t):
    t_crit_upper = round(norm.ppf(1-input_alpha/2),3)
    t_crit_lower = round(norm.ppf(input_alpha/2),3)
    
    if t_crit_lower < input_t < t_crit_upper:
        print(f't ∈ [{t_crit_lower},{t_crit_upper}]: Do not reject the null.')
    else:
        print(f't ∉ [{t_crit_lower},{t_crit_upper}]: Reject the null.')
        
'''
Exercise 1a: t-test based on real data
'''
print('Exercise 1a: t-test based on real data')

# define the list of t
time_list = np.linspace(1, len(annualAverage), len(annualAverage))

# estimate the model using OLS regression
matrix_X_original, estimate_beta_original, hat_y, hat_residuals = runRegressionModel(annualAverage, time_list)
# report the estimated coefficients
print(f'Annual: β_0 = {round(estimate_beta_original.iat[0,0],3)} & β_1 = {round(estimate_beta_original.iat[1,0],3)}')

# extract the estimated beta from actual data
theta_n = estimate_beta_original.iat[1,0]

# compute the standard errors
estimate_HACSE_original = getHACStandardError(matrix_X_original, annualAverage)[1]
estimate_SE_original = getStandardError(matrix_X_original, hat_residuals)[1]
estimate_HCSE_original = getHCStandardError(matrix_X_original, hat_residuals.squeeze())[1]

# compute t-statistic for β_1 under the null
t_statistic_beta_1 = estimate_beta_original.iat[1,0] / estimate_HACSE_original

alpha = 0.1 # significance level

# perform a t-test using the annual averages for β_1
print(f't-test for β_1 = {round(t_statistic_beta_1,3)}:', end = ' ')
t_test(alpha, t_statistic_beta_1)

'''
Exercise 1b: perform the t-test using the different bootstrap methods
'''
print('\nExercise 1b: t-test using different bootstrap methods')

numberOfSimulations = 999

# 
def pValue_approach(input_p_value, input_alpha):
    if input_p_value > input_alpha:
        print(f'p ∉ rejection region: Do not reject the null.')
    else:
        print(f'p ∈ rejection region: Reject the null.')
        
# calculation for bootstrap p-value
def computeBootstrapPValue(input_t_list, observed_t):
    result_p = 0    

    for bootstrap_t in input_t_list:
        if abs(bootstrap_t) > abs(observed_t):
            result_p += 1
            
    result_p = round(result_p / len(input_t_list),3)

    return result_p

# computation of statistics and storage of the bootstrap distribution
def computeAndStoreResults(input_beta1_list, input_t_list, input_beta, input_hat_beta_1, input_SE):
    # extract the bootstrap estimate β_1
    bootstrap_beta_1 = input_beta.iloc[1,0]
    input_beta1_list.append(float(bootstrap_beta_1 - input_hat_beta_1))
    
    # compute the bootstrap t-statistic for β_1 
    bootstrap_t_statistic_beta_1 = (bootstrap_beta_1 - input_hat_beta_1)/ input_SE
    input_t_list.append(float(bootstrap_t_statistic_beta_1))
    
    return input_beta1_list, input_t_list

# algorithm using Nonparametric residual bootstrap
def perform_iidBootstrap(input_numberOfSimulation, input_x, input_y, input_matrixX, input_beta, boolean_underTheNull):
    bootstrap_beta1_list = []
    bootstrap_t_list = []
    
    this_beta = input_beta.copy()
    
    # restricted β_1 = 0 if performing bootstrap under the null
    if boolean_underTheNull:
        this_beta.iloc[1,0] = 0
    
    # extract the β_1
    hat_beta_1 = this_beta.iat[1,0]
    
    # obtain the residuals
    hat_y = input_matrixX @ this_beta
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
        bootstrap_estimate_SE = getStandardError(matrix_X_star, hat_residuals_star) # use simple standard error
        bootstrap_SE_beta_1 = bootstrap_estimate_SE[1]

        # compute and store the bootstrap estimate and the bootstrap t-statistic for β_1 
        bootstrap_beta1_list, bootstrap_t_list = computeAndStoreResults(input_beta1_list=bootstrap_beta1_list, 
                                                                        input_t_list=bootstrap_t_list,
                                                                        input_beta=estimate_beta_star, input_hat_beta_1=hat_beta_1, 
                                                                        input_SE=bootstrap_SE_beta_1)
        
    return bootstrap_beta1_list, bootstrap_t_list

# algorithm using Wild bootstrap
def perform_wildBootstrap(errorDistribution, input_numberOfSimulation, input_x, input_y, input_matrixX, input_beta, boolean_underTheNull):
    bootstrap_beta1_list = []
    bootstrap_t_list = []
    
    this_beta = input_beta.copy()
    
    # restricted β_1 = 0 if performing bootstrap under the null
    if boolean_underTheNull:
        this_beta.iloc[1,0] = 0
    
    # extract the β_1
    hat_beta_1 = this_beta.iat[1,0]
        
    # obtain the residuals
    hat_y = input_matrixX @ this_beta
    hat_residuals = np.array(input_y) - np.array(hat_y.squeeze())
    
    for b in range(0,input_numberOfSimulation):
        # simulate new_residuals bootstrap errors
        if errorDistribution == 'Standard Normal':
            new_error = np.random.normal(loc=0, scale=1, size=len(hat_residuals)) # Standard normal distribution
            new_residuals = hat_residuals * new_error
        elif errorDistribution == 'Rademacher':
            new_error = np.random.binomial(n=1, p=0.5, size=len(hat_residuals)) # Rademacher distribution
            new_residuals = hat_residuals * (2 * new_error - 1)

        # calculate new_y 
        new_y = np.array(hat_y.squeeze()) + new_residuals

        # re-estimate the model
        matrix_X_star, estimate_beta_star, hat_y_star, hat_residuals_star = runRegressionModel(new_y, input_x)

        # compute the standard error of β_1
        bootstrap_estimate_SE = getHCStandardError(matrix_X_star, hat_residuals_star.squeeze()) # use HC standard error
        bootstrap_SE_beta_1 = bootstrap_estimate_SE[1]
        
        # compute and store the bootstrap estimate and the bootstrap t-statistic for β_1 
        bootstrap_beta1_list, bootstrap_t_list = computeAndStoreResults(input_beta1_list=bootstrap_beta1_list, 
                                                                        input_t_list=bootstrap_t_list,
                                                                        input_beta=estimate_beta_star, input_hat_beta_1=hat_beta_1, 
                                                                        input_SE=bootstrap_SE_beta_1)
    
    return bootstrap_beta1_list, bootstrap_t_list

# calculation for the block length
def calculate_L(n):
    return max(1, int(np.floor(n / 10)))

# methodology of building moving-blocks
def build_blocks(input_residuals, input_l):
    n = len(input_residuals)
    blocks = []
    
    for i in range(0, n - input_l + 1):
        window = input_residuals[i:i+input_l]
        blocks.append(window)
        
    return np.vstack(blocks)

# algorithm using Block Bootstrap
def perform_blockBootstrap(input_numberOfSimulation, input_x, input_y, input_matrixX, input_beta, boolean_underTheNull):
    bootstrap_beta1_list = []
    bootstrap_t_list = []
    
    this_beta = input_beta.copy()
    
    # restricted β_1 = 0 if performing bootstrap under the null
    if boolean_underTheNull:
        this_beta.iloc[1,0] = 0
    
    # extract the β_1
    hat_beta_1 = this_beta.iat[1,0]
    
    # obtain the residuals
    hat_y = input_matrixX @ this_beta
    hat_residuals = np.array(input_y) - np.array(hat_y.squeeze())
    
    # divide the residuals into overlapping blocks
    n = len(hat_residuals)
    blockLength = calculate_L(n)
    blocksList = build_blocks(hat_residuals, blockLength)
    
    numberOfBlocks = len(blocksList)
    k = int(np.ceil(n / blockLength))
    
    for b in range(0,input_numberOfSimulation):
        # draw randomly and with replacement from the blocks list
        selectedBlocksIndex = np.random.randint(0, numberOfBlocks - 1, k)
        selected_blocks = []
        
        # generate the new bootstrap residuals
        for blockIndex in selectedBlocksIndex:
            selected_blocks.append(blocksList[blockIndex])
        bootstrap_residuals = np.concatenate(selected_blocks)[:n]
        
        # calculate new_y
        new_y = np.array(hat_y.squeeze()) + np.array(bootstrap_residuals)
        
        # re-estimate the model
        matrix_X_star, estimate_beta_star, hat_y_star, hat_residuals_star = runRegressionModel(new_y, input_x)
        
        # compute the standard error of β_1
        bootstrap_estimate_SE = getHACStandardError(matrix_X_star, new_y) # use HAC standard error
        bootstrap_SE_beta_1 = bootstrap_estimate_SE[1]
        
        # compute and store the bootstrap estimate and the bootstrap t-statistic for β_1 
        bootstrap_beta1_list, bootstrap_t_list = computeAndStoreResults(input_beta1_list=bootstrap_beta1_list, 
                                                                        input_t_list=bootstrap_t_list,
                                                                        input_beta=estimate_beta_star, input_hat_beta_1=hat_beta_1, 
                                                                        input_SE=bootstrap_SE_beta_1)
        
    return bootstrap_beta1_list, bootstrap_t_list

def generateMatrixX(input_y, input_p, booleanIntercept):
    numberOfRows = len(input_y) - input_p
    result_matrix = []
    
    for rowNumber in range(0,numberOfRows):
        row = []
        
        if booleanIntercept:
            row.append(1)
        
        for t in range(0, input_p):
            row.append(input_y[input_p + rowNumber - t - 1])
    
        result_matrix.append(row)
     
    return pd.DataFrame(result_matrix)

def runRegressionModel_AR(input_y, input_p, booleanIntercept):
    # convert y to matrix y
    vector_y = pd.DataFrame(input_y[input_p:])
    vector_y = vector_y.reset_index(drop=True)
    # convert y to matrix X
    matrix_X = generateMatrixX(input_y, input_p, booleanIntercept)
    
    # calculate estimated beta by (X′ * X)^{−1} * X′* y
    estimate_beta = getEstimatedBeta(matrix_X, vector_y)

    # calculate estimated y by X * (estimated beta)
    estimate_y = matrix_X @ estimate_beta
    
    # calculate the residuals
    estimate_residuals = vector_y - estimate_y

    return estimate_beta.squeeze(), estimate_y.squeeze(), estimate_residuals.squeeze()

# algorithm using Sieve Bootstrap
def perform_sieveBootstrap(input_numberOfSimulation, input_x, input_y, input_matrixX, input_beta, boolean_underTheNull):
    bootstrap_beta1_list = []
    bootstrap_t_list = []

    this_beta = input_beta.copy()
    
    # restricted β_1 = 0 if performing bootstrap under the null
    if boolean_underTheNull:
        this_beta.iloc[1,0] = 0
    
    # extract the β_1
    hat_beta_1 = this_beta.iat[1,0]

    # obtain the residuals
    hat_y = input_matrixX @ this_beta
    hat_residuals = np.array(input_y) - np.array(hat_y.squeeze())

    # define the number of lags in AR(p) model
    default_p = 5
    
    # estimate an AR(p) model of ε_t
    estimate_rho_star, estimated_residuals, hat_eta = runRegressionModel_AR(input_y=hat_residuals, 
                                                                            input_p=default_p, booleanIntercept=False)

    for b in range(input_numberOfSimulation):
        # simulate new residuals as random variables from the empirical distribution of η
        eta_star = np.random.choice(hat_eta, len(hat_residuals), replace=True)
        # recenter the simulated values
        mean_eta_hat = np.average(hat_eta)
        eta_star_centered = eta_star - mean_eta_hat
        
        new_residuals_list = []
        
        # generate the new bootstrap residuals as the simulated innovations 
        for t in range(0, len(hat_residuals)):
            if t < default_p:
                new_residuals_list.append(hat_residuals[t])
            else: 
                this_residuals = estimated_residuals[t - default_p] + eta_star_centered[t - default_p]
                new_residuals_list.append(this_residuals)

        # calculate new_y
        new_y = np.array(hat_y.squeeze()) + np.array(new_residuals_list)

        # re-estimate the model
        matrix_X_star, estimate_beta_star, hat_y_star, hat_residuals_star = runRegressionModel(new_y, input_x)

        # compute the standard error of β_1
        bootstrap_estimate_SE = getHACStandardError(matrix_X_star, new_y) # use HAC standard error
        bootstrap_SE_beta_1 = bootstrap_estimate_SE[1]

        # compute and store the bootstrap estimate and the bootstrap t-statistic for β_1 
        bootstrap_beta1_list, bootstrap_t_list = computeAndStoreResults(input_beta1_list=bootstrap_beta1_list, 
                                                                        input_t_list=bootstrap_t_list,
                                                                        input_beta=estimate_beta_star, input_hat_beta_1=hat_beta_1, 
                                                                        input_SE=bootstrap_SE_beta_1)
        
    return bootstrap_beta1_list, bootstrap_t_list

# obtain the sequence of simulated test statistic using Nonparametric residual bootstrap
bootstrap_beta1_list_iid_annual, bootstrap_t_list_iid_annual = perform_iidBootstrap(input_numberOfSimulation=numberOfSimulations, 
                                                                                    input_x=time_list, input_y=annualAverage, 
                                                                                    input_matrixX=matrix_X_original, input_beta=estimate_beta_original, 
                                                                                    boolean_underTheNull=True)
# compute the bootstrap p-value       
bootstrap_p = computeBootstrapPValue(bootstrap_t_list_iid_annual, t_statistic_beta_1)
print(f'i.i.d. bootstrap: \nThe bootstrap p-value is {bootstrap_p}. ', end='') 
pValue_approach(input_p_value=bootstrap_p, input_alpha=alpha)

# obtain the sequence of simulated test statistic using Wild bootstrap using Standard Normal distribution
bootstrap_beta1_list_wild_annual, bootstrap_t_list_wild_annual = perform_wildBootstrap(errorDistribution='Standard Normal', 
                                                                                       input_numberOfSimulation=numberOfSimulations, 
                                                                                       input_x=time_list, input_y=annualAverage,
                                                                                       input_matrixX=matrix_X_original, input_beta=estimate_beta_original, 
                                                                                       boolean_underTheNull=True)
# compute the bootstrap p-value       
bootstrap_p = computeBootstrapPValue(bootstrap_t_list_wild_annual, t_statistic_beta_1)
print(f'Wild bootstrap (Standard Normal): \nThe bootstrap p-value is {bootstrap_p}. ', end='') 
pValue_approach(input_p_value=bootstrap_p, input_alpha=alpha) 

# obtain the sequence of simulated test statistic using Wild bootstrap using Rademacher distribution
bootstrap_beta1_list_wild_annual, bootstrap_t_list_wild_annual = perform_wildBootstrap(errorDistribution='Rademacher', 
                                                                                       input_numberOfSimulation=numberOfSimulations, 
                                                                                       input_x=time_list, input_y=annualAverage,
                                                                                       input_matrixX=matrix_X_original, input_beta=estimate_beta_original, 
                                                                                       boolean_underTheNull=True)
# compute the bootstrap p-value       
bootstrap_p = computeBootstrapPValue(bootstrap_t_list_wild_annual, t_statistic_beta_1)
print(f'Wild bootstrap (Rademacher): \nThe bootstrap p-value is {bootstrap_p}. ', end='') 
pValue_approach(input_p_value=bootstrap_p, input_alpha=alpha) 

# obtain the sequence of simulated test statistic using Block bootstrap
bootstrap_beta1_list_block_annual, bootstrap_t_list_block_annual = perform_blockBootstrap(input_numberOfSimulation=numberOfSimulations, 
                                                                                        input_x=time_list, input_y=annualAverage,
                                                                                        input_matrixX=matrix_X_original, input_beta=estimate_beta_original, 
                                                                                        boolean_underTheNull=True)
bootstrap_p = computeBootstrapPValue(bootstrap_t_list_block_annual, t_statistic_beta_1)
print(f'Block bootstrap: \nThe bootstrap p-value is {bootstrap_p}. ', end='') 
pValue_approach(input_p_value=bootstrap_p, input_alpha=alpha)

# obtain the sequence of simulated test statistic using Sieve bootstrap
bootstrap_beta1_list_sieve_annual, bootstrap_t_list_sieve_annual = perform_sieveBootstrap(input_numberOfSimulation=numberOfSimulations, 
                                                                                          input_x=time_list, input_y=annualAverage, 
                                                                                          input_matrixX=matrix_X_original, input_beta=estimate_beta_original, 
                                                                                          boolean_underTheNull=True)
bootstrap_p = computeBootstrapPValue(bootstrap_t_list_sieve_annual, t_statistic_beta_1)
print(f'Sieve bootstrap: \nThe bootstrap p-value is {bootstrap_p}. ', end='') 
pValue_approach(input_p_value=bootstrap_p, input_alpha=alpha)

'''
Exercise 2a: perform the t-test using the winter and summer averages
'''
print('\nExercise 2a: Winter Average')

# estimate the model using OLS regression
matrix_X, estimate_beta, hat_y, hat_residuals = runRegressionModel(winterAverage, time_list)
# report the estimated coefficients
print(f'Winter: β_0 = {round(estimate_beta.iat[0,0],3)} & β_1 = {round(estimate_beta.iat[1,0],3)}')

# compute the HAC standard errors
estimate_SE = getHACStandardError(matrix_X, winterAverage)

# compute t-statistic for β_1 under the null 
SE_beta_1 = estimate_SE[1]
t_statistic_beta_1 = estimate_beta.iat[1,0] / SE_beta_1

# perform a t-test using the winter averages for β_1
alpha = 0.1 # significance level
print(f't-test for β_1 = {round(t_statistic_beta_1,3)}:', end = ' ')
t_test(alpha, t_statistic_beta_1)

# obtain the sequence of simulated test statistic using Nonparametric residual bootstrap
bootstrap_beta1_list_iid_winter, bootstrap_t_list_iid_winter = perform_iidBootstrap(input_numberOfSimulation=numberOfSimulations, 
                                                                                    input_x=time_list, input_y=winterAverage, 
                                                                                    input_matrixX=matrix_X, input_beta=estimate_beta, 
                                                                                    boolean_underTheNull=True)
# compute the bootstrap p-value       
bootstrap_p = computeBootstrapPValue(bootstrap_t_list_iid_winter, t_statistic_beta_1)
print(f'i.i.d. bootstrap: \nThe bootstrap p-value is {bootstrap_p}. ', end='') 
pValue_approach(input_p_value=bootstrap_p, input_alpha=alpha)

# obtain the sequence of simulated test statistic using Wild bootstrap with Standard Normal distribution
bootstrap_beta1_list_wild_winter, bootstrap_t_list_wild_winter = perform_wildBootstrap(errorDistribution='Standard Normal', 
                                                                                       input_numberOfSimulation=numberOfSimulations, 
                                                                                       input_x=time_list, input_y=winterAverage, 
                                                                                       input_matrixX=matrix_X, input_beta=estimate_beta, 
                                                                                       boolean_underTheNull=True)
# compute the bootstrap p-value       
bootstrap_p = computeBootstrapPValue(bootstrap_t_list_wild_winter, t_statistic_beta_1)
print(f'Wild bootstrap (Standard Normal) : \nThe bootstrap p-value is {bootstrap_p}. ', end='') 
pValue_approach(input_p_value=bootstrap_p, input_alpha=alpha) 

# obtain the sequence of simulated test statistic using Wild bootstrap with Rademacher distribution
bootstrap_beta1_list_wild_winter, bootstrap_t_list_wild_winter = perform_wildBootstrap(errorDistribution='Rademacher', 
                                                                                       input_numberOfSimulation=numberOfSimulations, 
                                                                                       input_x=time_list, input_y=winterAverage, 
                                                                                       input_matrixX=matrix_X, input_beta=estimate_beta, 
                                                                                       boolean_underTheNull=True)
# compute the bootstrap p-value       
bootstrap_p = computeBootstrapPValue(bootstrap_t_list_wild_winter, t_statistic_beta_1)
print(f'Wild bootstrap (Rademacher): \nThe bootstrap p-value is {bootstrap_p}. ', end='') 
pValue_approach(input_p_value=bootstrap_p, input_alpha=alpha) 

# obtain the sequence of simulated test statistic using Block bootstrap
bootstrap_beta1_list_block_winter, bootstrap_t_list_block_winter = perform_blockBootstrap(input_numberOfSimulation=numberOfSimulations, 
                                                                                         input_x=time_list, input_y=winterAverage, 
                                                                                         input_matrixX=matrix_X, input_beta=estimate_beta, 
                                                                                         boolean_underTheNull=True)
# compute the bootstrap p-value       
bootstrap_p = computeBootstrapPValue(bootstrap_t_list_block_winter, t_statistic_beta_1)
print(f'Block bootstrap: \nThe bootstrap p-value is {bootstrap_p}. ', end='') 
pValue_approach(input_p_value=bootstrap_p, input_alpha=alpha)

# obtain the sequence of simulated test statistic using Sieve bootstrap
bootstrap_beta1_list_sieve_winter, bootstrap_t_list_sieve_winter = perform_sieveBootstrap(input_numberOfSimulation=numberOfSimulations, 
                                                                                          input_x=time_list, input_y=winterAverage, 
                                                                                          input_matrixX=matrix_X, input_beta=estimate_beta, 
                                                                                          boolean_underTheNull=True)
# compute the bootstrap p-value   
bootstrap_p = computeBootstrapPValue(bootstrap_t_list_sieve_winter, t_statistic_beta_1)
print(f'Sieve bootstrap: \nThe bootstrap p-value is {bootstrap_p}. ', end='') 
pValue_approach(input_p_value=bootstrap_p, input_alpha=alpha)

print('\nExercise 2a: Summer Average')

# estimate the model using OLS regression
matrix_X, estimate_beta, hat_y, hat_residuals = runRegressionModel(summerAverage, time_list)
# report the estimated coefficients
print(f'Summer: β_0 = {round(estimate_beta.iat[0,0],3)} & β_1 = {round(estimate_beta.iat[1,0],3)}')

# compute the HAC standard errors
estimate_SE = getHACStandardError(matrix_X, summerAverage)

# compute t-statistic for β_1 under the null 
SE_beta_1 = estimate_SE[1]
t_statistic_beta_1 = estimate_beta.iat[1,0] / SE_beta_1

# perform a t-test using the summer averages for β_1
alpha = 0.1 # significance level
print(f't-test for β_1 = {round(t_statistic_beta_1,3)}:', end = ' ')
t_test(alpha, t_statistic_beta_1)

# obtain the sequence of simulated test statistic using Nonparametric residual bootstrap
bootstrap_beta1_list_iid_summer, bootstrap_t_list_iid_summer = perform_iidBootstrap(input_numberOfSimulation=numberOfSimulations, 
                                                                                    input_x=time_list, input_y=summerAverage, 
                                                                                    input_matrixX=matrix_X, input_beta=estimate_beta, 
                                                                                    boolean_underTheNull=True)
# compute the bootstrap p-value       
bootstrap_p = computeBootstrapPValue(bootstrap_t_list_iid_summer, t_statistic_beta_1)
print(f'i.i.d. bootstrap: \nThe bootstrap p-value is {bootstrap_p}. ', end='') 
pValue_approach(input_p_value=bootstrap_p, input_alpha=alpha) 

# obtain the sequence of simulated test statistic using Wild bootstrap using Standard Normal distribution
bootstrap_beta1_list_wild_summer, bootstrap_t_list_wild_summer = perform_wildBootstrap(errorDistribution='Standard Normal', 
                                                                                       input_numberOfSimulation=numberOfSimulations, 
                                                                                       input_x=time_list, input_y=summerAverage, 
                                                                                       input_matrixX=matrix_X, input_beta=estimate_beta, 
                                                                                       boolean_underTheNull=True)
# compute the bootstrap p-value       
bootstrap_p = computeBootstrapPValue(bootstrap_t_list_wild_summer, t_statistic_beta_1)
print(f'Wild bootstrap (Standard Normal): \nThe bootstrap p-value is {bootstrap_p}. ', end='') 
pValue_approach(input_p_value=bootstrap_p, input_alpha=alpha)

# obtain the sequence of simulated test statistic using Wild bootstrap using Rademacher distribution
bootstrap_beta1_list_wild_summer, bootstrap_t_list_wild_summer = perform_wildBootstrap(errorDistribution='Rademacher', 
                                                                                       input_numberOfSimulation=numberOfSimulations, 
                                                                                       input_x=time_list, input_y=summerAverage, 
                                                                                       input_matrixX=matrix_X, input_beta=estimate_beta, 
                                                                                       boolean_underTheNull=True)
# compute the bootstrap p-value       
bootstrap_p = computeBootstrapPValue(bootstrap_t_list_wild_summer, t_statistic_beta_1)
print(f'Wild bootstrap (Rademacher): \nThe bootstrap p-value is {bootstrap_p}. ', end='') 
pValue_approach(input_p_value=bootstrap_p, input_alpha=alpha)

# obtain the sequence of simulated test statistic using Block bootstrap
bootstrap_beta1_list_block_summer, bootstrap_t_list_block_summer = perform_blockBootstrap(input_numberOfSimulation=numberOfSimulations, 
                                                                                          input_x=time_list, input_y=summerAverage, 
                                                                                          input_matrixX=matrix_X, input_beta=estimate_beta, 
                                                                                          boolean_underTheNull=True)
# compute the bootstrap p-value       
bootstrap_p = computeBootstrapPValue(bootstrap_t_list_block_summer, t_statistic_beta_1)
print(f'Block bootstrap: \nThe bootstrap p-value is {bootstrap_p}. ', end='') 
pValue_approach(input_p_value=bootstrap_p, input_alpha=alpha)

# obtain the sequence of simulated test statistic using Sieve bootstrap
bootstrap_beta1_list_sieve_summer, bootstrap_t_list_sieve_summer = perform_sieveBootstrap(input_numberOfSimulation=numberOfSimulations, 
                                                                                          input_x=time_list, input_y=summerAverage, 
                                                                                          input_matrixX=matrix_X, input_beta=estimate_beta, 
                                                                                          boolean_underTheNull=True)
# compute the bootstrap p-value  
bootstrap_p = computeBootstrapPValue(bootstrap_t_list_sieve_summer, t_statistic_beta_1)
print(f'Sieve bootstrap: \nThe bootstrap p-value is {bootstrap_p}. ', end='') 
pValue_approach(input_p_value=bootstrap_p, input_alpha=alpha)

'''
Exercise 3a: construct the confidence intervals
'''
print('\nExercise 3a: confidence intervals')

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

# construct all different types of confidence intervals
def getAllCI(input_beta1_list, input_t_list, input_alpha, input_theta, input_SE):
    etpi = getEqualTailedPercentileCI(input_alpha, input_beta1_list, input_theta)
    print(f'The equal-tailed percentile interval is {np.round(etpi,3)}.')
    
    etpti = getEqualTailedPercentileTCI(input_alpha, input_t_list, input_theta, input_SE)
    print(f'The equal-tailed percentile-t interval is {np.round(etpti,3)}.')
    
    spi = getSymmetricPercentileCI(input_alpha, np.abs(input_beta1_list), input_theta)
    print(f'The symmetric percentile interval is {np.round(spi,3)}.')
    
    spti = getSymmetricPercentileTCI(input_alpha, np.abs(input_t_list), input_theta, input_SE)
    print(f'The symmetric percentile-t interval is {np.round(spti,3)}.')

# obtain the sequence of simulated test statistic using iid Bootstrap (not under the null)
bootstrap_beta1_list_iid_annual, bootstrap_t_list_iid_annual = perform_iidBootstrap(input_numberOfSimulation=numberOfSimulations, 
                                                                                    input_x=time_list, input_y=annualAverage, 
                                                                                    input_matrixX=matrix_X_original, input_beta=estimate_beta_original, 
                                                                                    boolean_underTheNull=False)

# obtain the sequence of simulated test statistic using Wild Bootstrap with Standard Normal distribution (not under the null)
bootstrap_beta1_list_wild_SN_annual, bootstrap_t_list_wild_SN_annual = perform_wildBootstrap(errorDistribution='Standard Normal', 
                                                                                             input_numberOfSimulation=numberOfSimulations, 
                                                                                             input_x=time_list, input_y=annualAverage, 
                                                                                             input_matrixX=matrix_X_original, input_beta=estimate_beta_original, 
                                                                                             boolean_underTheNull=False)

# obtain the sequence of simulated test statistic using Wild Bootstrap with Rademacher distribution (not under the null)
bootstrap_beta1_list_wild_R_annual, bootstrap_t_list_wild_R_annual = perform_wildBootstrap(errorDistribution='Rademacher', 
                                                                                           input_numberOfSimulation=numberOfSimulations, 
                                                                                           input_x=time_list, input_y=annualAverage, 
                                                                                           input_matrixX=matrix_X_original, input_beta=estimate_beta_original, 
                                                                                           boolean_underTheNull=False)

# obtain the sequence of simulated test statistic using Block Bootstrap (not under the null)
bootstrap_beta1_list_block_annual, bootstrap_t_list_block_annual = perform_blockBootstrap(input_numberOfSimulation=numberOfSimulations, 
                                                                                          input_x=time_list, input_y=annualAverage, 
                                                                                          input_matrixX=matrix_X_original, input_beta=estimate_beta_original, 
                                                                                          boolean_underTheNull=False)

# obtain the sequence of simulated test statistic using Sieve Bootstrap (not under the null)
bootstrap_beta1_list_sieve_annual, bootstrap_t_list_sieve_annual = perform_sieveBootstrap(input_numberOfSimulation=numberOfSimulations, 
                                                                                          input_x=time_list, input_y=annualAverage, 
                                                                                          input_matrixX=matrix_X_original, input_beta=estimate_beta_original, 
                                                                                          boolean_underTheNull=False)

# construct different confidence intervals using Nonparametric residual bootstrap
print('i.i.d. bootstrap:')
getAllCI(input_beta1_list=bootstrap_beta1_list_iid_annual, input_t_list=bootstrap_t_list_iid_annual, 
         input_alpha=alpha, input_theta=theta_n, input_SE=estimate_SE_original)

# construct different confidence intervals using Wild bootstrap
print('Wild bootstrap (Standard Normal):')
getAllCI(input_beta1_list=bootstrap_beta1_list_wild_SN_annual, input_t_list=bootstrap_t_list_wild_SN_annual, 
         input_alpha=alpha, input_theta=theta_n, input_SE=estimate_HCSE_original)

print('Wild bootstrap (Rademacher):')
getAllCI(input_beta1_list=bootstrap_beta1_list_wild_R_annual, input_t_list=bootstrap_t_list_wild_R_annual, 
         input_alpha=alpha, input_theta=theta_n, input_SE=estimate_HCSE_original)

# construct different confidence intervals using Block bootstrap
print('Block bootstrap:')
getAllCI(input_beta1_list=bootstrap_beta1_list_block_annual, input_t_list=bootstrap_t_list_block_annual, 
         input_alpha=alpha, input_theta=theta_n, input_SE=estimate_HACSE_original)

# construct different confidence intervals using Sieve bootstrap
print('Sieve bootstrap:')
getAllCI(input_beta1_list=bootstrap_beta1_list_sieve_annual, input_t_list=bootstrap_t_list_sieve_annual, 
         input_alpha=alpha, input_theta=theta_n, input_SE=estimate_HACSE_original)

'''
Exercise 4: the empirical coverages
'''
print('\nExercise 4: the empirical coverages')

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
capitalN = 500   # number of simulation for Monte Carlo

# define new list of t
new_time_list = np.linspace(1, n, n)

iid_etpt_ci_list = []
iid_sp_ci_list = []
wild_etpt_ci_list = []
wild_sp_ci_list = []

# obtain confidence intervals from each bootstrap
for i in range(0, capitalN):
    errorTerms_list = []
    generatedYList = []
        
    # generate specific error terms σ_t which are depending on time
    for t in range(1, n+1):
        sigma_t = np.sqrt(1 + 2 * t + 4 * t ** 2)
        errorTerms_list.append(float(sigma_t))
    # generate bootstrap errors as ε_t = v_t * σ_t with v_t ∼ N(0,1)
    errorTerms_list = errorTerms_list * np.random.normal(loc=0, scale=1, size=len(errorTerms_list))

    # generate data from model with specific error terms
    for t in range(1, n+1):
        this_y = beta_0 + true_beta_1 * t + float(errorTerms_list[t-1])
        generatedYList.append(this_y)
    
    # estimate the model using OLS regression
    matrix_X, estimate_beta, hat_y, hat_residuals = runRegressionModel(generatedYList, new_time_list)

    # extract the estimated beta from actual data
    theta_n_newY = estimate_beta.iat[1,0]
    
    # compute the standard errors for β_1
    estimate_SE_iid = getStandardError(matrix_X, hat_residuals)[1]
    estimate_SE_wild = getHCStandardError(matrix_X, hat_residuals.squeeze())[1]
    
    # obtain the sequence of simulated test statistic using Nonparametric residual bootstrap
    bootstrap_beta1_list_iid_newY, bootstrap_t_list_iid_newY = perform_iidBootstrap(input_numberOfSimulation=numberOfSimulations, 
                                                                                    input_x=new_time_list, input_y=generatedYList, 
                                                                                    input_matrixX=matrix_X, input_beta=estimate_beta, 
                                                                                    boolean_underTheNull=False)
    # obtain the Equal-Tailed Percentile-t and Symmetric Percentile bootstrap confidence interval
    etpti_iid = getEqualTailedPercentileTCI(alpha, bootstrap_t_list_iid_newY, theta_n_newY, estimate_SE_iid)
    iid_etpt_ci_list.append(etpti_iid)
    spi_iid = getSymmetricPercentileCI(alpha, np.abs(bootstrap_beta1_list_iid_newY), theta_n_newY)
    iid_sp_ci_list.append(spi_iid)
    
    # obtain the sequence of simulated test statistic using Wild bootstrap with Standard Normal distribution
    bootstrap_beta1_list_wild_newY, bootstrap_t_list_wild_newY = perform_wildBootstrap(errorDistribution='Standard Normal', 
                                                                                       input_numberOfSimulation=numberOfSimulations, 
                                                                                       input_x=new_time_list, input_y=generatedYList, 
                                                                                       input_matrixX=matrix_X, input_beta=estimate_beta, 
                                                                                       boolean_underTheNull=False)
    # obtain the Equal-Tailed Percentile-t and Symmetric Percentile bootstrap confidence interval
    etpti_wild = getEqualTailedPercentileTCI(alpha, bootstrap_t_list_wild_newY, theta_n_newY, estimate_SE_wild)
    wild_etpt_ci_list.append(etpti_wild)
    spi_wild = getSymmetricPercentileCI(alpha, np.abs(bootstrap_beta1_list_wild_newY), theta_n_newY)
    wild_sp_ci_list.append(spi_wild)

# compute the empirical coverage of Equal-Tailed Percentile-t Interval using iid and Wild Bootstrap 
iid_etpt_Coverage = checkCoverage(iid_etpt_ci_list, true_beta_1)
wild_etpt_Coverage = checkCoverage(wild_etpt_ci_list, true_beta_1)
print(f'The empirical coverage of the equal-tailed percentile-t Interval using iid Bootstrap is {round(iid_etpt_Coverage, 3)}.')
print(f'The empirical coverage of the equal-tailed percentile-t Interval using Wild Bootstrap (Standard Normal) is {round(wild_etpt_Coverage, 3)}.')

# compute the empirical coverage of Symmetric Percentile Interval using iid and Wild Bootstrap
iid_sp_Coverage = checkCoverage(iid_sp_ci_list, true_beta_1)
wild_sp_Coverage = checkCoverage(wild_sp_ci_list, true_beta_1)
print(f'The empirical coverage of the symmetric percentile Interval using iid Bootstrap is {round(iid_sp_Coverage, 3)}.')
print(f'The empirical coverage of the symmetric percentile Interval using Wild Bootstrap (Standard Normal) is {round(wild_sp_Coverage, 3)}.')

# check the execution time
end_time = time.time()     # record end time
elapsed = end_time - start_time
print(f'Execution time: {elapsed:.2f} seconds')

'''
Expected output:

Exercise 1a: t-test based on real data
Annual: β_0 = 8.739 & β_1 = 0.013
t-test for β_1 = 6.035: t ∉ [-1.645,1.645]: Reject the null.

Exercise 1b: t-test using different bootstrap methods
i.i.d. bootstrap: 
The bootstrap p-value is 0.0. p ∈ rejection region: Reject the null.
Wild bootstrap (Standard Normal): 
The bootstrap p-value is 0.0. p ∈ rejection region: Reject the null.
Wild bootstrap (Rademacher): 
The bootstrap p-value is 0.0. p ∈ rejection region: Reject the null.
Block bootstrap: 
The bootstrap p-value is 0.002. p ∈ rejection region: Reject the null.
Sieve bootstrap: 
The bootstrap p-value is 0.081. p ∈ rejection region: Reject the null.

Exercise 2a: Winter Average
Winter: β_0 = 2.038 & β_1 = 0.012
t-test for β_1 = 2.723: t ∉ [-1.645,1.645]: Reject the null.
i.i.d. bootstrap: 
The bootstrap p-value is 0.012. p ∈ rejection region: Reject the null.
Wild bootstrap (Standard Normal) : 
The bootstrap p-value is 0.013. p ∈ rejection region: Reject the null.
Wild bootstrap (Rademacher): 
The bootstrap p-value is 0.004. p ∈ rejection region: Reject the null.
Block bootstrap: 
The bootstrap p-value is 0.032. p ∈ rejection region: Reject the null.
Sieve bootstrap: 
The bootstrap p-value is 0.096. p ∈ rejection region: Reject the null.

Exercise 2a: Summer Average
Summer: β_0 = 15.683 & β_1 = 0.012
t-test for β_1 = 5.265: t ∉ [-1.645,1.645]: Reject the null.
i.i.d. bootstrap: 
The bootstrap p-value is 0.0. p ∈ rejection region: Reject the null.
Wild bootstrap (Standard Normal): 
The bootstrap p-value is 0.0. p ∈ rejection region: Reject the null.
Wild bootstrap (Rademacher): 
The bootstrap p-value is 0.0. p ∈ rejection region: Reject the null.
Block bootstrap: 
The bootstrap p-value is 0.003. p ∈ rejection region: Reject the null.
Sieve bootstrap: 
The bootstrap p-value is 0.036. p ∈ rejection region: Reject the null.

Exercise 3a: confidence intervals
i.i.d. bootstrap:
The equal-tailed percentile interval is [0.01  0.016].
The equal-tailed percentile-t interval is [0.01  0.016].
The symmetric percentile interval is [0.009 0.017].
The symmetric percentile-t interval is [0.009 0.017].
Wild bootstrap (Standard Normal):
The equal-tailed percentile interval is [0.01  0.016].
The equal-tailed percentile-t interval is [0.01  0.016].
The symmetric percentile interval is [0.009 0.017].
The symmetric percentile-t interval is [0.009 0.017].
Wild bootstrap (Rademacher):
The equal-tailed percentile interval is [0.01  0.016].
The equal-tailed percentile-t interval is [0.01  0.016].
The symmetric percentile interval is [0.01  0.017].
The symmetric percentile-t interval is [0.009 0.017].
Block bootstrap:
The equal-tailed percentile interval is [0.008 0.017].
The equal-tailed percentile-t interval is [0.008 0.017].
The symmetric percentile interval is [0.008 0.018].
The symmetric percentile-t interval is [0.007 0.019].
Sieve bootstrap:
The equal-tailed percentile interval is [0.011 0.016].
The equal-tailed percentile-t interval is [0.011 0.017].
The symmetric percentile interval is [0.01  0.016].
The symmetric percentile-t interval is [0.009 0.017].

Exercise 4: the empirical coverages
The empirical coverage of the equal-tailed percentile-t Interval using iid Bootstrap is 0.87.
The empirical coverage of the equal-tailed percentile-t Interval using Wild Bootstrap (Standard Normal) is 0.904.
The empirical coverage of the symmetric percentile Interval using iid Bootstrap is 0.94.
The empirical coverage of the symmetric percentile Interval using Wild Bootstrap (Standard Normal) is 0.954.
Execution time: 1336.77 seconds
'''