import numpy as np
import pandas as pd

def read_csv(filename, hasHeader) -> str:
    output = pd.read_csv(filename, header = hasHeader)
    return output
    
raw_data = read_csv('../1_MonteCarlo/DataAssign1.csv', 0)

# remove the trends
difference_df = raw_data.diff()
raw_data.iloc[:, 1:]= difference_df.iloc[:, 1:] # replace the columns by the differences

# Exercise 2a
print('Exercise 2a:')

# construct matrix X
matrix_x = raw_data.copy()[['Cow milk production in 1000 t','Population in 1000']]
matrix_x['intercept'] = 1 # define a column for intercept
matrix_x = matrix_x.rename(columns={'Cow milk production in 1000 t':'x_1','Population in 1000':'x_2', 'intercept':'x_0'})
matrix_x = matrix_x[['x_0', 'x_1', 'x_2']]
matrix_x.dropna(inplace = True)

# construct vector y
matrix_y = raw_data.copy()[['Methane emissions in 1000t']]
matrix_y = matrix_y.rename(columns={'Methane emissions in 1000t':'y'})
matrix_y.dropna(inplace = True)

def getEstimatedBeta(input_matrix_X, input_vector_y):
    # calculate the inverse matrix (X′ * X)^{−1}
    df = input_matrix_X.T @ input_matrix_X
    df_inv = pd.DataFrame(np.linalg.inv(df.values),
                      index=df.columns,   # rows = original columns
                      columns=df.index) 
    
    # calculate estimated beta by (X′ * X)^{−1} * X′* y
    result_beta = df_inv @ input_matrix_X.T @ input_vector_y
    
    return result_beta

# calculate estimated beta by (X′ * X)^{−1} * X′* y
estimate_beta = getEstimatedBeta(matrix_x, matrix_y)

# calculate estimated y by X * (estimated beta)
matrix_y['hat_y'] = matrix_x @ estimate_beta

# calculate the residuals
matrix_y['hat_residuals'] = matrix_y['y'] - matrix_y['hat_y']

# obtain the Durbin-Watson test statistic
def getTestStatistic_d(residualsList):
    numerator = 0.0
    denominator = 0.0
    
    for i in range(1,len(residualsList)):
        numerator += (residualsList[i] - residualsList[i-1]) ** 2
        denominator += residualsList[i-1] ** 2
    
    hat_d = round(numerator / denominator, 3)
    
    return hat_d

# true d
true_d = getTestStatistic_d(list(matrix_y['hat_residuals']))

print(f'The true d is {true_d}.')

# perform a Monte Carlo simulation
numberOfSimulations_b = 9999
numberOfSamples = len(matrix_y)
monteCarlo_d_list = []

for i in range(1, numberOfSimulations_b):
    new_res = np.random.normal(loc=0, scale=1, size=numberOfSamples)
    
    this_d = getTestStatistic_d(new_res)
    monteCarlo_d_list.append(this_d)

def getQuantile(c_1_quantile, c_2_quantile, resultList):
    c_1 = round(np.quantile(resultList, c_1_quantile),3)
    c_2 = round(np.quantile(resultList, c_2_quantile),3)
    
    return c_1, c_2

# obtain the approximated rejection region using Monte Carlo approximations c_1^* and c_2^*
alpha = 0.1
c_1_quantile = alpha/2
c_2_quantile = 1 - alpha/2

# approximate the critical values
c_1_star, c_2_star = getQuantile(c_1_quantile, c_2_quantile, monteCarlo_d_list)

print(f'The {c_1_quantile}-quantile (c_1) is {c_1_star}.')
print(f'The {c_2_quantile}-quantile (c_2) is {c_2_star}.')

# obtain Monte Carlo p-values
def getPValue(inputList, true_t, B):
    
    left_counter = 0
    right_counter = 0

    for t_star in inputList:
        if t_star > true_t:
            right_counter += 1
            
        if t_star < true_t:
            left_counter += 1
    
    pValue = round(2 * min(left_counter / B, right_counter / B), 3)
    
    return pValue
          
# obtain the Monte Carlo p-values for a two-tailed equal-tailed test            
monteCarlo_p = getPValue(monteCarlo_d_list, true_d, numberOfSimulations_b)
      
print(f'The Monte-Carlo p-value is {monteCarlo_p}.')

# Exercise 2b
print('Exercise 2b:')

# invert the Durbin-Watson test with approximated rejection region
def getCIcorr(input_c_1, input_c_2, d):    
    lower_corr = round((input_c_1 - d)/2, 3)
    upper_corr = round((input_c_2 - d)/2, 3)
    
    return lower_corr, upper_corr

c_1_corr, c_2_corr = getCIcorr(c_1_star, c_2_star, true_d)

print(f'The (1-α) confidence interval for ρ is [{c_1_corr},{c_2_corr}].')

hat_residuals = matrix_y['hat_residuals']

# Exercise 3a
print('Exercise 3a:')

# perform a Monte Carlo simulation
numberOfSimulations_m = 10000
null_correlation = 0
hat_variance_residuals = np.var(hat_residuals) # error variance
monteCarlo_d_list = []
coverageCounter = 0   

for i in range(1,numberOfSimulations_m):
    # generate new error terms
    new_error = np.random.normal(loc=0, scale=hat_variance_residuals, size=numberOfSamples+1)
    new_res = []
    
    # compute the corresponding residuals
    for j in range(1, numberOfSamples+1):
        new_res.append(null_correlation * new_error[j-1] + new_error[j])
    
    # generate new y by assigning new residuals
    new_y = matrix_y['hat_y'] + new_res
    
    # run the regression to get new estimated residuals
    new_estimate_beta = getEstimatedBeta(matrix_x, new_y)
    new_hat_y = matrix_x @ new_estimate_beta
    new_hat_residuals = list(new_y - new_hat_y)
    
    this_d = getTestStatistic_d(new_hat_residuals)
        
    c_1_corr, c_2_corr = getCIcorr(c_1_star, c_2_star, this_d)
    
    if c_1_corr <= null_correlation <= c_2_corr:
        coverageCounter += 1
        
coveragePercentage = round(coverageCounter/numberOfSimulations_m * 100, 3)
        
print(f'For true ρ = {null_correlation}, the coverage of the confidence interval is {coveragePercentage}%.')

# Exercise 3b
print('Exercise 3b:')

null_correlation = 0.4
hat_variance_residuals = np.var(hat_residuals) * (1 - null_correlation ** 2) # error variance
monteCarlo_d_list = []
coverageCounter = 0
powerCounter = 0

for i in range(1,numberOfSimulations_m): 
    # generate new error terms   
    new_error = np.random.normal(loc=0, scale=hat_variance_residuals, size=numberOfSamples+1)
    new_res = []
    
    # compute the corresponding residuals
    for j in range(1, numberOfSamples+1):
        new_res.append(null_correlation * new_error[j-1] + new_error[j])
    
    # generate new y by assigning new residuals
    new_y = matrix_y['hat_y'] + new_res
    
    # run the regression to get new estimated residuals
    new_estimate_beta = getEstimatedBeta(matrix_x, new_y)
    new_hat_y = matrix_x @ new_estimate_beta
    new_hat_residuals = list(new_y - new_hat_y)
    
    this_d = getTestStatistic_d(new_hat_residuals)
        
    c_1_corr, c_2_corr = getCIcorr(c_1_star, c_2_star, this_d)
    
    if c_1_corr <= null_correlation <= c_2_corr:
        coverageCounter += 1
        
    if 0 <= this_d <= c_1_star or c_2_star <= this_d <= 4:
        powerCounter += 1

coveragePercentage = round(coverageCounter/numberOfSimulations_m * 100, 3)
print(f'For true ρ = {null_correlation}, the coverage of the confidence interval is {coveragePercentage}%.')

power = round(powerCounter/numberOfSimulations_m * 100, 3)
print(f'The approximated power of the test is {power}%.')