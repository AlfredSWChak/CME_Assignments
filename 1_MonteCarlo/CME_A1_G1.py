import numpy as np
import pandas as pd

def read_csv(filename, hasHeader) -> str:
    output = pd.read_csv(filename, header = hasHeader)
    return output
    
raw_data = read_csv('../1_MonteCarlo/DataAssign1.csv', 0)

# remove the trends
difference_df = raw_data.diff()
raw_data.iloc[:, 1:]= difference_df.iloc[:, 1:] # replace the columns by the differences

# construct matrix X
matrix_x = raw_data.copy()[['Cow milk production in 1000 t','Population in 1000']]
matrix_x['intercept'] = 1
matrix_x = matrix_x.rename(columns={'Cow milk production in 1000 t':'x_1','Population in 1000':'x_2', 'intercept':'x_0'})
matrix_x = matrix_x[['x_0', 'x_1', 'x_2']]
matrix_x.dropna(inplace = True)

# construct vector y
matrix_y = raw_data.copy()[['Methane emissions in 1000t']]
matrix_y = matrix_y.rename(columns={'Methane emissions in 1000t':'y'})
matrix_y.dropna(inplace = True)

# calculate estimated beta by (X′ * X)^{−1} * X′* y 
df = matrix_x.T @ matrix_x
df_inv = pd.DataFrame(np.linalg.inv(df.values),
                      index=df.columns,   # rows = original columns
                      columns=df.index) 

estimate_beta = df_inv @ matrix_x.T @ matrix_y

matrix_y['hat_y'] = matrix_x @ estimate_beta
matrix_y['hat_residuals'] = matrix_y['y'] - matrix_y['hat_y']

# obtain the Durbin-Watson test statistic
numerator = 0.0
denominator = 0.0

for i in range(1,len(matrix_y)):
    numerator += (matrix_y.iloc[i,2] - matrix_y.iloc[i-1,2]) ** 2
    denominator += matrix_y.iloc[i-1,2] ** 2
    
hat_d = numerator/denominator

print(f'The true d is {hat_d}.')

# perform a Monte Carlo simulation
numberOfSimulations_b = 9999
numberOfSamples = len(matrix_y)
mc_d = []

for i in range(1, numberOfSimulations_b):
    new_res = np.random.normal(loc=0, scale=1, size=numberOfSamples)
    
    this_numerator = 0.0
    this_denominator = 0.0
    
    for j in range(1,len(new_res)):
        this_numerator += (new_res[j] -new_res[j-1]) ** 2
        this_denominator += new_res[j-1] ** 2
    
    this_d = this_numerator / this_denominator
    mc_d.append(this_d)

# obtain the approximated rejection region using Monte Carlo approximations c_1^* and c_2^*
alpha = 0.1
c_1_quantile = alpha/2
c_2_quantile = 1 - alpha/2
c_1 = round(np.quantile(mc_d, c_1_quantile),3)
c_2 = round(np.quantile(mc_d, c_2_quantile),3)

print(f'The {c_1_quantile}-quantile (c_1) is {c_1}.')
print(f'The {c_2_quantile}-quantile (c_2) is {c_2}.')
    
left_counter = 0
right_counter = 0

for d in mc_d:
    if d > hat_d:
        right_counter += 1
        
    if d < hat_d:
        left_counter += 1
     
# obtain the Monte Carlo p-values for a two-tailed equal-tailed test   
mc_p = round(2 * min(left_counter / numberOfSimulations_b, right_counter / numberOfSimulations_b), 3)
        
print(f'The Monte-Carlo p-value is {mc_p}.')

# invert the Durbin-Watson test with approximated rejection region
approx_corr_upper = round(1 - (c_1 / 2),3)
approx_corr_lower = round(1 - (c_2 / 2),3)

print(f'The confidence interval for ρ [-1,{approx_corr_lower}] and [{approx_corr_upper},1].')

hat_residuals = matrix_y['hat_residuals']
hat_variance_residuals = np.var(hat_residuals)

numberOfSimulations_m = 10000
null_correlation = 0
mc_d = []

for i in range(1,numberOfSimulations_m):
    new_error = np.random.normal(loc=0, scale=hat_variance_residuals, size=numberOfSamples+1)
    
    new_res = []
    for j in range(1, numberOfSamples+1):
        new_res.append(null_correlation * new_error[j-1] + new_error[j])
        
    # print(len(new_error))
    # print(len(new_res))
    
    new_hat_res = list(matrix_y['y'] - (matrix_y['hat_y'] + new_res))
    
    # print(type(new_hat_res))
    
    this_numerator = 0.0
    this_denominator = 0.0
    
    for k in range(1,len(new_hat_res)):
        this_numerator += (new_hat_res[k] - new_hat_res[k-1]) ** 2
        this_denominator += new_hat_res[k-1] ** 2
    
    this_d = this_numerator / this_denominator
    mc_d.append(this_d)
    
# obtain the approximated rejection region using Monte Carlo approximations c_1^* and c_2^*
alpha = 0.1
c_1_quantile = alpha/2
c_2_quantile = 1 - alpha/2
c_1 = round(np.quantile(mc_d, c_1_quantile),3)
c_2 = round(np.quantile(mc_d, c_2_quantile),3)

print(f'The {c_1_quantile}-quantile (c_1) is {c_1}.')
print(f'The {c_2_quantile}-quantile (c_2) is {c_2}.')

# invert the Durbin-Watson test with approximated rejection region
approx_corr_upper = round(1 - (c_1 / 2),3)
approx_corr_lower = round(1 - (c_2 / 2),3)

print(f'The confidence interval for ρ [-1,{approx_corr_lower}] and [{approx_corr_upper},1].')

null_correlation = 0.4
hat_variance_residuals = np.var(hat_residuals) * (1 - null_correlation)
mc_d = []

for i in range(1,numberOfSimulations_m):
    new_error = np.random.normal(loc=0, scale=hat_variance_residuals, size=numberOfSamples+1)
    
    new_res = []
    for j in range(1, numberOfSamples+1):
        new_res.append(null_correlation * new_error[j-1] + new_error[j])
    
    new_hat_res = list(matrix_y['y'] - (matrix_y['hat_y'] + new_res))
    
    this_numerator = 0.0
    this_denominator = 0.0
    
    for k in range(1,len(new_hat_res)):
        this_numerator += (new_hat_res[k] - new_hat_res[k-1]) ** 2
        this_denominator += new_hat_res[k-1] ** 2
    
    this_d = this_numerator / this_denominator
    mc_d.append(this_d)
    
# obtain the approximated rejection region using Monte Carlo approximations c_1^* and c_2^*
alpha = 0.1
c_1_quantile = alpha/2
c_2_quantile = 1 - alpha/2
c_1 = round(np.quantile(mc_d, c_1_quantile),3)
c_2 = round(np.quantile(mc_d, c_2_quantile),3)

print(f'The {c_1_quantile}-quantile (c_1) is {c_1}.')
print(f'The {c_2_quantile}-quantile (c_2) is {c_2}.')  

# invert the Durbin-Watson test with approximated rejection region
approx_corr_upper = round(1 - (c_1 / 2),3)
approx_corr_lower = round(1 - (c_2 / 2),3)

print(f'The confidence interval for ρ [-1,{approx_corr_lower}] and [{approx_corr_upper},1].')