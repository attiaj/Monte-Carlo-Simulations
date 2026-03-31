
import numpy as np
import math

from numpy.random import exponential

x_0 = 0.03
k = 2
theta = 0.05
sigma = 0.5
N = 365
delta_t = 1/365
M = 1000


# part 1 ------------------------------------------------------------------------------------------------------------------------------------
def EM_scheme(N, M, dt, abs_value = False):
    #initalize dict to record our 1000 different paths
    #Each path has 366 values, initial plus 365 steps
    paths = {i: [0.0] * (N + 1) for i in range(1, M+1)}

    #Iterate over our number of paths (paths 1-1000)
    for j in range(1, M+1):

        #Initalize first value of each path as x_0
        paths[j][0] = x_0
        previous_x = x_0

        #Iterate over our number of steps
        for i in range(N):
            
            #Use the normality properties of Brownian motion increments to get a random sample using Normal dist
            delta_B = np.random.normal(0.0, math.sqrt(dt))
            
            #Add a conditional to implement abs(x) for part 3
            if abs_value == True:
                current_x = (1 - k*dt)*previous_x + k*theta*dt + sigma*math.sqrt(abs(previous_x))*delta_B
            else:
                current_x = (1 - k*dt)*previous_x + k*theta*dt + sigma*math.sqrt(max(previous_x, 0))*delta_B

            paths[j][i+1] = current_x

            previous_x = current_x

    return paths

# part 2 ------------------------------------------------------------------------------------------------------------------------------------

def Milstein_scheme(N, M, dt, abs_value = False):
    #initalize dict to record our 1000 different paths
    #Each path has 366 values, initial plus 365 steps
    paths = {i: [0.0] * (N + 1) for i in range(1, M+1)}

    #Iterate over our number of paths (paths 1-1000)
    for j in range(1, M+1):

        #Initalize first value of each path as x_0
        paths[j][0] = x_0
        previous_x = x_0

        #Iterate over our number of steps
        for i in range(N):
            
            #Use the normality properties of Brownian motion increments to get a random sample using Normal dist
            delta_B = np.random.normal(0.0, math.sqrt(dt))
            
            #Note: a(X_i) function = k*(theta-x), sigma(X_i) function = sigma*sqrt(x)
            a_function = k*(theta-previous_x)

            #Add conditional to implement abs(x) for part 3
            if abs_value == True:
                sigma_function = sigma*math.sqrt(abs(previous_x))
            else:
                sigma_function = sigma*math.sqrt(max(previous_x, 0))

            #Note: sigma(x)*sigma'(x) = sigma^2/2
            current_x = previous_x + a_function*dt + sigma_function*delta_B + ((sigma**2)/4)*(delta_B**2 - dt)

            paths[j][i+1] = current_x

            previous_x = current_x

    return paths

# part 3 ------------------------------------------------------------------------------------------------------------------------------------
#For part 3, just added conditionals within the existing functions to change x^+ to abs(x) when appropriate


def average_negatives(scheme):
    if scheme == "EM":
        paths = EM_scheme(N, M, delta_t, False)
    elif scheme == "Milstein":
        paths = Milstein_scheme(N, M, delta_t, False)
    elif scheme == "EM_abs":
        paths = EM_scheme(N, M, delta_t, True)
    elif scheme == "Milstein_abs":
        paths = Milstein_scheme(N, M, delta_t, True)

    total_negatives = 0
    for i in range(1, M+1):
        for j in range(0, N+1):
            if paths[i][j] < 0:
                total_negatives += 1

    return total_negatives / M

print(f'\nPart 1: In 1000 sampled paths of 365 steps each within the EM scheme, the average number of negative values per path = {average_negatives("EM")}\n')

print(f'Part 2: In 1000 sampled paths of 365 steps each within the Milstein scheme, the average number of negative values per path = {average_negatives("Milstein")}\n')

print(f'Part 3: In 1000 sampled paths of 365 steps each within the EM scheme replacing X^+ with abs(X), the average number of negative values per path = {average_negatives("EM_abs")}\n')

print(f'Part 3: In 1000 sampled paths of 365 steps each within the Milstein scheme replacing X^+ with abs(X), the average number of negative values per path = {average_negatives("Milstein_abs")}\n')

# part 4 ------------------------------------------------------------------------------------------------------------------------------------

#Chose M = 1000 because over repeated iterations it provides stable values for prices at varying T's, only showing minor variations across samples
#Also, M = 10000 significantly increased the runtime, to the point where the tradeoff in accuracy was not worth the cost of time

def ZCB_pricing(M):

    maturities_dict = {i: 0.0 for i in range(1, 11)}

    #We can use the Riemann sum method to approximate the integral exponent term

    for T in maturities_dict:
        num_steps = T*365

        #Generate X values using Milstein scheme:
        paths = Milstein_scheme(num_steps, M, delta_t, False)

        #Initalize list of Riemann sum terms (one sum for each iteration of M)
        summation_terms = [0.0] * M

        for i in range(1, M+1):
            
            #Take riemann sum
            summation_terms[i-1] = sum(paths[i][j]*delta_t for j in range(0, num_steps))

        #Find expected value:

        total = 0
        for term in summation_terms:
            exponent_term = math.exp(-term)
            total += exponent_term
        
        #Assign expected value calculation to its key in the dict
        maturities_dict[T] =  expected_value = total / M
    
    return maturities_dict


print(f'The computed values for ZCB prices with M = 1000 and original k, theta, sigma values are: \n{ZCB_pricing(1000)}\n')

for k_value in [0.5, 1, 2, 4]:
    k = k_value

    print(f'The computed values for ZCB prices with M = 1000 and k = {k_value}, with no change to theta or sigma are: \n{ZCB_pricing(1000)}\n')

k = 2

for theta_value in [0.02, 0.05, 0.08, 0.12]:
    theta = theta_value

    print(f'The computed values for ZCB prices with M = 1000 and theta = {theta_value}, with no change to k or sigma are: \n{ZCB_pricing(1000)}\n')

theta = 0.05

for sigma_value in [0.1, 0.3, 0.5, 0.8]:
    sigma = sigma_value

    print(f'The computed values for ZCB prices with M = 1000 and sigma = {sigma_value}, with no change to k or theta are: \n{ZCB_pricing(1000)}\n')

sigma = 0.5