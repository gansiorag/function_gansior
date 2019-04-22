import math

# двустороннее р-значение
def two_sided_p_value (x, mu=O , sigma=1) :
    if х >= mu :
        # если х больше среднего значения, то значения в хвосте больше
        return 2 * normal_probability_above (x, mu, sigma)
    else :
        # если х меньше среднего Значения, то значения в хвосте меньше х
        return 2 * normal_probability_below (x, mu, sigma)

def estimated_parameters(N,n):
    p=n/N
    sigma=math.sqrt(p*(1-p)/N)
    return p, sigma


def a_b_test_statistic(N_A,n_A,N_B,n_B):
    p_A, sigma_A = estimated_parameters(N_A,n_A)
    p_B, sigma_B = estimated_parameters(N_B,n_B)
    return (p_B-p_A)/math.sqrt(sigma_A**2+sigma_B**2)

if __name__=="__main__":
    z=a_b_test_statistic(1000,200,1000,180)
    print(z)    
