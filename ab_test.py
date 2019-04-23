import math
import random
# двустороннее р-значение
# def two_sided_p_value (x, mu=O , sigma=1) :
#     if х >= mu :
#         # если х больше среднего значения, то значения в хвосте больше
#         return 2 * normal_probability_above (x, mu, sigma)
#     else :
#         # если х меньше среднего Значения, то значения в хвосте меньше х
#         return 2 * normal_probability_below (x, mu, sigma)

def estimated_parameters(N,n):
    p=n/N
    sigma=math.sqrt(p*(1-p)/N)
    return p, sigma

# произвольно выбрать мальчика или девочку
def random_kid () :
    return random.choice(["boy", "girl"])

#Парадокс мальчил девочка
def paradocs_gerl_boys():
    both_girls = 0
    older_girl = 0
    either_girl =0
    #print(random.getstate())
    random.seed()
    for _ in range(1000000000):     # провести эксперимент на совокупности
        younger =random_kid()  # из 100 000 семей
        older = random_kid()
        if older == "girl" : # старшая?
            older_girl += 1
        if older == "girl" and younger == "girl" : # обе ?
            both_girls += 1
        if older == "girl" or younger == "girl" :# любая из двух?
            either_girl += 1

    print("Р(обе | старшая девочка):", both_girls / older_girl )    # 0.514 ~ 1/2
    print("Р(обе | любая девочка  ):", both_girls / either_girl) # 0.342 ~ 1/3




def a_b_test_statistic(N_A,n_A,N_B,n_B):
    p_A, sigma_A = estimated_parameters(N_A,n_A)
    p_B, sigma_B = estimated_parameters(N_B,n_B)
    return (p_B-p_A)/math.sqrt(sigma_A**2+sigma_B**2)

if __name__=="__main__":
    print("Парадокс мальчик-девочка - 1")
    z=input("Ваш выбор> ")
    if z=="1" :paradocs_gerl_boys()
    # z=a_b_test_statistic(1000,200,1000,180)
    # print(z)
