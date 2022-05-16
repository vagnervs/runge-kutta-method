# O código deste trabalho está baseado no algoritmo elaborado por Klopper(2014) -- Disponível em
# https://www.youtube.com/watch?v=cQSQCv0XBx8 -- e com fundamentação téorica na dissertação de Mestrado de RODRIGUES(2011):
# Modelagem matemática em câncer: dinâmica angiogênica e quimioterapia anti-neoplásica

# MODELO DE ANÁLISE EVOLUCIONAL DE CÉLULAS TUMORAIS VIA MÉTODO NUMÉRICO DE RUNGE-KUTTA

#obs: necessário atencão quanto a indentação no momento da execução.

import matplotlib.pyplot as plt

with open('solucao numerica.txt', 'w') as arquivo:

    K1 = 50 # Capacidade suporte: células tumorais
    K2 = 10 # Capacidade suporte: células normais
    p12 = 0.09 # Coeficiente de competição que mede as decorrências em CT geradas pela presença de CN
    r1 = 2.2  # Taxa de expansão do número de células tumorais
    r2 = 1.1  # Taxa de crescimento das células normais
    '''As taxas de crescimento são sempre maiores que ou iguais a zero, fazendo com que
    esses pontos sejam inicialmente instáveis
    '''
    p21 = 0.09  # Coeficiente de competição que mede as decorrências em CN geradas pela presença de CT
    TI = 0  # Tempo de início
    '''Considera-se que a quantidade de células tumorais seja pequena e a quantidade de células normais
    aproximada a sua capacidade suporte
    '''
    p = 1  # intervalo de passos
    CT = 1  # Células tumorais: Número inicial
    CN = 10 # Células normais: Número inicial

    while TI <= 45: # número de dias

        # ======Runge-Kutta de 1ª ordem=====
        mt1 = r1 * CT - (r1 * CT ** 2 + r1 * p12 * CT * CN) / K1
        #mt1 = r1 * CT * (1 - CT / K1)
        rk1 = r2 * CN - (r2 * CN ** 2 + r2 * p21 * CN * CT) / K2

        # ======Runge-Kutta de 2ª ordem======
        ft2 = TI + (p / 2)
        fx2 = CT + (p / 2) * mt1
        fy2 = CN + (p / 2) * rk1
        mt2 = r1 * fx2 - (r1 * fx2 ** 2 + r1 * p12 * fx2 * fy2) / K1
        gt2 = TI + (p / 2)
        gx2 = CT + (p / 2) * mt1
        gy2 = CN + (p / 2) * rk1
        rk2 = r2 * gy2 - (r2 * gy2 ** 2 + r2 * p21 * gy2 * gx2) / K2

        # ======Runge-Kutta de 3ª ordem======
        ft3 = TI + (p / 2)
        fx3 = CT + (p / 2) * mt2
        fy3 = CN + (p / 2) * rk2
        mt3 = r1 * fx3 - (r1 * fx3 ** 2 + r1 * p12 * fx3 * fy3) / K1
        gt3 = TI + (p / 2)
        gx3 = CT + (p / 2) * mt2
        gy3 = CN + (p / 2) * rk2
        rk3 = r2 * gy3 - (r2 * gy3 ** 2 + r2 * p21 * gy3 * gx3) / K2

        # ======Runge-Kutta de 4ª ordem======
        ft4 = TI + p
        fx4 = CT + p * mt3
        fy4 = CN + p * rk3
        mt4 = r1 * fx4 - (r1 * fx4 ** 2 + r1 * p12 * fx4 * fy4) / K1
        gt4 = TI + p
        gx4 = CT + p * mt3
        gy4 = CN + p * rk3
        rk4 = r2 * gy4 - (r2 * gy4 ** 2 + r2 * p21 * gy4 * gx4) / K2

        TI = TI + p  # indica o dia no intervalo compreendido de tempo
        CT = CT + (p / 6) * (mt1 + 2 * mt2 + 2 * mt3 + mt4)  # distribuição de células tumorais
        CN = CN + (p / 6) * (rk1 + 2 * rk2 + 2 * rk3 + rk4)  # distribuição de células normais

        print(f"{TI:.0f} \t \t {CT:.6f} \t\t {CN:.7f}")
        print(f"{TI:.0f} \t \t {CT:.6f} \t\t {CN:.7f}", file=arquivo) #criação de um arquivo de saída
        plt.plot(TI, CT, 'r.', markeredgewidth=1.5)
        plt.plot(TI, CN, 'g.', markeredgewidth=1.5)
        plt.xlabel('tempo em dias')
        plt.ylabel('Quantitativo de células')
        plt.legend(['Células tumorais', 'Células normais'])
        plt.pause(0.1)

    plt.grid()
    plt.show()
