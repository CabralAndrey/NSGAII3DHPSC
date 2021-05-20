# Required Libraries
import pandas as pd
import numpy  as np
import numpy
import math
import matplotlib.pyplot as plt
from decimal import Decimal
import random
import os
import numba
import time
from random import randint





inicio = time.time();





##PLOT
def plotagem (Matriz_Proteina):
    
    fig = plt.figure();
    ax = fig.add_subplot(111, projection='3d');
    X, Y, Z = numpy.mgrid[-1:1:10j, -1:1:10j, -1:1:10j];
    ax.set_ylim(-10, 10)
    ax.set_xlim(-10, 10)
    ax.set_zlim(-10, 10)    
    #plt.grid(linewidth=10);
    ax.grid(b = None)
    for i in range (len(Vetor_hpsc)):
        xs = Matriz_Proteina[i][0];
        ys = Matriz_Proteina[i][1];
        zs = Matriz_Proteina[i][2];
        if (Vetor_hpsc[i] == -1) :
            c='r';
            m='o';
        elif(Vetor_hpsc[i] == 1) :
            c = 'b';
            m = 'o';
        else:  
            c = 'g';
            m = 'o';        
        if(i==0):
            m='^';
            #s é o tamanho do scattter
        ax.scatter3D(xs , ys, zs ,zorder = 2, s = 80,zdir='z',c=c,marker=m);
    xline1 =[];
    zline1 = [];
    yline1 = [];
    u=0;
    while(u < len(Matriz_Proteina)-3):
        zline1.append(Matriz_Proteina[u][2]);
        xline1.append(Matriz_Proteina[u][0]);
        yline1.append(Matriz_Proteina[u][1]);
        zline1.append(Matriz_Proteina[u+1][2]);
        xline1.append(Matriz_Proteina[u+1][0])
        yline1.append(Matriz_Proteina[u+1][1])
        zline1.append(Matriz_Proteina[u][2])
        xline1.append(Matriz_Proteina[u][0])
        yline1.append(Matriz_Proteina[u][1])
        zline1.append(Matriz_Proteina[u+2][2])
        xline1.append(Matriz_Proteina[u+2][0])
        yline1.append(Matriz_Proteina[u+2][1])
        u=u+2;

    zline1.append(Matriz_Proteina[u][2]);
    xline1.append(Matriz_Proteina[u][0]);
    yline1.append(Matriz_Proteina[u][1]);
    zline1.append(Matriz_Proteina[u+1][2]);
    xline1.append(Matriz_Proteina[u+1][0])
    yline1.append(Matriz_Proteina[u+1][1])   
   
    ax.plot3D(xline1, yline1, zline1,'gray')  
    
      
    plt.show()
        
#Montando proteinas no espaço 3D
def Matriz_de_Proteina(Contador_Proteinas):

    Matriz_Proteina = numpy.zeros((Contador_Proteinas,3), dtype = numpy.int);
    return Matriz_Proteina

#monta as strings em um vetor com os valores zerados para x,y e z
    
@numba.jit
def vetor_strings(Vetor_hpsc,linhas):
#Contatos hidrofóbiocos =-1, Backbone = 0 , Polares = 1;    
    for caractere in linhas:
        if caractere == 'H':
            Vetor_hpsc.append(0);
            Vetor_hpsc.append(-1);
        if caractere == 'P':
            Vetor_hpsc.append(0);
            Vetor_hpsc.append(1); 
    return Vetor_hpsc;




#Declara Lista de movimentos
    
@numba.jit
def movimentos(vetor_movimentos,Matriz_Proteina,Vetor_hpsc):
#0 = ee
#1 = ef
#2 = ed
#3 = eb
#4 = ec
#5 = fe
#6 = ff
#7 = fd
#8 = fb
#9 = fc
#10 = de
#11 = df
#12 = dd
#13 = db
#14 = dc
#15 = be
#16 = bf
#17 = bd
#18 = bb
#19 = bc
#20 = ce
#21 = cf
#22 = cd
#23 = cb
#24 = cc
    Matriz_Proteina = numpy.zeros((len(Vetor_hpsc),3), dtype = numpy.int);
    #Primeiro Backbone e Proteinas nas posições 0,0,0 e -1,0,0 para referência(X,Y e Z)
    Matriz_Proteina[1][0] = -1;
    #Para o Vetor_Ref é o proximo movimento de referência(onde a proteina ou Backbone está olhando);
    #Nossa Frente de referencia é Frente x =1; Cioma = y = 1 ; Baixo = y = -1. Esquerda = z = -1; Direita z = 1;
    Vetor_Ref = numpy.zeros(8, dtype = numpy.int); 
    Vetor_Ref[0] =  1;#F
    Vetor_Ref[1] = 1;#C
    Vetor_Ref[2] = -1;#B
    Vetor_Ref[3] = -1;#E
    Vetor_Ref[4] = 1;#D
    Vetor_Ref[5] = 2;# Indica eixo do lado direito para REFERÊNCIA
    Vetor_Ref[6] = 0; # Indica o eixo 0 para x, 1 para y, 2 para z(FRente)
    Vetor_Ref[7] = 1; #Indica o "pé" para referência
  #  print(Vetor_Ref);
    #Começando a partir do segundo BB
    i=0;
    j=0;
    aux_mov_sc= 0;
    while(j < ((len(vetor_movimentos)))):
        
        
        if(j >0):
            #rECEBE A posição equivalente ao backbone anterior
            Matriz_Proteina [i+2][:] = Matriz_Proteina [i][:];        
    #movimento para frente
        
        if((((vetor_movimentos[j]) >= 5) and (vetor_movimentos[j]) <= 9)):
            Matriz_Proteina [i+2][Vetor_Ref[6]] =  (Matriz_Proteina [i+2] [Vetor_Ref[6]])  + Vetor_Ref[0];
            aux_mov_sc = vetor_movimentos[j]-5;            
    #movimento para direita    
        elif ((((vetor_movimentos[j]) >= 10) and (vetor_movimentos[j]) <= 14)):
            Matriz_Proteina [i+2][Vetor_Ref[5]] =  (Matriz_Proteina [i+2] [Vetor_Ref[5]])  + Vetor_Ref[4];
            aux_mov_sc = vetor_movimentos[j]-10;
            mov = 'd';
            Vetor_Ref = mov_ED(Vetor_Ref,mov);
            #movimento para  baixo
        elif ((((vetor_movimentos[j]) >= 15) and (vetor_movimentos[j]) <= 19)):  
            Matriz_Proteina [i+2][Vetor_Ref[7]] =  (Matriz_Proteina [i+2] [Vetor_Ref[7]])  + Vetor_Ref[2];
            aux_mov_sc = vetor_movimentos[j]-15;
            mov = 'b';
            Vetor_Ref = mov_CB(Vetor_Ref,mov);
    #movimento para cima   
        elif ((((vetor_movimentos[j]) >= 20) and (vetor_movimentos[j]) <= 24)):
            Matriz_Proteina [i+2][Vetor_Ref[7]] =  (Matriz_Proteina [i+2] [Vetor_Ref[7]]) + Vetor_Ref[1];
            aux_mov_sc = vetor_movimentos[j] - 20;  
            mov = 'c';
            Vetor_Ref = mov_CB(Vetor_Ref,mov);
    #movimento para esquerda
        else:
            Matriz_Proteina [i+2] [Vetor_Ref[5]] =  (Matriz_Proteina [i+2] [Vetor_Ref[5]])  + Vetor_Ref[3];
            mov = 'e';
            Vetor_Ref = mov_ED(Vetor_Ref,mov);      
            aux_mov_sc = vetor_movimentos[j];              
        #Matriz_Proteina [i+3] [Vetor_Ref[6]] = Movimento_SC(Matriz_Proteina,aux_mov_sc,Vetor_Ref,i);
        Movimento_SC(Matriz_Proteina,aux_mov_sc,Vetor_Ref,i);
#        print( Matriz_Proteina);
#        print("O Vetor Referência Atual é:",Vetor_Ref)
        i = i+2;        
        j = j+1;
#        print(i);
    return (Matriz_Proteina)

@numba.jit
def Movimento_SC(Matriz_Proteina, aux_mov_sc, Vetor_Ref,i):
    Matriz_Proteina [i+3][:] =   Matriz_Proteina[i+2][:];
    #ovimento para frente
    if(aux_mov_sc==1):
        Matriz_Proteina [i+3][Vetor_Ref[6]] = Matriz_Proteina [i+3][Vetor_Ref[6]]+  Vetor_Ref[0];
    #movimento para direita    
    elif (aux_mov_sc==2):
        Matriz_Proteina [i+3][Vetor_Ref[5]] = Matriz_Proteina [i+3][Vetor_Ref[5]] +  Vetor_Ref[4];
    #movimento para  baixo
    elif (aux_mov_sc==3):  
        Matriz_Proteina [i+3][Vetor_Ref[7]] = Matriz_Proteina [i+3][Vetor_Ref[7]]+   Vetor_Ref[2];
    #movimento para cima   
    elif (aux_mov_sc==4):
        Matriz_Proteina [i+3][Vetor_Ref[7]] = Matriz_Proteina [i+3][Vetor_Ref[7]] +  Vetor_Ref[1];
    #movimento para esquerda
    else:
       Matriz_Proteina [i+3] [Vetor_Ref[5]] = Matriz_Proteina [i+3] [Vetor_Ref[5]]+  Vetor_Ref[3];

    #return Matriz_Proteina [i+3] [Vetor_Ref[6]] ;
    return Matriz_Proteina ;

@numba.jit
def mov_ED(Vetor_Ref,mov):    
    Vetor_Atualiza_Ref = numpy.zeros(8, dtype = numpy.int); 
    if (mov == 'e') : 
        Vetor_Atualiza_Ref[0] = Vetor_Ref[3];
        Vetor_Atualiza_Ref[3] = Vetor_Ref[0]*(-1);
        Vetor_Atualiza_Ref[4] = Vetor_Ref[0];
    if(mov == 'd') : 
        Vetor_Atualiza_Ref[0] = Vetor_Ref[4];
        Vetor_Atualiza_Ref[3] = Vetor_Ref[0];
        Vetor_Atualiza_Ref[4] = Vetor_Ref[0]*(-1);  
    Vetor_Atualiza_Ref[1] = Vetor_Ref[1];
    Vetor_Atualiza_Ref[2] = Vetor_Ref[2];
    Vetor_Atualiza_Ref[7] = Vetor_Ref[7];    
    Vetor_Atualiza_Ref[6] = Vetor_Ref[5];
    Vetor_Atualiza_Ref[5] = Vetor_Ref[6];
    return Vetor_Atualiza_Ref;

@numba.jit
def mov_CB(Vetor_Ref,mov):    
    Vetor_Atualiza_Ref = numpy.zeros(8, dtype = numpy.int); 
    if (mov == 'c') : 
        Vetor_Atualiza_Ref[0] = Vetor_Ref[1];
        Vetor_Atualiza_Ref[1] = Vetor_Ref[0]*(-1);
        Vetor_Atualiza_Ref[2] = Vetor_Ref[0];
    if(mov == 'b') : 
        Vetor_Atualiza_Ref[0] = Vetor_Ref[2];
        Vetor_Atualiza_Ref[1] = Vetor_Ref[0];
        Vetor_Atualiza_Ref[2] = Vetor_Ref[0]*(-1);  
        #Vetor_Atualiza_Ref[6] = Vetor_Ref[5];
    Vetor_Atualiza_Ref[3] = Vetor_Ref[3];
    Vetor_Atualiza_Ref[4] = Vetor_Ref[4];
    Vetor_Atualiza_Ref[5] = Vetor_Ref[5];    
    Vetor_Atualiza_Ref[6] = Vetor_Ref[7];
    Vetor_Atualiza_Ref[7] = Vetor_Ref[6];
    
    return Vetor_Atualiza_Ref;
#contatos hidrofóbicos
#0 = e
#1 = f
#2 = d
#3 = b
#4 = c
#5 = t
@numba.jit    
def contatos(Matriz_Prot, indice):
    proximos =  numpy.zeros((6,3), dtype = numpy.int);
    for w in range (0,6):
        proximos[w] = Matriz_Prot;

    proximos[0][0] = proximos[0][0] + 1;
    proximos[1][0] = proximos[1][0] -1;
    proximos[2][1] = proximos[2][1] + 1;
    proximos[3][1] = proximos[3][1] -1;
    proximos[4][2] = proximos[4][2] + 1;
    proximos[5][2] = proximos[5][2] -1;
    #print(proximos);
    #print(Matriz_Prot);
    return proximos;


@numba.jit
def function1(Matriz_Proteina,Vetor_hpsc):
    contador1 = 0;
    proximos =  numpy.zeros((6,3), dtype = numpy.int);
    i=1;
   # print(Vetor_hpsc);
    while(i < len(Vetor_hpsc)):
        #print(Vetor_hpsc[i]);
        if (Vetor_hpsc[i] == -1):
            proximos = contatos(Matriz_Proteina[i],i);
            #compara o hidrofobico com o hidrofóbico
            j=i+2;
            while j < (len(Vetor_hpsc)):
                if(Vetor_hpsc[j] == -1):
                    for q in range(0,6):
                        if (proximos[q][0] == Matriz_Proteina[j][0] and proximos[q][1] == Matriz_Proteina[j][1] and proximos[q][2] == Matriz_Proteina[j][2] ):
                            contador1 = contador1 +1
                            
                j=j+2
        i = i+2
    return contador1


@numba.jit
def energia(Matriz_Proteina,Vetor_hpsc):
    contadorhh = 0;
    contadorhp = 0;
    contadorhb = 0;
    contadorpb = 0;
    contadorpp = 0;
    contadorbb = 0;
    proximos =  numpy.zeros((6,3), dtype = numpy.int);
    i=0;
    while(i < (len(Vetor_hpsc)-2)):
        proximos = contatos(Matriz_Proteina[i],i);
        if(Vetor_hpsc[i] == 0):
            j = i +3;
        else:
            j = i+1;
        while j < (len(Vetor_hpsc)):
           
            for q in range(0,6):
                if ((proximos[q][0] == Matriz_Proteina[j][0] and proximos[q][1] == Matriz_Proteina[j][1] and proximos[q][2] == Matriz_Proteina[j][2])):
                    if(Vetor_hpsc[i] == -1 and Vetor_hpsc[j] == -1):
                        contadorhh= contadorhh+1;
                    elif(Vetor_hpsc[i] == 1 and Vetor_hpsc[j] == 1):
                        contadorpp= contadorpp +1;
                    elif(Vetor_hpsc[i] == 0 and Vetor_hpsc[j] == 0):
                        contadorbb= contadorbb +1;
                    elif (Vetor_hpsc[i] == -1 and Vetor_hpsc[j] == 1) or (Vetor_hpsc[i] == 1 and Vetor_hpsc[j] == -1):
                        contadorhp= contadorhp+1;
                    elif (Vetor_hpsc[i] == 0 and Vetor_hpsc[j] == 1) or (Vetor_hpsc[i] == 1 and Vetor_hpsc[j] == 0):
                        contadorpb= contadorpb+1;
                    elif (Vetor_hpsc[i] == -1 and Vetor_hpsc[j] == 0) or (Vetor_hpsc[i] == 0 and Vetor_hpsc[j] == -1):
                        contadorhb= contadorhb+1;
            j=j+1;
        i = i+1;
    energy =(contadorhh*10)+(contadorhp*-3)+(contadorhb*-3)+(contadorpb)+(contadorpp)+(contadorbb); 
    return energy;

@numba.jit
def energia_best(Matriz_Proteina,Vetor_hpsc):
    contadorhh = 0;
    contadorhp = 0;
    contadorhb = 0;
    contadorpb = 0;
    contadorpp = 0;
    contadorbb = 0;
    proximos =  numpy.zeros((6,3), dtype = numpy.int);
    i=0;
    ppa = 0;
    while(i < (len(Vetor_hpsc)-2)):
        proximos = contatos(Matriz_Proteina[i],i);
        if(Vetor_hpsc[i] == 0):
            j = i +3;
        else:
            j = i+1;
        while j < (len(Vetor_hpsc)):
            for q in range(0,6):
                if ((proximos[q][0] == Matriz_Proteina[j][0] and
                     proximos[q][1] == Matriz_Proteina[j][1] and 
                     proximos[q][2] == Matriz_Proteina[j][2])):
                    ppa = ppa+1;
                    if(Vetor_hpsc[i] == -1 and Vetor_hpsc[j] == -1):
                        contadorhh= contadorhh +1;
                    elif(Vetor_hpsc[i] == 1 and Vetor_hpsc[j] == 1):
                        contadorpp= contadorpp +1;
                    elif(Vetor_hpsc[i] == 0 and Vetor_hpsc[j] == 0):
                        contadorbb= contadorbb +1;
                    elif (Vetor_hpsc[i] == -1 and Vetor_hpsc[j] == 1) or (Vetor_hpsc[i] == 1 and Vetor_hpsc[j] == -1):
                        contadorhp= contadorhp +1;
                    elif (Vetor_hpsc[i] == 0 and Vetor_hpsc[j] == 1) or (Vetor_hpsc[i] == 1 and Vetor_hpsc[j] == 0):
                        contadorpb= contadorpb +1;
                    elif (Vetor_hpsc[i] == -1 and Vetor_hpsc[j] == 0) or (Vetor_hpsc[i] == 0 and Vetor_hpsc[j] == -1):
                        contadorhb= contadorhb +1;
            j=j+1;
        i = i+1;
    vetor_energia =[contadorhh,contadorhp,contadorhb,contadorpb,contadorpp,contadorbb]; 
    return vetor_energia;


#Calculo de colisoes
@numba.jit
def colisoes(Matriz_Proteina, Vetor_hpsc):
    i=0;
    contador1 = 0;
    while(i < (len(Vetor_hpsc)-2)):
        j = i +1;
        #procura colisoes
        while (j < (len(Vetor_hpsc))) :
            if ((Matriz_Proteina[i][0] == Matriz_Proteina[j][0] and Matriz_Proteina[i][1] == Matriz_Proteina[j][1] and Matriz_Proteina[i][2] == Matriz_Proteina[j][2])):
                contador1 = contador1 +1;
                #print(" O ", i, " colidiu com o ", j , "\n")
            j=j+1;
        i = i+1;
        j=0;
    return contador1;

#converte para binario
@numba.jit
def dec2bin(n):
    b = ''
    if n == 0:
        b = '0'
    else:
        while n != 0:
            b = b + str(n % 2)
            n = int(n / 2)
    b=b[::-1]
    while(len(b)< 5):
        b= '0' + b
    z = [];
    for i in range(0, len(b)):
        z.append(int(b[i]))
    return z

#Verifica se o numero é maior que 24 e retorna convertido se for maior
@numba.jit
def bin2dec_24(n):
     probabilidade = random.random()
     if probabilidade > 0.5:
         n[0] = 0
     else:
         n[1] = 0
     return n;
    
    
#converte para decimal
@numba.jit
def bin2dec(n):
    numero = 0
    multiplicador=1
    x =len(n) - 1
    for i in range(x,-1,-1):
        numero = numero + multiplicador*n[i]
        multiplicador = multiplicador*2;
    return numero;    


#criando mutação        
@numba.jit
def mutacao(vetor_binario, tamanho):
    quantidade = random.randint(1,tamanho-2);
    
    for i in range (0,quantidade):
        j = random.randint(0,tamanho-2)
        prob = random.random();
        posicao = random.randint(0,4);
        if prob > 0.5:
            vetor_binario[j][posicao] = 1;
        else:
            vetor_binario[j][posicao] = 0;
        if (bin2dec(vetor_binario[j])> 24):
            bin2dec_24(vetor_binario[j]);
    return vetor_binario;
            
def mediaxyz(Matriz):
    vx=[]
    vy=[]
    vz = []
    
    for i in range (0,len(Matriz)):
        vx.append(Matriz[i][0])
        vy.append(Matriz[i][1])
        vz.append(Matriz[i][2])
    
    
    return [numpy.mean(vx),numpy.mean(vy), numpy.mean(vz)]  


#####
#####       FUNÇÕES DO SPEAII
#####




# Function 1
def func_1():
    return

# Function 2
def func_2():
    return

# Function: Initialize Variables
@numba.jit
def initial_population(population_size = 5, min_values = [-5,-5], max_values = [5,5], list_of_functions = [func_1, func_2]):
    population = pd.DataFrame(np.zeros((population_size, len(min_values))))
    for i in range (0, len(list_of_functions)):
        name = str(i+1)
        name = "Fitness_" + name
        population[name] = 0.0
    for i in range(0, population_size):
        for j in range(0, len(min_values)):
             population.iloc[i,j] = randint(min_values[j], max_values[j])      
        for k in range (1, len(list_of_functions) + 1):
            population.iloc[i,-k] = list_of_functions[-k](population.iloc[i,0:population.shape[1]-len(list_of_functions)])
    return population
    
# Function: Dominance
@numba.jit
def dominance_function(solution_1, solution_2, number_of_functions = 2):
    count = 0
    dominance = True
    for k in range (1, number_of_functions + 1):
        if (solution_1.iloc[-k] <= solution_2.iloc[-k]):
            count = count + 1
    if (count == number_of_functions):
        dominance = True
    else:
        dominance = False       
    return dominance

# Function: Raw Fitness
@numba.jit
def raw_fitness_function(population, number_of_functions = 2):    
    strength = pd.DataFrame(np.zeros((population.shape[0], 1)), columns = ['Strength'])
    raw_fitness = pd.DataFrame(np.zeros((population.shape[0], 1)), columns = ['Raw'])
    for i in range(0, population.shape[0]):
        for j in range(0, population.shape[0]):
            if(i != j):
                if dominance_function(solution_1 = population.iloc[i,:], solution_2 = population.iloc[j,:], number_of_functions = number_of_functions):
                    strength.iloc[i,0] = strength.iloc[i,0] + 1
    for i in range(0, population.shape[0]):
        for j in range(0, population.shape[0]):
            if(i != j):
                if dominance_function(solution_1 = population.iloc[i,:], solution_2 = population.iloc[j,:], number_of_functions = number_of_functions):
                    raw_fitness.iloc[j,0] = raw_fitness.iloc[j,0] + strength.iloc[i,0]
    return raw_fitness

# Function: Distance Calculations
@numba.jit
def euclidean_distance(x, y):       
    distance = 0
    for j in range(0, len(x)):   
        distance = (x.iloc[j] - y.iloc[j])**2 + distance   
    return distance**(1/2)  

# Function: Fitness
@numba.jit
def fitness_calculation(population, raw_fitness, number_of_functions = 2):
    k = int(len(population)**(1/2)) - 1
    fitness  = pd.DataFrame(np.zeros((population.shape[0], 1)), columns = ['Fitness'])
    distance = pd.DataFrame(np.zeros((population.shape[0], population.shape[0])))
    for i in range(0, population.shape[0]):
        for j in range(0, population.shape[0]):
            if(i != j):
                x = population.iloc[i, population.shape[1]-number_of_functions:].copy(deep = True)
                y = population.iloc[j, population.shape[1]-number_of_functions:].copy(deep = True)
                distance.iloc[i,j] =  euclidean_distance(x = x, y = y)                    
    for i in range(0, fitness.shape[0]):
        distance = distance.sort_values(by = i, axis = 1, ascending = True)
        fitness.iloc[i,0] = raw_fitness.iloc[i,0] + 1/(distance.iloc[i,k] + 2)
    return fitness

# Function: Sort Population by Fitness
@numba.jit
def sort_population_by_fitness(population, fitness):
    idx = np.argsort(fitness['Fitness'].values)
    fitness_new = pd.DataFrame(np.zeros((population.shape[0], 1)), columns = ['Fitness'])
    population_new = pd.DataFrame(np.zeros((population.shape[0], population.shape[1])))  
    for i in range(0, population.shape[0]):
        fitness_new.iloc[i,0] = fitness.iloc[(idx[i]),0] 
        for k in range(0, population.shape[1]):
            population_new.iloc[i,k] = population.iloc[(idx[i]),k]
    return population_new, fitness_new

# Function: Selection
#@numba.jit
#def roulette_wheel(fitness_new): 
#    fitness = pd.DataFrame(np.zeros((fitness_new.shape[0], 1)))
#    fitness['Probability'] = 0.0
#    for i in range(0, fitness.shape[0]):
#        fitness.iloc[i,0] = 1/(1+ fitness.iloc[i,0] + abs(fitness.iloc[:,0].min()))
#    fit_sum = fitness.iloc[:,0].sum()
#    fitness.iloc[0,1] = fitness.iloc[0,0]
#    for i in range(1, fitness.shape[0]):
#        fitness.iloc[i,1] = (fitness.iloc[i,0] + fitness.iloc[i-1,1])
#    for i in range(0, fitness.shape[0]):
#        fitness.iloc[i,1] = fitness.iloc[i,1]/fit_sum
#    ix = 0
#    random = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
#    for i in range(0, fitness.shape[0]):
#        if (random <= fitness.iloc[i, 1]):
#          ix = i
#          break
#    return ix
#
## Function: Offspring
#def breeding(population, fitness, min_values = [-5,-5], max_values = [5,5], mu = 1, list_of_functions = [func_1, func_2]):
#    offspring = population.copy(deep = True)
#    b_offspring = 0
#    for i in range (0, offspring.shape[0]):
#        parent_1, parent_2 = roulette_wheel(fitness), roulette_wheel(fitness)
#        while parent_1 == parent_2:
#            parent_2 = random.sample(range(0, len(population) - 1), 1)[0]
#        for j in range(0, offspring.shape[1] - len(list_of_functions)):
#            rand = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
#            rand_b = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)                                
#            if (rand <= 0.5):
#                b_offspring = 2*(rand_b)
#                b_offspring = b_offspring**(1/(mu + 1))
#            elif (rand > 0.5):  
#                b_offspring = 1/(2*(1 - rand_b))
#                b_offspring = b_offspring**(1/(mu + 1))       
#            offspring.iloc[i,j] = np.clip(((1 + b_offspring)*population.iloc[parent_1, j] + (1 - b_offspring)*population.iloc[parent_2, j])/2, min_values[j], max_values[j])           
#            if(i < population.shape[0] - 1):   
#                offspring.iloc[i+1,j] = np.clip(((1 - b_offspring)*population.iloc[parent_1, j] + (1 + b_offspring)*population.iloc[parent_2, j])/2, min_values[j], max_values[j]) 
#        for k in range (1, len(list_of_functions) + 1):
#            offspring.iloc[i,-k] = list_of_functions[-k](offspring.iloc[i,0:offspring.shape[1]-len(list_of_functions)])
#    return offspring 
#
## Function: Mutation
#@numba.jit
#def mutation(offspring, mutation_rate = 0.1, eta = 1, min_values = [-5,-5], max_values = [5,5], list_of_functions = [func_1, func_2]):
#    d_mutation = 0            
#    for i in range (0, offspring.shape[0]):
#        for j in range(0, offspring.shape[1] - len(list_of_functions)):
#            probability = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
#            if (probability < mutation_rate):
#                rand = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
#                rand_d = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)                                     
#                if (rand <= 0.5):
#                    d_mutation = 2*(rand_d)
#                    d_mutation = d_mutation**(1/(eta + 1)) - 1
#                elif (rand > 0.5):  
#                    d_mutation = 2*(1 - rand_d)
#                    d_mutation = 1 - d_mutation**(1/(eta + 1))                
#                offspring.iloc[i,j] = np.clip((offspring.iloc[i,j] + d_mutation), min_values[j], max_values[j])                        
#        for k in range (1, len(list_of_functions) + 1):
#            offspring.iloc[i,-k] = list_of_functions[-k](offspring.iloc[i,0:offspring.shape[1]-len(list_of_functions)])
#    return offspring 

# SPEA-2 Function

#def strength_pareto_evolutionary_algorithm_2(population_size = 5, archive_size = 5, mutation_rate = 0.1, min_values = [-5,-5],
#                                             max_values = [5,5], list_of_functions = [func_1, func_2], generations = 50, mu = 1, eta = 1):        
#    count = 0   
##    population = initial_population(population_size = population_size, min_values = min_values, max_values =
##                                    max_values, list_of_functions = list_of_functions) 
#    population = solution;
#    archive = initial_population(population_size = archive_size, min_values = min_values, max_values = 
#                                 max_values, list_of_functions = list_of_functions)  
#    
#
#    while (count <= generations):       
#        print("Generation = ", count)
#        population = pd.concat([population, archive])
#        raw_fitness   = raw_fitness_function(population, number_of_functions = len(list_of_functions))
#        fitness    = fitness_calculation(population, raw_fitness, number_of_functions = len(list_of_functions))        
#        population, fitness = sort_population_by_fitness(population, fitness)
#        population, archive, fitness = population.iloc[0:population_size,:],population.iloc[0:archive_size,:], fitness.iloc[0:archive_size,:]
#        population = breeding(population, fitness, mu = mu, min_values = min_values, max_values 
#                              = max_values, list_of_functions = list_of_functions)
#        population = mutation(population, mutation_rate = mutation_rate, eta = eta, min_values = min_values, max_values = 
#                              max_values, list_of_functions = list_of_functions)             
#        count = count + 1              
#    return archive

######################## Part 1 - Usage ####################################

## Schaffer Function 1
#def schaffer_f1(variables_values = [0]):
#    y = variables_values[0]**2
#    return y
#
## Schaffer Function 2
#def schaffer_f2(variables_values = [0]):
#    y = (variables_values[0]-2)**2
#    return y
#
## Shaffer Pareto Front
#schaffer = pd.DataFrame(np.arange(0.0, 2.0, 0.01))
#schaffer['Function 1'] = 0.0
#schaffer['Function 2'] = 0.0
#for i in range (0, schaffer.shape[0]):
#    schaffer.iloc[i,1] = schaffer_f1(variables_values = [schaffer.iloc[i,0]])
#    schaffer.iloc[i,2] = schaffer_f2(variables_values = [schaffer.iloc[i,0]])
#
#schaffer_1 = schaffer.iloc[:,1]
#schaffer_2 = schaffer.iloc[:,2]
#
    
#spea_2_schaffer = strength_pareto_evolutionary_algorithm_2(population_size = 200, archive_size = 50, mutation_rate = 0.05, min_values = [0], max_values = [24], list_of_functions = [schaffer_f1, schaffer_f2], generations = 100, mu = 5, eta = 5)
#
    

## Graph Pareto Front Solutions
#func_1_values = spea_2_schaffer.iloc[:,-2]
#func_2_values = spea_2_schaffer.iloc[:,-1]
#ax1 = plt.figure(figsize = (15,15)).add_subplot(111)
#plt.xlabel('Function 1', fontsize = 12)
#plt.ylabel('Function 2', fontsize = 12)
#ax1.scatter(func_1_values, func_2_values, c = 'red',   s = 25, marker = 'o', label = 'SPEA-2')
#ax1.scatter(schaffer_1,    schaffer_2,    c = 'black', s = 2,  marker = 's', label = 'Pareto Front')
#plt.legend(loc = 'upper right')
#plt.show()




#Inicio do 3DHP
arq = open( "C://3dhpsc//protein.pdb.txt",'r')
linhas = str(arq.read())
arq.close()

#Vai ler os caracters > que são padrao de inicio
Vetor_hpsc =[];
Vetor_hpsc = vetor_strings(Vetor_hpsc,linhas);
pop_size =len(linhas);
individuos = 200;
solution =  numpy.zeros((individuos ,pop_size-1), dtype = numpy.int);#
for w in range (0,individuos):
    solution[w]=[(randint(0,24)) for i in range(0,pop_size-1)];
    
function1_values = numpy.zeros(individuos, dtype = numpy.int);
function2_values = numpy.zeros(individuos, dtype = numpy.float);
energia_livre = numpy.zeros(individuos, dtype = numpy.int);
colision = numpy.zeros(individuos, dtype = numpy.int);
rgh =  numpy.zeros(individuos, dtype = numpy.float)
rgp = numpy.zeros(individuos, dtype = numpy.float)
radiusgh =  numpy.zeros(individuos, dtype = numpy.float)
radiusgp =  numpy.zeros(individuos, dtype = numpy.float) 
NH = linhas.count('H');
NP = linhas.count('P');


#preenchendo o vetor que calula a proteina esticada
solutionrgh =  numpy.zeros((pop_size-1), dtype = numpy.int);#
solutionrgh[0] = 24; 
for w in range (1,(pop_size-1)):
    solutionrgh[w]=9


#calculando maxrgh
xxx = numpy.zeros((len(Vetor_hpsc),3), dtype = numpy.int)
matriz_teste = movimentos(solutionrgh,xxx,Vetor_hpsc)
z=1
maxrgh = 0
vetor_media = mediaxyz(matriz_teste);
while(z<= len(Vetor_hpsc)):
            if(Vetor_hpsc[z]==-1):
                maxrgh = maxrgh + ((((matriz_teste[z][0]-vetor_media[0])**2)+ 
                   ((matriz_teste[z][1]-vetor_media[1])**2) +
                   ((matriz_teste[z][2]-vetor_media[2])**2)))
            z=z+2;
maxrgh= (maxrgh/NH)**0.5


Matriz_Proteina = numpy.zeros((individuos,len(Vetor_hpsc),3), dtype = numpy.int);

    
gen = 0
generations = 3000
population_size = individuos;
archive_size = round(individuos/4);
#inicio SPEAII
#Criando população inicial e Archive inicial aleatórios
#population = solution;
#archive =  numpy.zeros(((individuos/4) ,pop_size-1), dtype = numpy.int);
#for w in range (0,individuos/4):
#   archive[w]=[(randint(0,24)) for i in range(0,pop_size-1)];



#monta as matrizes no espaço  3D
for w in range(0, individuos):
    Matriz_Proteina[w] = movimentos(solution[w],Matriz_Proteina[w],Vetor_hpsc);
#Zera as funções   
function1_values = numpy.zeros(individuos, dtype = numpy.int);
function2_values = numpy.zeros(individuos, dtype = numpy.float);
energia_livre = numpy.zeros(individuos, dtype = numpy.int);
colision = numpy.zeros(individuos, dtype = numpy.int);   
#Cria Variáveis de raio de giração
rgh =  numpy.zeros(individuos, dtype = numpy.float)
rgp = numpy.zeros(individuos, dtype = numpy.float)
radiusgh =  numpy.zeros(individuos, dtype = numpy.float)
radiusgp =  numpy.zeros(individuos, dtype = numpy.float)    


#calculando o raio de giração 
for i in range (0, individuos):
    z = 1;
    vetor_media = []
    vetor_media = mediaxyz(Matriz_Proteina[i])
    while(z<= len(Vetor_hpsc)):
        if(Vetor_hpsc[z]==-1):
            rgh[i] = rgh[i] + ((((Matriz_Proteina[i][z][0]-vetor_media[0])**2)+ 
               ((Matriz_Proteina[i][z][1]-vetor_media[1])**2) +
               ((Matriz_Proteina[i][z][2]-vetor_media[2])**2)))
            
        elif(Vetor_hpsc[z]==1):
            rgp[i] = rgp[i] + ((((Matriz_Proteina[i][z][0]-vetor_media[0])**2)+ 
               ((Matriz_Proteina[i][z][1]-vetor_media[1])**2) +
               ((Matriz_Proteina[i][z][2]-vetor_media[2])**2)))
        z = z+2;
    rgh[i] = (rgh[i]/NH)**0.5    
    rgp[i] = (rgp[i]/NP)**0.5
    
    
    if (rgp[i] - rgh[i]) >= 0:
        radiusgp[i] = 1;
    else:
        radiusgp[i] = (1/(1-(rgp[i]- rgh[i])))
                 
    radiusgh[i] = maxrgh - rgh[i];
#calcula as 2 funções     
for i in range (0, individuos):
    energia1 = energia_best(Matriz_Proteina[i],Vetor_hpsc)
    function1_values [i] = energia1[0];
    energia_livre[i] = (energia1[0]*10)+(energia1[1]*-3)+(energia1[2]*-3)+(energia1[3])+(energia1[4])+(energia1[5])
    colision[i] = colisoes(Matriz_Proteina[i],Vetor_hpsc);
    #penalidades são iguais a 10
    function2_values[i] = (energia_livre[i] -(colision[i]*10))*radiusgp[i]*radiusgh[i];

vetor_energia_best = [];    
Melhor_Proteina = []
best_fitness1 = [];
best_fitness2 = [];
best3 = []
best4 = []
NH = linhas.count('H');
NP = linhas.count('P');
   

while (gen <= generations):       
    
        
    print("HNC: " , function1_values , " \n");
    print("Fitness: " , function2_values ," \n");
    print("Colisoes :" , colision ," \n");    
    print("Generation = ", gen,"\n");
    
    if(min(colision) == 0):
        conta_colision = [];
        for i in range(0, individuos):
            if colision[i] == 0:
                conta_colision.append(i);
        temporaria = 0;        
        for i in conta_colision:
            if temporaria == 0:
                melhor1 = (function1_values[i]);
                melhor2 = (function2_values[i]);
                melhor3 = i;
            else:
                if ((melhor1 <= function1_values[i]) and (melhor2 <= function2_values[i])):
                    melhor1 = (function1_values[i]);
                    melhor2 = (function2_values[i]);
                    melhor3 = i;
            temporaria = temporaria +1;             
                            
        
        if(len(Melhor_Proteina) == 0):
            Melhor_Proteina = Matriz_Proteina[melhor3]; 
            vetor_energia = energia_best(Matriz_Proteina[melhor3],Vetor_hpsc);
            f2= melhor2;
            nhc = melhor1;
            melhor_geracao = gen;
            best_fitness1.append(melhor1);
            best_fitness2.append(melhor2);
        elif(len(Melhor_Proteina) > 0 and ((nhc < melhor1 and f2 < melhor2) or (nhc <= melhor1 and f2 < melhor2) or (nhc < melhor1 and f2 <= melhor2) )):
            Melhor_Proteina = Matriz_Proteina[melhor3]; 
            vetor_energia = energia_best(Matriz_Proteina[melhor3],Vetor_hpsc);
            f2= melhor2;
            nhc = melhor1;
            melhor_geracao = gen;
            best_fitness1.append(melhor1);
            best_fitness2.append(melhor2);
        else:
            best_fitness1.append(best_fitness1[-1]);
            best_fitness2.append(best_fitness2[-1]);

        print("Vetor_Energia:" , vetor_energia, "\n")
        print("Fitness Best:" , f2, "\n")         
    else:
        if len(best_fitness1) == 0:
            best_fitness1.append(0);
            best_fitness2.append(0);
        else:
            best_fitness1.append(best_fitness1[-1]);
            best_fitness2.append(best_fitness2[-1]);    
    
    
    
    
    population = pd.DataFrame(np.arange(0,population_size, 1))
    archive =  pd.DataFrame(np.arange(0,archive_size, 1))
    population['Function 1'] = 0
    population['Function 2'] = 0
    archive['Function 1'] = 0
    archive['Function 2'] = 0
    for i in range (0, population.shape[0]):
        population.iloc[i,1] = function1_values[i]
        population.iloc[i,2] = function2_values[i]
    if(gen !=0):
        population = pd.concat([population, archive])
    raw_fitness   = raw_fitness_function(population, number_of_functions = 2)
    fitness    = fitness_calculation(population, raw_fitness, number_of_functions = 2)        
    population, fitness = sort_population_by_fitness(population, fitness)
    population, archive, fitness = population.iloc[0:population_size,:],population.iloc[0:archive_size,:], fitness.iloc[0:archive_size,:]

  
    ########Cruzamento e mutação

    if(gen<3000):
        solution2 =  numpy.zeros((individuos+(archive_size*2) ,pop_size-1), dtype = numpy.int);#
        for i in range (0, individuos):
            solution2[i] = solution[i];
            
        
        count = 0;
        count2 = individuos;
        while(count < (archive_size*2)):#lembre que mudou aqui no teste 3
            
            
            #operadores de cruzamento
            a1 = random.randint(0,round(pop_size/2))
            b1 = random.randint(a1+1,pop_size-2)
                    
            #selecionando quem vai cruzar
            c1 = random.randint(0,len(solution)-1)
            d1 = random.randint(0,len(solution)-1)
            
            #torneio
            tor=0;
            while tor <10 :
                z = random.randint(0,len(solution)-1)
                if colision[z] < colision[c1]:
                    c1 = z;
                tor = tor+1
            tor = 0
            while tor <10 :
                z = random.randint(0,len(solution)-1)
                if colision[z] < colision[d1]:
                    d1 = z;       
                tor = tor+1
    
            solution2[count2] = solution2[c1];
            solution2[count2+1] = solution[d1];
            #cruzamento
            solution2[count2][a1:b1]= solution2[d1][a1:b1]
            solution2[count2+1][a1:b1]=solution2[c1][a1:b1]
            
            #UNGER3
            #mutação
            probabilidade = random.random();
            if  probabilidade <= 0.08:
                #criando individuos binários
                solution_binario2 = numpy.zeros((2 ,pop_size-1,5), dtype = numpy.int); 
                for i in range(0,2):
                    for w in range(0,pop_size-1):
                        solution_binario2[i][w] = dec2bin(solution2[count2+i][w]) 
                    mutacao(solution_binario2[i],pop_size);
    
                #montando soluções no vetor solution2
                for i in range(0,2):
                    for w in range(0,pop_size-1):
                        solution2[count2 +i][w] = bin2dec(solution_binario2[i][w])       
            
            
            count = count+2;
            count2= count2 +2;
     
        
           
        #monta as matrizes no espaço 3D
        Matriz_Proteina2 = numpy.zeros((individuos+(archive_size*2),len(Vetor_hpsc),3), dtype = numpy.int);
        Matriz_Proteina2[0:individuos] = Matriz_Proteina
        for w in range(individuos, individuos+(archive_size*2)):
            Matriz_Proteina2[w] = movimentos(solution2[w],Matriz_Proteina2[w],Vetor_hpsc);        
        
        rgh2 =  numpy.zeros(individuos+(archive_size*2), dtype = numpy.float)
        rgp2  = numpy.zeros(individuos+(archive_size*2), dtype = numpy.float)
        radiusgh2 =  numpy.zeros(individuos+(archive_size*2), dtype = numpy.float)
        radiusgp2=  numpy.zeros(individuos+(archive_size*2), dtype = numpy.float) 
        rgh2[0:individuos] =rgh; 
        rgp2[0:individuos] =rgp;
        radiusgh2[0:individuos]=radiusgh;
        radiusgp2[0:individuos]=radiusgp;
        #calculando o raio de giração 
        for i in range (individuos, individuos+(archive_size*2)):
            z = 1;
            vetor_media = mediaxyz(Matriz_Proteina2[i]);
            while(z<= len(Vetor_hpsc)):
                if(Vetor_hpsc[z]==-1):
                    rgh2[i] = rgh2[i] + ((((Matriz_Proteina2[i][z][0]-vetor_media[0])**2)+ 
                       ((Matriz_Proteina2[i][z][1]-vetor_media[1])**2) +
                       ((Matriz_Proteina2[i][z][2]-vetor_media[2])**2)))
                    
                elif(Vetor_hpsc[z]==1):
                    rgp2[i] = rgp2[i] + ((((Matriz_Proteina2[i][z][0]-vetor_media[0])**2)+ 
                       ((Matriz_Proteina2[i][z][1]-vetor_media[1])**2) +
                       ((Matriz_Proteina2[i][z][2]-vetor_media[2])**2)))
                z = z+2;
            rgh2[i] = (rgh2[i]/NH)**0.5    
            rgp2[i] = (rgp2[i]/NP)**0.5
            
            
            if (rgp2[i] - rgh2[i]) >= 0:
                radiusgp2[i] = 1;
            else:
                radiusgp2[i] = (1/(1-(rgp2[i]- rgh2[i])))
                         
            radiusgh2[i] = maxrgh - rgh2[i];
        
        
        function1_values2 = numpy.zeros(individuos+(archive_size*2), dtype = numpy.int);
        function2_values2 = numpy.zeros(individuos+(archive_size*2), dtype = numpy.float);
        energia_livre2 = numpy.zeros(individuos+(archive_size*2), dtype = numpy.int);
        colision2 = numpy.zeros(individuos+(archive_size*2), dtype = numpy.int);
    
        function1_values2[0:individuos] = function1_values;
        energia_livre2[0:individuos] =  energia_livre;
        colision2[0:individuos] = colision;
        function2_values2[0:individuos] = function2_values;
        for i in range (individuos, individuos+(archive_size*2)):
            energia2 = energia_best(Matriz_Proteina2[i],Vetor_hpsc)
            function1_values2 [i] = energia2[0];
            energia_livre2[i] = (energia2[0]*10)+(energia2[1]*-3)+(energia2[2]*-3)+(energia2[3])+(energia2[4])+(energia2[5])
            colision2[i] = colisoes(Matriz_Proteina2[i],Vetor_hpsc);
            #penalidades são iguais a 10
            function2_values2[i] = energia_livre2[i] -(colision2[i]*10) *radiusgp2[i]*radiusgh2[i];      
        population_new = pd.DataFrame(np.arange(0,population_size+(archive_size*2), 1))
        population_new['Function 1'] = 0
        population_new['Function 2'] = 0
        for i in range (0,population_new.shape[0]):
            population_new.iloc[i,1] = function1_values2[i]
            population_new.iloc[i,2] = function2_values2[i]
        raw_fitness_new   = raw_fitness_function(population_new, number_of_functions = 2)
        fitness_new    = fitness_calculation(population_new, raw_fitness_new, number_of_functions = 2)        
        population_new, fitness_new = sort_population_by_fitness(population_new, fitness_new)
       
        for s in range(0,population.shape[0]):
            solution[s] = solution2[int(population_new.iloc[s,0])]
            function1_values[s] = function1_values2[int(population_new.iloc[s,0])]
            function1_values[s] = function2_values2[int(population_new.iloc[s,0])]
            Matriz_Proteina[s] = Matriz_Proteina2[int(population_new.iloc[s,0])]
            energia_livre[s] = energia_livre2[int(population_new.iloc[s,0])]
            colision[s] = colision2[int(population_new.iloc[s,0])]
          
    gen = gen + 1       

fim = time.time();

tempo  = (abs(fim-inicio))/3600



      
        
        





