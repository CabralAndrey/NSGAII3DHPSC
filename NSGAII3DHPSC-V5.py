import numpy
import math
from random import randint
import random
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pylab
import matplotlib.pyplot as plt
from decimal import Decimal
import time
import numba
from statistics import mean
from numpy import std

##PLOT
#
##PLOT
#
##PLOT
#
##PLOT
#
##PLOT
#
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
#    #ax.plot(xs , ys, zs,zdir='z');
    
#
#    zline1 = numpy.zeros(len(Vetor_hpsc), dtype = numpy.int);
#    xline1 = numpy.zeros(len(Vetor_hpsc), dtype = numpy.int);
#    yline1 = numpy.zeros(len(Vetor_hpsc), dtype = numpy.int);    
    
#    zline1 = numpy.zeros(6, dtype = numpy.int);
#    xline1 = numpy.zeros(6, dtype = numpy.int);
#    yline1 = numpy.zeros(6, dtype = numpy.int); 
    xline1 =[];
    zline1 = [];
    yline1 = [];
    u=0;
#    while u < 6 :
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
    print(u)
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
    Vetor_Ref = numpy.zeros(8, dtype = numpy.int) ;
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
        print(Vetor_Ref)    
        
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
    print(Vetor_Ref)
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



#Função para executar o tipo rápido não dominado do NSGA-II    
def fast_non_dominated_sort(values1, values2):
    S=[[] for i in range(0,len(values1))]
    front = [[]]
    n=[0 for i in range(0,len(values1))]
    rank = [0 for i in range(0, len(values1))]

    for p in range(0,len(values1)):
        S[p]=[]
        n[p]=0
        for q in range(0, len(values1)):
            if (values1[p] > values1[q] and values2[p] > values2[q]) or (values1[p] >= values1[q] and values2[p] > values2[q]) or (values1[p] > values1[q] and values2[p] >= values2[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[q] > values1[p] and values2[q] > values2[p]) or (values1[q] >= values1[p] and values2[q] > values2[p]) or (values1[q] > values1[p] and values2[q] >= values2[p]):
                n[p] = n[p] + 1
        if n[p]==0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)
                

    i = 0
    while(front[i] != []):
        Q=[]
        for p in front[i]:
            for q in S[p]:
                n[q] =n[q] - 1
                if( n[q]==0):
                    rank[q]=i+1
                    if q not in Q:
                        Q.append(q)
        i = i+1
        front.append(Q)

    del front[len(front)-1]
    return front    
    
def crowding_distance(values1, values2, front):
    distance = [0 for i in range(0,len(front))]
    sorted1 = sort_by_values(front, values1[:])
    sorted2 = sort_by_values(front, values2[:])
    distance[0] = 4444444444444444
    distance[len(front) - 1] = 4444444444444444
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted1[k+1]] - values2[sorted1[k-1]])/(max(values1)-min(values1))
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted2[k+1]] - values2[sorted2[k-1]])/(max(values2)-min(values2))
    return distance

def sort_by_values(list1,values):
    sorted_list = []
    values = values.astype(numpy.float);
    while(len(sorted_list)!=len(list1)):
        if index_of(min(values),values) in list1:
            sorted_list.append(index_of(min(values),values))
        values[index_of(min(values),values)] = math.inf;
    return sorted_list    

def sort_by_values2(list1,values):
    sorted_list = []
    #values = values.astype(numpy.float);
    while(len(sorted_list)!=len(list1)):
        if index_of(min(values),values) in list1:
            sorted_list.append(index_of(min(values),values))
        values[index_of(min(values),values)] = math.inf;
    return sorted_list  

#Function to find index of list
def index_of(a,list):
    for i in range(0,len(list)):
        if list[i] == a:
            return i
    return -1

#Function to carry out the crossover
#def crossover(a,b):
#    r=random.random()
#    if r>0.5:
#        return abs(round(((a+b)/2)))
#    else:
#        return abs(round(((a-b)/2)))

##Function to carry out the mutation operator
#def mutation(solucao):
#    mutation_prob = random.random()
#    if mutation_prob <0.08:
#        solucao = (randint(0,24))
#    return solucao

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
    
    
    
    
######Programa Começa Aqui!!!!
######Programa Começa Aqui!!!!
######Programa Começa Aqui!!!!
######Programa Começa Aqui!!!!
######Programa Começa Aqui!!!!
######Programa Começa Aqui!!!! #Le arquivo da proteina
#inicio = time.time()
#arq = open( "C://3dhpsc//protein.pdb.txt",'r')
#linhas = str(arq.read())
#arq.close()
#
#
#
##Vai ler os caracters > que são padrao de inicio
#Vetor_hpsc =[];
#Vetor_hpsc = vetor_strings(Vetor_hpsc,linhas);


##inicia NSGA2
##APenas pora teste    
linhas = 'HPH'
Vetor_hpsc =[];
Vetor_hpsc = vetor_strings(Vetor_hpsc,linhas);
Matriz_Proteina = numpy.zeros((individuos,len(Vetor_hpsc),3), dtype = numpy.int);
###################
pop_size =len(linhas);


individuos = 200;
min_x= 0
max_x= 24
contador = 0;
#define as soluções
solution =  numpy.zeros((individuos ,pop_size-1), dtype = numpy.int);#
for w in range (0,individuos):
    solution[w]=[(randint(0,24)) for i in range(0,pop_size-1)];
   
max_gen = 3000;
gen_no = 0;
    ####para solução 1
function1_values = numpy.zeros(individuos, dtype = numpy.int);
function2_values = numpy.zeros(individuos, dtype = numpy.float);
energia_livre = numpy.zeros(individuos, dtype = numpy.int);
colision = numpy.zeros(individuos, dtype = numpy.int);

#para solução2
function1_values2 = numpy.zeros(individuos*2, dtype = numpy.int);
function2_values2 = numpy.zeros(individuos*2, dtype = numpy.float);
energia_livre2 = numpy.zeros(individuos*2, dtype = numpy.int);
colision2 = numpy.zeros(individuos*2, dtype = numpy.int);

best_fitness1 = [];
best_fitness2 = [];
best3 = []
best4 = []



dizimacao_rate = round(individuos/2) + round(individuos/4);
dizimacao_cont = 0;\

vetor_energia_best = [];    
Melhor_Proteina = []


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
#
#
#
#
#
#
#
#
#
#
#
##inicia NSGA II
#Matriz_Proteina = numpy.zeros((individuos,len(Vetor_hpsc),3), dtype = numpy.int);
#Matriz_Proteina2 = numpy.zeros((individuos*2,len(Vetor_hpsc),3), dtype = numpy.int);
#
#while(gen_no<=max_gen):
#    #realiza a dizimação em individuos aleatórios sem repetir
#    if (dizimacao_cont == 20) and gen_no > 0:
#        result = []
#        while len(result) < dizimacao_rate:
#            r = randint(0, individuos -1)
#            if r not in result:
#                result.append(r)
#        solution_dizimacao =  numpy.zeros((dizimacao_rate ,pop_size-1), dtype = numpy.int);#
#        for w in range (0,dizimacao_rate):
#            solution_dizimacao[w]=[(randint(0,24)) for i in range(0,pop_size-1)];
#        for i in range(0,dizimacao_rate):
#            solution[result[i]] = solution_dizimacao[i]
#            
#        dizimacao_cont = 0;
#    print("solution :" , solution ," \n");
#    print("Dizimação contador:" , dizimacao_cont, "\n")
#    
#    
#    
#     #monta as matrizes no espaço  3D
#    for w in range(0, individuos):
#        Matriz_Proteina[w] = movimentos(solution[w],Matriz_Proteina[w],Vetor_hpsc);
#        
#        
#        
#        
#        
#        
#    function1_values = numpy.zeros(individuos, dtype = numpy.int);
#    function2_values = numpy.zeros(individuos, dtype = numpy.float);
#    energia_livre = numpy.zeros(individuos, dtype = numpy.int);
#    colision = numpy.zeros(individuos, dtype = numpy.int);   
#    
#    
#    
#    rgh =  numpy.zeros(individuos, dtype = numpy.float)
#    rgp = numpy.zeros(individuos, dtype = numpy.float)
#    radiusgh =  numpy.zeros(individuos, dtype = numpy.float)
#    radiusgp =  numpy.zeros(individuos, dtype = numpy.float)    
#    
#    
#    #calculando o raio de giração 
#    for i in range (0, individuos):
#        z = 1;
#        vetor_media = []
#        vetor_media = mediaxyz(Matriz_Proteina[i])
#        while(z<= len(Vetor_hpsc)):
#            if(Vetor_hpsc[z]==-1):
#                rgh[i] = rgh[i] + ((((Matriz_Proteina[i][z][0]-vetor_media[0])**2)+ 
#                   ((Matriz_Proteina[i][z][1]-vetor_media[1])**2) +
#                   ((Matriz_Proteina[i][z][2]-vetor_media[2])**2)))
#                
#            elif(Vetor_hpsc[z]==1):
#                rgp[i] = rgp[i] + ((((Matriz_Proteina[i][z][0]-vetor_media[0])**2)+ 
#                   ((Matriz_Proteina[i][z][1]-vetor_media[1])**2) +
#                   ((Matriz_Proteina[i][z][2]-vetor_media[2])**2)))
#            z = z+2;
#        rgh[i] = (rgh[i]/NH)**0.5    
#        rgp[i] = (rgp[i]/NP)**0.5
#        
#        
#        if (rgp[i] - rgh[i]) >= 0:
#            radiusgp[i] = 1;
#        else:
#            radiusgp[i] = (1/(1-(rgp[i]- rgh[i])))
#                     
#        radiusgh[i] = maxrgh - rgh[i];
#
#
#    for i in range (0, individuos):
#        energia1 = energia_best(Matriz_Proteina[i],Vetor_hpsc)
#        function1_values [i] = energia1[0];
#        energia_livre[i] = (energia1[0]*10)+(energia1[1]*-3)+(energia1[2]*-3)+(energia1[3])+(energia1[4])+(energia1[5])
#        colision[i] = colisoes(Matriz_Proteina[i],Vetor_hpsc);
#        #penalidades são iguais a 10
#        function2_values[i] = (energia_livre[i] -(colision[i]*10))*radiusgp[i]*radiusgh[i];
#    print("HNC: " , function1_values , " \n");
#    print("Fitness: " , function2_values ," \n");
#    print("Colisoes :" , colision ," \n");
#    
#    
#    if(min(colision) == 0):
#        conta_colision = [];
#        for i in range(0, individuos):
#            if colision[i] == 0:
#                conta_colision.append(i);
#        temporaria = 0;        
#        for i in conta_colision:
#            if temporaria == 0:
#                melhor1 = (function1_values[i]);
#                melhor2 = (function2_values[i]);
#                melhor3 = i;
#            else:
#                if ((melhor1 <= function1_values[i]) and (melhor2 <= function2_values[i])):
#                    melhor1 = (function1_values[i]);
#                    melhor2 = (function2_values[i]);
#                    melhor3 = i;
#            temporaria = temporaria +1;             
#                            
#        
#        if(len(Melhor_Proteina) == 0):
#            Melhor_Proteina = Matriz_Proteina[melhor3]; 
#            vetor_energia = energia_best(Matriz_Proteina[melhor3],Vetor_hpsc);
#            f2= melhor2;
#            nhc = melhor1;
#            dizimacao_cont = 0;
#            melhor_geracao = gen_no;
#            best_fitness1.append(melhor1);
#            best_fitness2.append(melhor2);
#        elif(len(Melhor_Proteina) > 0 and ((nhc < melhor1 and f2 < melhor2) or (nhc <= melhor1 and f2 < melhor2) or (nhc < melhor1 and f2 <= melhor2) )):
#            dizimacao_cont = 0;
#            Melhor_Proteina = Matriz_Proteina[melhor3]; 
#            vetor_energia = energia_best(Matriz_Proteina[melhor3],Vetor_hpsc);
#            f2= melhor2;
#            nhc = melhor1;
#            melhor_geracao = gen_no;
#            best_fitness1.append(melhor1);
#            best_fitness2.append(melhor2);
#        else:
#            dizimacao_cont = dizimacao_cont +1;
#            best_fitness1.append(best_fitness1[-1]);
#            best_fitness2.append(best_fitness2[-1]);
#
#        print("Vetor_Energia:" , vetor_energia, "\n")
#        print("Fitness Best:" , f2, "\n")         
#    else:
#        dizimacao_cont = dizimacao_cont +1;
#        if len(best_fitness1) == 0:
#            best_fitness1.append(0);
#            best_fitness2.append(0);
#        else:
#            best_fitness1.append(best_fitness1[-1]);
#            best_fitness2.append(best_fitness2[-1]);
#    
#    
#    
#    
#    
#    non_dominated_sorted_solution = fast_non_dominated_sort(function1_values[:],function2_values[:]);
##    print("The best front for Generation number ",gen_no, " is");
##    for valuez in non_dominated_sorted_solution[0]:
##        print(solution[valuez],end=" ")
##        print("\n")
#    crowding_distance_values=[];
#    for i in range(0,len(non_dominated_sorted_solution)):
#        crowding_distance_values.append(crowding_distance(function1_values[:],function2_values[:],non_dominated_sorted_solution[i][:]))
#    
#    #Generating offsprings
#    solution2 =  numpy.zeros((individuos*2 ,pop_size-1), dtype = numpy.int);#
#    for i in range (0, individuos):
#        solution2[i] = solution[i];
#        
#    
#    
#    
#    #convertendo para binario(codificação OK)
#    #solution_binario2 = numpy.zeros((individuos*2 ,pop_size-1,5), dtype = numpy.int); 
#    
##    for i in range(0,individuos):
##        for w in range(0,pop_size-1):
##            solution_binario2[i][w] = dec2bin(solution[i][w])  
#    
#    count = 0;
#    count2 = individuos;
#    while(count < individuos):#lembre que mudou aqui no teste 3
#        
#        
#        #operadores de cruzamento
#        a1 = random.randint(0,round(pop_size/2))
#        b1 = random.randint(a1+1,pop_size-2)
#                
#        #selecionando quem vai cruzar
#        c1 = random.randint(0,len(solution)-1)
#        d1 = random.randint(0,len(solution)-1)
#        
#        #torneio
#        tor=0;
#        while tor <10 :
#            z = random.randint(0,len(solution)-1)
#            if colision[z] < colision[c1]:
#                c1 = z;
#            tor = tor+1
#        tor = 0
#        while tor <10 :
#            z = random.randint(0,len(solution)-1)
#            if colision[z] < colision[d1]:
#                d1 = z;       
#            tor = tor+1
#
#        solution2[count2] = solution2[c1];
#        solution2[count2+1] = solution[d1];
#        #cruzamento
#        solution2[count2][a1:b1]= solution2[d1][a1:b1]
#        solution2[count2+1][a1:b1]=solution2[c1][a1:b1]
#        
#        #UNGER3
#        #mutação
#        probabilidade = random.random();
#        if  probabilidade <= 0.08:
#            #criando individuos binários
#            solution_binario2 = numpy.zeros((2 ,pop_size-1,5), dtype = numpy.int); 
#            for i in range(0,2):
#                for w in range(0,pop_size-1):
#                    solution_binario2[i][w] = dec2bin(solution2[count2+i][w]) 
#                mutacao(solution_binario2[i],pop_size);
#
#            #montando soluções no vetor solution2
#            for i in range(0,2):
#                for w in range(0,pop_size-1):
#                    solution2[count2 +i][w] = bin2dec(solution_binario2[i][w])       
#        
#        
#        count = count+2;
#        count2= count2 +2;
# 
#    
#       
#    #monta as matrizes no espaço 3D
#    Matriz_Proteina2 = numpy.zeros((individuos*2,len(Vetor_hpsc),3), dtype = numpy.int);
#    Matriz_Proteina2[0:individuos] = Matriz_Proteina
#    for w in range(individuos, individuos*2):
#        Matriz_Proteina2[w] = movimentos(solution2[w],Matriz_Proteina2[w],Vetor_hpsc);        
#    
#    rgh2 =  numpy.zeros(individuos*2, dtype = numpy.float)
#    rgp2  = numpy.zeros(individuos*2, dtype = numpy.float)
#    radiusgh2 =  numpy.zeros(individuos*2, dtype = numpy.float)
#    radiusgp2=  numpy.zeros(individuos*2, dtype = numpy.float) 
#    rgh2[0:individuos] =rgh; 
#    rgp2[0:individuos] =rgp;
#    radiusgh2[0:individuos]=radiusgh;
#    radiusgp2[0:individuos]=radiusgp;
#        #calculando o raio de giração 
#    for i in range (individuos, individuos*2):
#        z = 1;
#        vetor_media = mediaxyz(Matriz_Proteina2[i]);
#        while(z<= len(Vetor_hpsc)):
#            if(Vetor_hpsc[z]==-1):
#                rgh2[i] = rgh2[i] + ((((Matriz_Proteina2[i][z][0]-vetor_media[0])**2)+ 
#                   ((Matriz_Proteina2[i][z][1]-vetor_media[1])**2) +
#                   ((Matriz_Proteina2[i][z][2]-vetor_media[2])**2)))
#                
#            elif(Vetor_hpsc[z]==1):
#                rgp2[i] = rgp2[i] + ((((Matriz_Proteina2[i][z][0]-vetor_media[0])**2)+ 
#                   ((Matriz_Proteina2[i][z][1]-vetor_media[1])**2) +
#                   ((Matriz_Proteina2[i][z][2]-vetor_media[2])**2)))
#            z = z+2;
#        rgh2[i] = (rgh2[i]/NH)**0.5    
#        rgp2[i] = (rgp2[i]/NP)**0.5
#        
#        
#        if (rgp2[i] - rgh2[i]) >= 0:
#            radiusgp2[i] = 1;
#        else:
#            radiusgp2[i] = (1/(1-(rgp2[i]- rgh2[i])))
#                     
#        radiusgh2[i] = maxrgh - rgh2[i];
#    
#    
#    
#    
#    
#    
#    
#    
#    
#    function1_values2 = numpy.zeros(individuos*2, dtype = numpy.int);
#    function2_values2 = numpy.zeros(individuos*2, dtype = numpy.float);
#    energia_livre2 = numpy.zeros(individuos*2, dtype = numpy.int);
#    colision2 = numpy.zeros(individuos*2, dtype = numpy.int);
#
#    function1_values2[0:individuos] = function1_values;
#    energia_livre2[0:individuos] =  energia_livre;
#    colision2[0:individuos] = colision;
#    function2_values2[0:individuos] = function2_values;
#    for i in range (individuos, individuos*2):
#        energia2 = energia_best(Matriz_Proteina2[i],Vetor_hpsc)
#        function1_values2 [i] = energia2[0];
#        energia_livre2[i] = (energia2[0]*10)+(energia2[1]*-3)+(energia2[2]*-3)+(energia2[3])+(energia2[4])+(energia2[5])
#        colision2[i] = colisoes(Matriz_Proteina2[i],Vetor_hpsc);
#        #penalidades são iguais a 10
#        function2_values2[i] = (energia_livre2[i] -(colision2[i]*10)) *radiusgp2[i]*radiusgh2[i];
#        
#    non_dominated_sorted_solution2 = fast_non_dominated_sort(function1_values2[:],function2_values2[:]);
#     
##        
##        
#    #Verificando novas soluções   
#   
#    crowding_distance_values2=[]
#    for i in range(0,len(non_dominated_sorted_solution2)):
#        crowding_distance_values2.append(crowding_distance(function1_values2[:],function2_values2[:],non_dominated_sorted_solution2[i][:]))
#    new_solution= []
#    
#    
#    
#    for i in range(0,len(non_dominated_sorted_solution2)):
#        non_dominated_sorted_solution2_1 = [index_of(non_dominated_sorted_solution2[i][j],non_dominated_sorted_solution2[i] ) for j in range(0,len(non_dominated_sorted_solution2[i]))]
#        front22 = sort_by_values2(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])
#        front = [non_dominated_sorted_solution2[i][front22[j]] for j in range(0,len(non_dominated_sorted_solution2[i]))]
#        front.reverse()
#        for value in front:
#            new_solution.append(value)
#            if(len(new_solution)==individuos):
#                break
#        if (len(new_solution) == individuos):
#            break
#    j = 0;
#    for i in new_solution:
#        solution[j] = solution2[i]
#        j = j+1;
#    print(gen_no)
#    gen_no = gen_no +1;
#
#
#fim = time.time();
#
#tempo = abs(inicio-fim)





#
#best_fitness2a = [j * -1 for j in best_fitness2]
#plt.figure()
#plt.scatter(best_fitness1,best_fitness2a)
#plt.plot(best_fitness1,best_fitness2a)
#plt.xlabel('F1 - NCH', fontsize=15)
#plt.ylabel('F2 - Energia', fontsize=15)

#plt.figure()
#plt.scatter(best_fitness2,best_fitness1)
#plt.plot(best_fitness2,best_fitness1)
#plt.xlabel('F2 - Energia', fontsize=15)
#plt.ylabel('F1 - NCH', fontsize=15)
#
#
#
#plt.figure()
#geraca = numpy.arange(0,len(best_fitness1),1)
#plt.xlabel('Gerações', fontsize = 12)
#plt.ylabel('Function 1', fontsize = 12)
#plt.plot(geraca, best_fitness1, c = 'red', label = 'NSGA II')
#plt.show()
#
#plt.figure()
#
#geraca = numpy.arange(0,len(best_fitness1),1)
#plt.xlabel('Gerações', fontsize = 12)
#plt.ylabel('Function 2', fontsize = 12)
#plt.plot(geraca, best_fitness2, c = 'red', label = 'NSGA II')
#plt.show()
#
#
#
#
#plt.scatter(geraca, best_fitness1, c = 'red',   s = 25)


#ppp =  [j * -1 for j in best_fitness2019]
#plt.scatter(ppp,best_fitness1019, marker='x')
#
#
#
##
#best_fitness2a = [j * -1 for j in best_fitness2]
#plt.figure()
#plt.scatter(best_fitness1,best_fitness2a)
#plt.plot(best_fitness1,best_fitness2a)
#plt.xlabel('F1 - NCH', fontsize=15)
#plt.ylabel('F2 - Energia', fontsize=15)
#
#
### Graph Pareto Front Solutions
##func_1_values = spea_2_schaffer.iloc[:,-2]
##func_2_values = spea_2_schaffer.iloc[:,-1]
##ax1 = plt.figure(figsize = (15,15)).add_subplot(111)
##plt.xlabel('Function 1', fontsize = 12)
##plt.ylabel('Function 2', fontsize = 12)
##ax1.scatter(func_1_values, func_2_values, c = 'red',   s = 25, marker = 'o', label = 'SPEA-2')
##ax1.scatter(schaffer_1,    schaffer_2,    c = 'black', s = 2,  marker = 's', label = 'Pareto Front')
##plt.legend(loc = 'upper right')
##plt.show()
#
#
    


#

#mean - media
    #std - desvio padrão
    #var - variancia.


#mpediua e desvio padrão de hnc e f2
#media de tempo




#
#
#
##
#plt.figure()
#plt.scatter(best_fitness1,best_fitness2a)
#plt.plot(best_fitness1,best_fitness2a)
#plt.xlabel('HNC', fontsize=15)
#plt.ylabel('F2 ', fontsize=15)
#    













################Daqui começa os plots


# =============================================================================
# best_fitness2a = [j * -1 for j in best_fitness2000]
# best_fitness1 = best_fitness1000
# 
# plt.figure()
# plt.scatter(best_fitness2a,best_fitness1,marker = "x")
# #plt.plot(best_fitness2a,best_fitness1)
# plt.xlabel('Função f2', fontsize=15)
# plt.ylabel('F1 - NCH', fontsize=15)
# plt.show()
# 
# 
# plt.figure()
# geraca = numpy.arange(0,len(best_fitness1),1)
# plt.xlabel('Gerações', fontsize = 12)
# plt.ylabel('HNC', fontsize = 12)
# plt.scatter(geraca, best_fitness1, marker = "x")
# plt.show()
# 
# plt.figure()
# 
# geraca = numpy.arange(0,len(best_fitness2a),1)
# plt.xlabel('Gerações', fontsize = 12)
# plt.ylabel('Função f2', fontsize = 12)
# plt.scatter(geraca, best_fitness2a, marker = "x")
# plt.show()
# 
# 
# plotagem(Melhor_Proteina000)
# 
#  
# 
# 
# 
# 
# 
# =============================================================================


























