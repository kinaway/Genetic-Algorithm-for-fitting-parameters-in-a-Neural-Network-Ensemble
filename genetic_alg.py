from random import (choice, random, randint, seed)
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from read_data import read_data_cergy


data_set = read_data_cergy('./data_cergy/data_cergy/data_cergy.xlsx')
X = MinMaxScaler().fit_transform(data_set['points'])
y = data_set['targets'].transpose()
ind = []
seed(a=None)

#Creates Test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=None)

#Creates Training Sets
X_train_ensemble = []
y_train_ensemble = []
ind = 0
for set in range(0,10):
    X_train_ensemble.append([])
    y_train_ensemble.append([])
    for i in range(0,X_train.shape[0]):
        ind = randint(0, X_train.shape[0] - 1)
        X_train_ensemble[set].append(X_train[ind])
        y_train_ensemble[set].append(y_train[ind])
X_train_ensemble = np.array(X_train_ensemble)
y_train_ensemble = np.array(y_train_ensemble)


class Individuo:
    _numRedes = 10
    def __init__(self, gene):
        self.gene = gene
        self.fitness = 100

    def getFitness(self):
        ensemble_output = []
        ensemble_output.append(np.zeros(15))
        ensemble_output.append(np.zeros(15))
        ensemble_output = np.array(ensemble_output).transpose()
        for j in range(0,self._numRedes):
            if self.gene[j][0] == 0:
                solver_aux = 'lbfgs'
            else:
                solver_aux = 'lbfgs'

            if self.gene[j][1] == 0 and self.gene[j][2] == 0:
                activation_aux = 'identity'
            elif self.gene[j][1] == 0 and self.gene[j][2] == 1:
                activation_aux = 'logistic'
            elif self.gene[j][1] == 1 and self.gene[j][2] == 0:
                activation_aux = 'tanh'
            else:
                activation_aux = 'relu'
        
            if self.gene[j][3] == 0 and self.gene[j][4] == 0:
                alpha_aux = 0.0001
            elif self.gene[j][3] == 0 and self.gene[j][4] == 1:
                alpha_aux = 0.001
            elif self.gene[j][3] == 1 and self.gene[j][4] == 0:
                alpha_aux = 0.00001
            else:
                alpha_aux = 0.0015
        
            if self.gene[j][5] == 0 and self.gene[j][6] == 0:
                batch_aux = 60
            elif self.gene[j][5] == 0 and self.gene[j][6] == 1:
                batch_aux = 30
            elif self.gene[j][5] == 1 and self.gene[j][6] == 0:
                batch_aux = 45
            else:
                batch_aux = 15
        
            if self.gene[j][7] == 0:
                learning_aux = 'constant'
            else:
                learning_aux = 'adaptive'
            
            if self.gene[j][8] == 0 and self.gene[j][9] == 0:
                iter_aux = 200
            elif self.gene[j][8] == 0 and self.gene[j][9] == 1:
                iter_aux = 400
            elif self.gene[j][8] == 1 and self.gene[j][9] == 0:
                iter_aux = 600
            else:
                iter_aux = 1200
        
            if self.gene[j][10] == 0:
                shuffle_aux = True
            else:
                shuffle_aux = False
        
        
            if self.gene[j][11] == 0 and self.gene[j][12] == 0:
                momentum_aux = 0.2
            elif self.gene[j][11] == 0 and self.gene[j][12] == 1:
                momentum_aux = 0.5
            elif self.gene[j][11] == 1 and self.gene[j][12] == 0:
                momentum_aux = 0.9
            else:
                momentum_aux = 1
        
            if self.gene[j][13] == 0:
                nesterovs_aux = True
            else:
                nesterovs_aux = False
        
            if self.gene[j][14] == 0:
                early_aux = True
            else:
                early_aux = False
        
            if self.gene[j][15] == 0 and self.gene[j][16] == 0:
                validation_aux = 0.1
            elif self.gene[j][15] == 0 and self.gene[j][16] == 1:
                validation_aux = 0.15
            elif self.gene[j][15] == 1 and self.gene[j][16] == 0:
                validation_aux = 0.2
            else:
                validation_aux = 0
        
            if self.gene[j][17] == 0 and self.gene[j][18] == 0 and self.gene[j][19] == 0 and self.gene[j][20] == 0:
                hidden_aux = (10,)
            elif self.gene[j][17] == 0 and self.gene[j][18] == 0 and self.gene[j][19] == 0 and self.gene[j][20] == 1:
                hidden_aux = (50,)
            elif self.gene[j][17] == 0 and self.gene[j][18] == 0 and self.gene[j][19] == 1 and self.gene[j][20] == 0:
                hidden_aux = (100,)
            elif self.gene[j][17] == 0 and self.gene[j][18] == 0 and self.gene[j][19] == 1 and self.gene[j][20] == 1:
                hidden_aux = (10, 10,)
            elif self.gene[j][17] == 0 and self.gene[j][18] == 1 and self.gene[j][19] == 0 and self.gene[j][20] == 0:
                hidden_aux = (10, 50,)
            elif self.gene[j][17] == 0 and self.gene[j][18] == 1 and self.gene[j][19] == 0 and self.gene[j][20] == 1:
                hidden_aux = (10, 100,)
            elif self.gene[j][17] == 0 and self.gene[j][18] == 1 and self.gene[j][19] == 1 and self.gene[j][20] == 0:
                hidden_aux = (50, 10,)
            elif self.gene[j][17] == 0 and self.gene[j][18] == 1 and self.gene[j][19] == 1 and self.gene[j][20] == 1:
                hidden_aux = (50, 50,)
            elif self.gene[j][17] == 1 and self.gene[j][18] == 0 and self.gene[j][19] == 0 and self.gene[j][20] == 0:
                hidden_aux = (50, 100,)
            elif self.gene[j][17] == 1 and self.gene[j][18] == 0 and self.gene[j][19] == 0 and self.gene[j][20] == 1:
                hidden_aux = (100, 10,)
            elif self.gene[j][17] == 1 and self.gene[j][18] == 0 and self.gene[j][19] == 1 and self.gene[j][20] == 0:
                hidden_aux = (100, 50,)
            elif self.gene[j][17] == 1 and self.gene[j][18] == 0 and self.gene[j][19] == 1 and self.gene[j][20] == 1:
                hidden_aux = (100, 100,)
            elif self.gene[j][17] == 1 and self.gene[j][18] == 1 and self.gene[j][19] == 0 and self.gene[j][20] == 0:
                hidden_aux = (10, 10, 10,)
            elif self.gene[j][17] == 1 and self.gene[j][18] == 1 and self.gene[j][19] == 0 and self.gene[j][20] == 1:
                hidden_aux = (50, 50, 50,)
            elif self.gene[j][17] == 1 and self.gene[j][18] == 1 and self.gene[j][19] == 1 and self.gene[j][20] == 0:
                hidden_aux = (100, 100, 100,)
            else:
                hidden_aux = (10, 100, 50,)
        
            mlp = MLPRegressor(solver = solver_aux, hidden_layer_sizes = hidden_aux, activation = activation_aux, alpha = alpha_aux, batch_size = batch_aux, learning_rate = learning_aux, max_iter = iter_aux, shuffle = shuffle_aux, momentum = momentum_aux, nesterovs_momentum = nesterovs_aux, validation_fraction = validation_aux, random_state = None)
            mlp.fit(X_train_ensemble[j], y_train_ensemble[j])
            ensemble_output += mlp.predict(X_test)
            
        ensemble_output = ensemble_output/self._numRedes
        self.fitness =  np.mean(abs(ensemble_output - y_test)*100/y_test, axis=None)

    def mate(self, mate):
        pivot = randint(0, self._numRedes - 1)
        gene1 = np.zeros((self._numRedes, 21))
        gene2 = np.zeros((self._numRedes, 21))
        for i in range(0, self._numRedes):
            for j in range(0, 21):
                if i < pivot:
                    gene1[i][j] = self.gene[i][j]
                    gene2[i][j] = mate.gene[i][j]
                else:
                    gene1[i][j] = mate.gene[i][j]
                    gene2[i][j] = self.gene[i][j]
        return Individuo(gene1), Individuo(gene2)

    def mutate(self):
        for i in range(0, 40):
            rede = randint(0, self._numRedes - 1)
            atributo = randint(0, 20)
            if self.gene[rede][atributo] == 0:
                self.gene[rede][atributo] = 1
            else:
                self.gene[rede][atributo] = 0
    
    @staticmethod
    def gen_random():
        gene = np.random.random((Individuo._numRedes, 21))
        for i in range(0, Individuo._numRedes):
            for j in range(0, 21):
                if gene[i][j] > 0.5:
                    gene[i][j] = 1
                else:
                    gene[i][j] = 0
        ind = Individuo(gene)
        ind.getFitness()
        return ind



class Population:
    _tournamentSize = 2

    def __init__(self, size=10, crossover=0.8, mutation=0.03):
        self.size = size
        self.crossover = crossover
        self.mutation = mutation

        listInd = []
        for i in range(size):
            listInd.append(Individuo.gen_random())
        self.population = list(sorted(listInd, key=lambda x:x.fitness))

    def selectParents(self):
        best1 = choice(self.population)
        best2 = choice(self.population)
        for i in range(self._tournamentSize):
            cont = choice(self.population)
            if cont.fitness < best1.fitness:
                best1 = cont
            cont = choice(self.population)
            if cont.fitness < best2.fitness:
                best2 = cont
        return (best1, best2)
    
    def evolve(self):
        size = len(self.population)
        index = 1
        buf = self.population[:index] #Coloca melhor individuo na prox geracao
        while (index < size):
            if random() <= self.crossover:
                (p1, p2) = self.selectParents()
                children = p1.mate(p2)
                for c in children:
                    if random() <= self.mutation:
                        c.mutate()
                    c.getFitness()
                    buf.append(c)
                index += 2
            else:
                c = self.population[index]
                if random() <= self.mutation:
                    c.mutate()
                    c.getFitness()
                buf.append(c)
                index += 1
        self.population = list(sorted(buf[:size], key=lambda x: x.fitness))
                   
        

if __name__ == "__main__":
    maxGenerations = 1000
    pop = Population(size=10, crossover=0.8, mutation=0.03)

    for i in range(1, maxGenerations + 1):
        print("Generation %d: %f" % (i, pop.population[0].fitness))
        pop.evolve()
