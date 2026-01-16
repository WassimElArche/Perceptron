//
// Created by Wassim  on 13/01/2026.
//

#ifndef PERCEPTRON_H_
#define PERCEPTRON_H_
#include "dataSet.h"
typedef struct{
    double biais;
    int epoque;
    int nPoids;
    double *poids;
    double accuracy;
    double pasApprentissage;
} Perceptron;


Perceptron* createPerceptron(int n, int epoch);

int fonctionActivation(double somme);
void entrainerMultiClasse(Perceptron **perceptrons, int nbLabel, const DataSet *ds);
double predireProba(Perceptron *p , const double *entree);
int predireMulti(Perceptron **experts, int nbClasses, const double *entree);

double somme(const DataSet *data,const Perceptron *p , int n, int j);

void entrainerPerceptron(const DataSet *dataTrain , Perceptron *p);

int predire(Perceptron *p , const double *entree);

double accuracy(Perceptron *p , const DataSet *dataTeste);

Perceptron* chargerPerceptron(const char *file);

void libererPerceptron(Perceptron *p);

void sauvegarderPerceptron(const Perceptron *p , const char *file);
void listerFichiersPerceptron() ;

#endif //PERCEPTRON_H_
