#ifndef DATASET_H_
#define DATASET_H_

#include <stddef.h>

typedef struct DataSet {
    char *nom;
    double **tab_Data;
    double **tab_Teste;
    double **tab_Train;
    int nTrain;
    int nTest;
    int *sortieAttendue_Teste;
    int *sortieAttendue_train;
    int n;
    int nbColonne;
    char **nomColonne;
} DataSet;

DataSet* createDataSet(const char *fichier);
void melanger(const DataSet *data);
void libererDataSet(DataSet *data);

double moyenne(DataSet *d, int colIndex);
double ecartType(DataSet *d, int colIndex);
double mediane(DataSet *d, int colIndex);

void afficherDonnees(const DataSet *ds);
void sauvegarderSplit(const DataSet *ds, const char *nomDataset);
void sauvegarderDataSetSpecial(const DataSet *ds, const char *nomFichier);
DataSet* chargerDataSetSpecial(const char *nomFichier);

#endif