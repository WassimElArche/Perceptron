// created by wassim on 13/01/2026.

#include "dataSet.h"
#include <stdio.h>
#include <stdlib.h>
#include "perceptron.h"
#include "math.h"
#include <string.h>
#include <dirent.h>

// fonction de seuil (heaviside) retournant 1 si la somme est positive, sinon 0.
// utilisé pour la clasification binaire clasique.
int fonctionActivation(double somme) {
    if (somme < 0) return 0;
    return 1;
}

// fonction sigmoïde retournant une valeur continue entre 0 et 1.
// utilisé pour obtenir un indice de confiance ou probabilitée (multi-classe).
double fonctionActivationMultiClass(double somme) {
    return (double) 1 / (1 + exp(-somme));
}

// aloue la mémoire d'un nouveau perceptron et initialise ses parametres.
// les poids sont générés aléatoirement entre 0 et 1.
Perceptron* createPerceptron(int n, int epoch) {
    Perceptron* newPerceptron = malloc(sizeof(Perceptron));
    newPerceptron->biais = 1;
    newPerceptron->epoque = epoch;
    newPerceptron->accuracy = 0;
    newPerceptron->pasApprentissage = 0.001;
    newPerceptron->nPoids = n;
    if (n != 0) {
        newPerceptron->poids = malloc(n * sizeof(double));
        for (int i = 0; i < n ; i++) {
            newPerceptron->poids[i] = ((double)rand() / RAND_MAX * 0.1) - 0.05;
        }
    }
    return newPerceptron;
}

// réalise une clasification binaire (0 ou 1) pour une entrée donnée.
// calcule la somme pondérée des entrées et aplique la fonction de seuil.
int predire(Perceptron *p, const double *entree) {
    if (p == NULL || entree == NULL || p->poids == NULL) {
        printf("Erreur fatale : pointeur NULL dans predire\n");
        exit(1);
    }
    double somme = 0;
    for (int i = 0 ; i < p->nPoids; i++) {
        somme = somme + p->poids[i] * entree[i];
    }
    somme += p->biais;
    const int final = fonctionActivation(somme);
    return final;
}

// retourne un score de confiance (0.0 à 1.0) pour une entrée donnée.
// utilise la fonction d'activation sigmoïde au lieu du seuil brutale.
double predireProba(Perceptron *p, const double *entree) {
    if (p == NULL || entree == NULL || p->poids == NULL) {
        printf("Erreur fatale : pointeur NULL dans predire\n");
        exit(1);
    }
    double somme = 0;
    for (int i = 0 ; i < p->nPoids; i++) {
        somme = somme + p->poids[i] * entree[i];
    }
    somme += p->biais;
    const double final = fonctionActivationMultiClass(somme);
    return final;
}

// ajuste les poids et le biais du perceptron selon la regle d'apprentissage.
// s'arrête si le nombre d'époques est atteint ou si plus aucune ereur n'est détectée.
void entrainerPerceptron(const DataSet *dataTrain, Perceptron *p) {
    for (int i = 0; i < p->epoque ; i++) {
        int erreurTrouve = 0;
        for (int j = 0 ; j < dataTrain->nTrain ; j++) {
            if (dataTrain->tab_Train[j] == NULL) {
                printf("Erreur : tab_Train[%d] est NULL\n", j);
                return;
            }
            int prediction = predire(p, dataTrain->tab_Train[j]);
            int label = dataTrain->sortieAttendue_train[j];
            int erreur = label - prediction;
            if (erreur != 0) {
                erreurTrouve++;
                for (int z = 0; z < p->nPoids; z++) {
                    p->poids[z] += erreur * p->pasApprentissage * dataTrain->tab_Train[j][z];
                }
                p->biais = p->biais + erreur * p->pasApprentissage;
            }
        }
        if (erreurTrouve == 0 ) break;
    }
}

// entraine plusieur perceptrons selon la stratégie "one-vs-all".
// chaque perceptron devient un expert pour reconnaitre une classe spécifique.
void entrainerMultiClasse(Perceptron **perceptrons, int nbLabel, const DataSet *ds) {
    for (int i = 0; i < nbLabel; i++) {
        int *labelsOriginaux = malloc(ds->nTrain * sizeof(int));
        for (int k = 0; k < ds->nTrain; k++) {
            labelsOriginaux[k] = ds->sortieAttendue_train[k];
            ds->sortieAttendue_train[k] = (labelsOriginaux[k] == i) ? 1 : 0;
        }
        entrainerPerceptron(ds, perceptrons[i]);
        for (int k = 0; k < ds->nTrain; k++) {
            ds->sortieAttendue_train[k] = labelsOriginaux[k];
        }
        free(labelsOriginaux);
    }
}

// compare les scores de probabilité de chaque expert pour une entrée donnée.
// désigne comme gagnante la clase ayant obtenu la probabilité la plus élevée.
int predireMulti(Perceptron **experts, int nbClasses, const double *entree) {
    int gagnant = 0;
    double maxProba = -1.0;
    for (int i = 0; i < nbClasses; i++) {
        double p = predireProba(experts[i], entree);
        if (p > maxProba) {
            maxProba = p;
            gagnant = i;
        }
    }
    return gagnant;
}

// calcule le taux de réussite (0.0 à 1.0) sur les données de teste.
// compare les prédictions du modele avec les étiquettes réeles non vues durant l'entrainement.
double accuracy(Perceptron *p, const DataSet *dataTest) {
    int nombreDePrediction = dataTest->nTest;
    int nombreDeSucces = 0;
    if (dataTest->nTest == 0) {
        melanger(dataTest);
        nombreDePrediction = dataTest->nTest;
    }
    for (int i = 0; i < nombreDePrediction ; i++) {
        int prediction = predire(p, dataTest->tab_Teste[i]);
        int label = dataTest->sortieAttendue_Teste[i];
        if (label - prediction == 0) {
            nombreDeSucces++;
        }
    }
    return (double) nombreDeSucces / nombreDePrediction;
}

// lit un fichier texte pour restaurer les poids et le biais d'un modele.
// reconstruit dynamiquement la structure perceptron à partir des données sauvgardées.
Perceptron* chargerPerceptron(const char *file) {
    char cheminComplet[512];
    snprintf(cheminComplet, sizeof(cheminComplet), "Perceptron/%s", file);
    FILE *f = fopen(cheminComplet, "r");
    if (f == NULL) {
        printf("Fichier introuvable\n");
        return NULL;
    }
    char mot[256];
    int totalMots = 0;
    while (fscanf(f, "%255s", mot) != EOF) {
        totalMots++;
    }
    if (totalMots < 1) {
        printf("Fichier vide ou invalide\n");
        fclose(f);
        return NULL;
    }
    rewind(f);
    Perceptron* p = malloc(sizeof(Perceptron));
    p->nPoids = totalMots - 1;
    p->poids = malloc(p->nPoids * sizeof(double));
    char *endPtr;
    if (fscanf(f, "%255s", mot) != EOF) {
        p->biais = strtod(mot, &endPtr);
    }
    for (int j = 0; j < p->nPoids; j++) {
        if (fscanf(f, "%255s", mot) != EOF) {
            p->poids[j] = strtod(mot, &endPtr);
        }
    }
    fclose(f);
    return p;
}

// libere proprement la mémoire alouée pour le tableau de poids et la structure.
// esentiel pour éviter les fuites de mémoire lors des ré-entrainements.
void libererPerceptron(Perceptron *p) {
    p->biais = 0;
    p->epoque = 0;
    free(p->poids);
    p->accuracy = 0;
    free(p);
}

// parcourt le répertoire locale pour lister les modeles de perceptrons sauvgardés.
// permet à l'utilisateur de choisir quel fichier charger via le menu.
void listerFichiersPerceptron() {
    struct dirent *lecture;
    DIR *rep = opendir("Perceptron");
    printf("\n--- FICHIERS DISPONIBLES DANS /Perceptron ---\n");
    if (rep == NULL) {
        printf("[!] Dossier 'Perceptron' introuvable.\n");
        return;
    }
    int count = 0;
    while ((lecture = readdir(rep))) {
        if (lecture->d_name[0] != '.') {
            printf(" -> %s\n", lecture->d_name);
            count++;
        }
    }
    if (count == 0) printf(" (Aucun Perceptron trouvé)\n");
    printf("------------------------------------------\n");
    closedir(rep);
}

// enregistre le biais et les poids du modele actuelle dans un fichier texte.
// le fichier est stocké dans le sous-dosier dédié 'perceptron/'.
void sauvegarderPerceptron(const Perceptron *p, const char *file) {
    char chemin[255];
    snprintf(chemin, sizeof(chemin), "Perceptron/%s", file);
    FILE *f = fopen(chemin, "w");
    if (f == NULL) {
        printf("Erreur lors de la création du fichier de sauvegarde\n");
    }
    else {
        fprintf(f, "%f\n", p->biais);
        for (int i = 0 ; i < p->nPoids ; i++){
            fprintf(f, "%f\n", p->poids[i]);
        }
    }
    fclose(f);
}