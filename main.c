#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include "dataSet.h"
#include "perceptron.h"
#include "visual.h"

// afiche un petit graphique en texte dans la console pour voir les points.
void afficherNuagePoints(DataSet *ds) {
    const int LARGEUR = 50;
    const int HAUTEUR = 15;
    char grille[16][51];

    for (int i = 0; i <= HAUTEUR; i++) {
        for (int j = 0; j <= LARGEUR; j++) grille[i][j] = ' ';
        grille[i][LARGEUR] = '\0';
    }

    if (!ds || !ds->tab_Train || ds->nTrain <= 0) {
        printf("\n[!] Donnees non disponibles. Faites l'option 2 ou 13.\n");
        return;
    }

    for (int i = 0; i < ds->nTrain; i++) {
        // on utilise les colones 0 et 1 pour la visualisation par defaut
        int x = (int)((ds->tab_Train[i][0] / 8.0) * LARGEUR);
        int y = (int)((ds->tab_Train[i][1] / 5.0) * HAUTEUR);

        if (x >= 0 && x < LARGEUR && y >= 0 && y < HAUTEUR) {
            char symbole = (ds->sortieAttendue_train[i] == 0) ? 'S' :
                           (ds->sortieAttendue_train[i] == 1) ? 'V' : '@';
            grille[HAUTEUR - 1 - y][x] = symbole;
        }
    }

    printf("\n--- VISUALISATION DES DONNEES (ASCII) ---\n");
    for (int i = 0; i < HAUTEUR; i++) {
        printf("%s\n", grille[i]);
    }
}

int main(void) {
    srand((unsigned int)time(NULL));

    DataSet *ds = (DataSet *)calloc(1, sizeof(DataSet));
    if (!ds) return 1;
    ds->n = 0;

    Perceptron *pBin = NULL;
    Perceptron **experts = NULL;
    int nbClasses = 0;
    int choix = 0;
    char nomFichier[256];

    int epoques = 1000;
    double pasApprentissage = 0.01;

    while (choix != 16) {
        printf("\n============================================\n");
        printf("   PERCEPTRON SYSTEM - Version 4.4.2\n");
        printf("============================================\n");
        printf("--- MENU PRINCIPAL (%d CLASSES) ---\n", nbClasses);
        printf("1.  Afficher Nuage de Points (ASCII Simple)\n");
        printf("2.  Melanger & Split (80/20)\n");
        printf("3.  Entrainer le modele (Binaire ou Multi)\n");
        printf("4.  Calculer Accuracy\n");
        printf("5.  Sauvegarder Perceptron (.txt)\n");
        printf("6.  Charger Perceptron (.txt)\n");
        printf("7.  Visualisation Raylib (Frontiere 2D)\n");
        printf("8.  Aide & Documentation\n");
        printf("9.  Inspecter Valeur (Ligne/Col)\n");
        printf("10. Modifier Hyperparametres (Epoques/Pas)\n");
        printf("11. Statistiques (Moyenne/Ecart-Type)\n");
        printf("12. Sauvegarder DataSet Special (/DataSet)\n");
        printf("13. Charger DataSet Special (/DataSet)\n");
        printf("14. Charger CSV Standard\n");
        printf("15. Afficher le DataSet\n");
        printf("16. Quitter\n");
        printf("Choix : ");

        if (scanf("%d", &choix) != 1) break;

        switch (choix) {
            case 1:
                afficherNuagePoints(ds);
                break;

            case 2:
                if (ds->n > 0 && ds->tab_Data != NULL) {
                    melanger(ds);
                    printf("[OK] Split effectue.\n");
                } else {
                    printf("[!] Dataset vide.\n");
                }
                break;

            case 3:
                if (!ds->tab_Train) {
                    printf("[!] aucune donnee d'entrainement disponible.\n");
                    break;
                }
                if (pBin) { libererPerceptron(pBin); pBin = NULL; }
                if (experts) {
                    for(int i=0; i<nbClasses; i++) if(experts[i]) libererPerceptron(experts[i]);
                    free(experts); experts = NULL;
                }

                if (nbClasses <= 2) {
                    pBin = createPerceptron(ds->nbColonne, epoques);
                    pBin->pasApprentissage = pasApprentissage;
                    entrainerPerceptron(ds, pBin);
                    printf("[OK] Entrainement binaire fini.\n");
                } else {
                    printf("[INFO] Mode Multi-classe detecte. Creation de %d experts...\n", nbClasses);
                    experts = malloc(nbClasses * sizeof(Perceptron*));
                    for (int i = 0; i < nbClasses; i++) {
                        experts[i] = createPerceptron(ds->nbColonne, epoques);
                        experts[i]->pasApprentissage = pasApprentissage;
                    }
                    entrainerMultiClasse(experts, nbClasses, ds);
                    printf("[OK] Entrainement Multi-classe fini.\n");
                }
                break;

            case 4:
                if (nbClasses <= 2 && pBin && ds->tab_Teste) {
                    printf("Accuracy Binaire : %.2f%%\n", accuracy(pBin, ds) * 100.0);
                } else if (nbClasses > 2 && experts && ds->tab_Teste) {
                    int succes = 0;
                    for (int i = 0; i < ds->nTest; i++) {
                        int pred = predireMulti(experts, nbClasses, ds->tab_Teste[i]);
                        if (pred == ds->sortieAttendue_Teste[i]) succes++;
                    }
                    printf("Accuracy Multi-classe : %.2f%%\n", ((double)succes / ds->nTest) * 100.0);
                } else {
                    printf("[!] Modele non entraine ou donnees manquantes.\n");
                }
                break;

            case 5:
                if (pBin) {
                    printf("Nom fichier : "); scanf("%s", nomFichier);
                    sauvegarderPerceptron(pBin, nomFichier);
                }
                break;

            case 6:
                listerFichiersPerceptron();
                printf("Nom fichier : "); scanf("%s", nomFichier);
                if (pBin) libererPerceptron(pBin);
                pBin = chargerPerceptron(nomFichier);
                break;

            case 7:
                if (pBin && ds->nbColonne >= 2) {
                    visual_run_with_model_custom(ds, pBin, 0, 1);
                } else {
                    printf("[!] Visu Raylib dispo seulement pour le mode binaire.\n");
                }
                break;
            case 8:
                printf("\n==========================================================\n");
                printf("                GUIDE D'UTILISATION RAPIDE                \n");
                printf("==========================================================\n");
                printf("1. CHARGEMENT : utilise l'option [14] pour un CSV classique\n");
                printf("   ou l'option [13] pour un dataset special deja splitte.\n\n");
                printf("2. PREPARATION : l'option [2] est OBLIGATOIRE pour melanger\n");
                printf("   les donnees et creer les sets d'entrainement et de test.\n\n");
                printf("3. REGLAGES : l'option [10] permet de modifier les epoques\n");
                printf("   et le pas d'apprentissage (Learning Rate).\n\n");
                printf("4. APPRENTISSAGE : l'option [3] entraine le neurone. Elle\n");
                printf("   bascule toute seule en Multi-Classe si besoin.\n\n");
                printf("5. EVALUATION : l'option [4] calcul le pourcentage de reussite\n");
                printf("   sur des donnees que le modele n'a jamais vue.\n");
                printf("==========================================================\n");
                break;

                case 9:
                if (ds->n > 0 && ds->tab_Data) {
                    int l, c;
                    printf("ligne a inspecter (0-%d) : ", ds->n - 1);
                    scanf("%d", &l);
                    printf("colone a inspecter (0-%d) : ", ds->nbColonne - 1);
                    scanf("%d", &c);

                    // on verifie que l'utilisateur demande pas n'importe quoi
                    if (l >= 0 && l < ds->n && c >= 0 && c < ds->nbColonne) {
                        printf("\n[INSPECTION] Ligne %d | Colone %d (%s) : %f\n",
                                l, c, ds->nomColonne[c], ds->tab_Data[l][c]);
                        printf("[LABEL REEL] : %d\n", ds->sortieAttendue_train[l]);
                    } else {
                        printf("[!] erreur : index hors limite du dataset.\n");
                    }
                } else {
                    printf("[!] aucune donnee charger pour l'inspection.\n");
                }
                break;

            case 10:
                printf("Epoques (actuel %d) : ", epoques); scanf("%d", &epoques);
                printf("Pas d'apprentissage (actuel %f) : ", pasApprentissage); scanf("%lf", &pasApprentissage);
                break;

            case 11:
                if (ds->n > 0) {
                    for (int j = 0; j < ds->nbColonne; j++) {
                        printf("[%s] Moy: %.2f | E-T: %.2f\n", ds->nomColonne[j], moyenne(ds, j), ecartType(ds, j));
                    }
                }
                break;

            case 12:
                // sauvgarde de l'etat complet du dataset (train + teste)
                if (ds->tab_Train && ds->tab_Teste) {
                    printf("Nom du fichier de sauvegarde (ex: iris_split.txt) : ");
                    scanf("%s", nomFichier);
                    sauvegarderDataSetSpecial(ds, nomFichier);
                } else {
                    printf("[!] Tu dois d'abord faire le split (option 2) pour sauvgarder.\n");
                }
                break;

            case 13:
                listerFichiersDataSet();
                printf("Nom choisis : "); scanf("%s", nomFichier);
                DataSet *dsRelu = chargerDataSetSpecial(nomFichier);
                if (dsRelu) {
                    libererDataSet(ds);
                    ds = dsRelu;
                    int ml = -1;
                    for (int i = 0; i < ds->n; i++) {
                        if (ds->sortieAttendue_train[i] > ml) ml = ds->sortieAttendue_train[i];
                    }
                    nbClasses = ml + 1;
                    printf("[OK] Dataset charger. Classes detectées : %d\n", nbClasses);
                }
                break;

            case 14:
                printf("Chemin CSV : "); scanf("%s", nomFichier);
                DataSet *temp = createDataSet(nomFichier);
                if (temp) {
                    libererDataSet(ds);
                    ds = temp;
                    int ml = -1;
                    for (int i = 0; i < ds->n; i++) {
                        if (ds->sortieAttendue_train[i] > ml) ml = ds->sortieAttendue_train[i];
                    }
                    nbClasses = ml + 1;
                    printf("[OK] CSV charger. Classes : %d\n", nbClasses);
                }
                break;

            case 15:
                if (ds->n == 0) printf("dataset vide.\n");
                else afficherDonnees(ds);
                break;

            case 16:
                printf("Sortie du programme...\n");
                break;
        }
    }

    if (pBin) libererPerceptron(pBin);
    if (experts) {
        for(int i=0; i<nbClasses; i++) if(experts[i]) libererPerceptron(experts[i]);
        free(experts);
    }
    libererDataSet(ds);
    return 0;
}