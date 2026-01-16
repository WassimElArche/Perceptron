#include "dataSet.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <ctype.h>

/* ================= UTILITAIRES INTERNES ================= */

// alocation sécurisée avec malloc qui stop le programme si la ram est pleine.
static void *xmalloc(size_t n){
    void *p = malloc(n);
    if(!p){ perror("malloc"); exit(EXIT_FAILURE); }
    return p;
}

// alocation sécurisée avec caloc qui initialise tout à zéro.
static void *xcalloc(size_t n, size_t s){
    void *p = calloc(n, s);
    if(!p){ perror("calloc"); exit(EXIT_FAILURE); }
    return p;
}

// copie une chaine de caractere dans un nouvel espace mémoire aloué.
static char *xstrdup(const char *s){
    char *d = (char*)xmalloc(strlen(s)+1);
    strcpy(d, s);
    return d;
}

// retire les retours à la ligne \n et \r à la fin d'une chaine.
static void rstrip(char *s){
    int n = (int)strlen(s);
    while( n>0 && (s[n-1] == '\n' || s[n-1] == '\r')){ s[n-1] = '\0'; n-- ; }
}

// enlève les espaces au début et à la fin d'un texte.
static char *trim(char *s){
    while(isspace((unsigned char)*s)) s++;
    if(*s == 0) return s;
    char *end = s + strlen(s) - 1;
    while(end > s && isspace((unsigned char)*end)) end--;
    end[1] = '\0';
    return s;
}

// compte le nombre de virgules pour savoir combien il y a de colones.
static int countTokens(const char *l){
    int c = 1;
    for(int i = 0; l[i] ; i++ ) if(l[i] == ',') c++;
    return c;
}

// découpe une ligne csv en plusieur morceaux en utilisant la virgule.
static int splitCSV(char *line, char **tokens, int max){
    int k = 0;
    char *save = NULL;
    char *tok = strtok_r(line, ",", &save);
    while(tok && k < max){
        tokens[k++] = trim(tok);
        tok = strtok_r(NULL, ",", &save);
    }
    return k;
}

// aloue une matrice de double (tableau de tableaux).
static double **allocMat(int n, int m){
    double **t = xmalloc(sizeof(double*)*n);
    for(int i = 0;i < n;i++)
        t[i] = xcalloc((size_t)m, sizeof(double));
    return t;
}

// libere la mémoire d'une matrice de double.
static void freeMat(double **t, int n){
    if(!t) return;
    for(int i = 0;i<n;i++) free(t[i]);
    free(t);
}

// transforme un label texte en nombre entier unique pour le perceptron.
static int label_to_int(const char *s) {
    if (!s || !*s) return 0;
    char cleanS[256];
    int j = 0;
    for (int i = 0; s[i] != '\0' && j < 255; i++) {
        if (!isspace((unsigned char)s[i])) {
            cleanS[j++] = s[i];
        }
    }
    cleanS[j] = '\0';
    if (j == 0) return 0;
    static char nomsClasses[10][256];
    static int nbClassesEnregistrees = 0;
    for (int i = 0; i < nbClassesEnregistrees; i++) {
        if (strcmp(cleanS, nomsClasses[i]) == 0) return i;
    }
    if (nbClassesEnregistrees < 10) {
        strncpy(nomsClasses[nbClassesEnregistrees], cleanS, 255);
        return nbClassesEnregistrees++;
    }
    return 0;
}

// lit un fichier csv et crée l'objet dataset avec toute les données.
// il gère les erreurs si le fichier est vide ou mal formater.
DataSet* createDataSet(const char *fichier){
    FILE *f = fopen(fichier, "r");
    if(!f){
        printf("ERREUR Impossible d'ouvrir le fichier : %s\n", fichier);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    if (ftell(f) == 0) {
        printf("ERREUR Le fichier '%s' est vide.\n", fichier);
        fclose(f);
        exit(1);
    }
    rewind(f);
    DataSet *ds = xcalloc(1, sizeof(DataSet));
    ds->nom = xstrdup(fichier);
    char line[4096];
    if(!fgets(line, 4096, f)) {
        printf("ERREUR Impossible de lire la premiere ligne (header).\n");
        exit(1);
    }
    rstrip(line);
    int totalCols = countTokens(line);
    if (totalCols < 2) {
        printf("ERREUR Header corrompu : %d colonnes detectees.\n", totalCols);
        exit(1);
    }
    ds->nbColonne = totalCols - 1;
    ds->nomColonne = xcalloc((size_t)ds->nbColonne, sizeof(char*));
    char *tmp = xstrdup(line);
    char *tok[100];
    splitCSV(tmp, tok, totalCols);
    for(int i = 0; i < ds->nbColonne; i++)
        ds->nomColonne[i] = xstrdup(tok[i]);
    free(tmp);
    ds->n = 0;
    while(fgets(line, 4096, f)) {
        if (strlen(trim(line)) > 0) ds->n++;
    }
    if (ds->n == 0) {
        printf("ERREUR Le fichier ne contient aucune ligne de donnees.\n");
        exit(1);
    }
    ds->tab_Data = allocMat(ds->n, ds->nbColonne);
    ds->sortieAttendue_train = (int*)xmalloc(sizeof(int) * (size_t)ds->n);
    rewind(f);
    fgets(line, 4096, f);
    int i = 0;
    while(fgets(line, 4096, f)) {
        char *l = trim(line);
        if (strlen(l) == 0) continue;
        tmp = xstrdup(l);
        int lues = splitCSV(tmp, tok, totalCols);
        if (lues < totalCols) {
            printf("ERREUR Ligne %d : Manque de colonnes.\n", i + 2);
            exit(1);
        }
        for (int j = 0; j < ds->nbColonne; j++) {
            char *ptr_erreur;
            ds->tab_Data[i][j] = strtod(tok[j], &ptr_erreur);
            if (tok[j] == ptr_erreur || *ptr_erreur != '\0') {
                printf("ERREUR Ligne %d, Col %d : pas un nombre valide.\n", i+2, j);
                exit(1);
            }
        }
        ds->sortieAttendue_train[i] = label_to_int(tok[ds->nbColonne]);
        free(tmp);
        i++;
    }
    fclose(f);
    printf("[OK] Chargement robuste termine : %d lignes valides.\n", ds->n);
    return ds;
}

// mélange les lignes et sépare les données en 80% train et 20% teste.
void melanger(const DataSet *data){
    DataSet *ds = (DataSet*)data;
    if(ds->tab_Train) freeMat(ds->tab_Train, ds->nTrain);
    if(ds->tab_Teste) freeMat(ds->tab_Teste, ds->nTest);
    ds->nTrain = (int)(0.8 * ds->n);
    ds->nTest  = ds->n - ds->nTrain;
    ds->tab_Train = allocMat(ds->nTrain, ds->nbColonne);
    ds->tab_Teste = allocMat(ds->nTest,  ds->nbColonne);
    int *labels_all = xmalloc(sizeof(int) * ds->n);
    for(int i=0; i<ds->n; i++) labels_all[i] = ds->sortieAttendue_train[i];
    free(ds->sortieAttendue_train);
    ds->sortieAttendue_train = xmalloc(sizeof(int) * ds->nTrain);
    ds->sortieAttendue_Teste = xmalloc(sizeof(int) * ds->nTest);
    int *idx = xmalloc(ds->n * sizeof(int));
    for(int i = 0; i < ds->n; i++) idx[i] = i;
    for(int i = ds->n - 1; i > 0; i--){
        int j = rand() % (i + 1);
        int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
    }
    for(int i = 0; i < ds->nTrain; i++){
        for(int j = 0; j < ds->nbColonne; j++)
            ds->tab_Train[i][j] = ds->tab_Data[idx[i]][j];
        ds->sortieAttendue_train[i] = labels_all[idx[i]];
    }
    for(int i = 0; i < ds->nTest; i++){
        for(int j = 0; j < ds->nbColonne; j++)
            ds->tab_Teste[i][j] = ds->tab_Data[idx[i + ds->nTrain]][j];
        ds->sortieAttendue_Teste[i] = labels_all[idx[i + ds->nTrain]];
    }
    free(labels_all); free(idx);
}

// calcule la moyenne arithmétique d'une colone du dataset.
double moyenne(DataSet *d, int colIndex){
    if(colIndex < 0 || colIndex >= d->nbColonne) return 0;
    double s = 0;
    for(int i = 0;i < d->n;i++) s += d->tab_Data[i][colIndex];
    return s / d->n;
}

// calcule l'écart-type pour voir la dispersion des données.
double ecartType(DataSet *d, int colIndex){
    if(colIndex < 0 || colIndex >= d->nbColonne) return 0;
    double m = moyenne(d, colIndex), s=0;
    for(int i = 0;i < d->n;i++){
        double x = d->tab_Data[i][colIndex]-m;
        s += x*x;
    }
    return sqrt(s/d->n);
}

// petite fonction de comparaison pour le tri qsort.
static int cmp(const void *a,const void *b){
    double x =* (double*)a, y =* (double*)b;
    return (x>y)-(x<y);
}

// trouve la valeur médiane d'une colone après avoir trier les données.
double mediane(DataSet *d, int colIndex){
    if(colIndex < 0 || colIndex >= d->nbColonne) return 0;
    double *t = xmalloc((size_t)d->n * sizeof(double));
    for(int i = 0;i < d->n;i++) t[i] = d->tab_Data[i][colIndex];
    qsort(t,(size_t)d->n,sizeof(double),cmp);
    double m = (d->n%2)? t[d->n/2] : (t[d->n/2-1]+t[d->n/2])/2;
    free(t);
    return m;
}

// sauvegarde les données de train et de teste dans deux fichiers csv séparer.
void sauvegarderSplit(const DataSet *ds, const char *nomDataset) {
    char nTr[300], nTe[300];
    sprintf(nTr, "%s_TRAIN.csv", nomDataset);
    sprintf(nTe, "%s_TEST.csv", nomDataset);
    FILE *f1 = fopen(nTr, "w");
    if(f1) {
        for(int i = 0; i < ds->nTrain; i++) {
            for(int j = 0; j < ds->nbColonne; j++) fprintf(f1, "%f,", ds->tab_Train[i][j]);
            fprintf(f1, "%d\n", ds->sortieAttendue_train[i]);
        }
        fclose(f1);
    }
    FILE *f2 = fopen(nTe, "w");
    if(f2) {
        for(int i = 0; i < ds->nTest; i++) {
            for(int j = 0; j < ds->nbColonne; j++) fprintf(f2, "%f,", ds->tab_Teste[i][j]);
            fprintf(f2, "%d\n", ds->sortieAttendue_Teste[i]);
        }
        fclose(f2);
    }
    printf("Split enregistre: %s et %s\n", nTr, nTe);
}

// afiche un aperçu des données charger dans la console.
void afficherDonnees(const DataSet *ds) {
    printf("\n--- Apercu : %s ---\n", ds->nom);
    for(int i = 0; i < ds->n; i++) {
        for(int j = 0; j < ds->nbColonne; j++) printf("%.2f | ", ds->tab_Data[i][j]);
        printf("Label: %d\n", ds->sortieAttendue_train[i]);
    }
}

// libere toute la mémoire utiliser par le dataset pour éviter les fuites.
void libererDataSet(DataSet *d){
    if(!d) return;
    freeMat(d->tab_Data, d->n);
    if(d->tab_Train) freeMat(d->tab_Train, d->nTrain);
    if(d->tab_Teste) freeMat(d->tab_Teste, d->nTest);
    if(d->nom) free(d->nom);
    if(d->nomColonne) {
        for(int i=0;i<d->nbColonne;i++) free(d->nomColonne[i]);
        free(d->nomColonne);
    }
    if(d->sortieAttendue_train) free(d->sortieAttendue_train);
    if(d->sortieAttendue_Teste) free(d->sortieAttendue_Teste);
    free(d);
}

// sauvegarde le dataset complet dans un format spécial pour pouvoir le recharger plus tard.
void sauvegarderDataSetSpecial(const DataSet *ds, const char *nomFichier) {
    if (ds == NULL || ds->tab_Train == NULL || ds->tab_Teste == NULL) {
        printf("[!] Erreur : Dataset incomplet.\n");
        return;
    }
    char cheminComplet[512];
    snprintf(cheminComplet, sizeof(cheminComplet), "DataSet/%s", nomFichier);
    FILE *f = fopen(cheminComplet, "w");
    if (f == NULL) {
        printf("[!] Erreur : Impossible de creer le fichier.\n");
        return;
    }
    fprintf(f, "%d\n", ds->nTest);
    fprintf(f, "%d\n", ds->nTrain);
    for (int i = 0; i < ds->nTest; i++) {
        for (int j = 0; j < ds->nbColonne; j++) {
            fprintf(f, "%f%s", ds->tab_Teste[i][j], (j == ds->nbColonne - 1) ? "" : ",");
        }
        fprintf(f, "\n");
    }
    for (int i = 0; i < ds->nTrain; i++) {
        for (int j = 0; j < ds->nbColonne; j++) {
            fprintf(f, "%f%s", ds->tab_Train[i][j], (j == ds->nbColonne - 1) ? "" : ",");
        }
        fprintf(f, "\n");
    }
    for (int i = 0; i < ds->nTest; i++) fprintf(f, "%d\n", ds->sortieAttendue_Teste[i]);
    for (int i = 0; i < ds->nTrain; i++) fprintf(f, "%d\n", ds->sortieAttendue_train[i]);
    for (int j = 0; j < ds->nbColonne; j++) {
        fprintf(f, "%s%s", ds->nomColonne[j], (j == ds->nbColonne - 1) ? "" : ",");
    }
    fprintf(f, "\n%d\n", ds->nbColonne);
    fclose(f);
    printf("[OK] Sauvegarde effectuee : %s\n", cheminComplet);
}

// recharge un dataset sauvegarder avec le format spécial.
DataSet* chargerDataSetSpecial(const char *nomFichier) {
    char cheminComplet[512];
    snprintf(cheminComplet, sizeof(cheminComplet), "DataSet/%s", nomFichier);
    FILE *f = fopen(cheminComplet, "r");
    if (!f) {
        printf("[ERREUR FATALE] Impossible d'ouvrir : %s\n", cheminComplet);
        exit(1);
    }
    DataSet *ds = (DataSet *)calloc(1, sizeof(DataSet));
    if (!ds) exit(1);
    if (fscanf(f, "%d %d", &ds->nTest, &ds->nTrain) != 2) exit(1);
    ds->n = ds->nTest + ds->nTrain;
    fseek(f, -5, SEEK_END);
    if (fscanf(f, "%d", &ds->nbColonne) != 1) ds->nbColonne = 2;
    rewind(f);
    int dummy;
    fscanf(f, "%d %d", &dummy, &dummy);
    ds->tab_Teste = allocMat(ds->nTest, ds->nbColonne);
    ds->tab_Train = allocMat(ds->nTrain, ds->nbColonne);
    ds->tab_Data  = allocMat(ds->n, ds->nbColonne);
    ds->sortieAttendue_Teste = (int *)calloc(ds->nTest, sizeof(int));
    ds->sortieAttendue_train = (int *)calloc(ds->nTrain, sizeof(int));
    for (int i = 0; i < ds->nTest; i++) {
        for (int j = 0; j < ds->nbColonne; j++) {
            fscanf(f, " %lf ,", &ds->tab_Teste[i][j]);
            ds->tab_Data[i][j] = ds->tab_Teste[i][j];
        }
    }
    for (int i = 0; i < ds->nTrain; i++) {
        for (int j = 0; j < ds->nbColonne; j++) {
            fscanf(f, " %lf ,", &ds->tab_Train[i][j]);
            ds->tab_Data[i + ds->nTest][j] = ds->tab_Train[i][j];
        }
    }
    int *labels_complets = (int *)calloc(ds->n, sizeof(int));
    for (int i = 0; i < ds->nTest; i++) {
        fscanf(f, " %d", &ds->sortieAttendue_Teste[i]);
        labels_complets[i] = ds->sortieAttendue_Teste[i];
    }
    for (int i = 0; i < ds->nTrain; i++) {
        fscanf(f, " %d", &ds->sortieAttendue_train[i]);
        labels_complets[i + ds->nTest] = ds->sortieAttendue_train[i];
    }
    free(ds->sortieAttendue_train);
    ds->sortieAttendue_train = labels_complets;
    ds->nom = strdup(nomFichier);
    ds->nomColonne = (char**)calloc(ds->nbColonne, sizeof(char*));
    for(int i=0; i<ds->nbColonne; i++) ds->nomColonne[i] = strdup("Col");
    fclose(f);
    printf("[OK] Chargement %d lignes.\n", ds->n);
    return ds;
}