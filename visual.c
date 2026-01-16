#include "visual.h"
#include "raylib.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <dirent.h>

// ==================== UTILITAIRES ====================

static Color classColor(int c, bool light) {
    Color base;
    switch (c % 6) {
        case 0: base = RED; break;
        case 1: base = BLUE; break;
        case 2: base = GREEN; break;
        case 3: base = ORANGE; break;
        case 4: base = PURPLE; break;
        default: base = BROWN; break;
    }
    if (!light) return base;
    return (Color){ base.r, base.g, base.b, 70 };
}

static double calculerSommePonderee(const Perceptron *p, const double *entree) {
    double somme = 0;
    for (int i = 0; i < p->nPoids; i++) {
        somme += p->poids[i] * entree[i];
    }
    somme += p->biais;
    return somme;
}

// ==================== CALCUL DU CENTRE DE MASSE ====================

static double* calculerCentreMasse(const DataSet *ds) {
    double *centerPoint = malloc(ds->nbColonne * sizeof(double));

    for(int j = 0; j < ds->nbColonne; j++) {
        double somme = 0;
        for(int i = 0; i < ds->nTrain; i++) {
            somme += ds->tab_Train[i][j];
        }
        centerPoint[j] = somme / ds->nTrain;
    }

    return centerPoint;
}

// ==================== CALCUL DES BORNES ====================

static void calculerBornes(const DataSet *ds, int colX, int colY,
                          double *minX, double *maxX, double *minY, double *maxY) {
    *minX = DBL_MAX;
    *maxX = -DBL_MAX;
    *minY = DBL_MAX;
    *maxY = -DBL_MAX;

    for (int i = 0; i < ds->nTrain; i++) {
        double valX = ds->tab_Train[i][colX];
        double valY = ds->tab_Train[i][colY];

        if (valX < *minX) *minX = valX;
        if (valX > *maxX) *maxX = valX;
        if (valY < *minY) *minY = valY;
        if (valY > *maxY) *maxY = valY;
    }

    // Ajout d'une marge de 15%
    double spanX = *maxX - *minX;
    double spanY = *maxY - *minY;
    *minX -= spanX * 0.15;
    *maxX += spanX * 0.15;
    *minY -= spanY * 0.15;
    *maxY += spanY * 0.15;
}

// ==================== ANALYSE DES POIDS ====================

static void analyserPoids(const Perceptron *p, const DataSet *ds, int colX, int colY) {
    printf("\n========== DIAGNOSTIC PERCEPTRON ==========\n");
    printf("Biais: %.4f\n", p->biais);
    printf("\nPoids de chaque dimension:\n");

    double sumPoids = 0;
    for(int i = 0; i < p->nPoids; i++) {
        sumPoids += fabs(p->poids[i]);
    }

    for(int i = 0; i < p->nPoids; i++) {
        double importance = (fabs(p->poids[i]) / sumPoids) * 100.0;
        char marker = (i == colX || i == colY) ? '*' : ' ';
        printf("  %c [%s] w[%d] = %+.4f  (%.1f%%)\n",
               marker, ds->nomColonne[i], i, p->poids[i], importance);
    }

    double importanceVisible = (fabs(p->poids[colX]) + fabs(p->poids[colY])) / sumPoids * 100.0;
    printf("\n--- Projection choisie ---\n");
    printf("Axes: %s vs %s\n", ds->nomColonne[colX], ds->nomColonne[colY]);
    printf("Importance dimensions visibles: %.1f%%\n", importanceVisible);
    printf("Importance dimensions cachees: %.1f%%\n", 100.0 - importanceVisible);

    if(importanceVisible < 30) {
        printf("\n⚠️  ATTENTION: Dimensions cachees dominent!\n");
        printf("    Frontiere peut etre mal visible.\n");
    }
    printf("==========================================\n\n");
}

// ==================== CALCUL ACCURACY 2D ====================

static double calculerAccuracy2D(const DataSet *ds, const Perceptron *p,
                                 int colX, int colY, double *centerPoint, int *nbCorrect) {
    double *inputSimule = malloc(ds->nbColonne * sizeof(double));
    int correct = 0;

    for(int i = 0; i < ds->nTrain; i++) {
        // Initialiser avec le centre de masse
        for(int k = 0; k < ds->nbColonne; k++) {
            inputSimule[k] = centerPoint[k];
        }

        // Injecter les valeurs des 2 axes choisis
        inputSimule[colX] = ds->tab_Train[i][colX];
        inputSimule[colY] = ds->tab_Train[i][colY];

        // Prédire
        double score = calculerSommePonderee(p, inputSimule);
        int pred = (score >= 0) ? 1 : 0;

        if(pred == ds->sortieAttendue_train[i]) {
            correct++;
        }
    }

    free(inputSimule);
    *nbCorrect = correct;
    return (double)correct / ds->nTrain * 100.0;
}

// ==================== VÉRIFIER VISIBILITÉ FRONTIÈRE ====================

static bool frontiereEstVisible(const Perceptron *p, int colX, int colY,
                                double *centerPoint, int nbColonne,
                                double minX, double maxX, double minY, double maxY) {
    double w_x = p->poids[colX];
    double w_y = p->poids[colY];

    // Calcul du terme constant
    double reste = p->biais;
    for(int k = 0; k < nbColonne; k++) {
        if(k != colX && k != colY) {
            reste += p->poids[k] * centerPoint[k];
        }
    }

    bool visible = false;

    // Cas 1: Ligne non verticale (on trace y = f(x))
    if(fabs(w_y) > 0.001) {
        double yGauche = -(w_x * minX + reste) / w_y;
        double yDroit = -(w_x * maxX + reste) / w_y;

        visible = (yGauche >= minY && yGauche <= maxY) ||
                  (yDroit >= minY && yDroit <= maxY) ||
                  (yGauche < minY && yDroit > maxY) ||
                  (yGauche > maxY && yDroit < minY);
    }
    // Cas 2: Ligne verticale (on trace x = f(y))
    else if(fabs(w_x) > 0.001) {
        double xBas = -(w_y * minY + reste) / w_x;
        double xHaut = -(w_y * maxY + reste) / w_x;

        visible = (xBas >= minX && xBas <= maxX) ||
                  (xHaut >= minX && xHaut <= maxX) ||
                  (xBas < minX && xHaut > maxX) ||
                  (xBas > maxX && xHaut < minX);
    }

    return visible;
}

// ==================== RENDU ZONES DE DÉCISION ====================

static void dessinerZonesDecision(const Perceptron *p, int colX, int colY,
                                  double *centerPoint, int nbColonne,
                                  double minX, double maxX, double minY, double maxY,
                                  int W, int H) {
    double *inputSimule = malloc(nbColonne * sizeof(double));

    for (int px = 0; px < W; px += 4) {
        for (int py = 0; py < H; py += 4) {
            // Conversion pixel -> valeurs mathématiques
            double xVal = minX + (maxX - minX) * px / W;
            double yVal = maxY - (maxY - minY) * py / H;

            // Initialiser avec centre de masse
            for(int k = 0; k < nbColonne; k++) {
                inputSimule[k] = centerPoint[k];
            }

            // Injecter les valeurs des axes
            inputSimule[colX] = xVal;
            inputSimule[colY] = yVal;

            // Calculer la prédiction
            double score = calculerSommePonderee(p, inputSimule);
            int pred = (score >= 0) ? 1 : 0;

            DrawRectangle(px, py, 4, 4, classColor(pred, true));
        }
    }

    free(inputSimule);
}

// ==================== TRACÉ FRONTIÈRE ====================

static void tracerFrontiere(const Perceptron *p, int colX, int colY,
                           double *centerPoint, int nbColonne,
                           double minX, double maxX, double minY, double maxY,
                           int W, int H) {
    double w_x = p->poids[colX];
    double w_y = p->poids[colY];

    // Calcul terme constant
    double reste = p->biais;
    for(int k = 0; k < nbColonne; k++) {
        if(k != colX && k != colY) {
            reste += p->poids[k] * centerPoint[k];
        }
    }

    // Cas 1: Ligne non verticale
    if(fabs(w_y) > 0.001) {
        int prevScreenY = -1;

        for(int px = 0; px < W; px += 2) {
            double xVal = minX + (maxX - minX) * px / W;
            double yVal = -(w_x * xVal + reste) / w_y;

            int screenY = (int)((maxY - yVal) / (maxY - minY) * H);

            if(screenY >= 0 && screenY < H) {
                if(prevScreenY != -1) {
                    DrawLine(px - 2, prevScreenY, px, screenY, BLACK);
                }
                prevScreenY = screenY;
            } else {
                prevScreenY = -1;
            }
        }
    }
    // Cas 2: Ligne verticale
    else if(fabs(w_x) > 0.001) {
        int prevScreenX = -1;

        for(int py = 0; py < H; py += 2) {
            double yVal = maxY - (maxY - minY) * py / H;
            double xVal = -(w_y * yVal + reste) / w_x;

            int screenX = (int)((xVal - minX) / (maxX - minX) * W);

            if(screenX >= 0 && screenX < W) {
                if(prevScreenX != -1) {
                    DrawLine(prevScreenX, py - 2, screenX, py, BLACK);
                }
                prevScreenX = screenX;
            } else {
                prevScreenX = -1;
            }
        }
    }
}

void listerFichiersDataSet() {
    struct dirent *lecture;
    DIR *rep = opendir("DataSet");
    printf("\n--- FICHIERS DISPONIBLES DANS /DataSet ---\n");
    if (rep == NULL) {
        printf("[!] Dossier 'DataSet' introuvable.\n");
        return;
    }
    int count = 0;
    while ((lecture = readdir(rep))) {
        printf(" -> %s\n", lecture->d_name);
        count++;
    }
    if (count == 0) printf(" (Aucun DataSet trouvé)\n");
    printf("------------------------------------------\n");
    closedir(rep);
}

// ==================== DESSIN DES POINTS ====================

static void dessinerPoints(const DataSet *ds, const Perceptron *p,
                          int colX, int colY, double *centerPoint,
                          double minX, double maxX, double minY, double maxY,
                          int W, int H) {
    double *inputSimule = malloc(ds->nbColonne * sizeof(double));

    for (int i = 0; i < ds->nTrain; i++) {
        // Position écran
        int sx = (int)((ds->tab_Train[i][colX] - minX) / (maxX - minX) * W);
        int sy = (int)((maxY - ds->tab_Train[i][colY]) / (maxY - minY) * H);

        // Vérifier si mal classé
        for(int k = 0; k < ds->nbColonne; k++) {
            inputSimule[k] = centerPoint[k];
        }
        inputSimule[colX] = ds->tab_Train[i][colX];
        inputSimule[colY] = ds->tab_Train[i][colY];

        double score = calculerSommePonderee(p, inputSimule);
        int pred = (score >= 0) ? 1 : 0;
        bool malClasse = (pred != ds->sortieAttendue_train[i]);

        // Dessiner le point
        int label = ds->sortieAttendue_train[i];
        DrawCircle(sx, sy, 6, classColor(label, false));

        // Cercle rouge pour erreurs
        if(malClasse) {
            DrawCircleLines(sx, sy, 6, RED);
            DrawCircleLines(sx, sy, 7, RED);
            DrawCircleLines(sx, sy, 8, Fade(RED, 0.5f));
        } else {
            DrawCircleLines(sx, sy, 6, BLACK);
        }
    }

    free(inputSimule);
}

// ==================== FONCTION PRINCIPALE ====================

void visual_run_with_model_custom(const DataSet *ds, const Perceptron *p, int colX, int colY) {
    if (ds->nTrain <= 0) return;

    const int W = 1000, H = 700;
    if (!IsWindowReady()) InitWindow(W, H, "Neural Engine - 2D Projection");
    SetTargetFPS(60);

    // ===== ÉTAPE 1: CALCULS PRÉPARATOIRES =====
    double *centerPoint = calculerCentreMasse(ds);

    double minX, maxX, minY, maxY;
    calculerBornes(ds, colX, colY, &minX, &maxX, &minY, &maxY);

    // ===== ÉTAPE 2: DIAGNOSTIC =====
    analyserPoids(p, ds, colX, colY);

    int nbCorrect;
    double accuracy2D = calculerAccuracy2D(ds, p, colX, colY, centerPoint, &nbCorrect);
    printf("Accuracy sur projection 2D: %.1f%% (%d/%d)\n\n", accuracy2D, nbCorrect, ds->nTrain);

    bool frontiereVisible = frontiereEstVisible(p, colX, colY, centerPoint, ds->nbColonne,
                                                minX, maxX, minY, maxY);
    if(!frontiereVisible) {
        printf("⚠️  ATTENTION: Frontiere hors zone visible!\n");
        printf("    Essayez d'autres colonnes.\n\n");
    }

    // Calcul importance pour l'affichage
    double sumPoids = 0;
    for(int i = 0; i < p->nPoids; i++) sumPoids += fabs(p->poids[i]);
    double importanceVisible = (fabs(p->poids[colX]) + fabs(p->poids[colY])) / sumPoids * 100.0;

    // ===== ÉTAPE 3: BOUCLE DE RENDU =====
    while (!WindowShouldClose()) {
        BeginDrawing();
        ClearBackground(RAYWHITE);

        // Rendu zones de décision
        dessinerZonesDecision(p, colX, colY, centerPoint, ds->nbColonne,
                             minX, maxX, minY, maxY, W, H);

        // Tracé frontière
        tracerFrontiere(p, colX, colY, centerPoint, ds->nbColonne,
                       minX, maxX, minY, maxY, W, H);

        // Dessin des points
        dessinerPoints(ds, p, colX, colY, centerPoint, minX, maxX, minY, maxY, W, H);

        // ===== INTERFACE =====
        DrawRectangle(10, 10, 520, 95, Fade(BLACK, 0.75f));
        DrawText(TextFormat("AXES: [%s] vs [%s]", ds->nomColonne[colX], ds->nomColonne[colY]),
                 20, 20, 18, WHITE);

        Color accuracyColor = accuracy2D > 90 ? GREEN : (accuracy2D > 70 ? ORANGE : RED);
        DrawText(TextFormat("Accuracy 2D: %.1f%% (%d/%d)", accuracy2D, nbCorrect, ds->nTrain),
                 20, 45, 18, accuracyColor);

        Color importanceColor = importanceVisible > 70 ? GREEN : (importanceVisible > 40 ? ORANGE : RED);
        DrawText(TextFormat("Importance visible: %.1f%%", importanceVisible),
                 20, 65, 16, importanceColor);

        DrawText("Cercle rouge = mal classe", 20, 85, 14, LIGHTGRAY);

        if(!frontiereVisible) {
            DrawRectangle(W - 310, 10, 300, 40, Fade(RED, 0.8f));
            DrawText("⚠️  Frontiere hors de vue!", W - 300, 20, 16, WHITE);
        }

        EndDrawing();
    }

    // ===== NETTOYAGE =====
    free(centerPoint);
    CloseWindow();
}