#ifndef VISUAL_H_
#define VISUAL_H_

#include "dataSet.h"
#include "perceptron.h"

// Scatter simple (déjà fait)
void visual_run(const DataSet *ds);

// Scatter + frontière de décision (final)
void visual_run_with_model(const DataSet *ds, const Perceptron *p);
void visual_run_with_model_custom(const DataSet *ds, const Perceptron *p, int colX, int colY);
void listerFichiersDataSet();

#endif //VISUAL_H_