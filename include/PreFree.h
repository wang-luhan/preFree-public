#include "common.h"

void preFreeSpMV(valT *csrVal, int *csrRowPtr, int *csrColInd,
            valT *X_val, valT *Y_val, int rowA, int colA, int nnzA,
            double *cdTime, double *cdPre);