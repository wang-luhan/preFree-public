#include "common.h"

void cdspmv(valT *csrVal, int *csrRowPtr, int *csrColInd,
            valT *X_val, valT *Y_val, int rowA, int colA, int nnzA,
            double *cdTime, double *cdPre);

void preFreeSpMV(valT *csrVal, int *csrRowPtr, int *csrColInd,
            valT *X_val, valT *Y_val, int rowA, int colA, int nnzA,
            double *cdTime, double *cdPre);


void csrAdaptiveSpMV(valT *csrVal, int *csrRowPtr, int *csrColInd,
            valT *X_val, valT *Y_val, int rowA, int colA, int nnzA,
            double *cdTime, double *cdPre);

void flatSpMV(valT *csrVal, int *csrRowPtr, int *csrColInd,
            valT *X_val, valT *Y_val, int rowA, int colA, int nnzA,
            double *cdTime, double *cdPre);