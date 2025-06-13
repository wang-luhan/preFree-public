/* 
*   Matrix Market I/O library for ANSI C
*
*   See http://math.nist.gov/MatrixMarket for details.
*
*
*/
#include "mmio.h"

char *mm_strdup(const char *s)

{

    int len = strlen(s);

    char *s2 = (char *) malloc((len+1)*sizeof(char));

    return strcpy(s2, s);

}



char  *mm_typecode_to_str(MM_typecode matcode)

{

    char buffer[MM_MAX_LINE_LENGTH];

    char *types[4];

    char *mm_strdup(const char *);

    //int error =0;



    /* check for MTX type */

    if (mm_is_matrix(matcode))

        types[0] = (char *)MM_MTX_STR;

    //else

    //    error=1;



    /* check for CRD or ARR matrix */

    if (mm_is_sparse(matcode))

        types[1] = (char *)MM_SPARSE_STR;

    else

    if (mm_is_dense(matcode))

        types[1] = (char *)MM_DENSE_STR;

    else

        return NULL;



    /* check for element data type */

    if (mm_is_real(matcode))

        types[2] = (char *)MM_REAL_STR;

    else

    if (mm_is_complex(matcode))

        types[2] = (char *)MM_COMPLEX_STR;

    else

    if (mm_is_pattern(matcode))

        types[2] = (char *)MM_PATTERN_STR;

    else

    if (mm_is_integer(matcode))

        types[2] = (char *)MM_INT_STR;

    else

        return NULL;





    /* check for symmetry type */

    if (mm_is_general(matcode))

        types[3] = (char *)MM_GENERAL_STR;

    else

    if (mm_is_symmetric(matcode))

        types[3] = (char *)MM_SYMM_STR;

    else

    if (mm_is_hermitian(matcode))

        types[3] = (char *)MM_HERM_STR;

    else

    if (mm_is_skew(matcode))

        types[3] = (char *)MM_SKEW_STR;

    else

        return NULL;



    sprintf(buffer,"%s %s %s %s", types[0], types[1], types[2], types[3]);

    return mm_strdup(buffer);



}



int mm_read_mtx_crd(char *fname, int *M, int *N, int *nz, int **I, int **J,

        double **val, MM_typecode *matcode)

{

    int ret_code;

    FILE *f;



    if (strcmp(fname, "stdin") == 0) f=stdin;

    else

    if ((f = fopen(fname, "r")) == NULL)

        return MM_COULD_NOT_READ_FILE;





    if ((ret_code = mm_read_banner(f, matcode)) != 0)

        return ret_code;



    if (!(mm_is_valid(*matcode) && mm_is_sparse(*matcode) &&

            mm_is_matrix(*matcode)))

        return MM_UNSUPPORTED_TYPE;



    if ((ret_code = mm_read_mtx_crd_size(f, M, N, nz)) != 0)

        return ret_code;





    *I = (int *)  malloc(*nz * sizeof(int));

    *J = (int *)  malloc(*nz * sizeof(int));

    *val = NULL;



    if (mm_is_complex(*matcode))

    {

        *val = (double *) malloc(*nz * 2 * sizeof(double));

        ret_code = mm_read_mtx_crd_data(f, *M, *N, *nz, *I, *J, *val,

                *matcode);

        if (ret_code != 0) return ret_code;

    }

    else if (mm_is_real(*matcode))

    {

        *val = (double *) malloc(*nz * sizeof(double));

        ret_code = mm_read_mtx_crd_data(f, *M, *N, *nz, *I, *J, *val,

                *matcode);

        if (ret_code != 0) return ret_code;

    }



    else if (mm_is_pattern(*matcode))

    {

        ret_code = mm_read_mtx_crd_data(f, *M, *N, *nz, *I, *J, *val,

                *matcode);

        if (ret_code != 0) return ret_code;

    }



    if (f != stdin) fclose(f);

    return 0;

}



int mm_read_banner(FILE *f, MM_typecode *matcode)

{

    char line[MM_MAX_LINE_LENGTH];

    char banner[MM_MAX_TOKEN_LENGTH];

    char mtx[MM_MAX_TOKEN_LENGTH];

    char crd[MM_MAX_TOKEN_LENGTH];

    char data_type[MM_MAX_TOKEN_LENGTH];

    char storage_scheme[MM_MAX_TOKEN_LENGTH];

    char *p;





    mm_clear_typecode(matcode);



    if (fgets(line, MM_MAX_LINE_LENGTH, f) == NULL)

        return MM_PREMATURE_EOF;



    if (sscanf(line, "%s %s %s %s %s", banner, mtx, crd, data_type,

        storage_scheme) != 5)

        return MM_PREMATURE_EOF;



    for (p=mtx; *p!='\0'; *p=tolower(*p),p++);  /* convert to lower case */

    for (p=crd; *p!='\0'; *p=tolower(*p),p++);

    for (p=data_type; *p!='\0'; *p=tolower(*p),p++);

    for (p=storage_scheme; *p!='\0'; *p=tolower(*p),p++);



    /* check for banner */

    if (strncmp(banner, MatrixMarketBanner, strlen(MatrixMarketBanner)) != 0)

        return MM_NO_HEADER;



    /* first field should be "mtx" */

    if (strcmp(mtx, MM_MTX_STR) != 0)

        return  MM_UNSUPPORTED_TYPE;

    mm_set_matrix(matcode);





    /* second field describes whether this is a sparse matrix (in coordinate

            storgae) or a dense array */





    if (strcmp(crd, MM_SPARSE_STR) == 0)

        mm_set_sparse(matcode);

    else

    if (strcmp(crd, MM_DENSE_STR) == 0)

            mm_set_dense(matcode);

    else

        return MM_UNSUPPORTED_TYPE;





    /* third field */



    if (strcmp(data_type, MM_REAL_STR) == 0)

        mm_set_real(matcode);

    else

    if (strcmp(data_type, MM_COMPLEX_STR) == 0)

        mm_set_complex(matcode);

    else

    if (strcmp(data_type, MM_PATTERN_STR) == 0)

        mm_set_pattern(matcode);

    else

    if (strcmp(data_type, MM_INT_STR) == 0)

        mm_set_integer(matcode);

    else

        return MM_UNSUPPORTED_TYPE;





    /* fourth field */



    if (strcmp(storage_scheme, MM_GENERAL_STR) == 0)

        mm_set_general(matcode);

    else

    if (strcmp(storage_scheme, MM_SYMM_STR) == 0)

        mm_set_symmetric(matcode);

    else

    if (strcmp(storage_scheme, MM_HERM_STR) == 0)

        mm_set_hermitian(matcode);

    else

    if (strcmp(storage_scheme, MM_SKEW_STR) == 0)

        mm_set_skew(matcode);

    else

        return MM_UNSUPPORTED_TYPE;





    return 0;

}



int mm_read_mtx_crd_size(FILE *f, int *M, int *N, int *nz)

{

    char line[MM_MAX_LINE_LENGTH];

    int num_items_read;



    /* set return null parameter values, in case we exit with errors */

    *M = *N = *nz = 0;



    /* now continue scanning until you reach the end-of-comments */

    do

    {

        if (fgets(line,MM_MAX_LINE_LENGTH,f) == NULL)

            return MM_PREMATURE_EOF;

    }while (line[0] == '%');



    /* line[] is either blank or has M,N, nz */

    if (sscanf(line, "%d %d %d", M, N, nz) == 3)

        return 0;



    else

    do

    {

        num_items_read = fscanf(f, "%d %d %d", M, N, nz);

        if (num_items_read == EOF) return MM_PREMATURE_EOF;

    }

    while (num_items_read != 3);



    return 0;

}



int mm_read_mtx_array_size(FILE *f, int *M, int *N)

{

    char line[MM_MAX_LINE_LENGTH];

    int num_items_read;

    /* set return null parameter values, in case we exit with errors */

    *M = *N = 0;



    /* now continue scanning until you reach the end-of-comments */

    do

    {

        if (fgets(line,MM_MAX_LINE_LENGTH,f) == NULL)

            return MM_PREMATURE_EOF;

    }while (line[0] == '%');



    /* line[] is either blank or has M,N, nz */

    if (sscanf(line, "%d %d", M, N) == 2)

        return 0;



    else /* we have a blank line */

    do

    {

        num_items_read = fscanf(f, "%d %d", M, N);

        if (num_items_read == EOF) return MM_PREMATURE_EOF;

    }

    while (num_items_read != 2);



    return 0;

}



int mm_write_banner(FILE *f, MM_typecode matcode)

{

    char *str = mm_typecode_to_str(matcode);

    int ret_code;



    ret_code = fprintf(f, "%s %s\n", MatrixMarketBanner, str);

    free(str);

    if (ret_code !=2 )

        return MM_COULD_NOT_WRITE_FILE;

    else

        return 0;

}



int mm_write_mtx_crd_size(FILE *f, int M, int N, int nz)

{

    if (fprintf(f, "%d %d %d\n", M, N, nz) != 3)

        return MM_COULD_NOT_WRITE_FILE;

    else

        return 0;

}



int mm_write_mtx_array_size(FILE *f, int M, int N)

{

    if (fprintf(f, "%d %d\n", M, N) != 2)

        return MM_COULD_NOT_WRITE_FILE;

    else

        return 0;

}









int mm_is_valid(MM_typecode matcode)		/* too complex for a macro */

{

    if (!mm_is_matrix(matcode)) return 0;

    if (mm_is_dense(matcode) && mm_is_pattern(matcode)) return 0;

    if (mm_is_real(matcode) && mm_is_hermitian(matcode)) return 0;

    if (mm_is_pattern(matcode) && (mm_is_hermitian(matcode) ||

                mm_is_skew(matcode))) return 0;

    return 1;

}









/*  high level routines */



int mm_write_mtx_crd(char fname[], int M, int N, int nz, int I[], int J[],

         double val[], MM_typecode matcode)

{

    FILE *f;

    int i;



    if (strcmp(fname, "stdout") == 0)

        f = stdout;

    else

    if ((f = fopen(fname, "w")) == NULL)

        return MM_COULD_NOT_WRITE_FILE;



    /* print banner followed by typecode */

    fprintf(f, "%s ", MatrixMarketBanner);

    fprintf(f, "%s\n", mm_typecode_to_str(matcode));



    /* print matrix sizes and nonzeros */

    fprintf(f, "%d %d %d\n", M, N, nz);



    /* print values */

    if (mm_is_pattern(matcode))

        for (i=0; i<nz; i++)

            fprintf(f, "%d %d\n", I[i], J[i]);

    else

    if (mm_is_real(matcode))

        for (i=0; i<nz; i++)

            fprintf(f, "%d %d %20.16g\n", I[i], J[i], val[i]);

    else

    if (mm_is_complex(matcode))

        for (i=0; i<nz; i++)

            fprintf(f, "%d %d %20.16g %20.16g\n", I[i], J[i], val[2*i],

                        val[2*i+1]);

    else

    {

        if (f != stdout) fclose(f);

        return MM_UNSUPPORTED_TYPE;

    }



    if (f !=stdout) fclose(f);



    return 0;

}



int mm_read_mtx_crd_data(FILE *f, int M, int N, int nz, int I[], int J[],

        double val[], MM_typecode matcode)

{

    int i;

    if (mm_is_complex(matcode))

    {

        for (i=0; i<nz; i++)

            if (fscanf(f, "%d %d %lg %lg", &I[i], &J[i], &val[2*i], &val[2*i+1])

                != 4) return MM_PREMATURE_EOF;

    }

    else if (mm_is_real(matcode))

    {

        for (i=0; i<nz; i++)

        {

            if (fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i])

                != 3) return MM_PREMATURE_EOF;



        }

    }



    else if (mm_is_pattern(matcode))

    {

        for (i=0; i<nz; i++)

            if (fscanf(f, "%d %d", &I[i], &J[i])

                != 2) return MM_PREMATURE_EOF;

    }

    else

        return MM_UNSUPPORTED_TYPE;



    return 0;



}



int mm_read_mtx_crd_entry(FILE *f, int *I, int *J, double *real, double *imag,

            MM_typecode matcode)

{

    if (mm_is_complex(matcode))

    {

            if (fscanf(f, "%d %d %lg %lg", I, J, real, imag)

                != 4) return MM_PREMATURE_EOF;

    }

    else if (mm_is_real(matcode))

    {

            if (fscanf(f, "%d %d %lg\n", I, J, real)

                != 3) return MM_PREMATURE_EOF;



    }



    else if (mm_is_pattern(matcode))

    {

            if (fscanf(f, "%d %d", I, J) != 2) return MM_PREMATURE_EOF;

    }

    else

        return MM_UNSUPPORTED_TYPE;



    return 0;



}



int mm_read_unsymmetric_sparse(const char *fname, int *M_, int *N_, int *nz_,

                double **val_, int **I_, int **J_)

{

    FILE *f;

    MM_typecode matcode;

    int M, N, nz;

    int i;

    double *val;

    int *I, *J;



    if ((f = fopen(fname, "r")) == NULL)

            return -1;





    if (mm_read_banner(f, &matcode) != 0)

    {

        printf("mm_read_unsymetric: Could not process Matrix Market banner ");

        printf(" in file [%s]\n", fname);

        return -1;

    }







    if ( !(mm_is_real(matcode) && mm_is_matrix(matcode) &&

            mm_is_sparse(matcode)))

    {

        fprintf(stderr, "Sorry, this application does not support ");

        fprintf(stderr, "Market Market type: [%s]\n",

                mm_typecode_to_str(matcode));

        return -1;

    }

    /* find out size of sparse matrix: M, N, nz .... */

    if (mm_read_mtx_crd_size(f, &M, &N, &nz) !=0)
    {

        fprintf(stderr, "read_unsymmetric_sparse(): could not parse matrix size.\n");

        return -1;

    }

    *M_ = M;

    *N_ = N;

    *nz_ = nz;

    /* reseve memory for matrices */

    I = (int *) malloc(nz * sizeof(int));

    J = (int *) malloc(nz * sizeof(int));

    val = (double *) malloc(nz * sizeof(double));
    
    *val_ = val;

    *I_ = I;

    *J_ = J;

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */

    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */

    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (i=0; i<nz; i++)
    {

        int rt = fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]);

        I[i]--;  /* adjust from 1-based to 0-based */

        J[i]--;

    }

    fclose(f);

    return 0;

}


void exclusive_scan(int *input, int length)
{
    if (length == 0 || length == 1)
        return;

    int old_val, new_val;

    old_val = input[0];
    input[0] = 0;
    for (int i = 1; i < length; i++)
    {
        new_val = input[i];
        input[i] = old_val + input[i - 1];
        old_val = new_val;
    }
}


// read matrix infomation from mtx file

int mmio_info(int *m, int *n, int *nnz, int *isSymmetric, char *filename)

{

    int m_tmp, n_tmp, nnz_tmp;



    int ret_code;

    MM_typecode matcode;

    FILE *f;



    int nnz_mtx_report;

    int isInteger = 0, isReal = 0, isPattern = 0, isSymmetric_tmp = 0, isComplex = 0;



    // load matrix

    if ((f = fopen(filename, "r")) == NULL)

        return -1;



    if (mm_read_banner(f, &matcode) != 0)

    {

        printf("Could not process Matrix Market banner.\n");

        return -2;

    }



    if ( mm_is_pattern( matcode ) )  { isPattern = 1; /*printf("type = Pattern\n");*/ }

    if ( mm_is_real ( matcode) )     { isReal = 1; /*printf("type = real\n");*/ }

    if ( mm_is_complex( matcode ) )  { isComplex = 1; /*printf("type = real\n");*/ }

    if ( mm_is_integer ( matcode ) ) { isInteger = 1; /*printf("type = integer\n");*/ }



    /* find out size of sparse matrix .... */

    ret_code = mm_read_mtx_crd_size(f, &m_tmp, &n_tmp, &nnz_mtx_report);

    if (ret_code != 0)

        return -4;



    if ( mm_is_symmetric( matcode ) || mm_is_hermitian( matcode ) )

    {

        isSymmetric_tmp = 1;

        //printf("input matrix is symmetric = true\n");

    }

    else

    {

        //printf("input matrix is symmetric = false\n");

    }



    int *csrRowPtr_counter = (int *)malloc((m_tmp+1) * sizeof(int));

    memset(csrRowPtr_counter, 0, (m_tmp+1) * sizeof(int));



    int *csrRowIdx_tmp = (int *)malloc(nnz_mtx_report * sizeof(int));

    int *csrColIdx_tmp = (int *)malloc(nnz_mtx_report * sizeof(int));

    valT *csrVal_tmp    = (valT *)malloc(nnz_mtx_report * sizeof(valT));



    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */

    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */

    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */



    for (int i = 0; i < nnz_mtx_report; i++)

    {

        int idxi, idxj;

        double fval, fval_im;

        int ival;

        int returnvalue;



        if (isReal)

        {

            returnvalue = fscanf(f, "%d %d %lg\n", &idxi, &idxj, &fval);

        }

        else if (isComplex)

        {

            returnvalue = fscanf(f, "%d %d %lg %lg\n", &idxi, &idxj, &fval, &fval_im);

        }

        else if (isInteger)

        {

            returnvalue = fscanf(f, "%d %d %d\n", &idxi, &idxj, &ival);

            fval = ival;

        }

        else if (isPattern)

        {

            returnvalue = fscanf(f, "%d %d\n", &idxi, &idxj);

            fval = 1.0;

        }



        // adjust from 1-based to 0-based

        idxi--;

        idxj--;



        csrRowPtr_counter[idxi]++;

        csrRowIdx_tmp[i] = idxi;

        csrColIdx_tmp[i] = idxj;

        csrVal_tmp[i] = fval;

    }



    if (f != stdin)

        fclose(f);



    if (isSymmetric_tmp)

    {

        for (int i = 0; i < nnz_mtx_report; i++)

        {

            if (csrRowIdx_tmp[i] != csrColIdx_tmp[i])

                csrRowPtr_counter[csrColIdx_tmp[i]]++;

        }

    }



    // exclusive scan for csrRowPtr_counter

    int old_val, new_val;



    old_val = csrRowPtr_counter[0];

    csrRowPtr_counter[0] = 0;

    for (int i = 1; i <= m_tmp; i++)

    {

        new_val = csrRowPtr_counter[i];

        csrRowPtr_counter[i] = old_val + csrRowPtr_counter[i-1];

        old_val = new_val;

    }



    nnz_tmp = csrRowPtr_counter[m_tmp];



    *m = m_tmp;

    *n = n_tmp;

    *nnz = nnz_tmp;

    *isSymmetric = isSymmetric_tmp;



    // free tmp space

    free(csrColIdx_tmp);

    free(csrVal_tmp);

    free(csrRowIdx_tmp);

    free(csrRowPtr_counter);



    return 0;

}



// read matrix infomation from mtx file

int mmio_data(int *csrRowPtr, int *csrColIdx, valT *csrVal, char *filename)

{

    int m_tmp, n_tmp, nnz_tmp;



    int ret_code;

    MM_typecode matcode;

    FILE *f;



    int nnz_mtx_report;

    int isInteger = 0, isReal = 0, isPattern = 0, isSymmetric_tmp = 0, isComplex = 0;



    // load matrix

    if ((f = fopen(filename, "r")) == NULL)

        return -1;



    if (mm_read_banner(f, &matcode) != 0)

    {

        printf("Could not process Matrix Market banner.\n");

        return -2;

    }



    if ( mm_is_pattern( matcode ) )  { isPattern = 1; /*printf("type = Pattern\n");*/ }

    if ( mm_is_real ( matcode) )     { isReal = 1; /*printf("type = real\n");*/ }

    if ( mm_is_complex( matcode ) )  { isComplex = 1; /*printf("type = real\n");*/ }

    if ( mm_is_integer ( matcode ) ) { isInteger = 1; /*printf("type = integer\n");*/ }



    /* find out size of sparse matrix .... */

    ret_code = mm_read_mtx_crd_size(f, &m_tmp, &n_tmp, &nnz_mtx_report);

    if (ret_code != 0)

        return -4;



    if ( mm_is_symmetric( matcode ) || mm_is_hermitian( matcode ) )

    {

        isSymmetric_tmp = 1;

        //printf("input matrix is symmetric = true\n");

    }

    else

    {

        //printf("input matrix is symmetric = false\n");

    }



    int *csrRowPtr_counter = (int *)malloc((m_tmp+1) * sizeof(int));

    memset(csrRowPtr_counter, 0, (m_tmp+1) * sizeof(int));



    int *csrRowIdx_tmp = (int *)malloc(nnz_mtx_report * sizeof(int));

    int *csrColIdx_tmp = (int *)malloc(nnz_mtx_report * sizeof(int));

    valT *csrVal_tmp    = (valT *)malloc(nnz_mtx_report * sizeof(valT));



    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */

    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */

    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */



    for (int i = 0; i < nnz_mtx_report; i++)

    {

        int idxi, idxj;

        double fval, fval_im;

        int ival;

        int returnvalue;



        if (isReal)

        {

            returnvalue = fscanf(f, "%d %d %lg\n", &idxi, &idxj, &fval);

        }

        else if (isComplex)

        {

            returnvalue = fscanf(f, "%d %d %lg %lg\n", &idxi, &idxj, &fval, &fval_im);

        }

        else if (isInteger)

        {

            returnvalue = fscanf(f, "%d %d %d\n", &idxi, &idxj, &ival);

            fval = ival;

        }

        else if (isPattern)

        {

            returnvalue = fscanf(f, "%d %d\n", &idxi, &idxj);

            fval = 1.0;

        }



        // adjust from 1-based to 0-based

        idxi--;

        idxj--;



        csrRowPtr_counter[idxi]++;

        csrRowIdx_tmp[i] = idxi;

        csrColIdx_tmp[i] = idxj;

        csrVal_tmp[i] = fval;

    }



    if (f != stdin)

        fclose(f);



    if (isSymmetric_tmp)

    {

        for (int i = 0; i < nnz_mtx_report; i++)

        {

            if (csrRowIdx_tmp[i] != csrColIdx_tmp[i])

                csrRowPtr_counter[csrColIdx_tmp[i]]++;

        }

    }



    // exclusive scan for csrRowPtr_counter

    int old_val, new_val;



    old_val = csrRowPtr_counter[0];

    csrRowPtr_counter[0] = 0;

    for (int i = 1; i <= m_tmp; i++)

    {

        new_val = csrRowPtr_counter[i];

        csrRowPtr_counter[i] = old_val + csrRowPtr_counter[i-1];

        old_val = new_val;

    }



    nnz_tmp = csrRowPtr_counter[m_tmp];

    memcpy(csrRowPtr, csrRowPtr_counter, (m_tmp+1) * sizeof(int));

    memset(csrRowPtr_counter, 0, (m_tmp+1) * sizeof(int));



    if (isSymmetric_tmp)

    {

        for (int i = 0; i < nnz_mtx_report; i++)

        {

            if (csrRowIdx_tmp[i] != csrColIdx_tmp[i])

            {

                int offset = csrRowPtr[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];

                csrColIdx[offset] = csrColIdx_tmp[i];

                csrVal[offset] = csrVal_tmp[i];

                csrRowPtr_counter[csrRowIdx_tmp[i]]++;



                offset = csrRowPtr[csrColIdx_tmp[i]] + csrRowPtr_counter[csrColIdx_tmp[i]];

                csrColIdx[offset] = csrRowIdx_tmp[i];

                csrVal[offset] = csrVal_tmp[i];

                csrRowPtr_counter[csrColIdx_tmp[i]]++;

            }

            else

            {

                int offset = csrRowPtr[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];

                csrColIdx[offset] = csrColIdx_tmp[i];

                csrVal[offset] = csrVal_tmp[i];

                csrRowPtr_counter[csrRowIdx_tmp[i]]++;

            }

        }

    }

    else

    {

        for (int i = 0; i < nnz_mtx_report; i++)

        {

            int offset = csrRowPtr[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];

            csrColIdx[offset] = csrColIdx_tmp[i];

            csrVal[offset] = csrVal_tmp[i];

            csrRowPtr_counter[csrRowIdx_tmp[i]]++;

        }

    }



    // free tmp space

    free(csrColIdx_tmp);

    free(csrVal_tmp);

    free(csrRowIdx_tmp);

    free(csrRowPtr_counter);



    return 0;

}
// read matrix infomation from mtx file
int mmio_allinone(int *m, int *n, int *nnz, int *isSymmetric, 
                  int **csrRowPtr, int **csrColIdx, valT **csrVal, 
                  char *filename)
{
    int m_tmp, n_tmp;
    int nnz_tmp;

    int ret_code;
    MM_typecode matcode;
    FILE *f;

    int nnz_mtx_report;
    int isInteger = 0, isReal = 0, isPattern = 0, isSymmetric_tmp = 0, isComplex = 0;

    // load matrix
    if ((f = fopen(filename, "r")) == NULL)
        return -1;

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        return -2;
    }

    if ( mm_is_pattern( matcode ) )  { isPattern = 1; /*printf("type = Pattern\n");*/ }
    if ( mm_is_real ( matcode) )     { isReal = 1; /*printf("type = real\n");*/ }
    if ( mm_is_complex( matcode ) )  { isComplex = 1; /*printf("type = real\n");*/ }
    if ( mm_is_integer ( matcode ) ) { isInteger = 1; /*printf("type = integer\n");*/ }

    /* find out size of sparse matrix .... */
    ret_code = mm_read_mtx_crd_size(f, &m_tmp, &n_tmp, &nnz_mtx_report);
    if (ret_code != 0)
        return -4;

    if ( mm_is_symmetric( matcode ) || mm_is_hermitian( matcode ) )
    {
        isSymmetric_tmp = 1;
        //printf("input matrix is symmetric = true\n");
    }
    else
    {
        //printf("input matrix is symmetric = false\n");
    }

    int *csrRowPtr_counter = (int *)malloc((m_tmp+1) * sizeof(int));
    memset(csrRowPtr_counter, 0, (m_tmp+1) * sizeof(int));

    int *csrRowIdx_tmp = (int *)malloc(nnz_mtx_report * sizeof(int));
    int *csrColIdx_tmp = (int *)malloc(nnz_mtx_report * sizeof(int));
    valT *csrVal_tmp    = (valT *)malloc(nnz_mtx_report * sizeof(valT));

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (int i = 0; i < nnz_mtx_report; i++)
    {
        int idxi, idxj;
        double fval, fval_im;
        int ival;
        int returnvalue;

        if (isReal)
        {
            returnvalue = fscanf(f, "%d %d %lg\n", &idxi, &idxj, &fval);
        }
        else if (isComplex)
        {
            returnvalue = fscanf(f, "%d %d %lg %lg\n", &idxi, &idxj, &fval, &fval_im);
        }
        else if (isInteger)
        {
            returnvalue = fscanf(f, "%d %d %d\n", &idxi, &idxj, &ival);
            fval = ival;
        }
        else if (isPattern)
        {
            returnvalue = fscanf(f, "%d %d\n", &idxi, &idxj);
            fval = 1.0;
        }

        // adjust from 1-based to 0-based
        idxi--;
        idxj--;
        
        csrRowPtr_counter[idxi]++;
        csrRowIdx_tmp[i] = idxi;
        csrColIdx_tmp[i] = idxj;
        csrVal_tmp[i] = fval;
    }

    if (f != stdin)
        fclose(f);

    if (isSymmetric_tmp)
    {
        for (int i = 0; i < nnz_mtx_report; i++)
        {
            if (csrRowIdx_tmp[i] != csrColIdx_tmp[i])
                csrRowPtr_counter[csrColIdx_tmp[i]]++;
        }
    }

    // exclusive scan for csrRowPtr_counter
    exclusive_scan(csrRowPtr_counter, m_tmp+1);

    int *csrRowPtr_alias = (int *)malloc((m_tmp+1) * sizeof(int));
    nnz_tmp = csrRowPtr_counter[m_tmp];
    int *csrColIdx_alias = (int *)malloc(nnz_tmp * sizeof(int));
    valT *csrVal_alias    = (valT *)malloc(nnz_tmp * sizeof(valT));

    memcpy(csrRowPtr_alias, csrRowPtr_counter, (m_tmp+1) * sizeof(int));
    memset(csrRowPtr_counter, 0, (m_tmp+1) * sizeof(int));

    if (isSymmetric_tmp)
    {
        for (int i = 0; i < nnz_mtx_report; i++)
        {
            if (csrRowIdx_tmp[i] != csrColIdx_tmp[i])
            {
                int offset = csrRowPtr_alias[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
                csrColIdx_alias[offset] = csrColIdx_tmp[i];
                csrVal_alias[offset] = csrVal_tmp[i];
                csrRowPtr_counter[csrRowIdx_tmp[i]]++;

                offset = csrRowPtr_alias[csrColIdx_tmp[i]] + csrRowPtr_counter[csrColIdx_tmp[i]];
                csrColIdx_alias[offset] = csrRowIdx_tmp[i];
                csrVal_alias[offset] = csrVal_tmp[i];
                csrRowPtr_counter[csrColIdx_tmp[i]]++;
            }
            else
            {
                int offset = csrRowPtr_alias[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
                csrColIdx_alias[offset] = csrColIdx_tmp[i];
                csrVal_alias[offset] = csrVal_tmp[i];
                csrRowPtr_counter[csrRowIdx_tmp[i]]++;
            }
        }
    }
    else
    {
        for (int i = 0; i < nnz_mtx_report; i++)
        {            
            int offset = csrRowPtr_alias[csrRowIdx_tmp[i]] + csrRowPtr_counter[csrRowIdx_tmp[i]];
            csrColIdx_alias[offset] = csrColIdx_tmp[i];
            csrVal_alias[offset] = csrVal_tmp[i];
            csrRowPtr_counter[csrRowIdx_tmp[i]]++;
        }
    }
    
    *m = m_tmp;
    *n = n_tmp;
    *nnz = nnz_tmp;
    *isSymmetric = isSymmetric_tmp;

    *csrRowPtr = csrRowPtr_alias;
    *csrColIdx = csrColIdx_alias;
    *csrVal = csrVal_alias;

    // free tmp space
    free(csrColIdx_tmp);
    free(csrVal_tmp);
    free(csrRowIdx_tmp);
    free(csrRowPtr_counter);

    return 0;
}
