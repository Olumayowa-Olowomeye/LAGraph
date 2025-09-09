#include "LG_internal.h"
#include <LAGraphX.h>
#include <LAGraph.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#define DEBUG 0
#define dbg(x) \
    if (DEBUG) \
    GxB_print(x, 5)
// #define err(x, info)                                    \
//     if (!(info == GrB_SUCCESS || info == GrB_NO_VALUE)) \
//     {                                                   \
//         char **err;                                     \
//         GrB_error(err, x);                              \
//         printf("\ninfo: %d error: %s\n", info, err);    \
//     }

typedef struct Theta
{
    double *d;
    uint32_t *c; /* c arrays */
    double m;
    uint64_t seed;
} Theta;

#define THETA_DEFN                    \
    "typedef struct Theta"            \
    "{"                               \
    "    double *d;"                  \
    "    uint32_t *c; /* c arrays */" \
    "    double m;"                   \
    "    uint64_t seed;"              \
    "} Theta;"

typedef struct argmax_tup
{
    double score; /* change in modularity */
    int32_t k;    /* who */
    double tb;
} argmax_tup;

#define AM_TUP                                     \
    "typedef struct argmax_tup"                    \
    "{"                                            \
    "    double score; /* change in modularity */" \
    "    int64_t k;    /* who */"                  \
    "    double tb;"                               \
    "} argmax_tup;"

void make_argmax_tup(argmax_tup *z,
                     const double *x, GrB_Index ix, GrB_Index jx,
                     const double *y, GrB_Index iy, GrB_Index jy,
                     const void *theta)
{
    Theta *_theta = (Theta *)theta;
    uint64_t seed = _theta->seed + (*y + ix + iy + jy);
    seed ^= seed << 13;
    seed ^= seed >> 7;
    seed ^= seed << 17;
    z->k = (int32_t)jx;
    z->score = (*x) - ((_theta->d[ix]) * (*y)) / (2 * _theta->m);
    z->tb = seed;
}
#define MAKE_AM_TUP                                                       \
    "void make_argmax_tup(argmax_tup *z,\n"                               \
    "                     const double *x, GrB_Index ix, GrB_Index jx,\n" \
    "                     const double *y, GrB_Index iy, GrB_Index jy,\n" \
    "                     const void *theta)\n"                           \
    "{\n"                                                                 \
    "    Theta *_theta = (Theta *)theta;\n"                               \
    "    uint64_t seed = _theta->seed + (*y + ix + iy + jy);\n"           \
    "    seed ^= seed << 13;\n"                                           \
    "    seed ^= seed >> 7;\n"                                            \
    "    seed ^= seed << 17;\n"                                           \
    "    z->k = (int32_t)jx;\n"                                           \
    "    z->score = (*x) - ((_theta->d[ix]) * (*y)) / (2 * _theta->m);\n" \
    "    z->tb = seed;\n"                                                 \
    "}\n"

void argmax_op(argmax_tup *z, argmax_tup *x, argmax_tup *y)
{
    if (x->score > y->score)
    {
        z->score = x->score;
        z->k = x->k;
        z->tb = x->tb;
    }
    else if (x->score == y->score)
    {
        if (x->tb > y->tb)
        {
            z->score = x->score;
            z->k = x->k;
            z->tb = x->tb;
        }
        else
        {
            z->score = y->score;
            z->k = y->k;
            z->tb = y->tb;
        }
    }
    else
    {
        z->score = y->score;
        z->k = y->k;
        z->tb = y->tb;
    }
}
#define AM_OP                                                       \
    "void argmax_op(argmax_tup *z, argmax_tup *x, argmax_tup *y)\n" \
    "{\n"                                                           \
    "    if (x->score > y->score)\n"                                \
    "    {\n"                                                       \
    "        z->score = x->score;\n"                                \
    "        z->k = x->k;\n"                                        \
    "        z->tb = x->tb;\n"                                      \
    "    }\n"                                                       \
    "    else if (x->score == y->score)\n"                          \
    "    {\n"                                                       \
    "        if (x->tb > y->tb)\n"                                  \
    "        {\n"                                                   \
    "            z->score = x->score;\n"                            \
    "            z->k = x->k;\n"                                    \
    "            z->tb = x->tb;\n"                                  \
    "        }\n"                                                   \
    "        else\n"                                                \
    "        {\n"                                                   \
    "            z->score = y->score;\n"                            \
    "            z->k = y->k;\n"                                    \
    "            z->tb = y->tb;\n"                                  \
    "        }\n"                                                   \
    "    }\n"                                                       \
    "    else\n"                                                    \
    "    {\n"                                                       \
    "        z->score = y->score;\n"                                \
    "        z->k = y->k;\n"                                        \
    "        z->tb = y->tb;\n"                                      \
    "    }\n"                                                       \
    "}\n"
static GrB_Info build_argmax_operator(
    GrB_Type Theta_UDT, GrB_Type Tuple,
    double *d, uint32_t *c, double m, uint64_t seed,
    GrB_BinaryOp *MAKEAMTUP_Bop,
    GrB_BinaryOp *AM_Bop,
    GrB_Monoid *AM_mon,
    GrB_Semiring *AM_Semiring,
    GxB_IndexBinaryOp *MAKEAMTUP_op,
    GrB_Scalar *_0,
    char *msg)
{
    Theta theta_scalar;
    theta_scalar.d = d;
    theta_scalar.c = c;
    theta_scalar.seed = seed;
    theta_scalar.m = m;

    GRB_TRY(GrB_Scalar_setElement_UDT(*_0, (void *)&theta_scalar));
    GRB_TRY(GxB_IndexBinaryOp_new(MAKEAMTUP_op,
                                  (GxB_index_binary_function)make_argmax_tup,
                                  Tuple, GrB_FP64, GrB_FP64, Theta_UDT,
                                  "make_argmax_tup", MAKE_AM_TUP));
    GRB_TRY(GxB_BinaryOp_new_IndexOp(MAKEAMTUP_Bop, *MAKEAMTUP_op, *_0));

    argmax_tup id;
    memset(&id, 0, sizeof(argmax_tup));
    id.k = INT64_MAX;
    id.score = (double)(-INFINITY);

    GRB_TRY(GxB_BinaryOp_new(AM_Bop,
                             (GxB_binary_function)argmax_op, Tuple, Tuple, Tuple,
                             "argmax_op", AM_OP));

    GRB_TRY(GrB_Monoid_new_UDT(AM_mon, *AM_Bop, &id));

    GRB_TRY(GrB_Semiring_new(AM_Semiring, *AM_mon, *MAKEAMTUP_Bop));

    return GrB_SUCCESS;
}
void extract_k(void *out, const void *in)
{
    const argmax_tup *a = in;
    int64_t *k_out = out;
    *k_out = a->k;
}

#define EXTRACT_K_SRC                               \
    "void extract_k(void *out, const void *in) {\n" \
    "    const argmax_tup *a = in;\n"               \
    "    int64_t *k_out = out;\n"                   \
    "    *k_out = a->k;\n"                          \
    "}\n"

#undef LG_FREE_ALL
#define LG_FREE_ALL                  \
    {                                \
        GrB_free(&x);                \
        GrB_free(&y);                \
        GrB_free(&iset);             \
        GrB_free(&k);                \
        GrB_free(&neighbours);       \
        GrB_free(&S);                \
        GrB_free(&S_container);      \
        GrB_free(&A);                \
        GrB_free(&W);                \
        GrB_free(&Wy);               \
        GrB_free(&A_iset);           \
        GrB_free(&AM_Semiring);      \
        GrB_free(&AM_mon);           \
        GrB_free(&AS);               \
        GrB_free(&StAS);             \
        \  
                                                                                                                                      \
    }
int LAGraph_LouvainMIS(
    // output
    GrB_Matrix *S_result,
    // input
    LAGraph_Graph G,
    char *msg)
{
#if LG_SUITESPARSE_GRAPHBLAS_V10
    char MATRIX_TYPE[LAGRAPH_MSG_LEN];
    char *err;
    // if (DEBUG)
    GrB_set(GrB_GLOBAL, DEBUG, GxB_BURBLE);
    // Shortened monoids, Binary ops, and Semirings
    GrB_Monoid plusmon = GrB_PLUS_MONOID_FP64;
    GrB_Monoid timesmon = GrB_TIMES_MONOID_FP64;
    GrB_BinaryOp plusf64 = GrB_PLUS_FP64;
    GrB_BinaryOp timesf64 = GrB_TIMES_FP64;
    GrB_BinaryOp divf64 = GrB_DIV_FP64;
    GrB_Semiring stdmxm = GrB_PLUS_TIMES_SEMIRING_FP64;
    GrB_BinaryOp UDT_AM;

    // Declarations
    GrB_Descriptor rilist = NULL;
    GrB_Descriptor rvlist = NULL;
    GrB_Vector iset = NULL;
    GrB_Vector k = NULL;
    GrB_Vector x = NULL;
    GrB_Vector y = NULL;
    GrB_Vector neighbours = NULL;
    GrB_Matrix S = NULL;
    GxB_Container S_container = NULL;
    GrB_Matrix A = NULL;
    GrB_Matrix AS = NULL;
    GrB_Matrix StAS = NULL;
    GrB_Matrix A_iset = NULL;
    GrB_Index n;
    GrB_Matrix W = NULL;
    GrB_Vector Wy = NULL;
    GrB_Scalar argmax_0;
    GrB_Type Tuple = NULL;
    GxB_IndexBinaryOp MAKEAMTUP_op = NULL;
    GrB_BinaryOp MAKEAMTUP_Bop = NULL, AM_Bop = NULL;
    GrB_Monoid AM_mon = NULL;
    GrB_Semiring AM_Semiring = NULL;
    GrB_Vector k_values = NULL;
    GrB_Vector Si_old = NULL;
    // Initializing
    LG_TRY(LAGraph_CheckGraph(G, msg));
    LG_ASSERT(S_result != NULL, GrB_NULL_POINTER);
    GrB_Info info;
    A = G->A;
    dbg(A);
    uint64_t seed = 1231245;
    // printf("here");
    // -----------------------------Index Binary OP: AM--------------------------//

    GRB_TRY(GxB_Type_new(&Tuple, sizeof(argmax_tup), "argmax_tup", AM_TUP));

    // -------------------------------------------------------//

    GRB_TRY(GrB_Matrix_nrows(&n, A));
    GRB_TRY(GrB_Vector_new(&y, GrB_FP64, n));
    GRB_TRY(GrB_Vector_new(&x, GrB_FP64, n));
    GRB_TRY(GrB_Vector_new(&neighbours, GrB_FP64, n));
    GRB_TRY(GrB_Vector_new(&k, GrB_FP64, n));
    GRB_TRY(GrB_Matrix_new(&W, GrB_FP64, n, n));
    GRB_TRY(GrB_Vector_new(&Wy, Tuple, n));
    GRB_TRY(GrB_Matrix_new(&A_iset, GrB_FP64, n, n));
    GRB_TRY(GxB_Container_new(&S_container));
    GRB_TRY(GrB_Vector_new(&iset, GrB_FP64, n));
    GRB_TRY(GrB_Matrix_new(&AS, GrB_FP64, n, n));
    GRB_TRY(GrB_Matrix_new(&StAS, GrB_FP64, n, n));

    GRB_TRY(GrB_Descriptor_new(&rilist));
    GRB_TRY(GrB_set(rilist, GxB_USE_INDICES, GxB_ROWINDEX_LIST));
    GRB_TRY(GrB_Descriptor_new(&rvlist));
    GRB_TRY(GrB_set(rvlist, GxB_USE_VALUES, GxB_ROWINDEX_LIST));

    double m;

    GrB_Matrix iset_m = NULL;


    GrB_Index niset;
    GrB_Index ncols;
    GrB_Matrix_ncols(&ncols, A);
    GrB_Matrix A_rows;

    GxB_Container k_container = NULL;
    GRB_TRY(GxB_Container_new(&k_container));

    void *f = NULL;
    uint64_t f_size;
    uint64_t f_nvals = 0, f_nheld = 0;
    GrB_Type ftype = NULL;
    int f_handling;

    void *c = NULL;
    uint64_t c_size;
    uint64_t c_nvals = 0, c_nheld = 0;
    GrB_Type ctype = NULL;
    int c_handling;

    GrB_UnaryOp extract_k_op;
    GRB_TRY(GxB_UnaryOp_new(&extract_k_op, extract_k, GrB_INT64, Tuple, "extract_k", EXTRACT_K_SRC));

    GrB_Type Theta_UDT = NULL;
    GRB_TRY(GxB_Type_new(&Theta_UDT, sizeof(Theta), "Theta", THETA_DEFN));
    GRB_TRY(GrB_Scalar_new(&argmax_0, Theta_UDT));
    bool changed = false;

    // S <- I
    GRB_TRY(GrB_assign(x, NULL, NULL, 1.0, GrB_ALL, n, NULL));
    dbg(x);
    GRB_TRY(GrB_Matrix_diag(&S, x, 0));
    GRB_TRY(GrB_set(S, GxB_SPARSE, GxB_SPARSITY_CONTROL));
    dbg(S);
    GRB_TRY(GxB_unload_Matrix_into_Container(S, S_container, NULL));
    GRB_TRY(GrB_Vector_dup(&Si_old,S_container->i));
    GRB_TRY(GxB_load_Matrix_from_Container(S, S_container, NULL));
    dbg(S);
    // return 0;
    int iter = 0;
    
    // while(!changed)
    while(!changed){
        // k = [+_j A(:,j)]
        GRB_TRY(GrB_Matrix_reduce_Monoid(k, NULL, NULL, plusmon, A, NULL));
        GRB_TRY(GrB_set(k, GxB_SPARSE, GxB_SPARSITY_CONTROL));
        dbg(k);

        GRB_TRY(GrB_Vector_reduce_FP64(&m, NULL, plusmon, k, NULL));
        m *= 0.5;
        // printf("Total edge weight (m): %f\n", m);

        GRB_TRY(LAGraph_IsolateSets(&iset_m, A, seed, msg));
        // GxB_print(iset_m, 5);
        GrB_Index loop;
        GRB_TRY(GrB_Matrix_nrows(&loop, iset_m));
        for (int i = 0; i < loop; i++)
        {
            // printf("%u\n",loop);
            GRB_TRY(GrB_Col_extract(iset, NULL, NULL, iset_m, GrB_ALL, n, i, GrB_DESC_T0));
            dbg(iset);
            GrB_Vector_nvals(&niset, iset);
            GrB_Matrix_new(&A_rows, GrB_FP64, niset, ncols);
            info = GxB_Matrix_extract_Vector(A_rows, NULL, NULL, A, iset, NULL, rilist);

            // err(A_rows, info);
            dbg(A_rows);
            info = GxB_Matrix_assign_Vector(A_iset, NULL, NULL, A_rows, iset, NULL, rilist);

            GrB_Matrix_free(&A_rows);
            GRB_TRY(GrB_mxm(W, NULL, NULL, GrB_PLUS_TIMES_SEMIRING_FP64, A_iset, S, NULL));
            dbg(W);
            GRB_TRY(GrB_vxm(y, NULL, NULL, GrB_PLUS_TIMES_SEMIRING_FP64, k, S, GrB_DESC_T0));
            dbg(y);
            dbg(k);
            GRB_TRY(GxB_unload_Vector_into_Container(k, k_container, NULL));

            GRB_TRY(GxB_Vector_unload(k_container->x, &f, &ftype, &f_nheld, &f_size, &f_handling, NULL));
            GRB_TRY(GxB_unload_Matrix_into_Container(S, S_container, NULL));
            info = (GxB_Vector_unload(S_container->i, &c, &ctype, &c_nheld, &c_size, &c_handling, NULL));
            seed += i;
            GRB_TRY(build_argmax_operator(Theta_UDT, Tuple, f, c, m, seed, &MAKEAMTUP_Bop, &AM_Bop, &AM_mon, &AM_Semiring, &MAKEAMTUP_op, &argmax_0, msg));
            GRB_TRY(GrB_mxv(Wy, NULL, NULL, AM_Semiring, W, y, NULL));
            dbg(Wy);
            GRB_TRY(GxB_Vector_load(k_container->x, &f, ftype, f_nheld, f_size, f_handling, NULL));
            dbg(k_container->x);
            GRB_TRY(GxB_load_Vector_from_Container(k, k_container, NULL));
            dbg(k);
            GRB_TRY(GxB_Vector_load(S_container->i, &c, ctype, c_nheld, c_size, c_handling, NULL));
            // GxB_print(S_container->i, 5);
            GrB_Vector_new(&k_values, GrB_INT64, n);
            GRB_TRY(GrB_Vector_apply(k_values, NULL, NULL, extract_k_op, Wy, NULL));
            GRB_TRY(GrB_assign(S_container->i, k_values, NULL, k_values, GrB_ALL, n, NULL));
            // GxB_print(S_container->i, 5);
            GRB_TRY(GxB_load_Matrix_from_Container(S, S_container, NULL));

        }
        GRB_TRY(GxB_unload_Matrix_into_Container(S, S_container, NULL));
        GRB_TRY(LAGraph_Vector_IsEqual(&changed,Si_old,S_container->i,msg));
        GRB_TRY(GrB_Vector_dup(&Si_old,S_container->i));
        GRB_TRY(GxB_load_Matrix_from_Container(S, S_container, NULL));
        GRB_TRY(GrB_mxm(AS, NULL, NULL, GrB_PLUS_TIMES_SEMIRING_FP64, A, S, NULL));
        GRB_TRY(GrB_mxm(StAS, NULL, NULL, GrB_PLUS_TIMES_SEMIRING_FP64, S, AS, GrB_DESC_T0));
        dbg(StAS);
        // dbg(A);
        GrB_free(&A);
        GRB_TRY(GrB_Matrix_dup(&A, StAS));
        iter++;
    }
    GxB_print(S,5);
    double Q;
    double gamma = 1;
    GRB_TRY(LAGr_Modularity2(&Q, gamma, A, S, msg));
    printf("Iterations: %d\n", iter);
    printf("Q:%.15g\n", Q);
    LG_FREE_ALL;
#else
    LG_ASSERT(false, GrB_NOT_IMPLEMENTED);
#endif
    return 0;
}
