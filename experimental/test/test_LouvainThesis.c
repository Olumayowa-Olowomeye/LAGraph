#include <stdio.h>
#include <acutest.h>

#include <LAGraphX.h>
#include <LAGraph_test.h>
#include "LG_Xtest.h"

#define dbg(x) GxB_print(x, 5)
#define err(x, info)                                    \
    if (!(info == GrB_SUCCESS || info == GrB_NO_VALUE)) \
    {                                                   \
        char **err;                                     \
        GrB_error(err, x);                              \
        printf("\ninfo: %d error: %s\n", info, err);    \
    }
char msg[LAGRAPH_MSG_LEN];
LAGraph_Graph G;
GrB_Matrix A;
#define LEN 512
char filename[LEN + 1];
typedef struct
{
    const char *matrix_file; // Adjeancy matrix or graph
    const double mod;
} matrix_info;

const matrix_info files[] = {

    {"comm0.mtx", 0.357142857142857},
    {"res1.mtx", 0.0},
    {"karate.mtx", .42},
    {"50node.mtx", .42},
    {"", -1}};

void test_Louvain(void)
{
    LAGraph_Init(msg);
    // Lagraph+RAndom_init
    printf("\n");
    for (int k = 0;; k++)
    {
        if (strlen(files[k].matrix_file) == 0)
            break;
        snprintf(filename, LEN, LG_DATA_DIR "%s", files[k].matrix_file);
        FILE *f = fopen(filename, "r");
        TEST_CHECK(f != NULL);
        OK(LAGraph_MMRead(&A, f, msg));
        OK(LAGraph_New(&G, &A, LAGraph_ADJACENCY_DIRECTED, msg));
        TEST_CHECK(A == NULL);

        // check if the pattern is symmetric - if it isn't make it.
        OK(LAGraph_Cached_IsSymmetricStructure(G, msg));

        if (G->is_symmetric_structure == LAGraph_FALSE)
        {
            printf("This matrix is not symmetric. \n");
            // make the adjacency matrix symmetric
            OK(LAGraph_Cached_AT(G, msg));
            OK(GrB_eWiseAdd(G->A, NULL, NULL, GrB_LOR, G->A, G->AT, NULL));
            G->is_symmetric_structure = true;
            // consider the graph as directed
            G->kind = LAGraph_ADJACENCY_DIRECTED;
        }
        else
        {
            G->kind = LAGraph_ADJACENCY_UNDIRECTED;
        }
        GrB_Matrix S = NULL;
        double tsimple = LAGraph_WallClockTime();
        OK(LAGraph_Louvain(&S, G, msg));

        // OK(LAGraph_Louvain_res(&S,G,.3,msg));
        tsimple = LAGraph_WallClockTime() - tsimple;
        double Q = 0;
        double gamma = 1;
        dbg(S);
        dbg(G->A);
        // GrB_Index comms;
        GrB_Info info = (LAGr_Modularity2(&Q, gamma, G->A, S, msg));
        err(S,info);
        printf("Q:%f\n", Q);
        // printf("Number of Communities: %d",comms);
        printf(" time: %f\n", tsimple);
    }
}

void test_Louvain2(void)
{
    LAGraph_Init(msg);
    // Lagraph+RAndom_init
    printf("\n");
    for (int k = 0;; k++)
    {
        if (strlen(files[k].matrix_file) == 0)
            break;
        snprintf(filename, LEN, LG_DATA_DIR "%s", files[k].matrix_file);
        FILE *f = fopen(filename, "r");
        TEST_CHECK(f != NULL);
        OK(LAGraph_MMRead(&A, f, msg));
        OK(LAGraph_New(&G, &A, LAGraph_ADJACENCY_DIRECTED, msg));
        TEST_CHECK(A == NULL);

        // check if the pattern is symmetric - if it isn't make it.
        OK(LAGraph_Cached_IsSymmetricStructure(G, msg));

        if (G->is_symmetric_structure == LAGraph_FALSE)
        {
            printf("This matrix is not symmetric. \n");
            // make the adjacency matrix symmetric
            OK(LAGraph_Cached_AT(G, msg));
            OK(GrB_eWiseAdd(G->A, NULL, NULL, GrB_LOR, G->A, G->AT, NULL));
            G->is_symmetric_structure = true;
            // consider the graph as directed
            G->kind = LAGraph_ADJACENCY_DIRECTED;
        }
        else
        {
            G->kind = LAGraph_ADJACENCY_UNDIRECTED;
        }
        GrB_Matrix S = NULL;
        double tsimple = LAGraph_WallClockTime();
       GrB_Info info = (LAGraph_Louvain2(&S, G, msg));
        err(S,info);
        // OK(LAGraph_Louvain_res(&S,G,.3,msg));
        tsimple = LAGraph_WallClockTime() - tsimple;
        double Q =0;
        double gamma = 1;
        info = (LAGr_Modularity2(&Q, gamma, G->A, S, msg));
        err(S,info);
        printf("Q:%f\n", Q);
        printf(" time: %f\n", tsimple);
    }
}
void test_LouvainMIS(void)
{
    LAGraph_Init(msg);
    printf("\n");
    for (int k = 0;; k++)
    {
        if (strlen(files[k].matrix_file) == 0)
            break;
        snprintf(filename, LEN, LG_DATA_DIR "%s", files[k].matrix_file);
        FILE *f = fopen(filename, "r");
        TEST_CHECK(f != NULL);
        OK(LAGraph_MMRead(&A, f, msg));
        OK(LAGraph_New(&G, &A, LAGraph_ADJACENCY_DIRECTED, msg));
        TEST_CHECK(A == NULL);

        // check if the pattern is symmetric - if it isn't make it.
        OK(LAGraph_Cached_OutDegree(G, msg));
        OK(LAGraph_Cached_IsSymmetricStructure(G, msg));

        if (G->is_symmetric_structure == LAGraph_FALSE)
        {
            printf("This matrix is not symmetric. \n");
            // make the adjacency matrix symmetric
            OK(LAGraph_Cached_AT(G, msg));
            OK(GrB_eWiseAdd(G->A, NULL, NULL, GrB_LOR, G->A, G->AT, NULL));
            G->is_symmetric_structure = true;
            // consider the graph as directed
            G->kind = LAGraph_ADJACENCY_DIRECTED;
        }
        else
        {
            G->kind = LAGraph_ADJACENCY_UNDIRECTED;
        }
        GrB_Matrix S = NULL;
        double tsimple = LAGraph_WallClockTime();
        OK(LAGraph_LouvainIS(&S, G, msg));

        // OK(LAGraph_LouvainMIS_res(&S,G,.4,msg));
        tsimple = LAGraph_WallClockTime() - tsimple;
        GxB_print(S,5);
        double Q = 0;
        double gamma = 1;
        OK(LAGr_Modularity2(&Q, gamma, G->A, S, msg));
        printf("Q:%f\n", Q);
        printf(" time: %f\n", tsimple);
    }
}
TEST_LIST = {
    {"Louvain", test_Louvain},
    {"Louvain2", test_Louvain2},
    {"LouvainMIS", test_LouvainMIS},

    {NULL, NULL}};
