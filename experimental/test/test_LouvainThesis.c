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
GrB_Matrix A = NULL;
#define LEN 512
char filename[LEN + 1];
typedef struct
{
    const char *matrix_file; // Adjeancy matrix or graph
    const double mod;
} matrix_info;

const matrix_info files[] = {

    {"comm0.mtx", 0.357142857142857},
    {"karate.mtx", .42},
    {"50node.mtx", .42},
    {"", -1}};

void test_Louvain(void)
{
    LAGraph_Init(msg);

    for (int k = 0;; k++)
    {
        const char *aname = files[k].matrix_file;
        if (strlen(aname) == 0)
            break;
        printf("\n================================== %s:\n", aname);
        snprintf(filename, LEN, LG_DATA_DIR "%s", files[k].matrix_file);
        FILE *f = fopen(filename, "r");
        TEST_CHECK(f != NULL);
        OK(LAGraph_MMRead(&A, f, msg));
        fclose(f);

        OK(LAGraph_New(&G, &A, LAGraph_ADJACENCY_DIRECTED, msg));
        TEST_CHECK(A == NULL);


        OK(LAGraph_Cached_AT(G, msg));
        // check if the pattern is symmetric - if it isn't make it.
        OK(LAGraph_Cached_IsSymmetricStructure(G, msg));
        GrB_Matrix S = NULL;
        double tsimple = LAGraph_WallClockTime();
        OK(LAGraph_Louvain(&S, G, msg));

        // OK(LAGraph_Louvain_res(&S,G,.3,msg));
        tsimple = LAGraph_WallClockTime() - tsimple;
        double Q = 0;
        OK(LAGr_Modularity2(&Q, 1.0, G->A, S, msg));
        printf("Q:%f\n", Q);
        // printf("Number of Communities: %d",comms);
        printf(" time: %f\n", tsimple);
        OK(LAGraph_Delete(&G, msg));
    }
    LAGraph_Finalize(msg);
}

void test_Louvain2(void)
{
     LAGraph_Init(msg);

    for (int k = 0;; k++)
    {
        uint64_t seed = 12124231245;

        const char *aname = files[k].matrix_file;
        if (strlen(aname) == 0)
            break;
        printf("\n================================== %s:\n", aname);
        snprintf(filename, LEN, LG_DATA_DIR "%s", files[k].matrix_file);
        FILE *f = fopen(filename, "r");
        TEST_CHECK(f != NULL);
        OK(LAGraph_MMRead(&A, f, msg));
        fclose(f);

        OK(LAGraph_New(&G, &A, LAGraph_ADJACENCY_DIRECTED, msg));
        TEST_CHECK(A == NULL);


        OK(LAGraph_Cached_AT(G, msg));
        // check if the pattern is symmetric - if it isn't make it.
        OK(LAGraph_Cached_IsSymmetricStructure(G, msg));
        GrB_Matrix S = NULL;
        double tsimple = LAGraph_WallClockTime();
        OK(LAGraph_Louvain2(&S, G,seed, msg));

        // OK(LAGraph_Louvain_res(&S,G,.3,msg));
        tsimple = LAGraph_WallClockTime() - tsimple;
        double Q = 0;
        OK(LAGr_Modularity2(&Q, 1.0, G->A, S, msg));
        printf("Q:%f\n", Q);
        // printf("Number of Communities: %d",comms);
        printf(" time: %f\n", tsimple);
        OK(LAGraph_Delete(&G, msg));
    }
    LAGraph_Finalize(msg);
}
void test_LouvainIS(void)
{
     LAGraph_Init(msg);

    for (int k = 0;; k++)
    {
        uint64_t seed = rd();
        const char *aname = files[k].matrix_file;
        if (strlen(aname) == 0)
            break;
        printf("\n================================== %s:\n", aname);
        snprintf(filename, LEN, LG_DATA_DIR "%s", files[k].matrix_file);
        FILE *f = fopen(filename, "r");
        TEST_CHECK(f != NULL);
        OK(LAGraph_MMRead(&A, f, msg));
        fclose(f);

        OK(LAGraph_New(&G, &A, LAGraph_ADJACENCY_DIRECTED, msg));
        TEST_CHECK(A == NULL);


        OK(LAGraph_Cached_AT(G, msg));
        // check if the pattern is symmetric - if it isn't make it.
        OK(LAGraph_Cached_IsSymmetricStructure(G, msg));
        GrB_Matrix S = NULL;
        double tsimple = LAGraph_WallClockTime();
        OK(LAGraph_LouvainIS(&S,seed, G, msg));

        // OK(LAGraph_Louvain_res(&S,G,.3,msg));
        tsimple = LAGraph_WallClockTime() - tsimple;
        double Q = 0;
        OK(LAGr_Modularity2(&Q, 1.0, G->A, S, msg));
        printf("Q:%f\n", Q);
        // printf("Number of Communities: %d",comms);
        printf(" time: %f\n", tsimple);
        OK(LAGraph_Delete(&G, msg));
    }
    LAGraph_Finalize(msg);
}

TEST_LIST = {
    {"Louvain", test_Louvain},
    {"Louvain2", test_Louvain2},
    {"LouvainIS", test_LouvainIS},

    {NULL, NULL}};
