#include "LAGraph_demo.h"
#define NTHREAD_LIST 1
// #define NTHREAD_LIST 2
#define THREAD_LIST 0
#define LG_FREE_ALL              \
    {                            \
        GrB_free(&S);            \
        LAGraph_Delete(&G, msg); \
        if (f != NULL)           \
            fclose(f);           \
    }
#define dbg(x) \
    if (DEBUG) \
    GxB_print(x, 5)
#define err(x, info)                                    \
    if (!(info == GrB_SUCCESS || info == GrB_NO_VALUE)) \
    {                                                   \
        char **err;                                     \
        GrB_error(err, x);                              \
        printf("\ninfo: %d error: %s\n", info, err);    \
    }
#define MAX(a,b) ((a) > (b) ? (a) : (b))
int main(int argc, char **argv)
{
    char msg[LAGRAPH_MSG_LEN];

    LAGraph_Graph G = NULL;

    // GrB_Matrix A = NULL ;
    GrB_Matrix S = NULL;
    FILE *f = NULL;
    GrB_Info info;
    bool burble = false;
    demo_init(burble);

    int nt = NTHREAD_LIST;
    int Nthreads[20] = {0, THREAD_LIST};
    int nthreads_max, nthreads_outer, nthreads_inner;
    LAGRAPH_TRY(LAGraph_GetNumThreads(&nthreads_outer, &nthreads_inner, msg));
    nthreads_max = nthreads_outer * nthreads_inner;
    if (Nthreads[1] == 0)
    {
        Nthreads[1] = nthreads_max;
        for (int t = 2; t <= nt; t++)
        {
            Nthreads[t] = Nthreads[t - 1] / 2;
            if (Nthreads[t] == 0)
                nt = t - 1;
        }
    }
    printf("threads to test: ");
    for (int t = 1; t <= nt; t++)
    {
        int nthreads = Nthreads[t];
        if (nthreads > nthreads_max)
            continue;
        printf(" %d", nthreads);
    }
    printf("\n");

    char *matrix_name = (argc > 1) ? argv[1] : "stdin";
    LAGRAPH_TRY(readproblem(&G, NULL,
                            false, false, true, NULL, false, argc, argv));
    GrB_Index n, nvals;
    GRB_TRY(GrB_Matrix_nrows(&n, G->A));
    GRB_TRY(GrB_Matrix_nvals(&nvals, G->A));
    LAGRAPH_TRY(LAGraph_SetNumThreads(1, nthreads_max, msg));
    // GxB_print(G->A,5);
    uint64_t seed = 1224;

    double t1 = LAGraph_WallClockTime();
    LAGRAPH_TRY(LAGraph_Louvain2(&S, G, seed, msg));
    t1 = LAGraph_WallClockTime() - t1;
    // GxB_print(G->A,5);
    printf("warmup: %10.4f sec\n", t1);

    double Q =0;
    GrB_Index comms;

    LAGRAPH_TRY(LAGr_Modularity2(&Q, 1.0, G->A, S, msg));
    printf("Q:%f\n", Q);

    int ntrials = 16;
    printf("# of trials: %d\n", ntrials);

    for (int kk = 1; kk <= nt; kk++)
    {
        int nthreads = Nthreads[kk];
        if (nthreads > nthreads_max)
            continue;
        LAGRAPH_TRY(LAGraph_SetNumThreads(1, nthreads, msg));
        printf("\n--------------------------- nthreads: %2d\n", nthreads);

        double total_time = 0;
        double total_mod = 0;
        double max_mod = -INFINITY;
        
        for (int trial = 0; trial < ntrials; trial++)
        {
            GrB_free(&S);
            Q = 0;
            printf("seed: %ld\n",seed);
                        // GxB_print(G->A,5);
            double t1 = LAGraph_WallClockTime();
            LAGRAPH_TRY(LAGraph_Louvain2(&S, G, seed, msg));
            t1 = LAGraph_WallClockTime() - t1;
            printf("trial: %2d time: %10.8f sec\n", trial, t1);
            // GxB_print(G->A,5);
            // GxB_print(S,5);
            LAGRAPH_TRY(LAGr_Modularity2(&Q, 1.0, G->A, S, msg));
            printf("Q:%f\n", Q);

            // printf("Number of communities: %3d\n\n",comms);
            total_time += t1;
            total_mod += Q;
            max_mod = MAX(max_mod,Q);
        }
        // boop
        double t = total_time / ntrials;
        printf("Avg Modularity: %f, avg time: %10.8f, Max modularity %f\n", total_mod / ntrials, t, max_mod);
        fflush(stdout);
        fflush(stderr);
    }

    LG_FREE_ALL;
    LAGRAPH_TRY(LAGraph_Finalize(msg));
    return (GrB_SUCCESS);
}