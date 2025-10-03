#include "LAGraph_demo.h"
#define NTHREAD_LIST 6
#define THREAD_LIST 24, 16, 8, 4, 2, 1
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
#define MAX(a, b) ((a) > (b) ? (a) : (b))
// #define newseed(seed) ()

int main(int argc, char **argv)
{
    char msg[LAGRAPH_MSG_LEN];
    LAGraph_Graph G = NULL;
    GrB_Matrix S = NULL;
    FILE *f = NULL;
    double t_start, t_elapsed;
    uint64_t seed_base = 1224;
    int nt = NTHREAD_LIST;
    int Nthreads[20] = {0, THREAD_LIST};
    int nthreads_max, nthreads_outer, nthreads_inner;

    demo_init(false);
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
        if (nthreads > nthreads_max) continue;
        printf(" %d", nthreads);
    }
    printf("\n");

    char *matrix_name = (argc > 1) ? argv[1] : "stdin";
    LAGRAPH_TRY(readproblem(&G, NULL, false, false, true, NULL, false, argc, argv));

    // Warmup run
    LAGRAPH_TRY(LAGraph_SetNumThreads(1, nthreads_max, msg));
    seed_base = 12423259834;
    t_start = LAGraph_WallClockTime();
    LAGRAPH_TRY(LAGraph_LouvainIS(&S, seed_base, G, msg));
    t_elapsed = LAGraph_WallClockTime() - t_start;
    printf("Warmup: %10.4f sec\n", t_elapsed);

    int ntrials = 5;
    printf("# of trials per thread: %d\n", ntrials);

    for (int kk = 1; kk <= nt; kk++)
    {
        int nthreads = Nthreads[kk];
        if (nthreads > nthreads_max) continue;

        LAGRAPH_TRY(LAGraph_SetNumThreads(1, nthreads, msg));
        printf("\n--- nthreads: %2d ---\n", nthreads);

        double total_time = 0;
        double total_mod = 0;
        double max_mod = -INFINITY;

        for (int trial = 0; trial < ntrials; trial++)
        {
            GrB_free(&S);
            // seed_base += 13 << trial ;
            double Q = 0;

            t_start = LAGraph_WallClockTime();
            LAGRAPH_TRY(LAGraph_LouvainIS(&S, seed_base, G, msg));
            t_elapsed = LAGraph_WallClockTime() - t_start;

            LAGRAPH_TRY(LAGr_Modularity2(&Q, 1.0, G->A, S, msg));

            total_time += t_elapsed;
            total_mod += Q;
            max_mod = MAX(max_mod, Q);

            printf("Trial %2d: time %10.6f sec, Q %f\n", trial, t_elapsed, Q);
        }

        double avg_time = total_time / ntrials;
        double avg_mod = total_mod / ntrials;
        printf("Threads: %2d | Avg time: %10.10f sec | Avg modularity: %f | Max modularity: %f\n",
               nthreads, avg_time, avg_mod, max_mod);
        fflush(stdout);
    }

    LG_FREE_ALL;
    LAGRAPH_TRY(LAGraph_Finalize(msg));
    return 0;
}