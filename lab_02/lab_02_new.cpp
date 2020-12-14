#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <cmath>


MPI_Status status;
inline int get(int i, int j, int nCol)
{
    return i * nCol + j;
}



int numnodes, myid, mpi_err;
#define mpi_root 0

const int DIMENSION = 16 * 100;


void init_it(int* argc, char*** argv);

void init_it(int* argc, char*** argv) {
    mpi_err = MPI_Init(argc, argv);
    mpi_err = MPI_Comm_size(MPI_COMM_WORLD, &numnodes);
    mpi_err = MPI_Comm_rank(MPI_COMM_WORLD, &myid);
}

int main(int argc, char* argv[]) {
    int* local_a;
    int* local_c;
    int* local_b;
    int* b, * a = NULL, * c = NULL;

    double start = 0, finish = 0, partialTime = 0;

    init_it(&argc, &argv);

    MPI_Datatype columnsType;
    MPI_Datatype subMatrixType;
    MPI_Status status;


    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();


    b = (int*)malloc(sizeof(int) * DIMENSION * DIMENSION);




    int nColumns = (int) (DIMENSION / sqrt(numnodes - 1));
    int stride = DIMENSION;
    if (myid == mpi_root) {

        a = (int*)malloc(sizeof(int) * DIMENSION * DIMENSION);

        for (int i = 0; i < DIMENSION * DIMENSION; i++)
        {
            a[i] = 1;
            b[i] = 2;
        }
        c = (int*)malloc(sizeof(int) * DIMENSION * DIMENSION);



        MPI_Type_vector(DIMENSION, nColumns, stride, MPI_INT, &columnsType);
        MPI_Type_commit(&columnsType);

        for (int dest = 1; dest < numnodes; dest++)
        {
            int starterIndexB = ((dest - 1) * nColumns) % DIMENSION;
            int starterIndexA = (dest - 1) * nColumns * DIMENSION % (DIMENSION * DIMENSION);
            MPI_Send(&a[starterIndexA], nColumns * DIMENSION, MPI_INT, dest, 2, MPI_COMM_WORLD);
            MPI_Send(&b[starterIndexB], 1, columnsType, dest, 0, MPI_COMM_WORLD);
        }

    }


    local_a = (int*)malloc(sizeof(int) * nColumns * DIMENSION);
    local_b = (int*)malloc(sizeof(int) * nColumns * DIMENSION);
    local_c = (int*)malloc(sizeof(int) * nColumns * nColumns);


    if (myid != mpi_root)
    {

        MPI_Recv(local_a, nColumns * DIMENSION, MPI_INT, mpi_root, 2, MPI_COMM_WORLD, &status);
        MPI_Recv(local_b, DIMENSION * nColumns, MPI_INT, mpi_root, 0, MPI_COMM_WORLD, &status);

        for (int i = 0; i < nColumns; ++i) {
            for (int j = 0; j < nColumns; ++j) {

                local_c[get(i, j, nColumns)] = 0;
                for (int k = 0; k < DIMENSION; ++k) {
                    local_c[get(i, j, nColumns)] += local_a[get(i, k, DIMENSION)] * local_b[get(j, k, DIMENSION)];
                }


            }
        }

        MPI_Send(local_c, nColumns * nColumns, MPI_INT, mpi_root, 1, MPI_COMM_WORLD);



    }


    if (myid == mpi_root)
    {
        MPI_Type_vector(nColumns, nColumns, stride, MPI_INT, &subMatrixType);
        MPI_Type_commit(&subMatrixType);
        int starterIndex = 0;
        for (int source = 1; source < numnodes; ++source) {


            MPI_Recv(&c[starterIndex], 1, subMatrixType, source, 1, MPI_COMM_WORLD, &status);

            if ((starterIndex + nColumns) % DIMENSION == 0)
            {
                starterIndex += DIMENSION * (nColumns - 1);
            }

            starterIndex += nColumns;
        }
    }


    //    if (myid == mpi_root)
    //    {
    //        for (int i = 0; i< DIMENSION*DIMENSION; i++)
    //        {
    //            printf ("%d ", c[i]);
    //            if ((i+1)%DIMENSION ==0)
    //                printf ("\n");
    //        }
    //    }

    MPI_Barrier(MPI_COMM_WORLD);
    finish = MPI_Wtime();
    partialTime = finish - start;

    if (myid == mpi_root)
    {
        printf("Time = %f\n", partialTime);
    }

    mpi_err = MPI_Finalize();

    if (myid == mpi_root)
    {
        free(a);
        free(b);
        free(c);
    }
    free(local_a);
    free(local_b);
    free(local_c);
}