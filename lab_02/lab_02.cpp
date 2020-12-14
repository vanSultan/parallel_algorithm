#include <mpi.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>

#define MATRIX_SIZE 800

void print_matrix(double* a, int size) {
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++)
			printf("%7.4f\t", a[i * size + j]);
		printf("\n");
	}
	printf("\n");
}

void fill_matrix(double* a, int size) {
	for (int i = 0; i < size; i++)
		for (int j = 0; j < size; j++)
			a[i * size + j] = rand() / double(10000);
}

void transpose_matrix(double*& a, int size) {
	double temp;
	for (int i = 0; i < size; i++)
		for (int j = i + 1; j < size; j++) {
			temp = a[i * size + j];
			a[i * size + j] = a[j * size + i];
			a[j * size + i] = temp;
		}
}

void tape_multiplication(double*& a, double*& b, double*& c, int size) {
	MPI_Status status;
	double temp;
	int proc_size, proc_rank;
	int prev_proc, next_proc, cur_proc;
	MPI_Comm_size(MPI_COMM_WORLD, &proc_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

	int proc_part_size = size / proc_size;
	int proc_part_elem = proc_part_size * size;
	double* buffer_a = new double[size * proc_part_size];
	double* buffer_b = new double[size * proc_part_size];
	double* buffer_c = new double[size * proc_part_size];
	int proc_part = size / proc_size, part = proc_part * size;

	if (proc_rank == 0)
		transpose_matrix(b, size);

	MPI_Scatter(a, part, MPI_DOUBLE, buffer_a, part, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatter(b, part, MPI_DOUBLE, buffer_b, part, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	for (int i = 0; i < proc_part_size; i++)
		for (int j = 0; j < proc_part_size; j++) {
			temp = 0.;
			for (int k = 0; k < size; k++) {
				temp += buffer_a[i * size + k] * buffer_b[j * size + k];
			}
			buffer_c[i * size + j + proc_part_size * proc_rank] = temp;
		}
	
	for (int p = 1; p < proc_size; p++) {
		next_proc = proc_rank + 1;
		if (proc_rank == proc_size - 1)
			next_proc = 0;
		prev_proc = proc_rank - 1;
		if (proc_rank == 0)
			prev_proc = proc_size - 1;

		MPI_Sendrecv_replace(buffer_b, part, MPI_DOUBLE, next_proc, 0, prev_proc, 0, MPI_COMM_WORLD, &status);

		for (int i = 0; i < proc_part_size; i++)
			for (int j = 0; j < proc_part_size; j++) {
				temp = 0.;
				for (int k = 0; k < size; k++)
					temp += buffer_a[i * size + k] * buffer_b[j * size + k];
				if (proc_rank - p >= 0)
					cur_proc = proc_rank - p;
				else
					cur_proc = proc_size - p + proc_rank;
				buffer_c[i * size + j + cur_proc * proc_part_size] = temp;
			}
	}
	MPI_Gather(buffer_c, proc_part_elem, MPI_DOUBLE, c, proc_part_elem, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	delete[] buffer_a;
	delete[] buffer_b;
	delete[] buffer_c;
}

void parallel_matrix_mult(float local_A[], float local_B[], float local_C[], int n, int n_bar, int p) {
	float* B_cols;
	MPI_Datatype gather_mpi_t;
	int block;
	allocate_matrix(&B_cols, n, n_bar);
	MPI_Type_vector(n_bar, n_bar, n, MPI_FLOAT, &gather_mpi_t);
	MPI_Type_commit(&gather_mpi_t);
	for (block = 0; block < p; block++) {
		MPI_Allgather(local_B + block * n_bar, 1, gather_mpi_t, B_cols, n_bar * n_bar, MPI_FLOAT, MPI_COMM_WORLD);
		matrix_mult(local_A, B_cols, local_C, n_bar, n, block);
	}
	free(B_cols);
	MPI_Type_free(&gather_mpi_t);
}


int main(int argc, char** argv) {
	srand(time(NULL));
	double* a = NULL; double* b = NULL; double* c = NULL;
	double tick, tack;
	int proc_size, proc_rank, size = MATRIX_SIZE;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &proc_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);

	if (proc_rank == 0) {
		a = new double[size* size];
		b = new double[size* size];
		c = new double[size* size];

		fill_matrix(a, size); fill_matrix(b, size);
	}

	tick = MPI_Wtime();
	tape_multiplication(a, b, c, size);
	tack = MPI_Wtime();
	
	if (proc_rank == 0) {
		printf("time: %f7.4", tack - tick);
	}

	MPI_Finalize();

	delete[] a;
	delete[] b;
	delete[] c;

	return 0;
}
