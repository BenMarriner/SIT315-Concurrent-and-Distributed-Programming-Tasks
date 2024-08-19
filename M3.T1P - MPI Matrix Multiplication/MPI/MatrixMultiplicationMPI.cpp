// MPI Matrix Multiplication

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <time.h>
#include <fstream>
#include <mpich/mpi.h>

using namespace std;
using namespace std::chrono;

#define SIZE 10
#define DEBUG_OUTPUT_NODE 2

void randMatrix(int* m)
{
	for (size_t i = 0; i < SIZE; i++)
	{
		for (size_t j = 0; j < SIZE; j++)
		{
			m[i * SIZE + j] = rand() % 10;
		}
	}
}

void multMatrix(int* a, int* b, int *c)
{
	for (size_t i = 0; i < SIZE; i++)
	{
		for (size_t j = 0; j < SIZE; j++)
		{
			c[i * SIZE + j] = 0;

			// Find dot product
			for (size_t k = 0; k < SIZE; k++)
			{
				c[i * SIZE + j] += a[i * SIZE + k] * b[k * SIZE + j];
			}
		}
	}
}

void printMatrix(int* m)
{
	for (size_t i = 0; i < SIZE; i++)
	{
		cout << i << ": ";
		for (size_t j = 0; j < SIZE; j++)
		{
			cout << m[i * SIZE + j] << " ";
		}
		cout << "\n";
	}
	cout << endl;
}

int main(int argc, char **argv)
{
	int numtasks, rank, name_len, tag=1; 
    char name[MPI_MAX_PROCESSOR_NAME];
	int *m_A, *m_B, *m_C;
	std::chrono::_V2::system_clock::time_point start, stop;

	// Initialize the MPI environment
    MPI_Init(&argc,&argv);

    // Get the number of tasks/process
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    // Get the rank
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Find the processor name
    MPI_Get_processor_name(name, &name_len);

    srand(time(0));
	
	// Start timer
	if (rank == 0) start = high_resolution_clock::now();
	
	// Set up arrays
	m_B = (int*)malloc(SIZE * SIZE * sizeof(int));
	if (rank == 0)
	{	
		m_A = (int*)malloc(SIZE * SIZE * sizeof(int));
		m_C = (int*)malloc(SIZE * SIZE * sizeof(int));
		
		randMatrix(m_A);
		randMatrix(m_B);
	}

	int *m_ASub = (int*)malloc(SIZE * sizeof(int));
	int *m_CSub = (int*)malloc(SIZE * SIZE * sizeof(int));

	MPI_Scatter(m_A, (SIZE * SIZE) / numtasks, MPI_INT, m_ASub, (SIZE * SIZE) / numtasks, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(m_B, SIZE * SIZE, MPI_INT, 0, MPI_COMM_WORLD);

	multMatrix(m_ASub, m_B, m_CSub);

	MPI_Gather(m_CSub, SIZE, MPI_INT, m_C, SIZE, MPI_INT, 0, MPI_COMM_WORLD);

	if (rank == 0)
	{
		// Stop timer
		stop = high_resolution_clock::now();

		auto duration = duration_cast<microseconds>(stop - start);
		printf("Calculation time: %ld microseconds\n", duration.count());
		printf("Using %d nodes\n", numtasks);
	}	
	
	MPI_Finalize();

	return 0;
}