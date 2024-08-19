// CL program code (There was no file submission box on OnTrack for it. Please move this into a .cl file before compiling)
kernel void multiply(__global int* a, __global int* b, __global int* c, const int M, const int N, const int K)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);

    int value = 0;
    for (int k = 0; k < N; k++)
    {
        value += a[i * N + k] * b[k * N + j];
    }

    c[i * N + j] = value;
}

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <mpich/mpi.h>
#include <CL/cl.h>
#include <time.h>
#include <chrono>

#define PRINT 1

using namespace std;
using namespace std::chrono;

int SZ = 8;
const int TS = 4;
int *m_A, *m_B, *m_C;

// Buffer objects to be used by the device for storing local arrays
cl_mem bufM_A, bufM_B, bufM_C;

size_t local[2] = {TS, TS};
size_t global[2] = { (size_t)SZ, (size_t)SZ };

// ID of the device performing the calculations
cl_device_id device_id;
// Stores information regarding the environment the host and device programs are being executed in
cl_context context;
// The device program to be compiled at runtime for the device
cl_program program;
// The function in the device program to be executed on the device
cl_kernel kernel;
// The command queue where the device commands will be queued for execution
cl_command_queue queue;
cl_event event = NULL;

int err;

// Registers the device with the host program so it can be used
cl_device_id create_device();
// Does the bulk of the setup like intiialise the environment, device, context, queue and kernel
void setup_openCL_device_context_queue_kernel(char *filename, char *kernelname);
// Compiles the device program to be executed on the device
cl_program build_program(cl_context ctx, cl_device_id dev, const char *filename);
// Initalise kernel memory (i.e.: allocate memory buffers for device)
void setup_kernel_memory(int num_rows);
// Copies the corresponding parameter arguments for the kernel program
void copy_kernel_args(int num_rows);
// Free all memory from the device and the host when the program is finished
void free_memory();

void init(int *&A, int size);
void print(int *A, int size);
void head(int num_processes);
void node(int num_processes, int rank);


int main(int argc, char **argv)
{
    int numtasks, rank, name_len, tag=1; 
    char name[MPI_MAX_PROCESSOR_NAME];

	// Initialize the MPI environment
    MPI_Init(&argc,&argv);

    // Get the number of tasks/process
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    // Get the rank
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Find the processor name
    MPI_Get_processor_name(name, &name_len);

    srand(time(0));

    if (argc > 1)
        SZ = atoi(argv[1]);


    if (rank == 0)  head(numtasks);
    else            node(numtasks, rank);
    
    MPI_Finalize();

    //result vector
    print(m_C, SZ);


    //frees memory for device, kernel, queue, etc.
    //you will need to modify this to free your own buffers
    free_memory();
}

void head(int num_processes)
{
    init(m_A, SZ * SZ);
    init(m_B, SZ * SZ);
    init(m_C, SZ * SZ);

    //initial vector
    print(m_A, SZ * SZ);
    print(m_B, SZ * SZ);

    int num_rows = SZ / num_processes;
    int num_elements_for_bcast = SZ * SZ;
    int num_elements_for_scatter_gather = (SZ * SZ) / num_processes;

    MPI_Scatter(&m_A[0], num_elements_for_scatter_gather, MPI_INT, &m_A[0], 0, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&m_B[0], num_elements_for_bcast, MPI_INT, 0, MPI_COMM_WORLD);

    global[0] = num_rows; global[1] = SZ;
    local[0] = num_rows; local[1] = SZ;

    setup_openCL_device_context_queue_kernel((char *)"./MatrixMultiplicationOpenCL.cl", (char *)"multiply");
    setup_kernel_memory(num_rows);
    copy_kernel_args(num_rows);

    // This is the function used to enqueue a command to be executed on the device.
    // We provide the command queue object which points to the device. We also provide
    // the number of worker threads the device will use.
    
    // Start timer
    cout << "Starting timer..." << endl;
    auto start = high_resolution_clock::now();

    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, &event);
    clWaitForEvents(1, &event);

    // Reads a section of memory into a buffer. In this case, it is reading the master array into the buffer
    clEnqueueReadBuffer(queue, bufM_C, CL_TRUE, 0, num_rows * SZ * sizeof(int), &m_C[0], 0, NULL, NULL);

    MPI_Gather(MPI_IN_PLACE, num_elements_for_scatter_gather, MPI_INT, &m_C[0], num_elements_for_scatter_gather, MPI_INT, 0, MPI_COMM_WORLD);

    // Stop timer
    auto stop = high_resolution_clock::now();
    cout << "Stopping timer..." << endl;

    // Calculate execution time
    auto duration = duration_cast<microseconds>(stop - start);

    // Print execution time
    cout << "Time taken: " << duration.count() << " microseconds" << endl;
}

void node(int num_processes, int rank)
{
    int num_rows = SZ / num_processes;
    int num_elements = SZ * SZ;
    int num_elements_for_scatter_gather = (SZ * SZ) / num_processes;

    init(m_A, num_rows);
    init(m_B, SZ * SZ);
    init(m_C, num_rows);

    global[0] = num_rows; global[1] = SZ;
    local[0] = num_rows; local[1] = SZ;

    setup_openCL_device_context_queue_kernel((char *)"./MatrixMultiplicationOpenCL.cl", (char *)"multiply");
    setup_kernel_memory(num_rows);
    copy_kernel_args(num_rows);

    // This is the function used to enqueue a command to be executed on the device.
    // We provide the command queue object which points to the device. We also provide
    // the number of worker threads the device will use.

    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, &event);
    clWaitForEvents(1, &event);

    // Reads a section of memory into a buffer. In this case, it is reading the master array into the buffer
    clEnqueueReadBuffer(queue, bufM_C, CL_TRUE, 0, SZ * sizeof(int), &m_C[0], 0, NULL, NULL);
}

void init(int *&A, int size)
{
    A = (int *)malloc(sizeof(int) * size);

    for (long i = 0; i < size; i++)
    {
        A[i] = rand() % 100; // any number less than 100
    }
}

void print(int *A, int size)
{
    if (PRINT == 0)
    {
        return;
    }

    if (PRINT == 1 && size > 15)
    {
        for (long i = 0; i < 5; i++)
        {                        //rows
            printf("%d ", A[i]); // print the cell value
        }
        printf(" ..... ");
        for (long i = size - 5; i < size; i++)
        {                        //rows
            printf("%d ", A[i]); // print the cell value
        }
    }
    else
    {
        for (long i = 0; i < size; i++)
        {                        //rows
            printf("%d ", A[i]); // print the cell value
        }
    }
    printf("\n----------------------------\n");
}

void free_memory()
{
    //free the buffers
    clReleaseMemObject(bufM_A);
    clReleaseMemObject(bufM_B);
    clReleaseMemObject(bufM_C);

    //free opencl objects
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseProgram(program);
    clReleaseContext(context);

    free(m_A);
    free(m_B);
    free(m_C);
}


void copy_kernel_args(int num_rows)
{
    // Providing the parameter arguments for the kernel
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&bufM_A);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&bufM_B);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&bufM_C);
    
    clSetKernelArg(kernel, 3, sizeof(int), (void *)&num_rows);
    clSetKernelArg(kernel, 4, sizeof(int), (void *)&SZ);
    clSetKernelArg(kernel, 5, sizeof(int), (void *)&SZ);

    if (err < 0)
    {
        perror("Couldn't create a kernel argument");
        printf("error = %d", err);
        exit(1);
    }
}

void setup_kernel_memory(int num_rows)
{
    // The clCreateBuffer() function initialises a memory buffer that will be used by the device
    // Memory flags dictate how a buffer can be used. These are the following flags that exist:
    // CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY, CL_MEM_READ_ONLY, CL_MEM_USE_HOST_PTR,
    // CL_MEM_ALLOC_HOST_PTR, CL_MEM_COPY_HOST_PTR, CL_MEM_HOST_WRITE_ONLY, CL_MEM_HOST_READ_ONLY
    // CL_MEM_HOST_NO_ACCESS, CL_MEM_SVM_FINE_GRAIN_BUFFER, CL_MEM_SVM_ATOMICS and 
    // CL_MEM_KERNEL_READ_AND_WRITE
    bufM_A = clCreateBuffer(context, CL_MEM_READ_ONLY, num_rows * SZ * sizeof(int), NULL, NULL);
    bufM_B = clCreateBuffer(context, CL_MEM_READ_ONLY, SZ * SZ * sizeof(int), NULL, NULL);
    bufM_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, num_rows * SZ * sizeof(int), NULL, NULL);

    // Copy matrices to the GPU
    clEnqueueWriteBuffer(queue, bufM_A, CL_TRUE, 0, num_rows * SZ * sizeof(int), &m_A[0], 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufM_B, CL_TRUE, 0, SZ * SZ * sizeof(int), &m_B[0], 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufM_C, CL_TRUE, 0, num_rows * SZ * sizeof(int), &m_C[0], 0, NULL, NULL);
}

void setup_openCL_device_context_queue_kernel(char *filename, char *kernelname)
{
    device_id = create_device();
    cl_int err;

    // Initialises the environment the program is operating in
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    if (err < 0)
    {
        perror("Couldn't create a context");
        exit(1);
    }

    program = build_program(context, device_id, filename);

    // Initialises the command queue to refer to the device using its device ID
    queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
    if (err < 0)
    {
        perror("Couldn't create a command queue");
        exit(1);
    };


    kernel = clCreateKernel(program, kernelname, &err);
    if (err < 0)
    {
        perror("Couldn't create a kernel");
        printf("error =%d", err);
        exit(1);
    };
}

cl_program build_program(cl_context ctx, cl_device_id dev, const char *filename)
{

    cl_program program;
    FILE *program_handle;
    char *program_buffer, *program_log;
    size_t program_size, log_size;

    /* Read program file and place content into buffer */
    program_handle = fopen(filename, "r");
    if (program_handle == NULL)
    {
        perror("Couldn't find the program file");
        exit(1);
    }
    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);
    program_buffer = (char *)malloc(program_size + 1);
    program_buffer[program_size] = '\0';
    fread(program_buffer, sizeof(char), program_size, program_handle);
    fclose(program_handle);

    // Compiles kernel programs stored as source code files (files in the OpenCL format)
    program = clCreateProgramWithSource(ctx, 1,
                                        (const char **)&program_buffer, &program_size, &err);
    if (err < 0)
    {
        perror("Couldn't create the program");
        exit(1);
    }
    free(program_buffer);

    /* Build program 

   The fourth parameter accepts options that configure the compilation. 
   These are similar to the flags used by gcc. For example, you can 
   define a macro with the option -DMACRO=VALUE and turn off optimization 
   with -cl-opt-disable.
   */
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err < 0)
    {

        /* Find size of log and print to std output */
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
                              0, NULL, &log_size);
        program_log = (char *)malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
                              log_size + 1, program_log, NULL);
        printf("%s\n", program_log);
        free(program_log);
        exit(1);
    }

    return program;
}

cl_device_id create_device() {

   cl_platform_id platform;
   cl_device_id dev;
   int err;

   /* Identify a platform */
   err = clGetPlatformIDs(1, &platform, NULL);
   if(err < 0) {
      perror("Couldn't identify a platform");
      exit(1);
   } 

   // Access a device
   // GPU
   err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
   if(err == CL_DEVICE_NOT_FOUND) {
      // CPU
      printf("GPU not found\n");
      err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
   }
   if(err < 0) {
      perror("Couldn't access any devices");
      exit(1);   
   }

   return dev;
}