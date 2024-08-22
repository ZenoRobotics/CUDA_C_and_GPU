/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication as described in Chapter 3
 * of the programming guide.
 * It has been written for clarity of exposition to illustrate various CUDA
 * programming principles, not with the goal of providing the most
 * performant generic kernel for matrix multiplication.
 *
 * See also:
 * V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
 * in Proc. 2008 ACM/IEEE Conf. on Superconducting (SC '08),
 * Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
 */

//#define CUBLAS 1

// System includes
#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <helper_string.h>  // helper for shared functions common to CUDA SDK samples
#include <cuda_profiler_api.h>

// CUDA runtime
#include <cuda_runtime.h>
#ifdef CUBLAS
   #include <cublas_v2.h>
#endif

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>

#ifndef min
#define min(a,b) ((a < b) ? a : b)
#endif
#ifndef max
#define max(a,b) ((a > b) ? a : b)
#endif


typedef struct _matrixSize      // Optional Command-line multiplier for matrix sizes
{
    unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;
} sMatrixSize;

const int BLOCK_SIZE = 32;

#ifndef CUBLAS
/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
__global__ void
matrixMulCUDA(float *C, float *A, float *B, unsigned int wA, unsigned int wB)
{   
    // Block dimensions
    int bDimx = blockDim.x;
    int bDimy = blockDim.y;

    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * bDimy * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = bDimx; //BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = bx * bDimx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = bDimx * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    float Csub = 0;

    // Declaration of the shared memory array As used to
    // store the sub-matrix of A
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

    // Declaration of the shared memory array Bs used to
    // store the sub-matrix of B
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
         a <= aEnd;
         a += aStep, b += bStep)
    {

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];
        
        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll 

        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * bDimy * by + bDimx * bx;
    C[c + wB * ty + tx] = Csub;
    
}
#endif

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set matrix multiply on CPU
//! C = A * B
//! @param C          reference data, computed but preallocated
//! @param A          matrix A as provided to device
//! @param B          matrix B as provided to device
//! @param hA         height of matrix A
//! @param wB         width of matrix B
////////////////////////////////////////////////////////////////////////////////
void
matrixMulCPU(float *C, const float *A, const float *B, unsigned int hA, unsigned int wA, unsigned int wB)
{
    for (unsigned int i = 0; i < hA; ++i)
        for (unsigned int j = 0; j < wB; ++j)
        {
            float sum = 0;

            for (unsigned int k = 0; k < wA; ++k)
            {
                float a = A[i * wA + k];
                float b = B[k * wB + j];
                sum += a * b;
            }

            C[i * wB + j] = (float)sum;
        }
}

// Allocates a matrix with random float entries.
void randomInit(float *data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

void printDiff(float *data1, float *data2, int width, int height, int iListLength, float fListTol)
{
    printf("Listing first %d Differences > %.6f...\n", iListLength, fListTol);
    int i,j,k;
    int error_count=0;

    for (j = 0; j < height; j++)
    {
        if (error_count < iListLength)
        {
            printf("\n  Row %d:\n", j);
        }

        for (i = 0; i < width; i++)
        {
            k = j * width + i;
            float fDiff = fabs(data1[k] - data2[k]);

            if (fDiff > fListTol)
            {
                if (error_count < iListLength)
                {
                    printf("    Loc(%d,%d)\tCPU=%.5f\tGPU=%.5f\tDiff=%.6f\n", i, j, data1[k], data2[k], fDiff);
                }

                error_count++;
            }
        }
    }

    printf(" \n  Total Errors = %d\n", error_count);
}

void initializeCUDA(int argc, char **argv, int &devID, int &iSizeMultiple, sMatrixSize &matrix_size)
{
    // By default, we use device 0, otherwise we override the device ID based on what is provided at the command line
    cudaError_t error;
    devID = 0;

    if (checkCmdLineFlag(argc, (const char **)argv, "device"))
    {
        devID = getCmdLineArgumentInt(argc, (const char **)argv, "device");
        error = cudaSetDevice(devID);

        if (error != cudaSuccess)
        {
            printf("cudaSetDevice returned error code %d, line(%d)\n", error, __LINE__);
            exit(EXIT_FAILURE);
        }
    }

    // get number of SMs on this GPU
    error = cudaGetDevice(&devID);

    if (error != cudaSuccess)
    {
        printf("cudaGetDevice returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }


    if (checkCmdLineFlag(argc, (const char **)argv, "sizemult"))
    {
        iSizeMultiple = getCmdLineArgumentInt(argc, (const char **)argv, "sizemult");
    }

    iSizeMultiple = min(iSizeMultiple, 10);
    iSizeMultiple = max(iSizeMultiple, 1);

    cudaDeviceProp deviceProp;

    error = cudaGetDeviceProperties(&deviceProp, devID);

    if (error != cudaSuccess)
    {
        printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);

    // use a larger block size for Fermi and above
    //int block_size = (deviceProp.major < 2) ? 16 : 32;

    matrix_size.uiWA = 1024; //4 * block_size * iSizeMultiple;
    matrix_size.uiHA = 1024; //4 * block_size * iSizeMultiple;
    matrix_size.uiWB = 1024; //4 * block_size * iSizeMultiple;
    matrix_size.uiHB = 1024; //4 * block_size * iSizeMultiple;
    matrix_size.uiWC = matrix_size.uiWB;
    matrix_size.uiHC = matrix_size.uiHA;

    printf("MatrixA(%u,%u), MatrixB(%u,%u), MatrixC(%u,%u)\n",
           matrix_size.uiWA, matrix_size.uiHA,
           matrix_size.uiWB, matrix_size.uiHB,
           matrix_size.uiWC, matrix_size.uiHC);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test matrix multiply using CUBLAS
////////////////////////////////////////////////////////////////////////////////
int matrixMultiply(int argc, char **argv, int devID, sMatrixSize &matrix_size)
{
    cudaDeviceProp deviceProp;
    cudaError_t error;

    // execute the kernel
    int nIter = 30;

    error = cudaGetDeviceProperties(&deviceProp, devID);

    if (error != cudaSuccess)
    {
        printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    // use a larger block size for Fermi and above
    //int block_size = 32; //(deviceProp.major < 2) ? 16 : 32;

    // set seed for rand()
    srand(2006);

    // allocate host memory for matrices A and B
    unsigned int size_A = matrix_size.uiWA * matrix_size.uiHA;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float *h_A = (float *)malloc(mem_size_A);
    unsigned int size_B = matrix_size.uiWB * matrix_size.uiHB;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float *h_B = (float *)malloc(mem_size_B);

    // set seed for rand()
    srand(2006);

    // initialize host memory
    randomInit(h_A, size_A);
    randomInit(h_B, size_B);

    // allocate device memory
    float *d_A, *d_B, *d_C;
    unsigned int size_C = matrix_size.uiWC * matrix_size.uiHC;
    unsigned int mem_size_C = sizeof(float) * size_C;

    // allocate host memory for the result
    float *h_C      = (float *) malloc(mem_size_C);
    float *h_CUBLAS = (float *) malloc(mem_size_C);

    error = cudaMalloc((void **) &d_A, mem_size_A);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_A returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc((void **) &d_B, mem_size_B);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_B returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    // copy host memory to device
    error = cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy d_A h_A returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);

    if (error != cudaSuccess)
    {
        printf("cudaMemcpy d_B h_B returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc((void **) &d_C, mem_size_C);

    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_C returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    // setup execution parameters
    //dim3 threads(block_size, block_size);
    //dim3 grid(matrix_size.uiWC / threads.x, matrix_size.uiHC / threads.y);

    // Setup execution parameters
    int  wBlock = BLOCK_SIZE;  // Value may change based on dimsB.x (wB)
    int  hBlock = BLOCK_SIZE;  // Value may change based on wBlock and dimsA.x (wA) 
    //bool Done   = false;
/*
    if ((matrix_size.uiWB > 1) && ((matrix_size.uiWB%2) == 0)) 
    {
       while (!Done)
       {
          if ((matrix_size.uiWB%wBlock) > 0)
             wBlock = wBlock/2;
          else
             Done = true;
       }
       hBlock = 1024/wBlock;
    }
    else if (matrix_size.uiWB == 1)
    {
       wBlock = 1;
       hBlock = matrix_size.uiHB;  // Good for up to 1024
    }
    else
    {
       printf("Width of B Matrix is not a power of 2.\n");
       exit(EXIT_FAILURE);
    }
*/
    dim3 threads(wBlock, hBlock);                                             // (x = cols, y = rows)
    dim3 grid(matrix_size.uiWB / threads.x, matrix_size.uiHA / threads.y);    // (x = cols, y = rows)
    
    // Debug
    printf("\n");
    printf("hBlock = %d, wBlock = %d, hGrid = %d, wGrid = %d\n\n", hBlock, wBlock, grid.y, grid.x);

    #ifdef CUBLAS
       // create and start timer
       printf("Computing result using CUBLAS...");
    #else
       printf("Computing result using Homebrew Kernel...");
    #endif


    // Allocate CUDA events that we'll use for timing
    cudaEvent_t start;
    error = cudaEventCreate(&start);

    if (error != cudaSuccess)
    {
         fprintf(stderr, "Failed to create start event (error code %s)!\n", cudaGetErrorString(error));
         exit(EXIT_FAILURE);
    }

    cudaEvent_t stop;
    error = cudaEventCreate(&stop);

    if (error != cudaSuccess)
    {
         fprintf(stderr, "Failed to create stop event (error code %s)!\n", cudaGetErrorString(error));
         exit(EXIT_FAILURE);
    }

    // Record the start event
    error = cudaEventRecord(start, NULL);

    if (error != cudaSuccess)
    {
         fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
         exit(EXIT_FAILURE);
    }

    #ifdef CUBLAS
        cublasHandle_t handle;

        cublasStatus_t ret;

        ret = cublasCreate(&handle);

        if (ret != CUBLAS_STATUS_SUCCESS)
        {
            printf("cublasCreate returned error code %d, line(%d)\n", ret, __LINE__);
            exit(EXIT_FAILURE);
        }

        const float alpha = 1.0f;
        const float beta  = 0.0f;

    #endif
cudaProfilerStart();

        for (int j = 0; j < nIter; j++)
        {
            //note cublas is column primary!
            //need to transpose the order

            #ifdef CUBLAS
            ret = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrix_size.uiWB, matrix_size.uiHA, matrix_size.uiWA, 
                             &alpha, d_B, matrix_size.uiWB, d_A, matrix_size.uiWA, &beta, d_C, matrix_size.uiWA);
            
               if (ret != CUBLAS_STATUS_SUCCESS)
               {
                  printf("cublasSgemm returned error code %d, line(%d)\n", ret, __LINE__);
                  exit(EXIT_FAILURE);
               }
            
            #else
               matrixMulCUDA<<< grid, threads >>>(d_C, d_A, d_B, matrix_size.uiWA, matrix_size.uiWB);
            #endif
            
        }
cudaProfilerStop();

     printf("done.\n");

     // Record the stop event
     error = cudaEventRecord(stop, NULL);

     if (error != cudaSuccess)
     {
         fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
         exit(EXIT_FAILURE);
     }

     // Wait for the stop event to complete
     error = cudaEventSynchronize(stop);

     if (error != cudaSuccess)
     {
         fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
         exit(EXIT_FAILURE);
     }

     float msecTotal = 0.0f;
     error = cudaEventElapsedTime(&msecTotal, start, stop);

     if (error != cudaSuccess)
     {
         fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));
         exit(EXIT_FAILURE);
     }

     // Compute and print the performance
     float msecPerMatrixMul = msecTotal / nIter;
     double flopsPerMatrixMul = ((double)matrix_size.uiWA + ((double)matrix_size.uiWA - 1)) * (double)matrix_size.uiHA * (double)matrix_size.uiWB;
     double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
     printf("\n");
     printf(
         "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops\n\n",
         gigaFlops,
         msecPerMatrixMul,
         flopsPerMatrixMul);

     // copy result from device to host
     error = cudaMemcpy(h_CUBLAS, d_C, mem_size_C, cudaMemcpyDeviceToHost);

     if (error != cudaSuccess)
     {
         printf("cudaMemcpy h_CUBLAS d_C returned error code %d, line(%d)\n", error, __LINE__);
         exit(EXIT_FAILURE);
     }


    // compute reference solution
    printf("Computing result using host CPU...");
    float *reference = (float *)malloc(mem_size_C);
    clock_t tStart = clock();
    matrixMulCPU(reference, h_A, h_B, matrix_size.uiHA, matrix_size.uiWA, matrix_size.uiWB);
    clock_t tStop = clock();
    printf("done.\n");
    printf("Time taken for CPU version: %.2fs\n\n", (double)(tStop - tStart)/CLOCKS_PER_SEC);

    // check result (CUBLAS)
    bool resCUBLAS = sdkCompareL2fe(reference, h_CUBLAS, size_C, 1.0e-6f);

    if (resCUBLAS != true)
    {
        printDiff(reference, h_CUBLAS, matrix_size.uiWC, matrix_size.uiHC, 100, 1.0e-5f);
    }

    printf("Comparing CUBLAS Matrix Multiply with CPU results: %s\n", (true == resCUBLAS) ? "OK" : "FAIL");

    // clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(reference);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaDeviceReset();

    if (resCUBLAS == true)
    {
        return EXIT_SUCCESS;    // return value = 1
    }
    else
    {
        return EXIT_FAILURE;     // return value = 0
    }
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    printf("[Matrix Multiply] - Starting...\n");

    int devID = 0, sizeMult = 5;
    sMatrixSize matrix_size;

    initializeCUDA(argc, argv, devID, sizeMult, matrix_size);

    int matrix_result = matrixMultiply(argc, argv, devID, matrix_size);

    exit(matrix_result);
}


