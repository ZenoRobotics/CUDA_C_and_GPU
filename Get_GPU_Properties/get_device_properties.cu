//************************************************
//  Name:        get_device_properties.c
//  Copyright:   ZenoMachines, LLC
//  Author:      Peter J. Zeno
//  Date:        06/08/11 
//  Description: Gets/Displays all GPUs on the computer
//               along with their properties.
//               
//************************************************
//  USAGE:
//    ./get_device_properties   
//

#include "./common/book.h"

//output files 
//const char *OUTFILE       = "./common/gpu_arch_constants.h";
//const char *MAKEFILE      = "Makefile_gmatch_cu";

//file handles
//FILE *fout; 
//FILE *fmake;

int main( void ) {

    cudaDeviceProp prop;
    int count;
    int i=0;
/*
    //open output files
    if ((fout = fopen(OUTFILE,"w+")) == NULL)  
       printf("Cannot open %s for writing",OUTFILE); 
    if ((fmake = fopen(MAKEFILE,"w+")) == NULL)  
       printf("Cannot open %s for writing",MAKEFILE); 
*/
    HANDLE_ERROR( cudaGetDeviceCount( &count ) ) ;

    for (i=0; i< count; i++) {
        HANDLE_ERROR( cudaGetDeviceProperties( &prop, i ) ) ;
        
        printf( "\n\n" ) ;
        printf( "--- General Information for device %d ---\n\n" , i ) ;
        printf( "Name:                %s\n", prop.name ) ;
        printf( "Compute capability:  %d.%d\n" , prop.major, prop.minor ) ;  
        printf( "Device Clock rate:   %d MHz\n" , prop.clockRate/1000 ) ;       //comes in kilo-hertz
        printf( "Memory Clock rate:   %d MHz\n" , prop.memoryClockRate/1000 ) ; //comes in kilo_hertz
        printf( "Device copy overlap: " ) ;
        if ( prop.deviceOverlap)
            printf( "Enabled\n" ) ;
        else
            printf( "Disabled\n" ) ;
        printf( "Kernel execution timeout:  " ) ;
        if ( prop.kernelExecTimeoutEnabled)
            printf( "Enabled\n" ) ;
        else
            printf( "Disabled\n" ) ;
        printf( "\n" ) ;
        printf( "--- Memory Information for device %d ---\n\n" , i ) ;
        printf( "Total global mem:    %4.1f MBs\n" , (double) prop.totalGlobalMem/(1024 * 1024) ) ;
        printf( "Total constant Mem:  %lu KBs\n" , (long unsigned int) prop.totalConstMem/1024 ) ;
        printf( "Max mem pitch:       %lu MBs\n", (long unsigned int) prop.memPitch/(1024 * 1024) ) ;
        printf( "Texture Alignment:   %lu\n" , (long unsigned int) prop.textureAlignment ) ;
        printf( " \n" ) ;
        printf( "--- MP Information for device %d --- \n\n", i ) ;
        printf( "Multiprocessor count:   %d\n" , prop.multiProcessorCount ) ;  
        printf( "Shared mem per block:   %lu KBs\n", (long unsigned int) prop.sharedMemPerBlock/1024 ) ;
        printf( "Registers per block:    %d K\n", prop.regsPerBlock/1024 ) ;
        printf( "Threads in warp:        %d\n", prop.warpSize ) ;
        printf( "Max threads per block:  %d\n" , prop.maxThreadsPerBlock ) ; 
        printf( "Max threads per MP:     %d\n" , prop.maxThreadsPerMultiProcessor);  
        printf( "Max thread dimensions:  (%d, %d, %d) \n" ,
                    prop.maxThreadsDim[0] , prop.maxThreadsDim[1] ,
                    prop.maxThreadsDim[2] ) ;
        printf( "Max grid dimensions:    (%d, %d, %d) \n" ,
                    prop.maxGridSize[0] , prop.maxGridSize[1] ,
                    prop.maxGridSize[2] ) ;
        printf( " \n" ) ;
    }
/*
    //Create Makefile for gmatch
    fprintf(fmake,"# Makefile for gmatch.cu program. \n");
    fprintf(fmake,"# Created by get_device_properties program. \n\n\n");
    fprintf(fmake,"gmatch : gmatch.cu \n");
    fprintf(fmake,"	nvcc  -I. -I/usr/local/cuda/include/ -I/usr/local/cuda/include/crt/ -L/usr/local/cuda/lib64/ -lcuda --ptxas-options=-v -arch=sm_%d%d  gmatch.cu -o gmatch \n\n\n", prop.major, prop.minor);
    fprintf(fmake,"clean: \n");
    fprintf(fmake,"	rm -f *.o *~ core .depend gmatch \n\n\n");
    fprintf(fmake,"depend .depend dep: \n");
    fprintf(fmake,"	$(CC) $(CFLAGS) -M *.c > $@ \n\n\n");
    
    //Create GPU Specific Constants header file
    fprintf(fout,"//GPU Specific Constants Header File. \n");
    fprintf(fout,"// \n");
    fprintf(fout,"//Created by get_device_properties program. \n\n\n");
    fprintf(fout,"const int blocksPerGrid                 = %d;\n" , prop.multiProcessorCount);
    fprintf(fout,"const int max_num_of_threads_per_block  = %d;\n" , prop.maxThreadsPerBlock);
    fprintf(fout,"\n\n");


    //close output files
    fclose(fout);
    fclose(fmake);
*/
}
