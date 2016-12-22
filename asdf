#include<stdio.h>
#include<stdlib.h>
#include <math.h>
#include <time.h>
#include <Windows.h>

#include "cuda_multMatVec.cuh"

typedef float TIMER_T;

#define USE_CPU_TIMER 1
#define USE_GPU_TIMER 1


#if USE_CPU_TIMER == 1
__int64 start, freq, end;
#define CHECK_TIME_START() { QueryPerformanceFrequency((LARGE_INTEGER*)&freq); QueryPerformanceCounter((LARGE_INTEGER*)&start); }
#define CHECK_TIME_END(a) { QueryPerformanceCounter((LARGE_INTEGER*)&end); a = (float)((float)(end - start) / (freq / 1000.0f)); }
#else
#define CHECK_TIME_START()
#define CHECK_TIME_END(a)
#endif


#if USE_GPU_TIMER == 1
cudaEvent_t cuda_timer_start, cuda_timer_stop;
#define CUDA_STREAM_0 (0)

void create_device_timer()
{
	CUDA_CALL( cudaEventCreate( &cuda_timer_start ) );
	CUDA_CALL( cudaEventCreate( &cuda_timer_stop ) );
}

void destroy_device_timer()
{
	CUDA_CALL( cudaEventDestroy( cuda_timer_start ) );
	CUDA_CALL( cudaEventDestroy( cuda_timer_stop ) );
}

inline void start_device_timer()
{
	cudaEventRecord( cuda_timer_start, CUDA_STREAM_0 );
}

inline TIMER_T stop_device_timer()
{
	TIMER_T ms;
	cudaEventRecord( cuda_timer_stop, CUDA_STREAM_0 );
	cudaEventSynchronize( cuda_timer_stop );

	cudaEventElapsedTime( &ms, cuda_timer_start, cuda_timer_stop );
	return ms;
}

#define CHECK_TIME_INIT_GPU() { create_device_timer(); }
#define CHECK_TIME_START_GPU() { start_device_timer(); }
#define CHECK_TIME_END_GPU(a) { a = stop_device_timer(); }
#define CHECK_TIME_DEST_GPU() { destroy_device_timer(); }
#else
#define CHECK_TIME_INIT_GPU()
#define CHECK_TIME_START_GPU()
#define CHECK_TIME_END_GPU(a)
#define CHECK_TIME_DEST_GPU()
#endif

__host__ void cuda_error_check( const char * prefix, const char * postfix )
{
	if( cudaPeekAtLastError() != cudaSuccess )
	{
		printf( "%s%s%s", prefix, cudaGetErrorString( cudaGetLastError() ), postfix );
		cudaDeviceReset();
		//wait_exit();
		exit( 1 );
	}
}

void MultMatVec_CPU_WithPrefetch( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX, IN int n );
void MultMatVec_CPU_( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX, IN int n );
void MultMatVec_CPU_AndPrefetch( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX, IN int n );
void MultMatVec_CPU_AndUnrolling2( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX, IN int n );
void MultMatVec_CPU_AndUnrolling4( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX, IN int n );
void MultMatVec_CPU_AndUnrolling4_DoubleUnrolling2( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX, IN int n );
void MultMatVec_CPU_AndUnrolling4_DoubleUnrolling4( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX, IN int n );
void MultMatVec_CPU_PrefetchAndUnrolling4_DoubleUnrolling4( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX, IN int n );
void MultMatVec_CPU_AndUnrolling4_DoubleUnrolling8( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX, IN int n );
void MultMatVec_CPU_AndUnrolling8_DoubleUnrolling4( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX, IN int n );
void MultMatVec_CPU_AndUnrolling16( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX, IN int n );
void MultMatVec_CPU_AndUnrolling32( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX, IN int n );

inline float absIEEE754( float f)
{
	return ( float& )( ( int& )f &= 0x7fffffff );
}

float GetErrorRate( IN float* vecYcpuResult, IN float* vecYgpuResult, IN int numOfVectorElems )
{
	int cnt = 0;
	float epsilon = 0.000005f;
	for( int i = 0; i < numOfVectorElems; ++i )
	{
		if( absIEEE754( vecYcpuResult[ i ] - vecYgpuResult[ i ] ) > epsilon )
		{
			cnt++;
			//printf( "[%d][%d]: %f != %f\n", i / ELEM_PER_VECTOR, i % ELEM_PER_VECTOR, vecYcpuResult[ i ], vecYgpuResult[ i ] );
		}
	}

	//printf( " - Num of total elements: %d\n", numOfVectorElems );
	//printf( " - Num of error counts: %d\n", cnt );
	return float( cnt ) / numOfVectorElems * 100.f;
}

int main()
{
	float cpuTime, totalCPUtime;
	float gpuTime, totalGPUtime;

	float *vecX, *vecYcpuResult, *vecYgpuResult, ( *matA )[ ELEM_PER_VECTOR ];
	float *vecXcpu, *vecYcpu, ( *matAcpu )[ ELEM_PER_VECTOR ];
	float *vecXgpu, *vecYgpu, ( *matAgpu )[ ELEM_PER_VECTOR ];

	CHECK_TIME_INIT_GPU();

	FILE* fp = fopen( "gen.bin", "rb" );

	int numOfVectors, numOfVectorElems, numOfMatrixElems;
	fread( &numOfVectors, sizeof( int ), 1, fp );

	numOfVectorElems = numOfVectors * ELEM_PER_VECTOR;
	numOfMatrixElems = ELEM_PER_VECTOR * ELEM_PER_VECTOR;

	vecYcpuResult = new float[ numOfVectorElems ]();
	vecYgpuResult = new float[ numOfVectorElems ]();
	vecX = new float[ numOfVectorElems ]();
	matA = new float[ ELEM_PER_VECTOR ][ ELEM_PER_VECTOR ]();

	fread( vecX, sizeof( float ), numOfVectorElems, fp );
	fread( matA, sizeof( float ), numOfMatrixElems, fp );

#define CPU_FUNC_CALL(funcname, probname) \
	totalCPUtime = 0; \
	vecYcpu = new float[ numOfVectorElems ](); \
	vecXcpu = new float[ numOfVectorElems ]();						memcpy( vecXcpu, vecX, sizeof( float ) * numOfVectorElems ); \
	matAcpu = new float[ ELEM_PER_VECTOR ][ ELEM_PER_VECTOR ]();	memcpy( matAcpu, matA, sizeof( float ) * numOfMatrixElems ); \
	for( int i = 0; i < REPEAT_COUNT; ++i ) \
	{ \
		CHECK_TIME_START(); \
		funcname( vecYcpu, matAcpu, vecXcpu, numOfVectors ); \
		CHECK_TIME_END( cpuTime ); \
		totalCPUtime += cpuTime; \
	} \
	printf( "Elapsed Time by " #probname " is %f (s).\n\n", (totalCPUtime / REPEAT_COUNT)/1000.0 ); \
	memcpy( vecYcpuResult, vecYcpu, sizeof( float ) * numOfVectorElems ); \
	delete[] vecYcpu; \
	delete[] vecXcpu; \
	delete[] matAcpu;
#define CPU_FUNC_CALL__MACRO_END
	CPU_FUNC_CALL( MultMatVec_CPU_				, CPU );
	//CPU_FUNC_CALL( MultMatVec_CPU_AndUnrolling4 , CPU_Unrolling );

	size_t numThreads = ( 1 << 10 );
	size_t numBlocks = numOfVectors / numThreads;
	
	size_t _32Threads = ( 1 << 5 );
	size_t _32Blocks_perElement = ( numOfVectors * ELEM_PER_VECTOR ) / _32Threads;
	size_t _32Blocks_perVector = numOfVectors / _32Threads;
	size_t _32Blocks_perVector32threads = _32Blocks_perElement;

	size_t _64Threads = ( 1 << 6 );
	size_t _64Blocks_perElement = ( numOfVectors * ELEM_PER_VECTOR ) / _64Threads;
	size_t _64Blocks_perVector = numOfVectors / _64Threads;
	size_t _64Blocks_perVector32threads = _64Blocks_perElement;

	size_t _128Threads = ( 1 << 7 );
	size_t _128Blocks_perElement = ( numOfVectors * ELEM_PER_VECTOR ) / _128Threads;
	size_t _128Blocks_perVector = numOfVectors / _128Threads;
	size_t _128Blocks_perVector32threads = _128Blocks_perElement;

	size_t _256Threads = ( 1 << 8 );
	size_t _256Blocks_perElement = ( numOfVectors * ELEM_PER_VECTOR ) / _256Threads;
	size_t _256Blocks_perVector = numOfVectors / _256Threads;
	size_t _256Blocks_perVector32threads = _256Blocks_perElement;

	size_t _512Threads = ( 1 << 9 );
	size_t _512Blocks_perElement = ( numOfVectors * ELEM_PER_VECTOR ) / _512Threads;
	size_t _512Blocks_perVector = numOfVectors / _512Threads;
	size_t _512Blocks_perVector32threads = _512Blocks_perElement;

	size_t _1024Threads = ( 1 << 10 );
	size_t _1024Blocks_perElement = ( numOfVectors * ELEM_PER_VECTOR ) / _1024Threads;
	size_t _1024Blocks_perVector = numOfVectors / _1024Threads;
	size_t _1024Blocks_perVector32threads = _1024Blocks_perElement;

	GenerateConstantMatrix( matA );

	CUDA_CALL( cudaMalloc( &vecXgpu, numOfVectorElems * sizeof( float ) ) );
	CUDA_CALL( cudaMalloc( &vecYgpu, numOfVectorElems * sizeof( float ) ) );
	CUDA_CALL( cudaMalloc( &matAgpu, numOfMatrixElems * sizeof( float ) ) );
	CUDA_CALL( cudaMemcpy( vecXgpu, vecX, numOfVectorElems * sizeof( float ), cudaMemcpyHostToDevice ) );
	CUDA_CALL( cudaMemcpy( matAgpu, matA, numOfMatrixElems * sizeof( float ), cudaMemcpyHostToDevice ) );

#define GPU_FUNC_CALL_SHARED(funcname, num_block, num_thread, size_shared, probname) \
	totalGPUtime = 0; \
	CUDA_CALL( cudaMemset( vecYgpu, 0x00, numOfVectorElems * sizeof( float ) ) ); \
	for( int i = 0; i < REPEAT_COUNT; ++i ) \
	{ \
		CHECK_TIME_START_GPU(); \
		funcname <<< num_block, num_thread, size_shared >>> ( vecYgpu, matAgpu, vecXgpu ); \
		cuda_error_check( "ERROR: ", " when " #funcname "() was launched.\n" ); \
		CHECK_TIME_END_GPU( gpuTime ); \
		totalGPUtime += gpuTime; \
	} \
	CUDA_CALL( cudaMemcpy( vecYgpuResult, vecYgpu, numOfVectorElems * sizeof( float ), cudaMemcpyDeviceToHost ) ); \
	CUDA_CALL( cudaDeviceSynchronize() ); \
	printf( "Elapsed Time by " #probname " is %f (s). Error rate is %.2f%%\n\n", (totalGPUtime / REPEAT_COUNT)/1000.0, GetErrorRate( vecYcpuResult, vecYgpuResult, numOfVectorElems ) );

#define GPU_FUNC_CALL_SHARED__MACRO_END

#define GPU_FUNC_CALL(funcname, num_block, num_thread, probname) GPU_FUNC_CALL_SHARED(funcname, num_block, num_thread, 0, probname)
	//GPU_FUNC_CALL(			MultMatVec_GPU_GlobalMemoryWithoutRegister_Vector,									_1024Blocks_perVector,			_1024Threads,	GPU1   ); // GPU1
	GPU_FUNC_CALL(			MultMatVec_GPU_GlobalMemory_Vector,													_1024Blocks_perVector,			_1024Threads,	GPU1 ); // GPU1_2
	GPU_FUNC_CALL(			MultMatVec_GPU_SharedMemoryConstantMatrix_Element1024ThreadsPerBlock,				_1024Blocks_perElement,			_1024Threads,	GPU2   ); // GPU2
	GPU_FUNC_CALL(			MultMatVec_GPU_SharedMemoryConstantMatrix_Element1024ThreadsPerBlock_SOA,			_1024Blocks_perElement,			_1024Threads,	GPU3   ); // GPU3 SOA Elem version
	//GPU_FUNC_CALL(			MultMatVec_GPU_GlobalMemory_Vector_SOA,												_1024Blocks_perVector,			_1024Threads,	GPU3   ); // GPU3 SOA Vec versrion
	GPU_FUNC_CALL(			MultMatVec_GPU_Strided32VectorSharedMemoryConstantMatrix_Element1024ThreadsPerBlock,_1024Blocks_perElement,			_1024Threads,	GPU4   ); // GPU4

	CHECK_TIME_DEST_GPU();

	CUDA_CALL( cudaFree( vecXgpu ) );
	CUDA_CALL( cudaFree( vecYgpu ) );
	CUDA_CALL( cudaFree( matAgpu ) );

	int cnt = 0;
	float epsilon = 0.000005f;
	for( int i = 0; i < numOfVectorElems; ++i ){
		if( absIEEE754( vecYcpuResult[ i ] - vecYgpuResult[ i ] ) > epsilon ){
			cnt++;
		}
	}

	delete[] vecX;
	delete[] vecYcpuResult;
	delete[] vecYgpuResult;
	delete[] matA;
}


void MultMatVec_CPU_( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX, IN int n )
{
	for( int i = 0; i < n; ++i )
	{
		for( int j = 0; j < ELEM_PER_VECTOR; ++j )
		{
			float result = 0.0f;
			for( int k = 0; k < ELEM_PER_VECTOR; ++k )
			{
				result += matA[ j ][ k ] * vecX[ i * ELEM_PER_VECTOR + k ];
			}
			vecY[ i * ELEM_PER_VECTOR + j ] = result;
		}
	}
}

void MultMatVec_CPU_AndUnrolling4( OUT float* vecY, IN float( *matA )[ ELEM_PER_VECTOR ], IN float* vecX, IN int n )
{
	int HALF_ELEM_PER_VECTOR = ELEM_PER_VECTOR / 2;
	for( int i = 0; i < n; ++i )
	{
		// cache block size					= 64 bytes
		// ELEM_PER_VECTOR * sizeof(float)	= 128 bytes
		// 2 cache block needed to prefetching
		vecY[ i * ELEM_PER_VECTOR ] = vecY[ i * ELEM_PER_VECTOR + HALF_ELEM_PER_VECTOR ] = 0;
		for( int j = 0; j < ELEM_PER_VECTOR; ++j )
		{
			float result = 0.0f;
			for( int k = 0; k < ELEM_PER_VECTOR; k += 4 )
			{
				result +=
					+ matA[ j ][ k + 0 ] * vecX[ i * ELEM_PER_VECTOR + k + 0 ]
					+ matA[ j ][ k + 1 ] * vecX[ i * ELEM_PER_VECTOR + k + 1 ]
					+ matA[ j ][ k + 2 ] * vecX[ i * ELEM_PER_VECTOR + k + 2 ]
					+ matA[ j ][ k + 3 ] * vecX[ i * ELEM_PER_VECTOR + k + 3 ];
			}
			vecY[ i * ELEM_PER_VECTOR + j ] = result;
		}
	}
}
