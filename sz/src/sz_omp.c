/**
 *  @file sz_omp.c
 *  @author Xin Liang
 *  @date July, 2017
 *  @brief the implementation of openMP version
 *  (C) 2016 by Mathematics and Computer Science (MCS), Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include "sz_omp.h"
#include <math.h>
#include <time.h>

double sz_wtime(){
#ifdef _OPENMP
    return omp_get_wtime();
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);

    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
#endif
}

int sz_get_max_threads(){
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

int sz_get_thread_num(){
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

void sz_set_num_threads(int nthreads){
#ifdef _OPENMP
    omp_set_num_threads(nthreads);
#endif
}

unsigned char * SZ_compress_float_1D_MDQ_openmp(float *oriData, size_t r1, double realPrecision, size_t * comp_size){
	return NULL;
}
unsigned char * SZ_compress_float_2D_MDQ_openmp(float *oriData, size_t r1, size_t r2, double realPrecision, size_t * comp_size){
	return NULL;
}

unsigned char * SZ_compress_float_3D_MDQ_openmp(float *oriData, size_t r1, size_t r2, size_t r3, float realPrecision, size_t * comp_size, CPU_timing *cpu_timing){
	//jwang
	double elapsed_time = 0.0;

	unsigned int quantization_intervals;
	if(exe_params->optQuantMode==1)
	{
		quantization_intervals = optimize_intervals_float_3D_opt(oriData, r1, r2, r3, realPrecision);
		printf("3D number of bins: %d\nerror bound %.20f\n", quantization_intervals, realPrecision);
		updateQuantizationInfo(quantization_intervals);
	}	
	else{
		quantization_intervals = exe_params->intvCapacity;
	}

	int thread_num = sz_get_max_threads();
	//printf("thread_num=%d\n", thread_num);
	int thread_order = (int)log2(thread_num);
	size_t num_x = 0, num_y = 0, num_z = 0;
	{
		int block_thread_order = thread_order / 3;
		switch(thread_order % 3){
			case 0:{
				num_x = 1 << block_thread_order;
				num_y = 1 << block_thread_order;
				num_z = 1 << block_thread_order;
				break;
			}
			case 1:{
				num_x = 1 << (block_thread_order + 1);
				num_y = 1 << block_thread_order;
				num_z = 1 << block_thread_order;
				break;
			}
			case 2:{
				num_x = 1 << (block_thread_order + 1);
				num_y = 1 << (block_thread_order + 1);
				num_z = 1 << block_thread_order;
				break;
			}
		}
		thread_num = num_x * num_y * num_z;
	}
	//printf("thread_num=%d\n", thread_num);
	sz_set_num_threads(thread_num);
	// calculate block dims
	size_t split_index_x, split_index_y, split_index_z;
	size_t early_blockcount_x, early_blockcount_y, early_blockcount_z;
	size_t late_blockcount_x, late_blockcount_y, late_blockcount_z;
	SZ_COMPUTE_BLOCKCOUNT(r1, num_x, split_index_x, early_blockcount_x, late_blockcount_x);
	SZ_COMPUTE_BLOCKCOUNT(r2, num_y, split_index_y, early_blockcount_y, late_blockcount_y);
	SZ_COMPUTE_BLOCKCOUNT(r3, num_z, split_index_z, early_blockcount_z, late_blockcount_z);

	size_t max_num_block_elements = early_blockcount_x * early_blockcount_y * early_blockcount_z;
	size_t num_blocks = num_x * num_y * num_z;
	size_t num_elements = r1 * r2 * r3;
	size_t dim0_offset = r2 * r3;
	size_t dim1_offset = r3;

	size_t buffer_size = early_blockcount_y * early_blockcount_z * sizeof(float);
	int * result_type = (int *) malloc(num_elements * sizeof(int));

	size_t unpred_data_max_size = max_num_block_elements;
	float * result_unpredictable_data = (float *) malloc(unpred_data_max_size * sizeof(float) * num_blocks);
	unsigned int * unpredictable_count = (unsigned int *) malloc(num_blocks * sizeof(unsigned int));
	
	float * mean = malloc(num_blocks * sizeof(float));
	
	float * buffer0, * buffer1;
	buffer0 = (float *) malloc(buffer_size * thread_num);
	buffer1 = (float *) malloc(buffer_size * thread_num);
	
	unsigned char * result = (unsigned char *) malloc(num_elements * (sizeof(int) + sizeof(float)));
	size_t * unpred_offset = (size_t *) malloc(num_blocks * sizeof(size_t));
	unsigned char * encoding_buffer = (unsigned char *) malloc(max_num_block_elements * sizeof(int) * num_blocks);
	size_t * block_offset = (size_t *) malloc(num_blocks * sizeof(size_t));
	size_t *freq = (size_t *)malloc(thread_num*quantization_intervals*4*sizeof(size_t));
	memset(freq, 0, thread_num*quantization_intervals*4*sizeof(size_t));
	
	size_t stateNum = quantization_intervals*2;
	HuffmanTree* huffmanTree = createHuffmanTree(stateNum);	

	elapsed_time = -sz_wtime();
	int num_yz = num_y * num_z;
	#pragma omp parallel for
	for(int t=0; t<thread_num; t++){
		int id = sz_get_thread_num();
		int i = id/(num_yz);
		int j = (id % num_yz) / num_z;
		int k = id % num_z;
		size_t offset_x = (i < split_index_x) ? i * early_blockcount_x : i * late_blockcount_x + split_index_x;
		size_t offset_y = (j < split_index_y) ? j * early_blockcount_y : j * late_blockcount_y + split_index_y;
		size_t offset_z = (k < split_index_z) ? k * early_blockcount_z : k * late_blockcount_z + split_index_z;
		float * data_pos = oriData + offset_x * dim0_offset + offset_y * dim1_offset + offset_z; // current data block to be predicted.

		size_t current_blockcount_x = (i < split_index_x) ? early_blockcount_x : late_blockcount_x;
		size_t current_blockcount_y = (j < split_index_y) ? early_blockcount_y : late_blockcount_y;
		size_t current_blockcount_z = (k < split_index_z) ? early_blockcount_z : late_blockcount_z;
		size_t type_offset = offset_x * dim0_offset +  offset_y * current_blockcount_x * dim1_offset + offset_z * current_blockcount_x * current_blockcount_y;
		int * type = result_type + type_offset;

		float * unpredictable_data = result_unpredictable_data + id * unpred_data_max_size;
		float *P0, *P1; // buffer
		P0 = buffer0 + id * early_blockcount_y * early_blockcount_z;
		P1 = buffer1 + id * early_blockcount_y * early_blockcount_z;
		unpredictable_count[id] = SZ_compress_float_3D_MDQ_RA_block(data_pos, mean + id, r1, r2, r3, current_blockcount_x, current_blockcount_y, current_blockcount_z, realPrecision, P0, P1, type, unpredictable_data, cpu_timing);
	}
	(*cpu_timing).cfCost_omp = elapsed_time + sz_wtime();
	// ------------------------------------------------------------
	elapsed_time = -sz_wtime();
	size_t nodeCount = 0;
	Huffman_init_openmp(huffmanTree, result_type, num_elements, thread_num, freq, cpu_timing);
	printf("debug\n");
	(*cpu_timing).tree0_omp = elapsed_time + sz_wtime();
	// ------------------------------------------------------------
	elapsed_time = -sz_wtime();
	for (size_t i = 0; i < stateNum; i++)
		if (huffmanTree->code[i]) nodeCount++;
	nodeCount = nodeCount*2-1;
	(*cpu_timing).tree1_omp = elapsed_time + sz_wtime();
	(*cpu_timing).node_count = nodeCount;
	// ------------------------------------------------------------
	elapsed_time = -sz_wtime();
	unsigned char *treeBytes;
	unsigned int treeByteSize = convert_HuffTree_to_bytes_anyStates(huffmanTree, nodeCount, &treeBytes);
	(*cpu_timing).tree2_omp = elapsed_time + sz_wtime();
	(*cpu_timing).costTree_omp = (*cpu_timing).tree0_omp + (*cpu_timing).tree1_omp + (*cpu_timing).tree2_omp;
	// ------------------------------------------------------------
	unsigned int meta_data_offset = 3 + 1 + MetaDataByteLength;
	size_t total_unpred = 0;
	for(int i=0; i<num_blocks; i++){
		total_unpred += unpredictable_count[i];
	}
	(*cpu_timing).hit_ratio_omp = (float)(num_elements - total_unpred)/num_elements;

	unsigned char * result_pos = result;
	initRandomAccessBytes(result_pos);
	result_pos += meta_data_offset;

	size_t enCodeSize = 0;

	intToBytes_bigEndian(result_pos, thread_num);
	result_pos += 4;
	floatToBytes(result_pos, realPrecision);
	result_pos += sizeof(float);
	intToBytes_bigEndian(result_pos, quantization_intervals);
	result_pos += 4;
	intToBytes_bigEndian(result_pos, treeByteSize);
	result_pos += 4;
	intToBytes_bigEndian(result_pos, nodeCount);
	result_pos += 4;
	memcpy(result_pos, treeBytes, treeByteSize);
	result_pos += treeByteSize;

	memcpy(result_pos, unpredictable_count, num_blocks * sizeof(unsigned int));
	result_pos += num_blocks * sizeof(unsigned int);
	memcpy(result_pos, mean, num_blocks * sizeof(float));
	result_pos += num_blocks * sizeof(float);	

	unpred_offset[0] = 0;
	for(int t=1; t<thread_num; t++){
		unpred_offset[t] = unpredictable_count[t-1] + unpred_offset[t-1];
	}
	#pragma omp parallel for
	for(int t=0; t<thread_num; t++){
		int id = sz_get_thread_num();
		float * unpredictable_data = result_unpredictable_data + id * unpred_data_max_size;
		memcpy(result_pos + unpred_offset[id] * sizeof(float), unpredictable_data, unpredictable_count[id] * sizeof(float));		
	}
	result_pos += total_unpred * sizeof(float);
	// ------------------------------------------------------------
	elapsed_time = -sz_wtime();
	size_t * block_pos = (size_t *) result_pos;
	result_pos += num_blocks * sizeof(size_t);
	#pragma omp parallel for
	for(int t=0; t<thread_num; t++){
		int id = sz_get_thread_num();
		int i = id/(num_yz);
		int j = (id % num_yz) / num_z;
		int k = id % num_z;
		unsigned char * encoding_buffer_pos = encoding_buffer + id * max_num_block_elements * sizeof(int);
		size_t enCodeSize = 0;
		size_t offset_x = (i < split_index_x) ? i * early_blockcount_x : i * late_blockcount_x + split_index_x;
		size_t offset_y = (j < split_index_y) ? j * early_blockcount_y : j * late_blockcount_y + split_index_y;
		size_t offset_z = (k < split_index_z) ? k * early_blockcount_z : k * late_blockcount_z + split_index_z;
		size_t current_blockcount_x = (i < split_index_x) ? early_blockcount_x : late_blockcount_x;
		size_t current_blockcount_y = (j < split_index_y) ? early_blockcount_y : late_blockcount_y;
		size_t current_blockcount_z = (k < split_index_z) ? early_blockcount_z : late_blockcount_z;
		size_t current_block_elements = current_blockcount_x * current_blockcount_y * current_blockcount_z;
		size_t type_offset = offset_x * dim0_offset +  offset_y * current_blockcount_x * dim1_offset + offset_z * current_blockcount_x * current_blockcount_y;
		int * type = result_type + type_offset;
		encode(huffmanTree, type, current_block_elements, encoding_buffer_pos, &enCodeSize);
		block_pos[id] = enCodeSize;
	}
	(*cpu_timing).costEncode_omp = elapsed_time + sz_wtime();
	// ------------------------------------------------------------
	block_offset[0] = 0;
	for(int t=1; t<thread_num; t++){
		block_offset[t] = block_pos[t-1] + block_offset[t-1];
	}
	#pragma omp parallel for
	for(int t=0; t<thread_num; t++){
		int id = sz_get_thread_num();
		memcpy(result_pos + block_offset[id], encoding_buffer + t * max_num_block_elements * sizeof(int), block_pos[t]);		
	}
	result_pos += block_offset[thread_num - 1] + block_pos[thread_num - 1];
	// ------------------------------------------------------------
	result_pos += enCodeSize;
	size_t totalEncodeSize = 0;
	totalEncodeSize = result_pos - result;
	//printf("debug-1\n");
	free(freq);
	//printf("debug-2\n");
	free(buffer0);
	//printf("debug-3\n");
	free(buffer1);
	//printf("debug-4\n");
	free(treeBytes);
	//printf("debug-5\n");
	free(unpred_offset);
	//printf("debug-6\n");
	free(block_offset);
	//printf("debug-7\n");
	//free(encoding_buffer);
	//printf("debug-8\n");
	free(mean);
	//printf("debug-9\n");
	free(result_unpredictable_data);
	//printf("debug-10\n");
	free(unpredictable_count);
	//printf("debug-11\n");
	free(result_type);
	//printf("debug-12\n");
	SZ_ReleaseHuffman(huffmanTree);
	//printf("debug-13\n");
	*comp_size = totalEncodeSize;
	return result;
}

void decompressDataSeries_float_1D_openmp(float** data, size_t r1, unsigned char* comp_data){
}

void decompressDataSeries_float_2D_openmp(float** data, size_t r1, size_t r2, unsigned char* comp_data){
}

void decompressDataSeries_float_3D_openmp(float** data, size_t r1, size_t r2, size_t r3, unsigned char* comp_data){
	
	if(confparams_dec==NULL)
		confparams_dec = (sz_params*)malloc(sizeof(sz_params));
	memset(confparams_dec, 0, sizeof(sz_params));
	if(exe_params==NULL)
		exe_params = (sz_exedata*)malloc(sizeof(sz_exedata));
	memset(exe_params, 0, sizeof(sz_exedata));	
	
	// printf("num_block_elements %d num_blocks %d\n", max_num_block_elements, num_blocks);
	// fflush(stdout);
	double elapsed_time = 0.0;
	elapsed_time = -sz_wtime();

	size_t dim0_offset = r2 * r3;
	size_t dim1_offset = r3;
	size_t num_elements = r1 * r2 * r3;
	
	unsigned char * comp_data_pos = comp_data;
	//int meta_data_offset = 3 + 1 + MetaDataByteLength;
	//comp_data_pos += meta_data_offset;

	int thread_num = bytesToInt_bigEndian(comp_data_pos);
	comp_data_pos += 4;
	int thread_order = (int)log2(thread_num);
	size_t num_x = 0, num_y = 0, num_z = 0;
	{
		int block_thread_order = thread_order / 3;
		switch(thread_order % 3){
			case 0:{
				num_x = 1 << block_thread_order;
				num_y = 1 << block_thread_order;
				num_z = 1 << block_thread_order;
				break;
			}
			case 1:{
				num_x = 1 << (block_thread_order + 1);
				num_y = 1 << block_thread_order;
				num_z = 1 << block_thread_order;
				break;
			}
			case 2:{
				num_x = 1 << (block_thread_order + 1);
				num_y = 1 << (block_thread_order + 1);
				num_z = 1 << block_thread_order;
				break;
			}
		}
	}
	
	printf("number of blocks: %zu %zu %zu, thread_num %d\n", num_x, num_y, num_z, thread_num);
	sz_set_num_threads(thread_num);
	size_t split_index_x, split_index_y, split_index_z;
	size_t early_blockcount_x, early_blockcount_y, early_blockcount_z;
	size_t late_blockcount_x, late_blockcount_y, late_blockcount_z;
	SZ_COMPUTE_BLOCKCOUNT(r1, num_x, split_index_x, early_blockcount_x, late_blockcount_x);
	SZ_COMPUTE_BLOCKCOUNT(r2, num_y, split_index_y, early_blockcount_y, late_blockcount_y);
	SZ_COMPUTE_BLOCKCOUNT(r3, num_z, split_index_z, early_blockcount_z, late_blockcount_z);

	size_t num_blocks = num_x * num_y * num_z;
	size_t * unpred_offset = (size_t *) malloc(num_blocks * sizeof(size_t));
	*data = (float*)malloc(sizeof(float)*num_elements);
	int * result_type = (int *) malloc(num_elements * sizeof(int));
	size_t * block_offset = (size_t *) malloc(num_blocks * sizeof(size_t));

	float realPrecision = bytesToFloat(comp_data_pos);
	comp_data_pos += sizeof(float);
	unsigned int intervals = bytesToInt_bigEndian(comp_data_pos);
	comp_data_pos += sizeof(float);

	size_t stateNum = intervals*2;
        HuffmanTree* huffmanTree = createHuffmanTree(stateNum);

	updateQuantizationInfo(intervals);

	unsigned int tree_size = bytesToInt_bigEndian(comp_data_pos);
	comp_data_pos += sizeof(unsigned int);
	size_t huffman_nodes = bytesToInt_bigEndian(comp_data_pos);
	huffmanTree->allNodes = huffman_nodes;

	node root = reconstruct_HuffTree_from_bytes_anyStates(huffmanTree, comp_data_pos+4, huffmanTree->allNodes);

	comp_data_pos += 4 + tree_size;
	unsigned int * unpred_count = (unsigned int *) comp_data_pos;
	comp_data_pos += num_blocks * sizeof(unsigned int);
	float * mean_pos = (float *) comp_data_pos;
	comp_data_pos += num_blocks * sizeof(float);
	float * result_unpredictable_data = (float *) comp_data_pos;
	size_t total_unpred = 0;
	for(int i=0; i<num_blocks; i++){
		unpred_offset[i] = total_unpred;
		total_unpred += unpred_count[i];
	}
	comp_data_pos += total_unpred * sizeof(float);

	size_t * block_pos = (size_t *) comp_data_pos;
	comp_data_pos += num_blocks * sizeof(size_t);
	block_offset[0] = 0;
	for(int t=1; t<thread_num; t++){
		block_offset[t] = block_pos[t-1] + block_offset[t-1];
	}
	int num_yz = num_y * num_z;
	elapsed_time += sz_wtime();
	printf("Read data info elapsed time: %.4f\n", elapsed_time);
	elapsed_time = -sz_wtime();
	#pragma omp parallel for
	for(int t=0; t<thread_num; t++){
		int id = sz_get_thread_num();
		int i = id/(num_yz);
		int j = (id % num_yz) / num_z;
		int k = id % num_z;
		size_t offset_x = (i < split_index_x) ? i * early_blockcount_x : i * late_blockcount_x + split_index_x;
		size_t offset_y = (j < split_index_y) ? j * early_blockcount_y : j * late_blockcount_y + split_index_y;
		size_t offset_z = (k < split_index_z) ? k * early_blockcount_z : k * late_blockcount_z + split_index_z;
		size_t current_blockcount_x = (i < split_index_x) ? early_blockcount_x : late_blockcount_x;
		size_t current_blockcount_y = (j < split_index_y) ? early_blockcount_y : late_blockcount_y;
		size_t current_blockcount_z = (k < split_index_z) ? early_blockcount_z : late_blockcount_z;
		size_t type_offset = offset_x * dim0_offset +  offset_y * current_blockcount_x * dim1_offset + offset_z * current_blockcount_x * current_blockcount_y;
		int * type = result_type + type_offset;
		decode(comp_data_pos + block_offset[id], current_blockcount_x*current_blockcount_y*current_blockcount_z, root, type);
	}
	elapsed_time += sz_wtime();
	printf("Parallel Huffman decoding elapsed time: %.4f\n", elapsed_time);
	elapsed_time = -sz_wtime();

	#pragma omp parallel for
	for(int t=0; t<thread_num; t++){
		int id = sz_get_thread_num();
		int i = id/(num_yz);
		int j = (id % num_yz) / num_z;
		int k = id % num_z;
		// printf("%d: %d %d %d\n", sz_get_thread_num(), i, j, k);
		size_t offset_x = (i < split_index_x) ? i * early_blockcount_x : i * late_blockcount_x + split_index_x;
		size_t offset_y = (j < split_index_y) ? j * early_blockcount_y : j * late_blockcount_y + split_index_y;
		size_t offset_z = (k < split_index_z) ? k * early_blockcount_z : k * late_blockcount_z + split_index_z;
		float * data_pos = *data + offset_x * dim0_offset + offset_y * dim1_offset + offset_z;

		size_t current_blockcount_x = (i < split_index_x) ? early_blockcount_x : late_blockcount_x;
		size_t current_blockcount_y = (j < split_index_y) ? early_blockcount_y : late_blockcount_y;
		size_t current_blockcount_z = (k < split_index_z) ? early_blockcount_z : late_blockcount_z;
		size_t type_offset = offset_x * dim0_offset +  offset_y * current_blockcount_x * dim1_offset + offset_z * current_blockcount_x * current_blockcount_y;
		int * type = result_type + type_offset;

		float * unpredictable_data = result_unpredictable_data + unpred_offset[id];
		float mean = mean_pos[id];
		decompressDataSeries_float_3D_RA_block(data_pos, mean, r1, r2, r3, current_blockcount_x, current_blockcount_y, current_blockcount_z, realPrecision, type, unpredictable_data);
	}	
	elapsed_time += sz_wtime();
	printf("Parallel decompress elapsed time: %f\n", elapsed_time);

	free(block_offset);
	free(result_type);
	free(unpred_offset);
	SZ_ReleaseHuffman(huffmanTree);
}

//Double Precision

unsigned char * SZ_compress_double_1D_MDQ_openmp(double *oriData, size_t r1, double realPrecision, size_t * comp_size){
	return NULL;
}

unsigned char * SZ_compress_double_2D_MDQ_openmp(double *oriData, size_t r1, size_t r2, double realPrecision, size_t * comp_size){
	return NULL;
}

unsigned char * SZ_compress_double_3D_MDQ_openmp(double *oriData, size_t r1, size_t r2, size_t r3, double realPrecision, size_t * comp_size, CPU_timing *cpu_timing){

	// jwang
	double elapsed_time = 0.0;
	unsigned int quantization_intervals;
	if(exe_params->optQuantMode==1)
	{
		quantization_intervals = optimize_intervals_double_3D_opt(oriData, r1, r2, r3, realPrecision);
		printf("3D number of bins: %d\nerror bound %.20f\n", quantization_intervals, realPrecision);
		updateQuantizationInfo(quantization_intervals);
	}	
	else{
		quantization_intervals = exe_params->intvCapacity;
	}
	// ------------------------------------------------------------
	elapsed_time = -sz_wtime();
	int thread_num = sz_get_max_threads();
	int thread_order = (int)log2(thread_num);
	size_t num_x = 0, num_y = 0, num_z = 0;
	{
		int block_thread_order = thread_order / 3;
		switch(thread_order % 3){
			case 0:{
				num_x = 1 << block_thread_order;
				num_y = 1 << block_thread_order;
				num_z = 1 << block_thread_order;
				break;
			}
			case 1:{
				num_x = 1 << (block_thread_order + 1);
				num_y = 1 << block_thread_order;
				num_z = 1 << block_thread_order;
				break;
			}
			case 2:{
				num_x = 1 << (block_thread_order + 1);
				num_y = 1 << (block_thread_order + 1);
				num_z = 1 << block_thread_order;
				break;
			}
		}
		thread_num = num_x * num_y * num_z;
	}
	sz_set_num_threads(thread_num);

	size_t split_index_x, split_index_y, split_index_z;
	size_t early_blockcount_x, early_blockcount_y, early_blockcount_z;
	size_t late_blockcount_x, late_blockcount_y, late_blockcount_z;
	SZ_COMPUTE_BLOCKCOUNT(r1, num_x, split_index_x, early_blockcount_x, late_blockcount_x);
	SZ_COMPUTE_BLOCKCOUNT(r2, num_y, split_index_y, early_blockcount_y, late_blockcount_y);
	SZ_COMPUTE_BLOCKCOUNT(r3, num_z, split_index_z, early_blockcount_z, late_blockcount_z);

	size_t max_num_block_elements = early_blockcount_x * early_blockcount_y * early_blockcount_z;
	size_t num_blocks = num_x * num_y * num_z;
	size_t num_elements = r1 * r2 * r3;

	size_t dim0_offset = r2 * r3;
	size_t dim1_offset = r3;

	size_t buffer_size = early_blockcount_y * early_blockcount_z * sizeof(double);
	int * result_type = (int *) malloc(num_elements * sizeof(int));
	size_t unpred_data_max_size = max_num_block_elements;
	double * result_unpredictable_data = (double *) malloc(unpred_data_max_size * sizeof(double) * num_blocks);
	unsigned int * unpredictable_count = (unsigned int *) malloc(num_blocks * sizeof(unsigned int));
	double * mean = malloc(num_blocks * sizeof(double));
	double * buffer0, * buffer1;
	buffer0 = (double *) malloc(buffer_size * thread_num);
	buffer1 = (double *) malloc(buffer_size * thread_num);
	unsigned char * result = (unsigned char *) malloc(num_elements * (sizeof(int) + sizeof(double)));
	size_t * unpred_offset = (size_t *) malloc(num_blocks * sizeof(size_t));
	unsigned char * encoding_buffer = (unsigned char *) malloc(max_num_block_elements * sizeof(int) * num_blocks);
	size_t * block_offset = (size_t *) malloc(num_blocks * sizeof(size_t));
	size_t *freq = (size_t *)malloc(thread_num*quantization_intervals*4*sizeof(size_t));

	memset(freq, 0, thread_num*quantization_intervals*4*sizeof(size_t));
	
        size_t stateNum = quantization_intervals*2;
        HuffmanTree* huffmanTree = createHuffmanTree(stateNum);
	
	int num_yz = num_y * num_z;
	#pragma omp parallel for
	for(int t=0; t<thread_num; t++){
		int id = sz_get_thread_num();
		int i = id/(num_yz);
		int j = (id % num_yz) / num_z;
		int k = id % num_z;

		size_t offset_x = (i < split_index_x) ? i * early_blockcount_x : i * late_blockcount_x + split_index_x;
		size_t offset_y = (j < split_index_y) ? j * early_blockcount_y : j * late_blockcount_y + split_index_y;
		size_t offset_z = (k < split_index_z) ? k * early_blockcount_z : k * late_blockcount_z + split_index_z;
		double * data_pos = oriData + offset_x * dim0_offset + offset_y * dim1_offset + offset_z;

		size_t current_blockcount_x = (i < split_index_x) ? early_blockcount_x : late_blockcount_x;
		size_t current_blockcount_y = (j < split_index_y) ? early_blockcount_y : late_blockcount_y;
		size_t current_blockcount_z = (k < split_index_z) ? early_blockcount_z : late_blockcount_z;
		size_t type_offset = offset_x * dim0_offset +  offset_y * current_blockcount_x * dim1_offset + offset_z * current_blockcount_x * current_blockcount_y;
		int * type = result_type + type_offset;

		double * unpredictable_data = result_unpredictable_data + id * unpred_data_max_size;
		double *P0, *P1; // buffer

		P0 = buffer0 + id * early_blockcount_y * early_blockcount_z;
		P1 = buffer1 + id * early_blockcount_y * early_blockcount_z;
		unpredictable_count[id] = SZ_compress_double_3D_MDQ_RA_block(data_pos, mean + id, r1, r2, r3, current_blockcount_x, current_blockcount_y, current_blockcount_z, realPrecision, P0, P1, type, unpredictable_data);
	}
	// ------------------------------------------------------------
	// huffman encode
	size_t nodeCount = 0;
	Huffman_init_openmp(huffmanTree, result_type, num_elements, thread_num, freq, cpu_timing);
	// ------------------------------------------------------------
	for (size_t i = 0; i < stateNum; i++)
		if (huffmanTree->code[i]) nodeCount++;
	nodeCount = nodeCount*2-1;
	unsigned char *treeBytes;
	unsigned int treeByteSize = convert_HuffTree_to_bytes_anyStates(huffmanTree, nodeCount, &treeBytes);

	unsigned int meta_data_offset = 3 + 1 + MetaDataByteLength;
	size_t total_unpred = 0;
	for(int i=0; i<num_blocks; i++){
		total_unpred += unpredictable_count[i];
	}
	unsigned char * result_pos = result;
	initRandomAccessBytes(result_pos);
	result_pos += meta_data_offset;

	size_t enCodeSize = 0;

	intToBytes_bigEndian(result_pos, thread_num);
	result_pos += sizeof(int);
	doubleToBytes(result_pos, realPrecision);
	result_pos += sizeof(double);
	intToBytes_bigEndian(result_pos, quantization_intervals);
	result_pos += 4;
	intToBytes_bigEndian(result_pos, treeByteSize);
	result_pos += 4;
	intToBytes_bigEndian(result_pos, nodeCount);
	result_pos += 4;
	memcpy(result_pos, treeBytes, treeByteSize);
	result_pos += treeByteSize;

	memcpy(result_pos, unpredictable_count, num_blocks * sizeof(unsigned int));
	result_pos += num_blocks * sizeof(unsigned int);
	memcpy(result_pos, mean, num_blocks * sizeof(double));
	result_pos += num_blocks * sizeof(double);	
	unpred_offset[0] = 0;
	for(int t=1; t<thread_num; t++){
		unpred_offset[t] = unpredictable_count[t-1] + unpred_offset[t-1];
	}
	
	#pragma omp parallel for
	for(int t=0; t<thread_num; t++){ // t<thread_num
		int id = sz_get_thread_num();
		double * unpredictable_data = result_unpredictable_data + id * unpred_data_max_size;
		memcpy(result_pos + unpred_offset[id] * sizeof(double), unpredictable_data, unpredictable_count[id] * sizeof(double));
	}

	result_pos += total_unpred * sizeof(double);
	// ------------------------------------------------------------
	size_t * block_pos = (size_t *) result_pos;
	result_pos += num_blocks * sizeof(size_t);
	#pragma omp parallel for
	for(int t=0; t<thread_num; t++){ // t<thread_num
		int id = sz_get_thread_num();
		int i = id/(num_yz);
		int j = (id % num_yz) / num_z;
		int k = id % num_z;
		unsigned char * encoding_buffer_pos = encoding_buffer + id * max_num_block_elements * sizeof(int);
		size_t enCodeSize = 0;
		size_t offset_x = (i < split_index_x) ? i * early_blockcount_x : i * late_blockcount_x + split_index_x;
		size_t offset_y = (j < split_index_y) ? j * early_blockcount_y : j * late_blockcount_y + split_index_y;
		size_t offset_z = (k < split_index_z) ? k * early_blockcount_z : k * late_blockcount_z + split_index_z;
		size_t current_blockcount_x = (i < split_index_x) ? early_blockcount_x : late_blockcount_x;
		size_t current_blockcount_y = (j < split_index_y) ? early_blockcount_y : late_blockcount_y;
		size_t current_blockcount_z = (k < split_index_z) ? early_blockcount_z : late_blockcount_z;
		size_t current_block_elements = current_blockcount_x * current_blockcount_y * current_blockcount_z;
		size_t type_offset = offset_x * dim0_offset +  offset_y * current_blockcount_x * dim1_offset + offset_z * current_blockcount_x * current_blockcount_y;
		int * type = result_type + type_offset;
		encode(huffmanTree, type, current_block_elements, encoding_buffer_pos, &enCodeSize);
		block_pos[id] = enCodeSize;
	}
	// ------------------------------------------------------------
	block_offset[0] = 0;
	for(int t=1; t<thread_num; t++){
		block_offset[t] = block_pos[t-1] + block_offset[t-1];
	}
	#pragma omp parallel for
	for(int t=0; t<thread_num; t++){
		int id = sz_get_thread_num();
		memcpy(result_pos + block_offset[id], encoding_buffer + t * max_num_block_elements * sizeof(int), block_pos[t]);		
	}
	result_pos += block_offset[thread_num - 1] + block_pos[thread_num - 1];
	// ------------------------------------------------------------
	result_pos += enCodeSize;
	size_t totalEncodeSize = 0;
	totalEncodeSize = result_pos - result;

	free(freq);
	free(buffer0);
	free(buffer1);
	free(treeBytes);
	free(unpred_offset);
	free(encoding_buffer);
	free(mean);
	free(result_unpredictable_data);
	free(unpredictable_count);
	free(result_type);
	SZ_ReleaseHuffman(huffmanTree);

	free(block_offset);
	*comp_size = totalEncodeSize;
	
	return result;
}

void decompressDataSeries_double_1D_openmp(double** data, size_t r1, unsigned char* comp_data){
}

void decompressDataSeries_double_2D_openmp(double** data, size_t r1, size_t r2, unsigned char* comp_data){
}

void decompressDataSeries_double_3D_openmp(double** data, size_t r1, size_t r2, size_t r3, unsigned char* comp_data)
{
	if(confparams_dec==NULL)
		confparams_dec = (sz_params*)malloc(sizeof(sz_params));
	memset(confparams_dec, 0, sizeof(sz_params));
	if(exe_params==NULL)
		exe_params = (sz_exedata*)malloc(sizeof(sz_exedata));
	memset(exe_params, 0, sizeof(sz_exedata));	
	
	// printf("num_block_elements %d num_blocks %d\n", max_num_block_elements, num_blocks);
	// fflush(stdout);
	double elapsed_time = 0.0;
	elapsed_time = -sz_wtime();

	size_t dim0_offset = r2 * r3;
	size_t dim1_offset = r3;
	size_t num_elements = r1 * r2 * r3;
	
	unsigned char * comp_data_pos = comp_data;
	//int meta_data_offset = 3 + 1 + MetaDataByteLength;
	//comp_data_pos += meta_data_offset;

	int thread_num = bytesToInt_bigEndian(comp_data_pos);
	comp_data_pos += sizeof(int);
	int thread_order = (int)log2(thread_num);
	size_t num_x = 0, num_y = 0, num_z = 0;
	{
		int block_thread_order = thread_order / 3;
		switch(thread_order % 3){
			case 0:{
				num_x = 1 << block_thread_order;
				num_y = 1 << block_thread_order;
				num_z = 1 << block_thread_order;
				break;
			}
			case 1:{
				num_x = 1 << (block_thread_order + 1);
				num_y = 1 << block_thread_order;
				num_z = 1 << block_thread_order;
				break;
			}
			case 2:{
				num_x = 1 << (block_thread_order + 1);
				num_y = 1 << (block_thread_order + 1);
				num_z = 1 << block_thread_order;
				break;
			}
		}
	}
	
	printf("number of blocks: %zu %zu %zu, thread_num %d\n", num_x, num_y, num_z, thread_num);
	sz_set_num_threads(thread_num);
	size_t split_index_x, split_index_y, split_index_z;
	size_t early_blockcount_x, early_blockcount_y, early_blockcount_z;
	size_t late_blockcount_x, late_blockcount_y, late_blockcount_z;
	SZ_COMPUTE_BLOCKCOUNT(r1, num_x, split_index_x, early_blockcount_x, late_blockcount_x);
	SZ_COMPUTE_BLOCKCOUNT(r2, num_y, split_index_y, early_blockcount_y, late_blockcount_y);
	SZ_COMPUTE_BLOCKCOUNT(r3, num_z, split_index_z, early_blockcount_z, late_blockcount_z);

	size_t num_blocks = num_x * num_y * num_z;
	size_t * unpred_offset = (size_t *) malloc(num_blocks * sizeof(size_t));
	*data = (double*)malloc(sizeof(double)*num_elements);
	int * result_type = (int *) malloc(num_elements * sizeof(int));
	size_t * block_offset = (size_t *) malloc(num_blocks * sizeof(size_t));

	double realPrecision = bytesToDouble(comp_data_pos);
	comp_data_pos += sizeof(double);
	unsigned int intervals = bytesToInt_bigEndian(comp_data_pos);
	comp_data_pos += sizeof(double);

	size_t stateNum = intervals*2;
	HuffmanTree* huffmanTree = createHuffmanTree(stateNum);

	updateQuantizationInfo(intervals);
	// exe_params->intvRadius = (int)((tdps->intervals - 1)/ 2);

	unsigned int tree_size = bytesToInt_bigEndian(comp_data_pos);
	comp_data_pos += sizeof(unsigned int);
	size_t huffman_nodes = bytesToInt_bigEndian(comp_data_pos);
	huffmanTree->allNodes = huffman_nodes;
	// printf("Reconstruct huffman tree with node count %ld\n", nodeCount);
	// fflush(stdout);
	node root = reconstruct_HuffTree_from_bytes_anyStates(huffmanTree, comp_data_pos+4, huffmanTree->allNodes);

	comp_data_pos += 4 + tree_size;
	unsigned int * unpred_count = (unsigned int *) comp_data_pos;
	comp_data_pos += num_blocks * sizeof(unsigned int);
	double * mean_pos = (double *) comp_data_pos;
	comp_data_pos += num_blocks * sizeof(double);
	double * result_unpredictable_data = (double *) comp_data_pos;
	size_t total_unpred = 0;
	for(int i=0; i<num_blocks; i++){
		unpred_offset[i] = total_unpred;
		total_unpred += unpred_count[i];
	}
	comp_data_pos += total_unpred * sizeof(double);

	size_t * block_pos = (size_t *) comp_data_pos;
	comp_data_pos += num_blocks * sizeof(size_t);
	block_offset[0] = 0;
	for(int t=1; t<thread_num; t++){
		block_offset[t] = block_pos[t-1] + block_offset[t-1];
	}
	int num_yz = num_y * num_z;
	elapsed_time += sz_wtime();
	printf("Read data info elapsed time: %f\n", elapsed_time);
	elapsed_time = -sz_wtime();
	#pragma omp parallel for
	for(int t=0; t<thread_num; t++){
		int id = sz_get_thread_num();
		int i = id/(num_yz);
		int j = (id % num_yz) / num_z;
		int k = id % num_z;
		size_t offset_x = (i < split_index_x) ? i * early_blockcount_x : i * late_blockcount_x + split_index_x;
		size_t offset_y = (j < split_index_y) ? j * early_blockcount_y : j * late_blockcount_y + split_index_y;
		size_t offset_z = (k < split_index_z) ? k * early_blockcount_z : k * late_blockcount_z + split_index_z;
		size_t current_blockcount_x = (i < split_index_x) ? early_blockcount_x : late_blockcount_x;
		size_t current_blockcount_y = (j < split_index_y) ? early_blockcount_y : late_blockcount_y;
		size_t current_blockcount_z = (k < split_index_z) ? early_blockcount_z : late_blockcount_z;
		size_t type_offset = offset_x * dim0_offset +  offset_y * current_blockcount_x * dim1_offset + offset_z * current_blockcount_x * current_blockcount_y;
		int * type = result_type + type_offset;
		decode(comp_data_pos + block_offset[id], current_blockcount_x*current_blockcount_y*current_blockcount_z, root, type);
	}
	elapsed_time += sz_wtime();
	printf("Parallel Huffman decoding elapsed time: %f\n", elapsed_time);
	elapsed_time = -sz_wtime();

	#pragma omp parallel for
	for(int t=0; t<thread_num; t++){
		int id = sz_get_thread_num();
		int i = id/(num_yz);
		int j = (id % num_yz) / num_z;
		int k = id % num_z;
		// printf("%d: %d %d %d\n", sz_get_thread_num(), i, j, k);
		size_t offset_x = (i < split_index_x) ? i * early_blockcount_x : i * late_blockcount_x + split_index_x;
		size_t offset_y = (j < split_index_y) ? j * early_blockcount_y : j * late_blockcount_y + split_index_y;
		size_t offset_z = (k < split_index_z) ? k * early_blockcount_z : k * late_blockcount_z + split_index_z;
		double * data_pos = *data + offset_x * dim0_offset + offset_y * dim1_offset + offset_z;

		size_t current_blockcount_x = (i < split_index_x) ? early_blockcount_x : late_blockcount_x;
		size_t current_blockcount_y = (j < split_index_y) ? early_blockcount_y : late_blockcount_y;
		size_t current_blockcount_z = (k < split_index_z) ? early_blockcount_z : late_blockcount_z;
		size_t type_offset = offset_x * dim0_offset +  offset_y * current_blockcount_x * dim1_offset + offset_z * current_blockcount_x * current_blockcount_y;
		int * type = result_type + type_offset;

		double * unpredictable_data = result_unpredictable_data + unpred_offset[id];
		double mean = mean_pos[id];

		decompressDataSeries_double_3D_RA_block(data_pos, mean, r1, r2, r3, current_blockcount_x, current_blockcount_y, current_blockcount_z, realPrecision, type, unpredictable_data);
	}	
	elapsed_time += sz_wtime();
	printf("Parallel decompress elapsed time: %f\n", elapsed_time);

	free(block_offset);
	free(result_type);
	free(unpred_offset);
	SZ_ReleaseHuffman(huffmanTree);
}

void Huffman_init_openmp(HuffmanTree* huffmanTree, int *s, size_t length, int thread_num, size_t * freq, CPU_timing *cpu_timing){

	size_t i;
	size_t block_size = (length - 1)/ thread_num + 1;
	size_t block_residue = length - (thread_num - 1) * block_size;
	//printf("length=%lu, block_size=%lu, block_residue=%lu\n", length, block_size, block_residue);

	//printf("debug-1\n");
	double elapsed_time = -sz_wtime();
	//printf("thread_num=%d\n", thread_num);
	#pragma omp parallel for
	for(int t=0; t<thread_num; t++){
		int id = sz_get_thread_num();
		//printf("id-%d\n",id);
		//TODO
		int * s_pos = s + id * block_size;
		size_t * freq_pos = freq + id * huffmanTree->allNodes;
		if(id < thread_num - 1){
			for(size_t i=0; i<block_size; i++){
				freq_pos[s_pos[i]] ++;
			}
		}
		else{
			//printf("block_residue=%lu\n", block_residue);
			for(size_t i=0; i<block_residue; i++){
				freq_pos[s_pos[i]] ++;
			}
		}
	}
	//printf("debug-1.1\n");
	size_t * freq_pos = freq + huffmanTree->allNodes;
	//printf("debug-1.2\n");
	for(int t=1; t<thread_num; t++){
		for(i = 0; i<huffmanTree->allNodes; i++){
			freq[i] += freq_pos[i];
		}
		//printf("debug-1.3\n");
		freq_pos += huffmanTree->allNodes;
	}

	//printf("debug-2\n");
	(*cpu_timing).tree3_omp = elapsed_time + sz_wtime();

	elapsed_time = -sz_wtime();
	for (i = 0; i < huffmanTree->allNodes; i++)
		if (freq[i]) 
			qinsert(huffmanTree, new_node(huffmanTree, freq[i], i, 0, 0));
 	
	//printf("debug-3\n");
	while (huffmanTree->qend > 2) 
		qinsert(huffmanTree, new_node(huffmanTree, 0, 0, qremove(huffmanTree), qremove(huffmanTree)));
 
	(*cpu_timing).tree4_omp = elapsed_time + sz_wtime();

	//printf("debug-4\n");
	elapsed_time = -sz_wtime();
	build_code(huffmanTree, huffmanTree->qq[1], 0, 0, 0);
	//printf("debug-5\n");
	(*cpu_timing).tree5_omp = elapsed_time + sz_wtime();
	//printf("debug-6\n");
	//free(freq);
}

