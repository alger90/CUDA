/* *
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "header/table.h"
#include "header/error.h"
#include "header/constants.h"
#include <unistd.h>
#include <string.h>
#include <stdarg.h>
#include <time.h>

#include <png.h>

extern int width, height;

extern png_structp png_ptr;
extern png_infop info_ptr;
extern png_bytep * row_pointers;

extern void read_png_file(char* file_name);
extern void write_png_file(char* file_name);

__device__ float truncate(float i) {
	float r = round(i);
	if (r < 0.0 && r == i - 0.5) {
		return r + 1.0;
	}
	return r;
}

__device__ void RGB_YUV(float *rgb, float *yuv) {
	yuv[0] = 0.257 * rgb[0] + 0.504 * rgb[1] + 0.098 * rgb[2] + 16;
	yuv[1] = -0.148 * rgb[0] - 0.291 * rgb[1] + 0.439 * rgb[2] + 128;
	yuv[2] = 0.439 * rgb[0] - 0.368 * rgb[1] - 0.071 * rgb[2] + 128;
}

__device__ void YUV_RGB(float *yuv, float *rgb) {
	rgb[0] = 1.164 * (yuv[0] - 16) + 1.596 * (yuv[2] - 128);
	rgb[1] = 1.164 * (yuv[0] - 16) - 0.813 * (yuv[2] - 128)
			- 0.392 * (yuv[1] - 128);
	rgb[2] = 1.164 * (yuv[0] - 16) + 2.017 * (yuv[1] - 128);
	int k;
	for (k = 0; k < 3; k++) {
		if (rgb[k] < 0)
			rgb[k] = 0;
		else if (rgb[k] > 255)
			rgb[k] = 255;
	}
}

__global__ void FDCT(float *matrix, int width) {
	int i, j;

	__shared__ float mx[64];
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	mx[threadIdx.y * 8 + threadIdx.x] = *(matrix + index_y * width + index_x);

	__syncthreads();

	float sum = 0, cu = 0, cv = 0;

	if (!threadIdx.x)
		cu = sqrt((float) 1 / blockDim.x);
	else
		cu = sqrt((float) 2 / blockDim.x);

	if (!threadIdx.y)
		cv = sqrt((float) 1 / blockDim.y);
	else
		cv = sqrt((float) 2 / blockDim.y);

	for (i = 0; i < blockDim.x; i++) {
		for (j = 0; j < blockDim.y; j++) {
			sum += mx[j * 8 + i]
					* (cos(((2 * i + 1) * threadIdx.x * M_PI) / 16))
					* (cos(((2 * j + 1) * threadIdx.y * M_PI) / 16));
		}
	}

	index_x = blockIdx.x * blockDim.x + threadIdx.x;
	index_y = blockIdx.y * blockDim.y + threadIdx.y;
	*(matrix + index_y * width + index_x) = cv * cu * sum;

}

__global__ void IDCT(float *matrix, int width) {
	int u, v;

	__shared__ float mx[64];
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	mx[threadIdx.y * 8 + threadIdx.x] = *(matrix + index_y * width + index_x);

	__syncthreads();

	float sum = 0;
	for (u = 0; u < blockDim.x; u++) {
		for (v = 0; v < blockDim.y; v++) {
			float cu;
			float cv;
			if (!u)
				cu = sqrt((float) 1 / blockDim.x);
			else
				cu = sqrt((float) 2 / blockDim.x);

			if (!v)
				cv = sqrt((float) 1 / blockDim.y);
			else
				cv = sqrt((float) 2 / blockDim.y);

			sum += cu * cv * mx[v * 8 + u]
					* (cos(((2 * threadIdx.x + 1) * u * M_PI) / 16))
					* (cos(((2 * threadIdx.y + 1) * v * M_PI) / 16));
		}
	}

	index_x = blockIdx.x * blockDim.x + threadIdx.x;
	index_y = blockIdx.y * blockDim.y + threadIdx.y;
	*(matrix + index_y * width + index_x) = sum;

}

__global__ void QUANTIZATION(float *matrix, int width, int quality) {
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	float divided;

	int s = (quality < 50) ? 5000 / quality : 200 - 2 * quality;
	int val = (s * Qy_[threadIdx.y * 8 + threadIdx.x] + 50) / 100;
	divided = *(matrix + index_y * width + index_x) / val;
	divided = truncate(divided);
	*(matrix + index_y * width + index_x) = (int) divided;
}

__global__ void DEQUANTIZATION(float *matrix, int width, int quality) {
	float divided;

	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	int s = (quality < 50) ? 5000 / quality : 200 - 2 * quality;
	int val = (s * Qy_[threadIdx.y * 8 + threadIdx.x] + 50) / 100;
	divided = *(matrix + index_y * width + index_x) * val;
	*(matrix + index_y * width + index_x) = (int) divided;

}

__global__ void shiftBlock(float *matrix, int width) {
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	*(matrix + index_y * width + index_x) = (*(matrix + index_y * width
			+ index_x) - 128);
}

__global__ void ishiftBlock(float *matrix, int width) {
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	*(matrix + index_y * width + index_x) = (*(matrix + index_y * width
			+ index_x) + 128);
}

__global__ void convertColorSpace_rgb2yuv(float *r, float *g, float *b,
		int width, int height) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	float rgb[3] = { r[j * width + i], g[j * width + i], b[j * width + i] };
	float yuv[3];
	RGB_YUV(rgb, yuv);
	r[j * width + i] = yuv[0];
	g[j * width + i] = yuv[1];
	b[j * width + i] = yuv[2];
}

__global__ void convertColorSpace_yuv2rgb(float *y, float *cb, float *cr,
		int width, int height) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	float rgb[3];
	float yuv[3] = { y[j * width + i], cb[j * width + i], cr[j * width + i] };
	YUV_RGB(yuv, rgb);
	y[j * width + i] = rgb[0];
	cb[j * width + i] = rgb[1];
	cr[j * width + i] = rgb[2];
}

__device__ void zigzag(float *matrix, UNIT16 side, float *sequence) {
	int i = 0;
	int j = 0;

	int index = 0;
	sequence[index++] = TABLE_ELEMENT(matrix, side, 0, 0);

	//for upper triangle of matrix
	do {
		j++;
		sequence[index++] = TABLE_ELEMENT(matrix, side, i, j);

		while (j != 0) {
			i++;
			j--;
			sequence[index++] = TABLE_ELEMENT(matrix, side, i, j);
		}

		i++;
		if (i > 7) {
			i--;
			break;
		}
		sequence[index++] = TABLE_ELEMENT(matrix, side, i, j);

		while (i != 0) {
			i--;
			j++;
			sequence[index++] = TABLE_ELEMENT(matrix, side, i, j);
		}
	} while (true);

	//for lower triangle of matrix
	do {
		j++;
		sequence[index++] = TABLE_ELEMENT(matrix, side, i, j);

		while (j != 7) {
			j++;
			i--;
			sequence[index++] = TABLE_ELEMENT(matrix, side, i, j);
		}
		i++;
		if (i > 7) {
			i--;
			break;
		}
		sequence[index++] = TABLE_ELEMENT(matrix, side, i, j);

		while (i != 7) {
			i++;
			j--;
			sequence[index++] = TABLE_ELEMENT(matrix, side, i, j);
		}
	} while (true);
}

__device__ int sizeofNumber(float number) {
	int k = 0;
	for (k = 0; k < 12; k++) {
		if ((number < (1 << k)) && (number > -(1 << k))) {
			return k;
		}
	}
	return 0;
}

__device__ void countHuffAcCode(float *matrix, int block_x, int block_y,
		int width, UNIT32 *count) {
	int i, j, k, m, n;
	int idx_x, idx_y;
	float mx2[64];
	float mx[64];
	for (i = 0; i < block_x; i++) {
		for (k = 0; k < block_y; k++) {

			for (m = 0; m < 8; m++) {
				for (n = 0; n < 8; n++) {
					idx_y = k * 8 + n;
					idx_x = i * 8 + m;
					mx[n * 8 + m] = *(matrix + idx_y * width
							+ idx_x);

				}
			}

			zigzag(mx, 8, mx2);

			int zc = 0;
			int size = 0;
			//skip DC code
			for (j = 1; j < 64; j++) {

				if (mx2[j] == 0) {
					zc++;
					if (zc != 16) {
						if (j == 63) {
							UNIT8 idx = ((zc & 0xF) << 4) | (0 & 0xF);
							count[idx]++;
						}
						continue;
					} else {
						zc = 15;
						size = 0;
					}
				} else {
					size = sizeofNumber(mx2[j]);
				}

				UNIT8 idx = ((zc & 0xF) << 4) | (size & 0xF);
				count[idx]++;
				zc = 0;
				size = 0;
			}
			count[0]++;
		}
	}
}

__device__ void countHuffDcCode(float *matrix, int block_x, int block_y,
		int width, UNIT32 *count) {
	int i, k, m, n;
	int idx_x, idx_y;
	float mx[64];

	int prevdc = DEFAULT_DC;

	for (i = 0; i < block_x; i++) {
		for (k = 0; k < block_y; k++) {

			for (m = 0; m < 8; m++) {
				for (n = 0; n < 8; n++) {
					idx_y = k * 8 + n;
					idx_x = i * 8 + m;
					mx[n * 8 + m] = *(matrix + idx_y * width
							+ idx_x);

				}
			}

			int diff = mx[0] - prevdc;
			int size = sizeofNumber(diff);
			UNIT8 idx = size & 0xF;
			count[idx]++;
			prevdc = mx[0];
		}
	}
}

__global__ void huffCode(float *y, float *cb, float *cr, UNIT32 *counts_y_dc,
		UNIT32 *counts_y_ac, UNIT32 *counts_b_dc, UNIT32 *counts_b_ac,
		UNIT32 *counts_r_dc, UNIT32 *counts_r_ac, int block_x, int block_y,
		size_t pitch_y, size_t pitch_b, size_t pitch_r) {

	switch (threadIdx.x) {
	case 0:
		countHuffDcCode(y, block_x, block_y, pitch_y, counts_y_dc);
		break;
	case 1:
		countHuffAcCode(y, block_x, block_y, pitch_y, counts_y_ac);
		break;
	case 2:
		countHuffDcCode(cb, block_x, block_y, pitch_b, counts_b_dc);
		break;
	case 3:
		countHuffAcCode(cb, block_x, block_y, pitch_b, counts_b_ac);
		break;
	case 4:
		countHuffDcCode(cr, block_x, block_y, pitch_r, counts_r_dc);
		break;
	case 5:
		countHuffAcCode(cr, block_x, block_y, pitch_r, counts_r_ac);
		break;
	}

}

__host__ void process_file(int quality) {

	int y, x, skip;

	//UNIT8 *QY, *QC;
	float *ypointers;
	float *cbpointers;
	float *crpointers;

	UNIT32 *counts_y_dc, *counts_y_ac, *counts_b_dc, *counts_b_ac, *counts_r_dc,
			*counts_r_ac;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	float seconds;

	if (png_get_color_type(png_ptr, info_ptr) != PNG_COLOR_TYPE_RGBA
			&& png_get_color_type(png_ptr, info_ptr) != PNG_COLOR_TYPE_RGB)
		abort_(
				"[process_file] color_type of input file must be PNG_COLOR_TYPE_RGBA or PNG_COLOR_TYPE_RGB (%d) (is %d)",
				PNG_COLOR_TYPE_RGBA, png_get_color_type(png_ptr, info_ptr));

	if (png_get_color_type(png_ptr, info_ptr) == PNG_COLOR_TYPE_RGB)
		skip = 3;
	else
		skip = 4;

	if (width % block_side != 0 || height % block_side != 0 || width > 20000
			|| height > 20000)
		abort_("Invalid and unsupported image width %d and height %d", width,
				height);

	if (quality > 100 || quality < 1)
		abort_("Invalid quality, the range is [0,100]");

	/* allocate memory space for each component y cb cr*/
	cudaEventRecord(start, 0);
	int imgsize = sizeof(float) * height * width;
	cudaMalloc(&ypointers, imgsize);
	cudaMalloc(&cbpointers, imgsize);
	cudaMalloc(&crpointers, imgsize);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&seconds, start, stop);
	printf("%f milliseconds to allocate device memory space\n", seconds);

	/* divide original 4-dimension matrix into three components*/
	float *yp, *bp, *rp;
	yp = (float *) malloc(imgsize);
	bp = (float *) malloc(imgsize);
	rp = (float *) malloc(imgsize);

	for (y = 0; y < height; y++) {
		png_byte* row = row_pointers[y];
		for (x = 0; x < width; x++) {
			png_byte* ptr = &(row[x * skip]);
			// set red value to 0 and green value to the blue one
			yp[y * width + x] = ptr[0];
			bp[y * width + x] = ptr[1];
			rp[y * width + x] = ptr[2];
		}
	}

	/* load each component into GPU global memory */
	cudaEventRecord(start, 0);
	cudaMemcpy(ypointers, yp, imgsize, cudaMemcpyHostToDevice);
	cudaMemcpy(cbpointers, bp, imgsize, cudaMemcpyHostToDevice);
	cudaMemcpy(crpointers, rp, imgsize, cudaMemcpyHostToDevice);

	cudaMalloc(&counts_y_dc, 256 * sizeof(UNIT32));
	cudaMemset(counts_y_dc, 0, sizeof(UNIT32) * 256);
	cudaMalloc(&counts_y_ac, 256 * sizeof(UNIT32));
	cudaMemset(counts_y_ac, 0, sizeof(UNIT32) * 256);
	cudaMalloc(&counts_b_dc, 256 * sizeof(UNIT32));
	cudaMemset(counts_b_dc, 0, sizeof(UNIT32) * 256);
	cudaMalloc(&counts_b_ac, 256 * sizeof(UNIT32));
	cudaMemset(counts_b_ac, 0, sizeof(UNIT32) * 256);
	cudaMalloc(&counts_r_dc, 256 * sizeof(UNIT32));
	cudaMemset(counts_r_dc, 0, sizeof(UNIT32) * 256);
	cudaMalloc(&counts_r_ac, 256 * sizeof(UNIT32));
	cudaMemset(counts_r_ac, 0, sizeof(UNIT32) * 256);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&seconds, start, stop);
	printf("%f milliseconds to upload data to device \n", seconds);
	/* empty local buffer in main memory */
	memset(yp, 0, imgsize);
	memset(bp, 0, imgsize);
	memset(rp, 0, imgsize);

	/*
	 * create block with size 8 X 8, e.g each block has 64 threads
	 * create grid with width/8 + height/8 size.
	 * */
	dim3 dimBlock2(8, 8);
	dim3 dimGrid2((dimBlock2.x - 1 + width) / dimBlock2.x,
			(dimBlock2.y - 1 + height) / dimBlock2.y);

	/*
	 * convert color space from rgb to yuv
	 * */
	cudaEventRecord(start, 0);
	convertColorSpace_rgb2yuv<<<dimGrid2, dimBlock2>>>(ypointers, cbpointers,
			crpointers, width, height);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&seconds, start, stop);
	printf("%f milliseconds for color space conversion from rgb to yuv\n",
			seconds);

	cudaEventRecord(start, 0);
	shiftBlock<<<dimGrid2, dimBlock2>>>(ypointers, width);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&seconds, start, stop);
	printf("%f milliseconds for data shift\n", seconds * 3);

	cudaEventRecord(start, 0);
	FDCT<<<dimGrid2, dimBlock2>>>(ypointers, width);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&seconds, start, stop);
	printf("%f milliseconds for DCT transform\n", seconds * 3);

	cudaEventRecord(start, 0);
	QUANTIZATION<<<dimGrid2, dimBlock2>>>(ypointers, width, quality);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&seconds, start, stop);
	printf("%f milliseconds for Quantization\n", seconds * 3);

	shiftBlock<<<dimGrid2, dimBlock2>>>(cbpointers, width);
	FDCT<<<dimGrid2, dimBlock2>>>(cbpointers, width);
	QUANTIZATION<<<dimGrid2, dimBlock2>>>(cbpointers, width, quality);

	shiftBlock<<<dimGrid2, dimBlock2>>>(crpointers, width);
	FDCT<<<dimGrid2, dimBlock2>>>(crpointers, width);
	QUANTIZATION<<<dimGrid2, dimBlock2>>>(crpointers, width, quality);

	cudaEventRecord(start, 0);
	huffCode<<<6, 1>>>(ypointers, cbpointers, crpointers, counts_y_dc,
			counts_y_ac, counts_b_dc, counts_b_ac, counts_r_dc, counts_r_ac,
			width / 8, height / 8, (size_t)width, (size_t)width, (size_t)width);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&seconds, start, stop);
	printf("%f milliseconds for encoding\n", seconds);

	cudaEventRecord(start, 0);
	DEQUANTIZATION<<<dimGrid2, dimBlock2>>>(ypointers, width, quality);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&seconds, start, stop);
	printf("%f milliseconds for Dequantization\n", seconds * 3);

	cudaEventRecord(start, 0);
	IDCT<<<dimGrid2, dimBlock2>>>(ypointers, width);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&seconds, start, stop);
	printf("%f milliseconds for IDCT transform\n", seconds * 3);

	cudaEventRecord(start, 0);
	ishiftBlock<<<dimGrid2, dimBlock2>>>(ypointers, width);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&seconds, start, stop);
	printf("%f milliseconds for reverse data shift\n", seconds * 3);

	DEQUANTIZATION<<<dimGrid2, dimBlock2>>>(cbpointers, width, quality);
	IDCT<<<dimGrid2, dimBlock2>>>(cbpointers, width);
	ishiftBlock<<<dimGrid2, dimBlock2>>>(cbpointers, width);

	DEQUANTIZATION<<<dimGrid2, dimBlock2>>>(crpointers, width, quality);
	IDCT<<<dimGrid2, dimBlock2>>>(crpointers, width);
	ishiftBlock<<<dimGrid2, dimBlock2>>>(crpointers, width);

	/*
	 * convert color space back from yuv to rgb
	 * */
	cudaEventRecord(start, 0);
	convertColorSpace_yuv2rgb<<<dimGrid2, dimBlock2>>>(ypointers, cbpointers,
			crpointers, width, height);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&seconds, start, stop);
	printf("%f milliseconds for color space conversion from yuv to rgb\n",
			seconds);

	cudaEventRecord(start, 0);
	cudaMemcpy(yp, ypointers, imgsize, cudaMemcpyDeviceToHost);
	cudaMemcpy(bp, cbpointers, imgsize, cudaMemcpyDeviceToHost);
	cudaMemcpy(rp, crpointers, imgsize, cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&seconds, start, stop);
	printf("%f milliseconds used to download data from device \n", seconds);

	for (y = 0; y < height; y++) {
		png_byte* row = row_pointers[y];
		for (x = 0; x < width; x++) {
			png_byte* ptr = &(row[x * skip]);
			// set red value to 0 and green value to the blue one
			ptr[0] = yp[y * width + x];
			ptr[1] = bp[y * width + x];
			ptr[2] = rp[y * width + x];
		}
	}

	cudaEventRecord(start, 0);
	cudaFree(ypointers);
	cudaFree(cbpointers);
	cudaFree(crpointers);
	cudaFree(counts_y_dc);
	cudaFree(counts_y_ac);
	cudaFree(counts_b_dc);
	cudaFree(counts_b_ac);
	cudaFree(counts_r_dc);
	cudaFree(counts_r_ac);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&seconds, start, stop);
	printf("%f milliseconds used to free allocated memory on device \n",
			seconds);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	free(yp);
	free(bp);
	free(rp);

}

void printBlock_float(float *matrix, int width, int block_x, int block_y) {
	int u, v;

	for (u = 0; u < block_side; u++) {
		for (v = 0; v < block_side; v++) {
			int index_x = block_x * width * block_side + u * width;
			int index_y = block_y * block_side + v;
			printf("%5.1f\t", *(matrix + index_x + index_y));
		}
		printf("\n");
	}
	printf("\n");
}

int main(int argc, char *argv[]) {

	if (argc < 4)
		abort_(
				"Usage: program_name file_path_of_input_file file_path_of_output_file quality");

	read_png_file(argv[1]);
	process_file(atoi(argv[3]));
	write_png_file(argv[2]);

}

