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
#include "header/jpeg.h"
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

void printBlock_int(int *matrix, int width, int block_x, int block_y);

extern void read_png_file(char* file_name);
extern void write_png_file(char* file_name);

__device__ float truncate(float i) {
	float r = round(i);
	if (r < 0.0 && r == i - 0.5) {
		return r + 1.0;
	}
	return r;
}

__device__ void RGB_YUV(int *rgb, int *yuv) {
	yuv[0] = (UNIT8) (0.257 * rgb[0] + 0.504 * rgb[1] + 0.098 * rgb[2] + 16);
	yuv[1] = (UNIT8) (-0.148 * rgb[0] - 0.291 * rgb[1] + 0.439 * rgb[2] + 128);
	yuv[2] = (UNIT8) (0.439 * rgb[0] - 0.368 * rgb[1] - 0.071 * rgb[2] + 128);

	int k;
	for (k = 0; k < 3; k++) {
		if (yuv[k] < 0)
			yuv[k] = 0;
		else if (yuv[k] > 255)
			yuv[k] = 255;
	}
}

__device__ void YUV_RGB(int *yuv, int *rgb) {
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

__global__ void FDCT(int *matrix, int width) {
	int i;

	__shared__ float mx[64];
	__shared__ float temp[64];
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	mx[threadIdx.y * 8 + threadIdx.x] = *((int *) ((char *) matrix
			+ index_y * width) + index_x);

	__syncthreads();

	float cu = 0, cv = 0;

	temp[threadIdx.y * 8 + threadIdx.x] = 0;
	for (i = 0; i < 8; i++) {
		temp[threadIdx.y * 8 + threadIdx.x] += COS[threadIdx.y * 8 + i]
				* mx[i * 8 + threadIdx.x];
	}

	__syncthreads();

	mx[threadIdx.y * 8 + threadIdx.x] = 0;
	for (i = 0; i < 8; i++) {
		mx[threadIdx.y * 8 + threadIdx.x] += temp[threadIdx.y * 8 + i]
				* COS_T[i * 8 + threadIdx.x];
	}
	__syncthreads();

	index_x = blockIdx.x * blockDim.x + threadIdx.x;
	index_y = blockIdx.y * blockDim.y + threadIdx.y;

	if (!threadIdx.x)
		cu = sqrt((float) 1 / blockDim.x);
	else
		cu = sqrt((float) 2 / blockDim.x);

	if (!threadIdx.y)
		cv = sqrt((float) 1 / blockDim.y);
	else
		cv = sqrt((float) 2 / blockDim.y);

	int val = (int) (cv * cu * mx[threadIdx.y * 8 + threadIdx.x]);
	if (val > 1023)
		val = 1023;
	else if (val < -1024)
		val = -1024;
	*((int *) ((char *) matrix + index_y * width) + index_x) = val;

}

__global__ void IDCT(int *matrix, int width) {

	int i;

	__shared__ float mx[64];
	__shared__ float temp[64];
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	mx[threadIdx.y * 8 + threadIdx.x] = *((int *) ((char *) matrix
			+ index_y * width) + index_x);

	__syncthreads();

	float cu = 0, cv = 0;

	temp[threadIdx.y * 8 + threadIdx.x] = 0;
	for (i = 0; i < 8; i++) {
		if (!i)
			cv = sqrt((float) 1 / blockDim.x);
		else
			cv = sqrt((float) 2 / blockDim.x);


		temp[threadIdx.y * 8 + threadIdx.x] += cv
				* COS_T[threadIdx.y * 8 + i] * mx[i * 8 + threadIdx.x];
	}
	__syncthreads();

	mx[threadIdx.y * 8 + threadIdx.x] = 0;
	for (i = 0; i < 8; i++) {
		if (!i)
			cu = sqrt((float) 1 / blockDim.x);
		else
			cu = sqrt((float) 2 / blockDim.x);

		mx[threadIdx.y * 8 + threadIdx.x] += cu * temp[threadIdx.y * 8 + i]
				* COS[i * 8 + threadIdx.x];
	}
	__syncthreads();

	index_x = blockIdx.x * blockDim.x + threadIdx.x;
	index_y = blockIdx.y * blockDim.y + threadIdx.y;

	int val = (int) (mx[threadIdx.y * 8 + threadIdx.x]);
	if (val > 1023)
		val = 1023;
	else if (val < -1024)
		val = -1024;
	*((int *) ((char *) matrix + index_y * width) + index_x) = val;

}

__global__ void QUANTIZATION(int *matrix, int width, int quality) {
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	float divided;

	float s = (quality < 50) ? 5000 / quality : 200 - 2 * quality;
	if(s < 1)
		s = 1;
	float val = (s * Qy_[threadIdx.y * 8 + threadIdx.x] + 50) / 100;
	divided = *((int *) ((char *) matrix + index_y * width) + index_x) / val;
	divided = truncate(divided);
	*((int *) ((char *) matrix + index_y * width) + index_x) = (int) divided;
}

__global__ void DEQUANTIZATION(int *matrix, int width, int quality) {
	float divided;

	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;

	float s = (quality < 50) ? 5000 / quality : 200 - 2 * quality;
	if(s < 1)
			s = 1;
	float val = (s * Qy_[threadIdx.y * 8 + threadIdx.x] + 50) / 100;
	divided = *((int *) ((char *) matrix + index_y * width) + index_x) * val;
	*((int *) ((char *) matrix + index_y * width) + index_x) = (int) divided;

}

__global__ void shiftBlock(int *matrix, int width) {
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	*((int *) ((char *) matrix + index_y * width) + index_x) =
			(*((int *) ((char *) matrix + index_y * width) + index_x) - 128);
}

__global__ void ishiftBlock(int *matrix, int width) {
	int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	int index_y = blockIdx.y * blockDim.y + threadIdx.y;
	*((int *) ((char *) matrix + index_y * width) + index_x) =
			(*((int *) ((char *) matrix + index_y * width) + index_x) + 128);
}

__global__ void convertColorSpace_rgb2yuv(int *r, int *g, int *b, int width_y,
		int width_b, int width_r) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int rgb[3] = { *((int *) ((char *) r + j * width_y) + i),
			*((int *) ((char *) g + j * width_b) + i), *((int *) ((char *) b
					+ j * width_r) + i) };
	int yuv[3];
	RGB_YUV(rgb, yuv);
	*((int *) ((char *) r + j * width_y) + i) = yuv[0];
	*((int *) ((char *) g + j * width_b) + i) = yuv[1];
	*((int *) ((char *) b + j * width_r) + i) = yuv[2];
}

__global__ void convertColorSpace_yuv2rgb(int *y, int *cb, int *cr, int width_y,
		int width_b, int width_r) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int rgb[3];
	int yuv[3] = { *((int *) ((char *) y + j * width_y) + i),
			*((int *) ((char *) cb + j * width_b) + i), *((int *) ((char *) cr
					+ j * width_r) + i) };
	YUV_RGB(yuv, rgb);
	*((int *) ((char *) y + j * width_y) + i) = rgb[0];
	*((int *) ((char *) cb + j * width_b) + i) = rgb[1];
	*((int *) ((char *) cr + j * width_r) + i) = rgb[2];
}

__device__ void zigzag(int *matrix, UNIT16 side, int *sequence) {
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

__device__ int sizeofNumber(int number) {
	int k = 0;
	for (k = 0; k < 12; k++) {
		if ((number < (1 << k)) && (number > -(1 << k))) {
			return k;
		}
	}
	return 0;
}

__device__ void countHuffAcCode(int *matrix, int block_x, int block_y,
		int width, UNIT32 *count) {
	int i, j, k, m, n;
	int idx_x, idx_y;
	int mx2[64];
	int mx[64];
	for (i = 0; i < block_x; i++) {
		for (k = 0; k < block_y; k++) {

			for (m = 0; m < 8; m++) {
				for (n = 0; n < 8; n++) {
					idx_y = k * 8 + n;
					idx_x = i * 8 + m;
					mx[n * 8 + m] = *((int *) ((char *) matrix + idx_y * width)
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

__device__ void countHuffDcCode(int *matrix, int block_x, int block_y,
		int width, UNIT32 *count) {
	int i, k, m, n;
	int idx_x, idx_y;
	int mx[64];

	int prevdc = DEFAULT_DC;

	for (i = 0; i < block_x; i++) {
		for (k = 0; k < block_y; k++) {

			for (m = 0; m < 8; m++) {
				for (n = 0; n < 8; n++) {
					idx_y = k * 8 + n;
					idx_x = i * 8 + m;
					mx[n * 8 + m] = *((int *) ((char *) matrix + idx_y * width)
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

__global__ void huffCode(int *y, int *cb, int *cr, UNIT32 *counts_y_dc,
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
	int *ypointers;
	int *cbpointers;
	int *crpointers;

	UNIT32 *counts_y_dc, *counts_y_ac, *counts_b_dc, *counts_b_ac, *counts_r_dc,
			*counts_r_ac;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	size_t pitch_y, pitch_b, pitch_r;
	float seconds;
	float cs[64];
	float cs_t[64];

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
		abort_("Invalid or unsupported image width and height");

	if (quality > 100 || quality < 1)
		abort_("Invalid quality, the range is [0,100]");

	for (y = 0; y < 8; y++) {
		for (x = 0; x < 8; x++) {
			cs[y * 8 + x] = cos(((2 * x + 1) * y * M_PI) / (float) 16);
		}
	}

	for (y = 0; y < 8; y++) {
		for (x = 0; x < 8; x++) {
			cs_t[x * 8 + y] = cos(((2 * x + 1) * y * M_PI) / (float) 16);
		}
	}

	cudaMemcpyToSymbol(COS, cs, 64 * sizeof(float), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(COS_T, cs_t, 64 * sizeof(float), 0,
			cudaMemcpyHostToDevice);

	/* allocate memory space for each component y cb cr*/
	cudaEventRecord(start, 0);
	cudaMallocPitch(&ypointers, &pitch_y, sizeof(int) * width, height);
	cudaMallocPitch(&cbpointers, &pitch_b, sizeof(int) * width, height);
	cudaMallocPitch(&crpointers, &pitch_r, sizeof(int) * width, height);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&seconds, start, stop);
	printf("%f milliseconds to allocate device memory space\n", seconds);

	int *yp, *bp, *rp;
	size_t imgsize = sizeof(int) * height * width;
	yp = (int *) malloc(imgsize);
	bp = (int *) malloc(imgsize);
	rp = (int *) malloc(imgsize);

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

	cudaEventRecord(start, 0);
	cudaMemcpy2D(ypointers, pitch_y, yp, sizeof(int) * width,
			sizeof(int) * width, height, cudaMemcpyHostToDevice);
	cudaMemcpy2D(cbpointers, pitch_b, bp, sizeof(int) * width,
			sizeof(int) * width, height, cudaMemcpyHostToDevice);
	cudaMemcpy2D(crpointers, pitch_r, rp, sizeof(int) * width,
			sizeof(int) * width, height, cudaMemcpyHostToDevice);
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
	 * create block with size 16 X 16, e.g each block has 256 threads
	 * create grid with width/16 + height/16 size.
	 * */
	dim3 dimBlock(16, 16);
	dim3 dimGrid((dimBlock.x - 1 + width) / dimBlock.x,
			(dimBlock.y - 1 + height) / dimBlock.y);

	/*
	 * convert color space from rgb to yuv
	 * */
	cudaEventRecord(start, 0);
	convertColorSpace_rgb2yuv<<<dimGrid, dimBlock>>>(ypointers, cbpointers,
			crpointers, pitch_y, pitch_b, pitch_r);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&seconds, start, stop);
	printf("%f milliseconds for color space conversion from rgb to yuv\n",
			seconds);

	/*
	 * create block with size 8 X 8 and corresponding Grid, which would
	 *  be used as the basic unit of our JPEG compression method
	 * */
	dim3 dimBlock2(8, 8);
	dim3 dimGrid2((dimBlock2.x - 1 + width) / dimBlock2.x,
			(dimBlock2.y - 1 + height) / dimBlock2.y);

	cudaEventRecord(start, 0);
	shiftBlock<<<dimGrid2, dimBlock2>>>(ypointers, pitch_y);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&seconds, start, stop);
	printf("%f milliseconds for data shift\n", seconds * 3);

	cudaEventRecord(start, 0);
	FDCT<<<dimGrid2, dimBlock2>>>(ypointers, pitch_y);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&seconds, start, stop);
	printf("%f milliseconds for DCT transform\n", seconds * 3);

	cudaEventRecord(start, 0);
	QUANTIZATION<<<dimGrid2, dimBlock2>>>(ypointers, pitch_y, quality);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&seconds, start, stop);
	printf("%f milliseconds for Quantization\n", seconds * 3);

	shiftBlock<<<dimGrid2, dimBlock2>>>(cbpointers, pitch_b);
	FDCT<<<dimGrid2, dimBlock2>>>(cbpointers, pitch_b);
	QUANTIZATION<<<dimGrid2, dimBlock2>>>(cbpointers, pitch_b, quality);

	shiftBlock<<<dimGrid2, dimBlock2>>>(crpointers, pitch_r);
	FDCT<<<dimGrid2, dimBlock2>>>(crpointers, pitch_r);
	QUANTIZATION<<<dimGrid2, dimBlock2>>>(crpointers, pitch_r, quality);

	cudaEventRecord(start, 0);
	huffCode<<<6, 1>>>(ypointers, cbpointers, crpointers, counts_y_dc,
			counts_y_ac, counts_b_dc, counts_b_ac, counts_r_dc, counts_r_ac,
			width / 8, height / 8, pitch_y, pitch_b, pitch_r);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&seconds, start, stop);
	printf("%f milliseconds for encoding\n", seconds);

	cudaEventRecord(start, 0);
	DEQUANTIZATION<<<dimGrid2, dimBlock2>>>(ypointers, pitch_y, quality);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&seconds, start, stop);
	printf("%f milliseconds for Dequantization\n", seconds * 3);

	cudaEventRecord(start, 0);
	IDCT<<<dimGrid2, dimBlock2>>>(ypointers, pitch_y);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&seconds, start, stop);
	printf("%f milliseconds for IDCT transform\n", seconds * 3);

	cudaEventRecord(start, 0);
	ishiftBlock<<<dimGrid2, dimBlock2>>>(ypointers, pitch_y);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&seconds, start, stop);
	printf("%f milliseconds for reverse data shift\n", seconds * 3);

	DEQUANTIZATION<<<dimGrid2, dimBlock2>>>(cbpointers, pitch_b, quality);
	IDCT<<<dimGrid2, dimBlock2>>>(cbpointers, pitch_b);
	ishiftBlock<<<dimGrid2, dimBlock2>>>(cbpointers, pitch_b);

	DEQUANTIZATION<<<dimGrid2, dimBlock2>>>(crpointers, pitch_r, quality);
	IDCT<<<dimGrid2, dimBlock2>>>(crpointers, pitch_r);
	ishiftBlock<<<dimGrid2, dimBlock2>>>(crpointers, pitch_r);

	/*
	 * convert color space back from yuv to rgb
	 * */
	cudaEventRecord(start, 0);
	convertColorSpace_yuv2rgb<<<dimGrid, dimBlock>>>(ypointers, cbpointers,
			crpointers, pitch_y, pitch_b, pitch_r);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&seconds, start, stop);
	printf("%f milliseconds for color space conversion from yuv to rgb\n",
			seconds);

	cudaEventRecord(start, 0);
	cudaMemcpy2D(yp, sizeof(int) * width, ypointers, pitch_y,
			sizeof(int) * width, height, cudaMemcpyDeviceToHost);
	cudaMemcpy2D(bp, sizeof(int) * width, cbpointers, pitch_b,
			sizeof(int) * width, height, cudaMemcpyDeviceToHost);
	cudaMemcpy2D(rp, sizeof(int) * width, crpointers, pitch_r,
			sizeof(int) * width, height, cudaMemcpyDeviceToHost);
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

void printBlock_int(int *matrix, int width, int block_x, int block_y) {
	int u, v;

	for (u = 0; u < block_side; u++) {
		for (v = 0; v < block_side; v++) {
			int index_x = block_x * width * block_side + u * width;
			int index_y = block_y * block_side + v;
			printf("%d\t", *(matrix + index_x + index_y));
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

