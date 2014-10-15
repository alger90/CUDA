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

int *ypointers;
int *cbpointers;
int *crpointers;

UNIT8 *QY;
UNIT8 *QC;

extern int width, height;

extern png_structp png_ptr;
extern png_infop info_ptr;
extern png_bytep * row_pointers;

extern void read_png_file(char* file_name);
extern void write_png_file(char* file_name);

void RGB_YUV(int *rgb, int *yuv) {
	yuv[0] = 0.257 * rgb[0] + 0.504 * rgb[1] + 0.098 * rgb[2] + 16;
	yuv[1] = -0.148 * rgb[0] - 0.291 * rgb[1] + 0.439 * rgb[2] + 128;
	yuv[2] = 0.439 * rgb[0] - 0.368 * rgb[1] - 0.071 * rgb[2] + 128;
	int k;
	for (k = 0; k < 3; k++) {
		if (yuv[k] < 0)
			yuv[k] = 0;
		else if (yuv[k] > 255)
			yuv[k] = 255;
	}
}

void YUV_RGB(int *yuv, int *rgb) {
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

void FDCT(int *matrix, int width, int block_x, int block_y, int *nmatrix) {
	int i, j, u, v;

	for (u = 0; u < block_side; u++) {
		for (v = 0; v < block_side; v++) {
			float sum = 0, cu = 0, cv = 0;
			for (i = 0; i < block_side; i++) {
				for (j = 0; j < block_side; j++) {
					int index_x = block_x * width * block_side + i * width;
					int index_y = block_y * block_side + j;
					sum += *(matrix + index_x + index_y)
							* (cos(((2 * i + 1) * u * M_PI) / 16))
							* (cos(((2 * j + 1) * v * M_PI) / 16));
				}
			}
			if (!u)
				cu = sqrt((float) 1 / block_side);
			else
				cu = sqrt((float) 2 / block_side);

			if (!v)
				cv = sqrt((float) 1 / block_side);
			else
				cv = sqrt((float) 2 / block_side);

			int index_x = block_x * width * block_side + u * width;
			int index_y = block_y * block_side + v;
			int val = (int) (cv * cu * sum);
			if (val > 1023)
				val = 1023;
			else if (val < -1024)
				val = -1024;
			*(nmatrix + index_x + index_y) = val;
		}
	}

}

void IDCT(int *matrix, int width, int block_x, int block_y, int *nmatrix) {
	int i, j, u, v;

	for (i = 0; i < block_side; i++) {
		for (j = 0; j < block_side; j++) {
			float sum = 0;
			for (u = 0; u < block_side; u++) {
				for (v = 0; v < block_side; v++) {
					float cu;
					float cv;
					if (!u)
						cu = sqrt((float) 1 / block_side);
					else
						cu = sqrt((float) 2 / block_side);

					if (!v)
						cv = sqrt((float) 1 / block_side);
					else
						cv = sqrt((float) 2 / block_side);

					int index_x = block_x * width * block_side + u * width;
					int index_y = block_y * block_side + v;
					sum += cu * cv * (*(matrix + index_x + index_y))
							* (cos(((2 * i + 1) * u * M_PI) / 16))
							* (cos(((2 * j + 1) * v * M_PI) / 16));
				}
			}

			int index_x = block_x * width * block_side + i * width;
			int index_y = block_y * block_side + j;
			*(nmatrix + index_x + index_y) = sum;
		}
	}

}

void QUANTIZATION(int *matrix, int width, int block_x, int block_y,
		const UNIT8 *quan_table, int *nmatrix) {
	int i, j;
	float divided;

	//quantization loop
	for (j = 0; j < block_side; j++)
		for (i = 0; i < block_side; i++) {
			int index_x = block_x * width * block_side + i * width;
			int index_y = block_y * block_side + j;
			divided = *(matrix + index_x + index_y) / quan_table[i * 8 + j];
			divided = truncate(divided);
			*(nmatrix + index_x + index_y) = (int) divided;
		}
}

void DEQUANTIZATION(int *matrix, int width, int block_x, int block_y,
		const UNIT8 *quan_table, int *nmatrix) {
	int i, j;
	float divided;

	//quantization loop
	for (j = 0; j < block_side; j++)
		for (i = 0; i < block_side; i++) {
			int index_x = block_x * width * block_side + i * width;
			int index_y = block_y * block_side + j;
			divided = *(matrix + index_x + index_y) * quan_table[i * 8 + j];
			*(nmatrix + index_x + index_y) = (int) divided;
		}
}

void shiftBlock(int *matrix, int width, int block_x, int block_y) {
	int u, v;
	for (u = 0; u < block_side; u++) {
		for (v = 0; v < block_side; v++) {
			int index_x = block_x * width * block_side + u * width;
			int index_y = block_y * block_side + v;
			*(matrix + index_x + index_y) = *(matrix + index_x + index_y) - 128;
		}
	}

}

void ishiftBlock(int *matrix, int width, int block_x, int block_y) {
	int u, v;
	for (u = 0; u < block_side; u++) {
		for (v = 0; v < block_side; v++) {
			int index_x = block_x * width * block_side + u * width;
			int index_y = block_y * block_side + v;
			*(matrix + index_x + index_y) = *(matrix + index_x + index_y) + 128;
		}
	}

}

void dct_component(int *matrix, int width, int height) {
	int bx_num = width / block_side;
	int by_num = height / block_side;
	int length = sizeof(int) * width * height;
	int *nmatrix = (int *) malloc(length);
	int i, j;
	for (i = 0; i < by_num; i++) {
		for (j = 0; j < bx_num; j++) {
			shiftBlock(matrix, width, i, j);
			FDCT(matrix, width, i, j, nmatrix);
		}
	}
	memcpy(matrix, nmatrix, length);
	free(nmatrix);
}

void idct_component(int *matrix, int width, int height) {
	int bx_num = width / block_side;
	int by_num = height / block_side;
	int length = sizeof(int) * width * height;
	int *nmatrix = (int *) malloc(length);
	int i, j;
	for (i = 0; i < by_num; i++) {
		for (j = 0; j < bx_num; j++) {
			IDCT(matrix, width, i, j, nmatrix);
			ishiftBlock(nmatrix, width, i, j);
		}
	}
	memcpy(matrix, nmatrix, length);
	free(nmatrix);
}

void quan_component(int *matrix, int width, int height,
		const UNIT8 *quan_table) {
	int bx_num = width / block_side;
	int by_num = height / block_side;
	int length = sizeof(int) * width * height;
	int *nmatrix = (int *) malloc(length);
	int i, j;
	for (i = 0; i < by_num; i++) {
		for (j = 0; j < bx_num; j++) {
			QUANTIZATION(matrix, width, i, j, quan_table, nmatrix);
		}
	}
	memcpy(matrix, nmatrix, length);
	free(nmatrix);
}

void de_quan_component(int *matrix, int width, int height,
		const UNIT8 *quan_table) {
	int bx_num = width / block_side;
	int by_num = height / block_side;
	int length = sizeof(int) * width * height;
	int *nmatrix = (int *) malloc(length);
	int i, j;
	for (i = 0; i < by_num; i++) {
		for (j = 0; j < bx_num; j++) {
			DEQUANTIZATION(matrix, width, i, j, quan_table, nmatrix);
		}
	}
	memcpy(matrix, nmatrix, length);
	free(nmatrix);
}

void zigzag(int *matrix, UNIT16 side, int *sequence) {
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

int sizeofNumber(int number) {
	int k = 0;
	for (k = 0; k < 12; k++) {
		if ((number < (1 << k)) && (number > -(1 << k))) {
			return k;
		}
	}
	return 0;
}

void countHuffAcCode(int *matrix, int block_x, int block_y, int width,
		UNIT32 *count) {
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

void countHuffDcCode(int *matrix, int block_x, int block_y, int width,
		UNIT32 *count) {
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

void huffCode(int *y, int *cb, int *cr, UNIT32 *counts_y_dc,
		UNIT32 *counts_y_ac, UNIT32 *counts_b_dc, UNIT32 *counts_b_ac,
		UNIT32 *counts_r_dc, UNIT32 *counts_r_ac, int block_x, int block_y,
		size_t pitch_y, size_t pitch_b, size_t pitch_r) {

	countHuffDcCode(y, block_x, block_y, pitch_y, counts_y_dc);

	countHuffAcCode(y, block_x, block_y, pitch_y, counts_y_ac);

	countHuffDcCode(cb, block_x, block_y, pitch_b, counts_b_dc);

	countHuffAcCode(cb, block_x, block_y, pitch_b, counts_b_ac);

	countHuffDcCode(cr, block_x, block_y, pitch_r, counts_r_dc);

	countHuffAcCode(cr, block_x, block_y, pitch_r, counts_r_ac);

}

void process_file() {

	int x, y, skip;
	UNIT32 counts_y_dc[256];
	UNIT32 counts_y_ac[256];
	UNIT32 counts_b_dc[256];
	UNIT32 counts_b_ac[256];
	UNIT32 counts_r_dc[256];
	UNIT32 counts_r_ac[256];

	if (png_get_color_type(png_ptr, info_ptr) != PNG_COLOR_TYPE_RGBA
			&& png_get_color_type(png_ptr, info_ptr) != PNG_COLOR_TYPE_RGB)
		abort_(
				"[process_file] color_type of input file must be PNG_COLOR_TYPE_RGBA or PNG_COLOR_TYPE_RGB (%d) (is %d)",
				PNG_COLOR_TYPE_RGBA, png_get_color_type(png_ptr, info_ptr));

	if (png_get_color_type(png_ptr, info_ptr) == PNG_COLOR_TYPE_RGB)
		skip = 3;
	else
		skip = 4;

	if (width % block_side != 0 || height % block_side != 0)
		abort_("Invalid image width and height");

	/* allocate memory space for each component y cb cr*/
	int imgsize = sizeof(int) * height * width;
	ypointers = (int *) malloc(imgsize);
	memset(ypointers, 0, imgsize);
	cbpointers = (int *) malloc(imgsize);
	memset(cbpointers, 0, imgsize);
	crpointers = (int *) malloc(imgsize);
	memset(crpointers, 0, imgsize);

	clock_t start, end;
	double seconds;
	start = clock();

	/* convert rgb space of original image to YcbCr space */
	for (y = 0; y < height; y++) {
		png_byte* row = row_pointers[y];
		for (x = 0; x < width; x++) {
			png_byte* ptr = &(row[x * skip]);
			int rgb[3] = { ptr[0], ptr[1], ptr[2] };
			int yuv[3];

			/* convert rgb color space into ycrcb color space */
			RGB_YUV(rgb, yuv);

			/* Separate each component */
			ypointers[y * width + x] = yuv[0];
			cbpointers[y * width + x] = yuv[1];
			crpointers[y * width + x] = yuv[2];
		}
	}

	end = clock();
	seconds = 1000 * ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("%f milliseconds for color space conversion\n", seconds);

	start = clock();
	dct_component(ypointers, width, height);
	dct_component(cbpointers, width, height);
	dct_component(crpointers, width, height);
	end = clock();
	seconds = 1000 * ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("%f milliseconds for dct transformation\n", seconds);

	start = clock();
	quan_component(ypointers, width, height, QY);
	quan_component(cbpointers, width, height, QC);
	quan_component(crpointers, width, height, QC);
	end = clock();
	seconds = 1000 * ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("%f milliseconds for quantization\n", seconds);

	start = clock();
	huffCode(ypointers, cbpointers, crpointers, counts_y_dc, counts_y_ac,
			counts_b_dc, counts_b_ac, counts_r_dc, counts_r_ac, width / 8,
			height / 8, (size_t) width, (size_t) width, (size_t) width);
	end = clock();
	seconds = 1000 * ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("%f milliseconds for encoding\n", seconds);

	start = clock();
	de_quan_component(ypointers, width, height, QY);
	de_quan_component(cbpointers, width, height, QC);
	de_quan_component(crpointers, width, height, QC);
	end = clock();
	seconds = 1000 * ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("%f milliseconds for dequantization\n", seconds);

	start = clock();
	idct_component(ypointers, width, height);
	idct_component(cbpointers, width, height);
	idct_component(crpointers, width, height);
	end = clock();
	seconds = 1000 * ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("%f milliseconds for reverse dct transformation\n", seconds);

	/* convert YcbCr space of image back to RGB space */
	for (y = 0; y < height; y++) {
		png_byte* row = row_pointers[y];
		for (x = 0; x < width; x++) {
			png_byte* ptr = &(row[x * skip]);
			int rgb[3];
			int yuv[3] = { ypointers[y * width + x], cbpointers[y * width + x],
					crpointers[y * width + x] };

			/* convert YcbCr color space into rgb color space */
			YUV_RGB(yuv, rgb);

			/* Separate each component */
			ptr[0] = (png_byte) rgb[0];
			ptr[1] = (png_byte) rgb[1];
			ptr[2] = (png_byte) rgb[2];

		}
	}

	free(ypointers);
	free(cbpointers);
	free(crpointers);
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

	if (argc < 3)
		abort_(
				"Usage: program_name file_path_of_input_file file_path_of_output_file quality(1,2)");

	read_png_file(argv[1]);

	if (atoi(argv[3]) == 1) {
		QY = Qy_;
		QC = Qc_;
	} else if (atoi(argv[2]) == 2) {
		QY = Qy_mid_;
		QC = Qc_mid_;
	} else if (atoi(argv[3]) == 3) {
		QY = Qy_low_;
		QC = Qc_low_;
	}

	process_file();

	write_png_file(argv[2]);

}

