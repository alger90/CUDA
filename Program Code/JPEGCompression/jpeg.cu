/* *
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */

#include <stdlib.h>
#include <string.h>
#include "header/jpeg.h"
#include "header/error.h"
#include "header/table.h"
#include <stdio.h>
#include <math.h>

#ifndef SEEK_SET
#define SEEK_SET 0
#endif
#ifndef SEEK_CUR
#define	SEEK_CUR	1	/* set file offset to current plus offset */
#endif

void printJFIFHeader(JFIF_header *header) {
	printf("marker %x %x\n", header->marker[0], header->marker[1]);
	printf("length %x %x\n", header->length[0], header->length[1]);
	printf("jfif %x %x %x %x %x\n", header->jfif[0], header->jfif[1],
			header->jfif[2], header->jfif[3], header->jfif[4]);
	printf("version %x %x\n", header->version[0], header->version[1]);
	printf("density_used %x\n", header->density_used);
	printf("X_SCAL %x %x\n", header->X_SCAL[0], header->X_SCAL[1]);
	printf("Y_SCAL %x %x\n", header->Y_SCAL[0], header->Y_SCAL[1]);
	printf("t_width %x\n", header->t_width);
	printf("t_height %x\n", header->t_height);
}

void printQuanTable(QuanTable *table) {

	UNIT16 length = ((table->header.length[0] << 8) | table->header.length[1])
			- 3;
	printf("marker %x %x\n", table->header.marker[0], table->header.marker[1]);
	printf("length %x %x\n", table->header.length[0], table->header.length[1]);
	printf("id %x\n", table->header.id);

	UNIT16 side = sqrt(length);

	int w = 0;
	int h = 0;
	for (w = 0; w < side; w++) {
		for (int h = 0; h < side; h++) {
			printf("%2x\t", TABLE_ELEMENT(table->table, side, w, h));
		}
		printf("\n");
	}

}

void printStart_of_frame(SOF *sof) {
	printf("marker %x %x\n", sof->header.marker[0], sof->header.marker[1]);
	printf("length %x %x\n", sof->header.length[0], sof->header.length[1]);
	printf("precision %x\n", sof->header.percision);
	printf("height %x %x\n", sof->header.height[0], sof->header.height[1]);
	printf("width %x %x\n", sof->header.width[0], sof->header.width[1]);
	printf("component number %x\n", sof->header.component_num);
	int i;
	for (i = 0; i < sof->header.component_num; i++) {
		printf("component %x %x %x\n", sof->components[i].id,
				sof->components[i].sample, sof->components[i].quant_table_id);
	}
}

void printHuffmanTable(HuffTable *table) {
	int i;
	UNIT16 length;
	length = ((table->header.length[0] << 8) | table->header.length[1]) - 3;
	printf("marker %x %x\n", table->header.marker[0], table->header.marker[1]);
	printf("length %x %x\n", table->header.length[0], table->header.length[1]);
	printf("id %x\n", table->header.id);

	for (i = 0; i < 16; i++) {
		printf("%d: %d\t", i, table->header.code_num_by_length[i]);
	}
	printf("\n");
	for (i = 0; i < 256; i++) {
		if (table->codes[i].length > 0) {
			printf("%X %d %X\n", i, table->codes[i].length,
					table->codes[i].code);
		}
	}
	printf("\n");
}

void bind_src(JPEG *jpg, FILE *fp) {
	jpg->fp = fp;
}

void zigzag(UNIT8 *matrix, UNIT16 side, UNIT8 *sequence) {
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

void antizigzag(UNIT8 *matrix, UNIT16 side, UNIT8 *sequence) {
	int i = 0, j = 0;

	int index = 0;
	SET_TABLE_ELEMENT(matrix, side, 0, 0, sequence[index++]);

	//for upper triangle of matrix
	do {
		j++;
		SET_TABLE_ELEMENT(matrix, side, i, j, sequence[index++]);

		while (j != 0) {
			i++;
			j--;
			SET_TABLE_ELEMENT(matrix, side, i, j, sequence[index++]);
		}

		i++;
		if (i > 7) {
			i--;
			break;
		}
		SET_TABLE_ELEMENT(matrix, side, i, j, sequence[index++]);

		while (i != 0) {
			i--;
			j++;
			SET_TABLE_ELEMENT(matrix, side, i, j, sequence[index++]);
		}
	} while (true);

	//for lower triangle of matrix
	do {
		j++;
		SET_TABLE_ELEMENT(matrix, side, i, j, sequence[index++]);

		while (j != 7) {
			j++;
			i--;
			SET_TABLE_ELEMENT(matrix, side, i, j, sequence[index++]);
		}
		i++;
		if (i > 7) {
			i--;
			break;
		}
		SET_TABLE_ELEMENT(matrix, side, i, j, sequence[index++]);

		while (i != 7) {
			i++;
			j--;
			SET_TABLE_ELEMENT(matrix, side, i, j, sequence[index++]);
		}
	} while (true);
}

int read_JFIF_header(JFIF_header *head, FILE *fp) {
	size_t size;
	size = fread(head, sizeof(JFIF_header), 1, fp);
	if (!size)
		return INVALID_JFIF_HEADER;

	return 0;
}

int write_JFIF_header(JFIF_header *head, FILE *fp) {
	size_t size;
	size = fwrite(head, sizeof(JFIF_header), 1, fp);
	if (!size)
		return INVALID_JFIF_HEADER;

	return 0;
}

int read_start_of_frame(SOF *sof, FILE *fp) {
	size_t size;
	size = fread(&sof->header, sizeof(sof->header), 1, fp);
	if (!size)
		return INVALID_START_OF_FRAME;

	sof->components = (component *) malloc(
			sizeof(component) * sof->header.component_num);
	size = fread(sof->components, sizeof(component), sof->header.component_num,
			fp);
	if (!size)
		return INVALID_START_OF_FRAME;

	return 0;
}

int write_start_of_frame(SOF *sof, FILE *fp) {
	size_t size;
	int i;
	size = fwrite(&sof->header, sizeof(sof->header), 1, fp);
	if (!size)
		return INVALID_START_OF_FRAME;

	for (i = 0; i < sof->header.component_num; i++) {
		size = fwrite(&sof->components[i], sizeof(component), 1, fp);
		if (!size)
			return INVALID_START_OF_FRAME;
	}

	return 0;
}

int read_quantization_table(QuanTable *table, FILE *fp) {
	size_t size;
	UNIT16 length, side;
	UNIT8 *buff;
	/* read quantization table header */
	size = fread(&table->header, sizeof(table->header), 1, fp);
	if (!size)
		return INVALID_QUANT_TBLE;

	/* read the quantization table sequence */
	length = ((table->header.length[0] << 8) | table->header.length[1]) - 3;
	side = sqrt(length);
	if (side * side != length)
		return -90;
	table->table = (UNIT8 *) malloc(sizeof(UNIT8) * length);
	size = fread(table->table, sizeof(UNIT8), length, fp);

	/* anti zig-zag to get original quantization table */
	buff = (UNIT8 *) malloc(sizeof(UNIT8) * length);
	antizigzag(buff, side, table->table);
	memcpy(table->table, buff, length);
	free(buff);

	return 0;
}

int write_quantization_table(QuanTable *table, FILE *fp) {
	size_t size;
	UNIT16 length, side;
	UNIT8 *buff;
	/* write quantization table header */
	size = fwrite(&table->header, sizeof(table->header), 1, fp);
	if (!size)
		return INVALID_QUANT_TBLE;

	/* write the quantization table sequence */
	length = ((table->header.length[0] << 8) | table->header.length[1]) - 3;
	side = sqrt(length);
	if (side * side != length)
		return -90;

	/* zig-zag to get sequenced quantization table */
	buff = (UNIT8 *) malloc(sizeof(UNIT8) * length);
	zigzag(table->table, side, buff);
	size = fwrite(buff, sizeof(UNIT8), length, fp);
	if (!size)
		return INVALID_QUANT_TBLE;
	free(buff);

	return 0;
}

int read_huffman_table(HuffTable *table, FILE *fp) {
	size_t size;
	UNIT16 length;
	UNIT8 *buff, index = 0;
	int i, j;
	UNIT32 codevalue = 0;

	/* read huffman table header */
	size = fread(&table->header, sizeof(table->header), 1, fp);
	if (!size)
		return INVALID_HUFF_TBLE;

	length = ((table->header.length[0] << 8) | table->header.length[1]) - 19;

	/* read categories */
	table->categories = (UNIT8 *) malloc(sizeof(UNIT8) * length);
	size = fread(table->categories, sizeof(UNIT8), length, fp);
	if (!size)
		return INVALID_HUFF_TBLE;

	for (i = 0; i < 16; i++) {
		for (j = 1; j <= table->header.code_num_by_length[i]; j++) {
			table->codes[table->categories[index]].length = i;
			table->codes[table->categories[index++]].code = codevalue;
			codevalue++;
		}
		codevalue = codevalue * 2;
	}

	return 0;
}

int write_huffman_table(HuffTable *table, FILE *fp){
	size_t size;
	UNIT16 length;

	/* write huffman table header */
	size = fwrite(&table->header, sizeof(table->header), 1, fp);
	if (!size)
		return INVALID_HUFF_TBLE;

	length = ((table->header.length[0] << 8) | table->header.length[1]) - 19;

	/* write categories */
	size = fwrite(&table->categories, sizeof(UNIT8), length, fp);
	if (!size)
		return INVALID_HUFF_TBLE;

	return 0;
}

int read_header(struct JPEG_t *self) {
	int ret;
	size_t size;
	QuanTable *table;
	HuffTable *htable;
	UNIT16 length;

	UNIT8 marker[2] = { 0x00, 0x00 };
	size = fread(&self->SOI, sizeof(UNIT8), 2, self->fp);
	if (size <= 0 || !(self->SOI[0] == 0xFF && self->SOI[1] == 0xD8))
		return START_OF_IMAGE_INVALID;

	long int pos = ftell(self->fp);
	size = fread(&marker, sizeof(UNIT8), 2, self->fp);
	fseek(self->fp, pos, SEEK_SET);
	while (size > 0) {

		if (marker[0] != 0xFF)
			return -91;
		printf("%X\n", marker[1]);
		switch (marker[1]) {
		/* Read JFIF header */
		case JPEG_JFIF:

			if ((ret = read_JFIF_header(&self->jfif, self->fp)) < 0)
				return ret;

			/* Get next marker */
			pos = ftell(self->fp);
			size = fread(&marker, sizeof(UNIT16), 1, self->fp);
			fseek(self->fp, pos, SEEK_SET);
			break;

		case JPEG_SOF:

			if ((ret = read_start_of_frame(&self->sof, self->fp)) < 0)
				return ret;

			/* Get next marker */
			pos = ftell(self->fp);
			size = fread(&marker, sizeof(UNIT16), 1, self->fp);
			fseek(self->fp, pos, SEEK_SET);
			break;

		case JPEG_HUFF:
			/* find a huffman table entry */
			int i;
			for (i = 0; i < 4; i++) {
				if (!(self->htables[0].header.marker[0] == 0xFF
						&& self->htables[0].header.marker[1] == 0xC4)) {
					htable = &self->htables[i];
					break;
				}
			}

			/* no more quantization table entry */
			if (htable == NULL)
				return INVALID_NUMBER_OF_HUFF_TABLE;

			if ((ret = read_huffman_table(htable, self->fp)) < 0)
				return ret;

			/* Get next marker */
			pos = ftell(self->fp);
			size = fread(&marker, sizeof(UNIT16), 1, self->fp);
			fseek(self->fp, pos, SEEK_SET);
			break;

		case JPEG_QUAN:
			table = NULL;

			/* find the empty quantization table entry */
			for (i = 0; i < 4; i++) {
				if (!(self->qtables[i].header.marker[0] == 0xFF
						&& self->qtables[i].header.marker[1] == 0xDB)) {
					table = &self->qtables[i];
					break;
				}
			}

			/* no more quantization table entry */
			if (table == NULL)
				return INVALID_NUMBER_OF_QUANT_TABLE;

			if ((ret = read_quantization_table(table, self->fp)) < 0)
				return ret;

			/* Get next marker */
			pos = ftell(self->fp);
			size = fread(&marker, sizeof(UNIT16), 1, self->fp);
			fseek(self->fp, pos, SEEK_SET);
			break;

		case JPEG_SOS:
			size = fread(&self->sos.marker, sizeof(UNIT8), 2, self->fp);
			if (!size)
				return INVALID_START_OF_SCAN;

			size = fread(&self->sos.length, sizeof(UNIT8), 2, self->fp);

			size = fread(&self->sos.component_num, sizeof(UNIT8), 1, self->fp);
			if (!size)
				return INVALID_START_OF_SCAN;

			self->sos.components = (sos_component *)malloc(sizeof(sos_component) * self->sos.component_num);
			if (!size)
				return INVALID_START_OF_SCAN;

			size = fread(self->sos.components, sizeof(sos_component), self->sos.component_num, self->fp);
			if (!size)
				return INVALID_START_OF_SCAN;

			size = fread(self->sos.skip, sizeof(UNIT8), 3, self->fp);
			if (!size)
				return INVALID_START_OF_SCAN;

			printf("Okay!!!!!!!! %X\n", self->sos.component_num);
			return 0;

		default:
			while (!(marker[1] == 0xE0 || marker[1] == 0xC0 || marker[1] == 0xC4
					|| marker[1] == 0xDB || marker[1] == 0xDA) && size != 0) {
				marker[0] = marker[1];
				size = fread(&marker[1], sizeof(UNIT8), 1, self->fp);
			}

			if (size == 0)
				return -1;

			/* Get next marker */
			pos = ftell(self->fp);
			fseek(self->fp, pos - 2, SEEK_SET);
			break;
		}
	}
	return 0;
}

JPEG *mallocJPEG() {
	JPEG *ptr = (JPEG *) malloc(sizeof(JPEG));
	memset(ptr, 0, sizeof(JPEG));
	return ptr;
}

void destroyJPEG(JPEG **ptr) {
	free(*ptr);
	*ptr = NULL;
}

/**
int main(void) {
	// input file pointer
	FILE *input;
	JPEG *jpg = mallocJPEG();
	int ret;

	input = fopen("resource/jpeg1.jpg", "rb");
	if (!input)
		sys_error(INPUT_FILE_OPEN_ERROR);

	bind_src(jpg, input);
	ret = read_header(jpg);

	destroyJPEG(&jpg);
	fclose(input);

	return 0;
}
*/
