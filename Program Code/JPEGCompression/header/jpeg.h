/*
 * jpeg.h
 *
 *  Created on: May 5, 2013
 *      Author: hanggao
 */

#ifndef JPEG_H_
#define JPEG_H_

#include <stdio.h>

#define START_OF_IMAGE_INVALID -32
#define INVALID_JFIF_HEADER -33
#define INVALID_NUMBER_OF_QUANT_TABLE -34
#define INVALID_QUANT_TBLE -35
#define INVALID_START_OF_FRAME -36
#define INVALID_NUMBER_OF_HUFF_TABLE -37
#define INVALID_HUFF_TBLE -38
#define INVALID_START_OF_SCAN -39


#define JPEG_SOS 0xDA
#define JPEG_SOI 0xD8
#define JPEG_JFIF 0xE0
#define JPEG_QUAN 0xDB
#define JPEG_HUFF 0xC4
#define JPEG_SOF 0xC0
#define JPEG_EOI 0xD9

typedef unsigned char UNIT8;
typedef unsigned short UNIT16;
typedef unsigned int UNIT32;
typedef unsigned long UNIT64;


typedef struct{
	UNIT8 marker[2];
	UNIT8 length[2];
	UNIT8 jfif[5];
	UNIT8 version[2];
	UNIT8 density_used;
	UNIT8 X_SCAL[2];
	UNIT8 Y_SCAL[2];
	UNIT8 t_width;
	UNIT8 t_height;
} JFIF_header;

typedef struct{
	UNIT8 marker[2];
	UNIT8 length[2];
	UNIT8 id;
} QuanTable_header;

typedef struct{
	QuanTable_header header;
	UNIT8 *table;
} QuanTable;

typedef struct{
	UNIT8 id;
	UNIT8 sample;
	UNIT8 quant_table_id;
} component;

typedef struct{
	UNIT8 marker[2];
	UNIT8 length[2];
	UNIT8 percision;
	UNIT8 height[2];
	UNIT8 width[2];
	UNIT8 component_num;
} SOF_header;

typedef struct{
	SOF_header header;
	component * components;
} SOF;

typedef struct{
	UNIT32 length;
	UNIT32 code;
} HuffCode;

typedef struct{
	UNIT8 marker[2];
	UNIT8 length[2];
	UNIT8 id;
	UNIT8 code_num_by_length[16];
} HuffTable_header;

typedef struct{
	HuffTable_header header;
	UNIT8 *categories;
	HuffCode codes[256];
} HuffTable;

typedef struct{
	UNIT8 id;
	UNIT8 huffid;
} sos_component;

typedef struct{
	UNIT8 marker[2];
	UNIT8 length[2];
	UNIT8 component_num;
	sos_component *components;
	UNIT8 skip[3];
} SOS_header;

typedef struct JPEG_t{
	UNIT8 SOI[2];
	JFIF_header jfif;
	QuanTable qtables[4];
	HuffTable htables[4];
	SOF sof;
	SOS_header sos;
	FILE *fp;
} JPEG;

JPEG *mallocJPEG();
void destroyJPEG(JPEG **ptr);
void bind_src(struct JPEG_t *self, FILE *fp);
int read_header(struct JPEG_t *self);
int read_JFIF_header(JFIF_header *head, FILE *fp);
int read_start_of_frame(SOF *sof, FILE *fp);
int read_quantization_table(QuanTable *table, FILE *fp);
int read_huffman_table(HuffTable *table, FILE *fp);
void zigzag(UNIT8 *matrix, UNIT16 side, UNIT8 *sequence);
void antizigzag(UNIT8 *matrix, UNIT16 side, UNIT8 *sequence);

#endif /* JPEG_H_ */
