#ifndef CONSTANTS_H
#define CONSTANTS_H
#include "jpeg.h"

const UNIT32 block_side = 8,
               block_length = block_side * block_side;

/* Quantization tables */
__constant__ UNIT8 Qy_[] = {
    16,  11,  10,  16,  24,  40,  51,  61,
    12,  12,  14,  19,  26,  58,  60,  55,
    14,  13,  16,  24,  40,  57,  69,  56,
    14,  17,  22,  29,  51,  87,  80,  62,
    18,  22,  37,  56,  68, 109, 103,  77,
    24,  35,  55,  64,  81, 104, 113,  92,
    49,  64,  78,  87, 103, 121, 120, 101,
    72,  92,  95,  98, 112, 100, 103,  99
};

__constant__ UNIT8 Qy_mid_[] = {
	6, 4, 4, 6, 10, 16, 20, 24,
	5, 5, 6, 8, 10, 23, 24, 22,
	6, 5, 6, 10, 16, 23, 28, 22,
	6, 7, 9, 12, 20, 35, 32, 25,
	7, 9, 15, 22, 27, 44, 41, 31,
	10, 14, 22, 26, 32, 42, 45, 37,
	20, 26, 31, 35, 41, 48, 48, 40,
	29, 37, 38, 39, 45, 40, 41, 40
};

__constant__ UNIT8 Qy_low_[] = {
	2,	1,	1,	 1,	 1,	 1,	 2,	 1,
	1,	1,	2,	 2,	 2,	 2,	 2,	 4,
	3,	2,	2,	 2,	 2,	 5,	 4,	 4,
	3, 	4,	6,	 5,	 6,	 6,	 6,	 5,
	6,	6,	6,	 7,	 9,	 8,	 6,	 7,
	9,	7,	6,	 6,	 8,	 11, 8,  9,
	10, 10, 10,  10, 10, 6,  8, 11,
	12, 11, 10, 12, 9, 10, 10, 10
};

__constant__ UNIT8 Qc_[] = {
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99
};

__constant__ UNIT8 Qc_mid_[] = {
	6, 4, 4, 6, 10, 16, 20, 24,
	5, 5, 6, 8, 10, 23, 24, 22,
	6, 5, 6, 10, 16, 23, 28, 22,
	6, 7, 9, 12, 20, 35, 32, 25,
	7, 9, 15, 22, 27, 44, 41, 31,
	10, 14, 22, 26, 32, 42, 45, 37,
	20, 26, 31, 35, 41, 48, 48, 40,
	29, 37, 38, 39, 45, 40, 41, 40
};

__constant__ UNIT8 Qc_low_[] = {
	2,	1,	1,	 1,	 1,	 1,	 2,	 1,
	1,	1,	2,	 2,	 2,	 2,	 2,	 4,
	3,	2,	2,	 2,	 2,	 5,	 4,	 4,
	3, 	4,	6,	 5,	 6,	 6,	 6,	 5,
	6,	6,	6,	 7,	 9,	 8,	 6,	 7,
	9,	7,	6,	 6,	 8,	 11, 8,  9,
	10, 10, 10,  10, 10, 6,  8, 11,
	12, 11, 10, 12, 9, 10, 10, 10
};

__constant__ int DEFAULT_DC = 255;

/* Huffman tables */
UNIT8 Hy_dc_bits[] = {
    0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0
};

UNIT8 Hy_dc_val[] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
};

UNIT8 Hy_ac_bits[] = {
    0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d
};

UNIT8 Hy_ac_val[] = {
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12,
    0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08,
    0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16,
    0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
    0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
    0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
    0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
    0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
    0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6,
    0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
    0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4,
    0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
    0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea,
    0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa
};

UNIT8 Hc_dc_bits[] = {
    0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0
};

UNIT8 Hc_dc_val[] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
};

UNIT8 Hc_ac_bits[] = {
    0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77
};

UNIT8 Hc_ac_val[] = {
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21,
    0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
    0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91,
    0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
    0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34,
    0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
    0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38,
    0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
    0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,
    0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96,
    0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
    0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4,
    0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
    0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2,
    0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
    0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9,
    0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa
};

#endif