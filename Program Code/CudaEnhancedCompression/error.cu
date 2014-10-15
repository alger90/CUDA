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
#include <stdio.h>
#include <stdarg.h>
#include <math.h>
#include "header/error.h"

#define INPUT_FILE_OPEN_ERROR -10

void sys_error(int err){
	switch(err){
	case INPUT_FILE_OPEN_ERROR:
		printf("ERROR %d: cannot open input file!\n", INPUT_FILE_OPEN_ERROR);
		break;
	default:
		break;
	}
	exit(0);
}

void abort_(const char * s, ...)
{
        va_list args;
        va_start(args, s);
        vfprintf(stderr, s, args);
        fprintf(stderr, "\n");
        va_end(args);
        abort();
}


float quality_factor(float quality)
{
    return (quality <= 50
             ? 50.0 / (float) quality
               : 2.0 - (float) quality / 50.0);
}
