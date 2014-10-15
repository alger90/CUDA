/*
 * error.h
 *
 *  Created on: May 6, 2013
 *      Author: hanggao
 */

#ifndef JPEG_ERROR_H
#define JPEG_ERROR_H

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>


#define INPUT_FILE_OPEN_ERROR -10

void sys_error(int err);

void abort_(const char * s, ...);

float quality_factor(float quality);

#endif /* ERROR_H_ */
