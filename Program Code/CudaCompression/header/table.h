/*
 * table.h
 *
 *  Created on: May 6, 2013
 *      Author: hanggao
 */

#ifndef TABLE_H_
#define TABLE_H_

#define TABLE_ELEMENT(table, width, x, y) *(table + x*width + y)

#define TABLE_ELEMENT_PTR(table, width, x, y) (table + x*width + y)

#define SET_TABLE_ELEMENT(table, width, x, y, ele)	*(table + x*width + y) = ele

#endif /* TABLE_H_ */
