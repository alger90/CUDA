################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../Huffman.cu \
../compression.cu \
../error.cu \
../png.cu 

CU_DEPS += \
./Huffman.d \
./compression.d \
./error.d \
./png.d 

OBJS += \
./Huffman.o \
./compression.o \
./error.o \
./png.o 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -I/opt/local/include -I/opt/local/include/libpng15 -O3 -m64 -gencode arch=compute_30,code=sm_30 -odir "" -M -o "$(@:%.o=%.d)" "$<"
	nvcc --compile -I/opt/local/include -I/opt/local/include/libpng15 -O3 -gencode arch=compute_30,code=compute_30 -gencode arch=compute_30,code=sm_30 -m64  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


