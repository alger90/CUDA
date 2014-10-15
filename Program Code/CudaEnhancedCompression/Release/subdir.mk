################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../compression.cu \
../error.cu \
../png.cu 

CU_DEPS += \
./compression.d \
./error.d \
./png.d 

OBJS += \
./compression.o \
./error.o \
./png.o 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -I/opt/local/include/libpng15 -I/opt/local/include -O3 -m64 -gencode arch=compute_10,code=sm_10 -odir "" -M -o "$(@:%.o=%.d)" "$<"
	nvcc --compile -I/opt/local/include/libpng15 -I/opt/local/include -O3 -gencode arch=compute_10,code=compute_10 -gencode arch=compute_10,code=sm_10 -m64  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


