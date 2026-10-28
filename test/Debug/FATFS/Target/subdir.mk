################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (13.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../FATFS/Target/user_diskio.c 

OBJS += \
./FATFS/Target/user_diskio.o 

C_DEPS += \
./FATFS/Target/user_diskio.d 


# Each subdirectory must supply rules for building sources it contributes
FATFS/Target/%.o FATFS/Target/%.su FATFS/Target/%.cyclo: ../FATFS/Target/%.c FATFS/Target/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DREGION_AU915 -DSOFT_SE -DUSE_HAL_DRIVER -DSTM32L476xx -c -I../USB_DEVICE/App -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/lora" -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/lora/apps" -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/lora/apps/LmHandler" -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/lora/apps/LmHandler/packages" -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/lora/Board" -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/lora/Board/EEPROM_Emul/Core" -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/lora/Board/EEPROM_Emul/Porting/STM32L4" -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/lora/Crypto" -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/lora/Inc" -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/lora/Mac" -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/lora/Mac/region" -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/lora/Radio" -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/lora/Radio/sx126x" -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/lora/System" -I../USB_DEVICE/Target -I../Core/Inc -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Drivers/STM32L4xx_HAL_Driver/Inc -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Drivers/STM32L4xx_HAL_Driver/Inc/Legacy -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Middlewares/ST/STM32_USB_Device_Library/Core/Inc -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Middlewares/ST/STM32_USB_Device_Library/Class/CDC/Inc -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Drivers/CMSIS/Device/ST/STM32L4xx/Include -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Drivers/CMSIS/Include -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares" -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node" -I../FATFS/Target -I../FATFS/App -IC:/Users/daisy/STM32Cube/Repository/STM32Cube_FW_L4_V1.18.2/Middlewares/Third_Party/FatFs/src -I"C:/Users/daisy/Downloads/test (1)/test/Middlewares_node/Inc" -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-FATFS-2f-Target

clean-FATFS-2f-Target:
	-$(RM) ./FATFS/Target/user_diskio.cyclo ./FATFS/Target/user_diskio.d ./FATFS/Target/user_diskio.o ./FATFS/Target/user_diskio.su

.PHONY: clean-FATFS-2f-Target

