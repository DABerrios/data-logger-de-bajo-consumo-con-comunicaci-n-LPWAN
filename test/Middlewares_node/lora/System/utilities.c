/*!
 * \file      utilities.c
 *
 * \brief     Helper functions implementation (memcpy1, memset1, Crc32, etc.)
 *
 * \copyright Revised BSD License, see section \ref LICENSE.
 *
 * \code
 *                ______                              _
 *               / _____)             _              | |
 *              ( (____  _____ ____ _| |_ _____  ____| |__
 *               \____ \| ___ |    (_   _) ___ |/ ___)  _ \
 *               _____) ) ____| | | || |_| ____( (___| | | |
 *              (______/|_____)_|_|_| \__)_____)\____)_| |_|
 *              (C)2013-2017 Semtech
 *
 * \endcode
 */
#include <stdlib.h>
#include <stdint.h>
#include "utilities.h"

/*!
 * Redefinition of rand() and srand() standard C functions.
 * These functions are redefined in order to get the same behavior
 * across different compiler toolchains implementations.
 */
// Linear congruential generator
static uint32_t next = 1;

void srand1( uint32_t seed )
{
    next = seed;
}

int32_t randr( int32_t min, int32_t max )
{
    next = next * 1103515245 + 12345;
    return ( int32_t )min + ( ( int32_t )( next / 65536 ) % ( max - min + 1 ) );
}

void memcpy1( uint8_t *dst, const uint8_t *src, uint16_t size )
{
    while( size-- )
    {
        *dst++ = *src++;
    }
}

void memcpyr( uint8_t *dst, const uint8_t *src, uint16_t size )
{
    dst = dst + ( size - 1 );
    while( size-- )
    {
        *dst-- = *src++;
    }
}

void memset1( uint8_t *dst, uint8_t value, uint16_t size )
{
    while( size-- )
    {
        *dst++ = value;
    }
}

int8_t Nibble2HexChar( uint8_t a )
{
    if( a < 10 )
    {
        return '0' + a;
    }
    else if( a < 16 )
    {
        return 'A' + ( a - 10 );
    }
    else
    {
        return '?';
    }
}

/*
 * CRC32 CCITT implementation
 */
#define CRC_POLY 0x04C11DB7

uint32_t Crc32( uint8_t *buffer, uint16_t length )
{
    uint32_t crc = Crc32Init( );
    crc = Crc32Update( crc, buffer, length );
    crc = Crc32Finalize( crc );
    return crc;
}

uint32_t Crc32Init( void )
{
    return 0xFFFFFFFF;
}

uint32_t Crc32Update( uint32_t crcInit, uint8_t *buffer, uint16_t length )
{
    uint32_t crc = crcInit;

    for( uint16_t i = 0; i < length; i++ )
    {
        crc ^= ( uint32_t )buffer[i] << 24;
        for( uint8_t j = 0; j < 8; j++ )
        {
            if( crc & 0x80000000 )
            {
                crc = ( crc << 1 ) ^ CRC_POLY;
            }
            else
            {
                crc <<= 1;
            }
        }
    }
    return crc;
}

uint32_t Crc32Finalize( uint32_t crc )
{
    return crc ^ 0xFFFFFFFF;
}
