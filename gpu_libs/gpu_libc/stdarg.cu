#include "stdarg.cuh"

#include "assert.cuh"
#include "mp4_printf.cuh"

__device__ int __gpu_vfprintf(__gpu_FILE *stream, const char *format, va_list ap) {
    int bufSize = 256;
    char* buffer = (char*) malloc(bufSize);

    int writtenSymbols = -1;
    while (1) {
        writtenSymbols = vsnprintf_(buffer, bufSize, format, ap);
        if (writtenSymbols < bufSize) {
            break;
        }
        bufSize *= 2;
        free(buffer);
        buffer = (char*) malloc(bufSize);
    }
    
    // TODO redirect to real file
    // currently it redirects everything on standard output
    if (writtenSymbols > 0) {
        printf(buffer);
    }
    
    free(buffer);
    
    return writtenSymbols;
    
    
//     HostDeviceMessage* msg = gHostDeviceCommunicator.currentThreadMemoryPosition();
//     
//     
//     
//     while (msg->mIsOwnedByHost) {}
//     
//     msg->mMessageType = HostDeviceMessage::FPRINTF;
// 
//     strcpy(msg->mMemory);
//     
//     __threadfence_system();
//     msg->mIsOwnedByHost = 0;
//     
//     return 0;
}
