#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdio.h>
#include <assert.h>
#include <string.h>

#define _DEBUG

#define __FILENAME__                                                           \
    (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define p(fmt, args...) fprintf(stdout, fmt, ##args)
#define lp(fmt, args...)                                                       \
    p("[ %-10.10s | %-15.15s | %3d ] ", __FILENAME__, __FUNCTION__, __LINE__);        \
    p(fmt, ##args)

#if defined(_DEBUG)

#define dp(fmt, args...) fprintf(stderr, fmt, ##args)
#define dlp(fmt, args...)                                                      \
    dp("[ %-15.15s | %-20.20s | %3d ] ", __FILENAME__, __FUNCTION__, __LINE__);       \
    dp(fmt, ##args)

#else

#define dlp(fmt, args...)
#define dp(fmt, args...)

#endif

#define debug dlp("check for debug!!\n")
#define not_reachable assert(0)
#define elp(fmt, args...)                                                      \
    {                                                                          \
        fprintf(stderr, "[Error at %-10s %-15s %3d] " fmt, __FILENAME__,      \
                __FUNCTION__, __LINE__, ##args);                               \
        not_reachable;                                                             \
    }

#endif /* ifndef _COMMON_H_ */
