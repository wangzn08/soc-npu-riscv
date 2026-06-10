// Soft-float libgcc stubs for PicoRV32
// Provides division/multiplication routines needed by the firmware
#pragma GCC diagnostic ignored "-Wmissing-prototypes"

// 64-bit logical shift right
unsigned long long __lshrdi3(unsigned long long a, unsigned int b) {
    if (b >= 64) return 0;
    unsigned int hi = (unsigned int)(a >> 32);
    unsigned int lo = (unsigned int)a;
    if (b >= 32) { hi = 0; lo = (unsigned int)(a >> 32) >> (b - 32); }
    else { hi >>= b; lo = (lo >> b) | ((unsigned int)(a >> 32) << (32 - b)); }
    return ((unsigned long long)hi << 32) | lo;
}

// 64-bit arithmetic shift left
unsigned long long __ashldi3(unsigned long long a, unsigned int b) {
    if (b >= 64) return 0;
    unsigned int hi = (unsigned int)(a >> 32);
    unsigned int lo = (unsigned int)a;
    if (b >= 32) { lo = 0; hi = (unsigned int)a << (b - 32); }
    else { lo <<= b; hi = (hi << b) | ((unsigned int)a >> (32 - b)); }
    return ((unsigned long long)hi << 32) | lo;
}

// Unsigned 32-bit division: quotient
unsigned int __udivsi3(unsigned int a, unsigned int b) {
    if (b == 0) return 0;
    unsigned int q = 0, r = 0;
    for (int i = 31; i >= 0; i--) {
        r = (r << 1) | ((a >> i) & 1);
        if (r >= b) { r -= b; q |= (1u << i); }
    }
    return q;
}

// Signed 32-bit division: quotient
int __divsi3(int a, int b) {
    int neg = (a < 0) ^ (b < 0);
    unsigned int ua = (a < 0) ? -a : a;
    unsigned int ub = (b < 0) ? -b : b;
    int q = (int)__udivsi3(ua, ub);
    return neg ? -q : q;
}

// Unsigned 32-bit division: remainder
unsigned int __umodsi3(unsigned int a, unsigned int b) {
    if (b == 0) return 0;
    unsigned int r = 0;
    for (int i = 31; i >= 0; i--) {
        r = (r << 1) | ((a >> i) & 1);
        if (r >= b) r -= b;
    }
    return r;
}

// Signed 32-bit division: remainder
int __modsi3(int a, int b) {
    unsigned int ua = (a < 0) ? -a : a;
    unsigned int ub = (b < 0) ? -b : b;
    unsigned int r = __umodsi3(ua, ub);
    return (a < 0) ? -(int)r : (int)r;
}

// Unsigned 64-bit division: quotient
unsigned long long __udivdi3(unsigned long long a, unsigned long long b) {
    if (b == 0) return 0;
    unsigned long long q = 0, r = 0;
    for (int i = 63; i >= 0; i--) {
        r = (r << 1) | ((a >> i) & 1);
        if (r >= b) { r -= b; q |= (1ull << i); }
    }
    return q;
}

// Signed 64-bit division: quotient
long long __divdi3(long long a, long long b) {
    int neg = (a < 0) ^ (b < 0);
    unsigned long long ua = (a < 0) ? -a : a;
    unsigned long long ub = (b < 0) ? -b : b;
    long long q = (long long)__udivdi3(ua, ub);
    return neg ? -q : q;
}

// Unsigned 64-bit division: remainder
unsigned long long __umoddi3(unsigned long long a, unsigned long long b) {
    if (b == 0) return 0;
    unsigned long long r = 0;
    for (int i = 63; i >= 0; i--) {
        r = (r << 1) | ((a >> i) & 1);
        if (r >= b) r -= b;
    }
    return r;
}

// Signed 64-bit division: remainder
long long __moddi3(long long a, long long b) {
    unsigned long long ua = (a < 0) ? -a : a;
    unsigned long long ub = (b < 0) ? -b : b;
    unsigned long long r = __umoddi3(ua, ub);
    return (a < 0) ? -(long long)r : (long long)r;
}

// 32-bit multiply: result = a * b
unsigned int __mulsi3(unsigned int a, unsigned int b) {
    unsigned int result = 0;
    while (b) {
        if (b & 1) result += a;
        a <<= 1;
        b >>= 1;
    }
    return result;
}

// 64-bit multiply: result = a * b
// Split into 16-bit chunks to avoid needing hardware multiply
long long __muldi3(long long a, long long b) {
    int neg = 0;
    if (a < 0) { a = -a; neg = !neg; }
    if (b < 0) { b = -b; neg = !neg; }
    unsigned long long ua = (unsigned long long)a;
    unsigned long long ub = (unsigned long long)b;
    unsigned long long result = 0;
    while (ub) {
        if (ub & 1) result += ua;
        ua <<= 1;
        ub >>= 1;
    }
    return neg ? -(long long)result : (long long)result;
}

// String functions (no libc available)
void *memcpy(void *dst, const void *src, unsigned int n) {
    unsigned char *d = (unsigned char *)dst;
    const unsigned char *s = (const unsigned char *)src;
    while (n--) *d++ = *s++;
    return dst;
}

void *memset(void *dst, int c, unsigned int n) {
    unsigned char *d = (unsigned char *)dst;
    while (n--) *d++ = (unsigned char)c;
    return dst;
}
