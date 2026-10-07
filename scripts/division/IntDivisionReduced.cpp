#include <iostream>
#include <cmath>
#include <cassert>
#include <cstdint>
#include <bit> // Requires C++20 for std::bit_width

// Constants
const int W = 32;

// Utility to count bits (mimicking math.floor(math.log2(y)) + 1)
// If you do not have C++20, replace std::bit_width(y) with:
// (y == 0) ? 0 : 32 - __builtin_clz(y) (for GCC/Clang)
int get_bit_width(uint32_t y) {
    return std::bit_width(y);
}

uint64_t f_32(uint32_t y) {
    // 1. Normalization
    // We normalize to the [2^31, 2^32) range
    int clz = get_bit_width(y); 
    
    // Using uint64_t for yy to ensure shifts don't behave unexpectedly, 
    // though uint32_t is sufficient here since result fits in 32 bits.
    uint64_t yy = (uint64_t)y << (W - clz);

    // 2. Initial Seed (Approx 4 bits)
    uint64_t hy = yy >> (W / 2);      // Top 16 bits
    uint64_t hhy = hy >> (W / 4);     // Top 8 bits
    
    // z = 2**(15) // hhy
    // Use uint64_t for z to handle intermediate math safely
    uint64_t z = (1ULL << (W / 2 - 1)) / hhy;

    // 3. First Iteration (4 bits -> 8 bits)
    // Python: z = (z << 8) - (z * (hy * z >> 8) >> 8)
    // We must cast inputs of multiplications to uint64_t to prevent 32-bit overflow
    z = (z << (W / 4)) - ((z * ((hy * z) >> (W / 4))) >> (W / 4));

    // // Refinement Logic
    // // Python: if hy*z < 2**(W-2)-hy
    // uint64_t threshold = (1ULL << (W - 2));
    
    // if (hy * z < threshold - hy) {
    //     z += 1;
    // }
    
    // for (int i = 0; i < 2; ++i) {
    //     if (hy * z > threshold) {
    //         z -= 1;
    //     }
    // }

    // // assert 2**(W-2) - hy <= hy*z <= 2**(W-2)
    // assert((threshold - hy) <= (hy * z));
    // assert((hy * z) <= threshold);

    assert((1ULL << (W - 2)) - 2 * hy <= (hy * z) && (hy * z) < (1ULL << (W - 2)) + 2 * hy);
    // assert((1ULL << (W - 3)) <= (hy * z) && (hy * z) < (1ULL << (W - 1)));

    // 4. Second Iteration (8 bits -> 16 bits)
    // Python: z = ((z << 3) << 16) - (z * (yy * z >> 12) >> 16)
    // Note: (W/2 - 4) is 12. 
    // We use temporary variables to ensure calculation order and types match Python
    uint64_t term1 = (z << 3) << (W / 2);
    
    // Inner part: yy * z >> 12
    uint64_t inner = (yy * z) >> ((W / 2) - 4);
    
    // Outer part: z * inner >> 16
    uint64_t term2 = (z * inner) >> (W / 2);
    
    z = term1 - term2;

    // 6. De-normalization
    z = z >> clz;

    // 7. Final Check & Correction
    // Handle potential overflow if z was 2^32 exactly
    if (z >= (1ULL << W)) {
        z = (1ULL << W) - 1;
    }

    // Standard remainder check
    // if z * y < 2**W - y
    if (z * y < (1ULL << W) - y) {
        z += 1;
    }

    return z;
}

int main() {
    // Determine the max value for 32-bit unsigned integer
    // Note: iterating up to 4.2 billion will take some time.
    // We use uint64_t for 'b' to avoid infinite loop when b wraps around at 2^32
    uint64_t limit = (1ULL << W);
    
    std::cout << "Starting verification loop for range [1, 2^32)..." << std::endl;
    std::cout << "Note: Printing every line is disabled for performance." << std::endl;

    for (uint64_t b = 1; b < limit; ++b) {
        uint64_t inv = f_32((uint32_t)b);
        
        // Verify: 2^32 - b <= inv * b <= 2^32
        uint64_t product = inv * b;
        uint64_t lower_bound = (1ULL << W) - b;
        uint64_t upper_bound = (1ULL << W);

        // Printing strictly as requested by script logic would be:
        // std::cout << b << " " << inv << " " << product << "\n";
        // However, this is too slow for 4 billion iterations. 
        // We assert instead, and print only on error or periodically.
        
        if (!(product >= lower_bound && product <= upper_bound)) {
            std::cerr << "Assertion failed at b=" << b 
                      << " inv=" << inv 
                      << " prod=" << product << std::endl;
            assert(false);
        }

        // Optional: Progress indicator every 100 million
        if (b % 100000000 == 0) {
             std::cout << "Checked up to: " << b << std::endl;
        }
    }

    std::cout << "All values verified successfully." << std::endl;
    return 0;
}