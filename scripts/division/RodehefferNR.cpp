#include <iostream>
#include <cmath>
#include <cstdint>

constexpr uint W = 32;
uint numiter = 3;
constexpr uint iterlimit = 10;

int InvMicrosoftNR(uint32_t y) {
    uint clz = W - static_cast<uint>(floor(log2(y))) - 1;
    uint32_t z = 1U << clz;
    uint32_t my = (1ULL << W) - y;
    for (int i = 0; i < numiter; i++) {
        z += ((z * (static_cast<uint64_t>(z) * my % (1ULL << W))) >> W) % (1ULL << W);
    }
    return z;
}

std::pair<uint32_t, uint32_t> DivWithInv(uint32_t x, uint32_t y, uint32_t z) {
    uint32_t q = (static_cast<uint64_t>(x) * z) >> W;
    uint32_t r = x - y * q;
    for (int i = 0; i < 2; i++) {
        if (r >= y) {
            r -= y;
            q += 1;
        }
    }
    return std::make_pair(q, r);
}

int main() {
    // #pragma omp parallel for shared(numiter)
    for (uint32_t b = 1; b < (1ULL << W); b++) {
        int inv = InvMicrosoftNR(b);
        std::cout << "START: " << b << " " << numiter << std::endl;
        for (uint32_t a = 0; a < (1ULL << W); a++) {
            // std::cout << a << " " << b << " " << numiter << std::endl;
	    std::pair<uint32_t, uint32_t> tres = {a / b, a % b};
	    std::pair<uint32_t, uint32_t> res = DivWithInv(a, b, inv);
            while (tres != res) {
                std::cout << "ERROR" << std::endl;
                std::cout << tres.first << " " << tres.second << std::endl;
                std::cout << res.first << " " << res.second << std::endl;
                if (numiter < iterlimit) {
		    #pragma omp atomic
		    numiter += 1;
                    inv = InvMicrosoftNR(b);
                    res = DivWithInv(a, b, inv);
                } else {
                    exit(1);
                }
            }
        }
        std::cout << "END: " << b << " " << numiter << std::endl;
    }
    std::cout << numiter << std::endl;
    return 0;
}
