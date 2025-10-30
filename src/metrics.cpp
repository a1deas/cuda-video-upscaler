#include "metrics.hpp"
#include <cmath>

namespace upscaler {

static inline void accumRGB(
    const uint8_t* p, int n, double m[3], double v[3])
{
    for (int c = 0; c < 3; ++c){ m[c] = 0.0; v[c] = 0.0; }
    for (int i = 0;i < n; ++i){
        for (int c = 0; c < 3; ++c){
            double x = p[3 * i + c];
            m[c] += x;
            v[c] += x * x;
        }
    }
    for (int c = 0; c < 3; ++c){
        m[c] /= n;
        v[c] /= n;
        v[c] -= m[c] * m[c]; // variance
        if (v[c] < 0) v[c] = 0;
    }
}

double psnrRGB(const uint8_t* a, const uint8_t* b, int width, int height){
    const int N = width * height * 3;
    double mse = 0.0;
    for (int i = 0; i < N; ++i){
        double d = double(a[i]) - double(b[i]);
        mse += d * d;
    }
    mse /= N;
    if (mse <= 1e-12) return 1e9;
    const double MAXI = 255.0;
    return 10.0 * std::log10((MAXI * MAXI) / mse);
}

double ssimRGB(const uint8_t* a, const uint8_t* b, int width, int height){
    const int n = width * height;
    double ma[3], va[3], mb[3], vb[3], cab[3] = {0, 0, 0};
    accumRGB(a, n, ma, va);
    accumRGB(b, n, mb, vb);

    for (int i = 0; i < n; ++i){
        for (int c = 0; c < 3; ++c){
            double da = double(a[3 * i + c]) - ma[c];
            double db = double(b[3 * i + c]) - mb[c];
            cab[c] += da * db;
        }
    }
    for (int c = 0; c < 3; ++c) cab[c] /= n;

    const double C1 = (0.01 * 255) * (0.01 * 255);
    const double C2 = (0.03 * 255) * (0.03 * 255);

    double ssimC[3];
    for (int c = 0; c < 3; ++c){
        double num = (2 * ma[c] * mb[c] + C1) * (2 * cab[c] + C2);
        double den = (ma[c] * ma[c] + mb[c] * mb[c] + C1) * (va[c] + vb[c] + C2);
        ssimC[c] = den > 0 ? num / den : 1.0;
    }
    return (ssimC[0] + ssimC[1] + ssimC[2]) / 3.0;
}

} // namespace upscaler
