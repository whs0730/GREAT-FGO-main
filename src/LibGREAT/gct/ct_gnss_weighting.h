#ifndef GREAT_GCT_CT_GNSS_WEIGHTING_H
#define GREAT_GCT_CT_GNSS_WEIGHTING_H

#include <algorithm>
#include <cmath>

namespace gfgomsf
{
namespace gct
{

// 输入由后端检查：q>=0、previous_scale>=1、lambda>=0、maximum_scale>=1，且均有限。
// 在 s=log(a) 上最小化 q*exp(-s)+3*s+lambda*(s-log(a_prev))^2。
inline double EstimateGnssCovarianceScale(double q, double previous_scale,
    double lambda, double maximum_scale)
{
    double previous = std::log(previous_scale);
    double lower = 0.0;
    double upper = std::log(maximum_scale);
    double derivative_lower = -q + 3.0 - 2.0 * lambda * previous;
    if (derivative_lower >= 0.0 || maximum_scale == 1.0)
    {
        return 1.0;
    }
    double derivative_upper = -q / maximum_scale + 3.0 + 2.0 * lambda * (upper - previous);
    if (derivative_upper <= 0.0)
    {
        return maximum_scale;
    }
    // 对 s 的二阶导数为 q*exp(-s)+2*lambda；内部驻点唯一。
    for (int i = 0; i < 60; i++)
    {
        double middle = 0.5 * (lower + upper);
        double derivative = -q * std::exp(-middle) + 3.0 + 2.0 * lambda * (middle - previous);
        if (derivative > 0.0)
        {
            upper = middle;
        }
        else
        {
            lower = middle;
        }
    }
    return std::min(maximum_scale, std::exp(0.5 * (lower + upper)));
}

}  // namespace gct
}  // namespace gfgomsf

#endif
