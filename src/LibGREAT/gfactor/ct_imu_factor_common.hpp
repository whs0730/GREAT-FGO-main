#ifndef GREAT_CT_IMU_FACTOR_COMMON_HPP
#define GREAT_CT_IMU_FACTOR_COMMON_HPP

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef GLOG_NO_ABBREVIATED_SEVERITIES
#define GLOG_NO_ABBREVIATED_SEVERITIES
#endif
#endif

#include "gct/ct_types.h"
#include "gct/gspline/spline_segment.h"
#include <ceres/dynamic_autodiff_cost_function.h>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

namespace gfgomsf
{
namespace ct_imu_detail
{

struct SplineLocation
{
    std::size_t offset;
    double u;
    double inverse_dt;
};

// 元数据须覆盖观测时刻所需的 Order 个相邻控制点，offset 按前序段的参数块数累计。
template <int Order>
SplineLocation Locate(const SplineMeta<Order>& meta, double time_s)
{
    if (meta.segments.empty() || !std::isfinite(time_s))
    {
        throw std::invalid_argument("CT factor requires nonempty spline metadata and finite time");
    }
    for (std::size_t i = 0; i < meta.segments.size(); i++)
    {
        const SplineSegmentMeta<Order>& segment = meta.segments[i];
        if (!std::isfinite(segment.t0) || !std::isfinite(segment.dt)
            || segment.dt <= 0.0 || segment.n < static_cast<std::size_t>(Order)
            || !std::isfinite(segment.MaxTime()))
        {
            throw std::invalid_argument("CT factor received invalid spline metadata");
        }
    }
    std::size_t offset = 0;
    for (std::size_t i = 0; i < meta.segments.size(); i++)
    {
        const SplineSegmentMeta<Order>& segment = meta.segments[i];
        if (time_s >= segment.MinTime() && time_s < segment.MaxTime())
        {
            double u = 0.0;
            std::size_t index = 0;
            if (segment.ComputeTIndex(time_s, u, index)
                && index + Order <= segment.n)
            {
                SplineLocation location;
                location.offset = offset + index;
                location.u = u;
                location.inverse_dt = 1.0 / segment.dt;
                return location;
            }
        }
        offset += segment.NumParameters();
    }
    throw std::out_of_range("CT timestamp is outside the supplied spline metadata");
}

inline void ValidateMeasurement(const gct::CtImuSample& sample, double weight)
{
    if (!sample.IsValid())
    {
        throw std::invalid_argument("CT IMU factor received a nonfinite sample");
    }
    if (!std::isfinite(weight) || weight <= 0.0)
    {
        throw std::invalid_argument("CT IMU residual weight must be finite and positive");
    }
}

// 返回的代价函数包含参数块定义和三个残差，AddResidualBlock 后由 Ceres 默认接管所有权。
template <class Functor>
ceres::DynamicAutoDiffCostFunction<Functor>* MakeCost(
    Functor* functor, const std::vector<int>& block_sizes)
{
    // 参数块配置完成前保持所有权，构造失败时释放代价函数及其仿函数。
    std::unique_ptr<ceres::DynamicAutoDiffCostFunction<Functor>> cost(
        new ceres::DynamicAutoDiffCostFunction<Functor>(functor));

    for (std::size_t i = 0; i < block_sizes.size(); i++)
    {
        cost->AddParameterBlock(block_sizes[i]);
    }
    cost->SetNumResiduals(3);
    return cost.release();
}

}  // namespace ct_imu_detail
}  // namespace gfgomsf

#endif  // GREAT_CT_IMU_FACTOR_COMMON_HPP
