// Copyright (c) 2023. Created on 10/31/23 12:49 PM by shlchen@whu.edu.cn (Shuolong Chen), who received the B.S. degree in
// geodesy and geomatics engineering from Wuhan University, Wuhan China, in 2023. He is currently a master candidate at
// the school of Geodesy and Geomatics, Wuhan University. His area of research currently focuses on integrated navigation
// systems and multi-sensor fusion.

#ifndef IMU_GYRO_FACTOR_HPP
#define IMU_GYRO_FACTOR_HPP

#include "gfactor/ct_imu_factor_common.hpp"
#include "gct/gspline/ceres_spline_helper.h"
#include <sophus/so3.hpp>
#include <typeinfo>

namespace gfgomsf {

// R maps body vectors to the reference frame. VelocityBody is angular rate
// expressed in the body frame. Bias and measurements are in rad/s.
// weight is scalar square-root information (inverse measurement std deviation).
template <int Order>
struct IMUGyroFactor {
    static_assert(Order >= 2, "Gyroscope factors require a differentiable spline");
    using SplineMetaType = SplineMeta<Order>;

    IMUGyroFactor(const SplineMetaType& so3_meta, const SplineMetaType& bg_meta,
                  const gct::CtImuSample& sample, double weight)
        : sample_(sample), weight_(weight) {
        ct_imu_detail::ValidateMeasurement(sample_, weight_);
        so3_ = ct_imu_detail::Locate(so3_meta, sample_.time_s);
        bg_ = ct_imu_detail::Locate(bg_meta, sample_.time_s);
        bg_offset_ = so3_meta.NumParameters() + bg_.offset;
    }

    // Blocks: all SO3 knots [qx,qy,qz,qw], then all 3-vector gyro-bias knots.
    static auto Create(const SplineMetaType& so3_meta, const SplineMetaType& bg_meta,
                       const gct::CtImuSample& sample, double weight) {
        std::vector<int> sizes(so3_meta.NumParameters(), 4);
        sizes.insert(sizes.end(), bg_meta.NumParameters(), 3);
        return ct_imu_detail::MakeCost(new IMUGyroFactor(so3_meta, bg_meta, sample, weight), sizes);
    }

    static std::size_t TypeHashCode() { return typeid(IMUGyroFactor).hash_code(); }

    template <class T>
    bool operator()(T const* const* blocks, T* residuals) const {
        typename Sophus::SO3<T>::Tangent omega;
        CeresSplineHelper<Order>::template EvaluateLie<T, Sophus::SO3>(
            blocks + so3_.offset, so3_.u, so3_.inverse_dt, nullptr, &omega);
        Eigen::Matrix<T, 3, 1> bias;
        CeresSplineHelper<Order>::template Evaluate<T, 3, 0>(
            blocks + bg_offset_, bg_.u, bg_.inverse_dt, &bias);
        Eigen::Map<Eigen::Matrix<T, 3, 1>> error(residuals);
        error = T(weight_) * (omega + bias - sample_.angular_rate_rad_s.template cast<T>());
        return true;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    // Snapshot the measurement so later buffer changes cannot change this factor.
    gct::CtImuSample sample_;
    double weight_;
    ct_imu_detail::SplineLocation so3_{}, bg_{};
    std::size_t bg_offset_ = 0;
};

template <int Order>
struct IMUGyroFactorWithConstBias {
    static_assert(Order >= 2, "Gyroscope factors require a differentiable spline");
    using SplineMetaType = SplineMeta<Order>;

    IMUGyroFactorWithConstBias(const SplineMetaType& so3_meta,
                             const gct::CtImuSample& sample, double weight)
        : sample_(sample), weight_(weight), bg_offset_(so3_meta.NumParameters()) {
        ct_imu_detail::ValidateMeasurement(sample_, weight_);
        so3_ = ct_imu_detail::Locate(so3_meta, sample_.time_s);
    }

    // Blocks: all SO3 knots [qx,qy,qz,qw], then one 3-vector gyro bias.
    static auto Create(const SplineMetaType& so3_meta, const gct::CtImuSample& sample, double weight) {
        std::vector<int> sizes(so3_meta.NumParameters(), 4);
        sizes.push_back(3);
        return ct_imu_detail::MakeCost(new IMUGyroFactorWithConstBias(so3_meta, sample, weight), sizes);
    }

    static std::size_t TypeHashCode() { return typeid(IMUGyroFactorWithConstBias).hash_code(); }

    template <class T>
    bool operator()(T const* const* blocks, T* residuals) const {
        typename Sophus::SO3<T>::Tangent omega;
        CeresSplineHelper<Order>::template EvaluateLie<T, Sophus::SO3>(
            blocks + so3_.offset, so3_.u, so3_.inverse_dt, nullptr, &omega);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> bias(blocks[bg_offset_]);
        Eigen::Map<Eigen::Matrix<T, 3, 1>> error(residuals);
        error = T(weight_) * (omega + bias - sample_.angular_rate_rad_s.template cast<T>());
        return true;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    gct::CtImuSample sample_;
    double weight_;
    ct_imu_detail::SplineLocation so3_{};
    std::size_t bg_offset_;
};

}  // namespace gfgomsf

#endif  // IMU_GYRO_FACTOR_HPP
