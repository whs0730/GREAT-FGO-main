// Copyright (c) 2023. Created on 10/31/23 12:49 PM by shlchen@whu.edu.cn (Shuolong Chen), who received the B.S. degree in
// geodesy and geomatics engineering from Wuhan University, Wuhan China, in 2023. He is currently a master candidate at
// the school of Geodesy and Geomatics, Wuhan University. His area of research currently focuses on integrated navigation
// systems and multi-sensor fusion.

#ifndef IMU_ACCE_FACTOR_HPP
#define IMU_ACCE_FACTOR_HPP

#include "gfactor/ct_imu_factor_common.hpp"
#include "gct/gspline/ceres_spline_helper.h"
#include <sophus/so3.hpp>
#include <typeinfo>

namespace gfgomsf {

// Position knots and gravity use the same reference frame; R maps body to it.
// The measured specific force is R^-1 (p'' - g) + ba, in m/s^2.
// Both variants use POSITION knots and their second derivative. A rotating
// reference frame such as ECEF needs consistent inertial corrections in the
// future navigation adapter/model; this factor keeps the teacher's basic model.
// weight is scalar square-root information (inverse measurement std deviation).
template <int Order>
struct IMUAcceFactor {
    static_assert(Order >= 3, "Accelerometer factors require a twice differentiable spline");
    using SplineMetaType = SplineMeta<Order>;

    IMUAcceFactor(const SplineMetaType& so3_meta, const SplineMetaType& pos_meta,
                  const SplineMetaType& ba_meta, const gct::CtImuSample& sample, double weight)
        : sample_(sample), weight_(weight) {
        ct_imu_detail::ValidateMeasurement(sample_, weight_);
        so3_ = ct_imu_detail::Locate(so3_meta, sample_.time_s);
        pos_ = ct_imu_detail::Locate(pos_meta, sample_.time_s);
        ba_ = ct_imu_detail::Locate(ba_meta, sample_.time_s);
        pos_offset_ = so3_meta.NumParameters() + pos_.offset;
        ba_offset_ = so3_meta.NumParameters() + pos_meta.NumParameters() + ba_.offset;
        gravity_offset_ = so3_meta.NumParameters() + pos_meta.NumParameters() + ba_meta.NumParameters();
    }

    // Blocks: SO3 knots [qx,qy,qz,qw], position knots, accel-bias knots, gravity.
    // Each block after SO3 is a 3-vector. The factory is fully configured.
    static auto Create(const SplineMetaType& so3_meta, const SplineMetaType& pos_meta,
                       const SplineMetaType& ba_meta, const gct::CtImuSample& sample, double weight) {
        std::vector<int> sizes(so3_meta.NumParameters(), 4);
        sizes.insert(sizes.end(), pos_meta.NumParameters() + ba_meta.NumParameters(), 3);
        sizes.push_back(3);
        return ct_imu_detail::MakeCost(new IMUAcceFactor(so3_meta, pos_meta, ba_meta, sample, weight), sizes);
    }

    static std::size_t TypeHashCode() { return typeid(IMUAcceFactor).hash_code(); }

    template <class T>
    bool operator()(T const* const* blocks, T* residuals) const {
        Sophus::SO3<T> rotation;
        CeresSplineHelper<Order>::template EvaluateLie<T, Sophus::SO3>(
            blocks + so3_.offset, so3_.u, so3_.inverse_dt, &rotation);
        Eigen::Matrix<T, 3, 1> acceleration, bias;
        CeresSplineHelper<Order>::template Evaluate<T, 3, 2>(
            blocks + pos_offset_, pos_.u, pos_.inverse_dt, &acceleration);
        CeresSplineHelper<Order>::template Evaluate<T, 3, 0>(
            blocks + ba_offset_, ba_.u, ba_.inverse_dt, &bias);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> gravity(blocks[gravity_offset_]);
        Eigen::Map<Eigen::Matrix<T, 3, 1>> error(residuals);
        error = T(weight_) * (rotation.inverse() * (acceleration - gravity) + bias
                             - sample_.specific_force_m_s2.template cast<T>());
        return true;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    gct::CtImuSample sample_;
    double weight_;
    ct_imu_detail::SplineLocation so3_{}, pos_{}, ba_{};
    std::size_t pos_offset_ = 0, ba_offset_ = 0, gravity_offset_ = 0;
};

template <int Order>
struct IMUAcceFactorWithConstBias {
    static_assert(Order >= 3, "Accelerometer factors require a twice differentiable spline");
    using SplineMetaType = SplineMeta<Order>;

    IMUAcceFactorWithConstBias(const SplineMetaType& so3_meta, const SplineMetaType& pos_meta,
                              const gct::CtImuSample& sample, double weight)
        : sample_(sample), weight_(weight) {
        ct_imu_detail::ValidateMeasurement(sample_, weight_);
        so3_ = ct_imu_detail::Locate(so3_meta, sample_.time_s);
        pos_ = ct_imu_detail::Locate(pos_meta, sample_.time_s);
        pos_offset_ = so3_meta.NumParameters() + pos_.offset;
        ba_offset_ = so3_meta.NumParameters() + pos_meta.NumParameters();
        gravity_offset_ = ba_offset_ + 1;
    }

    // Blocks: SO3 knots [qx,qy,qz,qw], position knots, one accel bias, gravity.
    static auto Create(const SplineMetaType& so3_meta, const SplineMetaType& pos_meta,
                       const gct::CtImuSample& sample, double weight) {
        std::vector<int> sizes(so3_meta.NumParameters(), 4);
        sizes.insert(sizes.end(), pos_meta.NumParameters(), 3);
        sizes.push_back(3);
        sizes.push_back(3);
        return ct_imu_detail::MakeCost(new IMUAcceFactorWithConstBias(so3_meta, pos_meta, sample, weight), sizes);
    }

    static std::size_t TypeHashCode() { return typeid(IMUAcceFactorWithConstBias).hash_code(); }

    template <class T>
    bool operator()(T const* const* blocks, T* residuals) const {
        Sophus::SO3<T> rotation;
        CeresSplineHelper<Order>::template EvaluateLie<T, Sophus::SO3>(
            blocks + so3_.offset, so3_.u, so3_.inverse_dt, &rotation);
        Eigen::Matrix<T, 3, 1> acceleration;
        CeresSplineHelper<Order>::template Evaluate<T, 3, 2>(
            blocks + pos_offset_, pos_.u, pos_.inverse_dt, &acceleration);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> bias(blocks[ba_offset_]);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> gravity(blocks[gravity_offset_]);
        Eigen::Map<Eigen::Matrix<T, 3, 1>> error(residuals);
        error = T(weight_) * (rotation.inverse() * (acceleration - gravity) + bias
                             - sample_.specific_force_m_s2.template cast<T>());
        return true;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    gct::CtImuSample sample_;
    double weight_;
    ct_imu_detail::SplineLocation so3_{}, pos_{};
    std::size_t pos_offset_ = 0, ba_offset_ = 0, gravity_offset_ = 0;
};

}  // namespace gfgomsf

#endif  // IMU_ACCE_FACTOR_HPP
