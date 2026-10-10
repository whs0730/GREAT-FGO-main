#ifndef CT_GNSS_POSITION_FACTOR_H
#define CT_GNSS_POSITION_FACTOR_H
/**
 * @file         ct_gnss_position_factor.h
 * @author       GREAT-WHU (https://github.com/GREAT-WHU)
 * @brief        Continuous-time adaptation of the existing loosely coupled GNSS/RTK position factor.
 * @version      1.0
 * @date         2026-08
 *
 * @copyright Copyright (c) 2025, Wuhan University. All rights reserved.
 *
 */
#include "gfactor/ct_imu_factor_common.hpp"
#include "gct/gspline/ceres_spline_helper.h"
#include <Eigen/Cholesky>
#include <sophus/so3.hpp>
#include <typeinfo>

namespace gfgomsf
{
    // Based on GnssPositionFactor: antenna prediction and covariance weighting
    // are retained; the discrete pose parameter becomes SO3 and position knots.
    // R maps body to ECEF. Position knots use ECEF axes and are relative to the
    // fixed position_origin_ecef_m supplied when constructing the factor.
    template <int Order>
    class CtGnssPositionFactor
    {
        static_assert(Order >= 2, "CT GNSS position factors require a spline");

    public:
        using SplineMetaType = SplineMeta<Order>;

    private:
        Eigen::Vector3d _gnss_position;
        Eigen::Matrix3d _sqrt_info;
        Eigen::Vector3d _lever_arm;
        ct_imu_detail::SplineLocation _so3_location{}, _position_location{};
        std::size_t _position_offset = 0;

    public:
        CtGnssPositionFactor(const SplineMetaType& so3_meta,
                             const SplineMetaType& position_meta,
                             const gct::CtGnssPosition& observation,
                             const Eigen::Vector3d& lever_arm_body_m,
                             const Eigen::Vector3d& position_origin_ecef_m = Eigen::Vector3d::Zero()): _lever_arm(lever_arm_body_m)
        {
            if (!observation.IsValid() || !_lever_arm.allFinite()
                || !position_origin_ecef_m.allFinite()) 
            {
                throw std::invalid_argument("CT GNSS factor requires a valid observation, finite lever arm and origin");
            }
            // Subtract absolute ECEF coordinates once, in double precision,
            // before evaluating relative knots and automatic-differentiation Jets.
            _gnss_position = observation.position_ecef_m - position_origin_ecef_m;
            if (!_gnss_position.allFinite()) 
            {
                throw std::invalid_argument("CT GNSS position minus origin must be finite");
            }
            _so3_location = ct_imu_detail::Locate(so3_meta, observation.time_s);
            _position_location = ct_imu_detail::Locate(position_meta, observation.time_s);
            _position_offset = so3_meta.NumParameters() + _position_location.offset;

            // Diagonal covariance gives diag(1 / sqrt(Rpos)), as in the original
            // factor. Full covariance retains correlations: Sigma = L L^T,
            // and sqrt_info = L^-1. Invalid covariance is rejected by the CT type.
            const Eigen::LLT<Eigen::Matrix3d> decomposition(observation.position_covariance_ecef_m2);
            _sqrt_info = decomposition.matrixL().solve(Eigen::Matrix3d::Identity());
            if (!_sqrt_info.allFinite()) 
            {
                throw std::invalid_argument("CT GNSS covariance produces nonfinite square-root information");
            }
        }

        // Fully configured cost: SO3 blocks [qx,qy,qz,qw] first, followed by
        // 3-vector relative-position blocks. Do not add blocks to the cost again.
        static auto Create(const SplineMetaType& so3_meta,
                           const SplineMetaType& position_meta,
                           const gct::CtGnssPosition& observation,
                           const Eigen::Vector3d& lever_arm_body_m,
                           const Eigen::Vector3d& position_origin_ecef_m = Eigen::Vector3d::Zero())
        {
            std::vector<int> sizes(so3_meta.NumParameters(), 4);
            sizes.insert(sizes.end(), position_meta.NumParameters(), 3);
            return ct_imu_detail::MakeCost(new CtGnssPositionFactor(
                so3_meta, position_meta, observation, lever_arm_body_m, position_origin_ecef_m), sizes);
        }

        static std::size_t TypeHashCode() { return typeid(CtGnssPositionFactor).hash_code(); }

        // r = sqrt_info * [p_origin + p(t) + R(t) * lever - z_GNSS].
        // p_origin has already been subtracted from _gnss_position above.
        template <class T>
        bool operator()(T const* const* parameters, T* residuals) const
        {
            Sophus::SO3<T> Ri;
            CeresSplineHelper<Order>::template EvaluateLie<T, Sophus::SO3>(
                parameters + _so3_location.offset, _so3_location.u,
                _so3_location.inverse_dt, &Ri);
            Eigen::Matrix<T, 3, 1> Pi;
            CeresSplineHelper<Order>::template Evaluate<T, 3, 0>(
                parameters + _position_offset, _position_location.u,
                _position_location.inverse_dt, &Pi);

            const Eigen::Matrix<T, 3, 1> prediction_position =
                Pi + Ri * _lever_arm.template cast<T>();
            Eigen::Map<Eigen::Matrix<T, 3, 1>> residual(residuals);
            residual = _sqrt_info.template cast<T>()
                * (prediction_position - _gnss_position.template cast<T>());
            return true;
        }

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };
}  // namespace gfgomsf

#endif  // CT_GNSS_POSITION_FACTOR_HPP
