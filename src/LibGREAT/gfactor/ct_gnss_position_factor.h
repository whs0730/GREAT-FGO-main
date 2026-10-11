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
    // GNSS position factor using SO3 and position spline control points.
    // R maps body to ECEF; position knots are relative to a fixed ECEF origin.
    template <int Order>
    class CtGnssPositionFactor
    {
        static_assert(Order >= 2, "CT GNSS position factors require a spline");

    private:
        /// GNSS antenna position relative to the fixed ECEF origin, in metres.
        Eigen::Vector3d _gnss_position;
        /// Position square-root information matrix.
        Eigen::Matrix3d _sqrt_info;
        /// IMU-to-GNSS lever arm in the body frame, in metres.
        Eigen::Vector3d _lever_arm;
        ct_imu_detail::SplineLocation _so3_location;
        ct_imu_detail::SplineLocation _position_location;
        std::size_t _position_offset;

    public:
        CtGnssPositionFactor(const SplineMeta<Order>& so3_meta,
                             const SplineMeta<Order>& position_meta,
                             const gct::CtGnssPosition& observation,
                             const Eigen::Vector3d& lever_arm_body_m,
                             const Eigen::Vector3d& position_origin_ecef_m = Eigen::Vector3d::Zero())
            : _lever_arm(lever_arm_body_m), _position_offset(0)
        {
            if (!observation.IsValid() || !_lever_arm.allFinite()
                || !position_origin_ecef_m.allFinite()) 
            {
                throw std::invalid_argument("CT GNSS factor requires a valid observation, finite lever arm and origin");
            }
            // Use the same fixed ECEF origin as the position control points.
            _gnss_position = observation.position_ecef_m - position_origin_ecef_m;
            if (!_gnss_position.allFinite()) 
            {
                throw std::invalid_argument("CT GNSS position minus origin must be finite");
            }
            _so3_location = ct_imu_detail::Locate(so3_meta, observation.time_s);
            _position_location = ct_imu_detail::Locate(position_meta, observation.time_s);
            _position_offset = so3_meta.NumParameters() + _position_location.offset;

            // Sigma = L * L^T; sqrt_info = L^-1.
            const Eigen::LLT<Eigen::Matrix3d> decomposition(observation.position_covariance_ecef_m2);
            Eigen::Matrix3d identity = Eigen::Matrix3d::Identity();
            _sqrt_info = decomposition.matrixL().solve(identity);
            if (!_sqrt_info.allFinite()) 
            {
                throw std::invalid_argument("CT GNSS covariance produces nonfinite square-root information");
            }
        }

        // Fully configured cost: SO3 blocks [qx,qy,qz,qw] first, followed by
        // 3-vector relative-position blocks. Do not add blocks to the cost again.
        static ceres::DynamicAutoDiffCostFunction<CtGnssPositionFactor<Order>>* Create(
                           const SplineMeta<Order>& so3_meta,
                           const SplineMeta<Order>& position_meta,
                           const gct::CtGnssPosition& observation,
                           const Eigen::Vector3d& lever_arm_body_m,
                           const Eigen::Vector3d& position_origin_ecef_m = Eigen::Vector3d::Zero())
        {
            std::vector<int> sizes;
            std::size_t so3_count = so3_meta.NumParameters();
            std::size_t position_count = position_meta.NumParameters();
            for (std::size_t i = 0; i < so3_count; i++)
            {
                sizes.push_back(4);
            }
            for (std::size_t i = 0; i < position_count; i++)
            {
                sizes.push_back(3);
            }

            CtGnssPositionFactor<Order>* factor = new CtGnssPositionFactor<Order>(so3_meta, position_meta, observation, lever_arm_body_m, position_origin_ecef_m);
            return ct_imu_detail::MakeCost(factor, sizes);
        }

        static std::size_t TypeHashCode()
        {
            return typeid(CtGnssPositionFactor<Order>).hash_code();
        }

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

            Eigen::Matrix<T, 3, 1> lever_arm = _lever_arm.template cast<T>();
            Eigen::Matrix<T, 3, 1> gnss_position = _gnss_position.template cast<T>();
            Eigen::Matrix<T, 3, 3> sqrt_info = _sqrt_info.template cast<T>();

            Eigen::Matrix<T, 3, 1> prediction_position = Pi + Ri * lever_arm;
            Eigen::Matrix<T, 3, 1> position_error = prediction_position - gnss_position;
            Eigen::Map<Eigen::Matrix<T, 3, 1>> residual(residuals);
            residual = sqrt_info * position_error;
            return true;
        }

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };
}  // namespace gfgomsf

#endif 
