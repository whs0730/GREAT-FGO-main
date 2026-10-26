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
    // R 为机体系到 ECEF 的旋转，位置控制点为 IMU 相对固定 ECEF 原点的位置。
    template <int Order>
    class CtGnssPositionFactor
    {
        static_assert(Order >= 2, "CT GNSS position factors require a spline");

    private:
        /// GNSS 天线相对固定 ECEF 原点的位置（m）
        Eigen::Vector3d _gnss_position;
        /// ECEF 协方差满足 Sigma=L*L^T，平方根信息矩阵为 L^-1
        Eigen::Matrix3d _sqrt_info;
        /// 机体系下 IMU 指向天线的杆臂（m）
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
            : _lever_arm(lever_arm_body_m)
        {
            if (!std::isfinite(observation.time_s) || !observation.position_ecef_m.allFinite()
                || !observation.position_covariance_ecef_m2.allFinite()
                || !observation.position_covariance_ecef_m2.isApprox(
                    observation.position_covariance_ecef_m2.transpose(), 1e-12)
                || !_lever_arm.allFinite() || !position_origin_ecef_m.allFinite())
            {
                throw std::invalid_argument("CT GNSS factor requires a valid observation, finite lever arm and origin");
            }
            _gnss_position = observation.position_ecef_m - position_origin_ecef_m;
            if (!_gnss_position.allFinite()) 
            {
                throw std::invalid_argument("CT GNSS position minus origin must be finite");
            }
            _so3_location = ct_imu_detail::Locate(so3_meta, observation.time_s);
            _position_location = ct_imu_detail::Locate(position_meta, observation.time_s);
            _position_offset = so3_meta.NumParameters() + _position_location.offset;

            const Eigen::LLT<Eigen::Matrix3d> decomposition(observation.position_covariance_ecef_m2);
            if (decomposition.info() != Eigen::Success)
            {
                throw std::invalid_argument("CT GNSS covariance must be positive definite");
            }
            Eigen::Matrix3d identity = Eigen::Matrix3d::Identity();
            _sqrt_info = decomposition.matrixL().solve(identity);
            if (!_sqrt_info.allFinite()) 
            {
                throw std::invalid_argument("CT GNSS covariance produces nonfinite square-root information");
            }
        }

        // 参数块先放 SO3 控制点 [qx,qy,qz,qw]，再放三维相对位置控制点。
        static ceres::DynamicAutoDiffCostFunction<CtGnssPositionFactor<Order>>* Create(
                           const SplineMeta<Order>& so3_meta,
                           const SplineMeta<Order>& position_meta,
                           const gct::CtGnssPosition& observation,
                           const Eigen::Vector3d& lever_arm_body_m,
                           const Eigen::Vector3d& position_origin_ecef_m = Eigen::Vector3d::Zero())
        {
            std::vector<int> sizes(so3_meta.NumParameters(), 4);
            sizes.insert(sizes.end(), position_meta.NumParameters(), 3);

            CtGnssPositionFactor<Order>* factor = new CtGnssPositionFactor<Order>(so3_meta, position_meta, observation, lever_arm_body_m, position_origin_ecef_m);
            return ct_imu_detail::MakeCost(factor, sizes);
        }

        static std::size_t TypeHashCode()
        {
            return typeid(CtGnssPositionFactor<Order>).hash_code();
        }

        // 观测与位置样条使用同一原点；残差为 L^-1*(p(t)+R(t)*lever-z_GNSS)。
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
