#ifndef GNSS_POSITION_FACTOR_H
#define GNSS_POSITION_FACTOR_H
/**
 * @file         gimu_factor.h
 * @author       GREAT-WHU (https://github.com/GREAT-WHU)
 * @brief        Construction of imu factors for factor graph optimization.
 * @version      1.0
 * @date         2026-08
 *
 * @copyright Copyright (c) 2025, Wuhan University. All rights reserved.
 *
 */
#include"gfgo/gutility.h"
namespace gfgo
{
	class LibGREAT_LIBRARY_EXPORT GnssPositionFactor : public ceres::SizedCostFunction<3, 7> 
	{
    private:

        /// GNSS antenna position in ECEF
        Eigen::Vector3d _gnss_position;

        /// Position square-root information matrix
        Eigen::Matrix3d _sqrt_info;

        /// IMU-GNSS lever arm in body frame
        Eigen::Vector3d _lever_arm;
	public:
        GnssPositionFactor(const Eigen::Vector3d& gnss_position, const Eigen::Vector3d& position_variance, const Eigen::Vector3d& lever_arm) 
            :_gnss_position(gnss_position),_lever_arm(lever_arm)
        {
            _sqrt_info.setZero();

            for (int i = 0; i < 3; i++) 
            {
                double variance = position_variance[i];
                if (!std::isfinite(variance) || variance <= 0.0) {
                    variance = 1.0;
                }
                _sqrt_info(i,i) = 1.0 / std::sqrt(variance);
            }

        }


        virtual bool Evaluate(double const*const* parameters, double* residuals, double** jacobians)const override
        {
            Eigen::Vector3d Pi(
                parameters[0][0],
                parameters[0][1],
                parameters[0][2]
            );
            Eigen::Quaterniond Qi(
                parameters[0][6],
                parameters[0][3],
                parameters[0][4],
                parameters[0][5]

            );

            Eigen::Matrix3d Ri = Qi.toRotationMatrix();

            Eigen::Vector3d prediction_position = Pi + Ri * _lever_arm;

            Eigen::Map<Eigen::Vector3d> residual(residuals);

            residual = _sqrt_info * (prediction_position - _gnss_position);

            if (jacobians) {
                if (jacobians[0]) {
                    Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>>jacobian_pose(jacobians[0]);

                    jacobian_pose.setZero();

                    jacobian_pose.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();

                    jacobian_pose.block<3, 3>(0, 3)
                        = -Ri *
                        t_gfgo_utility::skewSymmetric(_lever_arm);

                    jacobian_pose.col(6).setZero();

                    jacobian_pose = _sqrt_info * jacobian_pose;

                }
            }

            return true;
        }

	};
}
#endif