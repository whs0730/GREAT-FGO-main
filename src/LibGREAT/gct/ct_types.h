#ifndef GREAT_GCT_CT_TYPES_H
#define GREAT_GCT_CT_TYPES_H
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace gfgomsf {
namespace gct {

// Body-frame angular rate (rad/s) and specific force (m/s^2).
// Timestamps share a continuous seconds axis with GNSS and splines, without
// GPS-week wrap. Calibration and axis conversion belong to the caller.
struct CtImuSample {
    double time_s = std::numeric_limits<double>::quiet_NaN();
    Eigen::Vector3d angular_rate_rad_s = Eigen::Vector3d::Zero();
    Eigen::Vector3d specific_force_m_s2 = Eigen::Vector3d::Zero();

    bool IsValid() const {
        return std::isfinite(time_s) && angular_rate_rad_s.allFinite()
            && specific_force_m_s2.allFinite();
    }

    static CtImuSample FromRates(double timestamp_s,
                                const Eigen::Vector3d& angular_rate,
                                const Eigen::Vector3d& specific_force) 
    {
        CtImuSample sample;
        sample.time_s = timestamp_s;
        sample.angular_rate_rad_s = angular_rate;
        sample.specific_force_m_s2 = specific_force;
        if (!sample.IsValid()) 
        {
            throw std::invalid_argument("CT IMU sample must contain finite SI measurements and time");
        }
        return sample;
    }

    // interval_s must span these increments; axes and timestamps are not converted.
    static CtImuSample FromIncrements(double timestamp_s, double interval_s,
                                     const Eigen::Vector3d& delta_angle_rad,
                                     const Eigen::Vector3d& delta_velocity_m_s) 
    {
        if (!std::isfinite(interval_s) || interval_s <= 0.0) 
        {
            throw std::invalid_argument("CT IMU integration interval must be finite and positive");
        }
        return FromRates(timestamp_s, delta_angle_rad / interval_s,
                         delta_velocity_m_s / interval_s);
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// Absolute GNSS antenna position (m) and covariance (m^2), both in ECEF.
// Timestamps use the same continuous seconds as IMU samples and spline knots.
struct CtGnssPosition {
    double time_s = std::numeric_limits<double>::quiet_NaN();
    Eigen::Vector3d position_ecef_m = Eigen::Vector3d::Zero();
    Eigen::Matrix3d position_covariance_ecef_m2 = Eigen::Matrix3d::Zero();

    bool IsValid() const 
    {
        if (!std::isfinite(time_s) || !position_ecef_m.allFinite()
            || !position_covariance_ecef_m2.allFinite()
            || !position_covariance_ecef_m2.isApprox(
                position_covariance_ecef_m2.transpose(), 1e-12)) 
        {
            return false;
        }
        // Positive definiteness is required for covariance-based whitening.
        const Eigen::LLT<Eigen::Matrix3d> decomposition(position_covariance_ecef_m2);
        return decomposition.info() == Eigen::Success;
    }

    static CtGnssPosition FromCovariance(double timestamp_s,
                                         const Eigen::Vector3d& position_ecef,
                                         const Eigen::Matrix3d& covariance_ecef) 
    {
        CtGnssPosition observation;
        observation.time_s = timestamp_s;
        observation.position_ecef_m = position_ecef;
        observation.position_covariance_ecef_m2 = covariance_ecef;
        if (!observation.IsValid()) 
        {
            throw std::invalid_argument(
                "CT GNSS position requires finite time/position and symmetric positive-definite ECEF covariance");
        }
        return observation;
    }

    // t_gposdata::data_pos::Rpos contains ECEF variances (m^2), not standard deviations.
    static CtGnssPosition FromVariances(double timestamp_s,
                                        const Eigen::Vector3d& position_ecef,
                                        const Eigen::Vector3d& position_variances_ecef_m2) 
    {
        return FromCovariance(timestamp_s, position_ecef,
                              position_variances_ecef_m2.asDiagonal());
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}  // namespace gct
}  // namespace gfgomsf

#endif  // GREAT_GCT_CT_TYPES_H
