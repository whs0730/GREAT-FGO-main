#ifndef GREAT_GCT_CT_TYPES_H
#define GREAT_GCT_CT_TYPES_H

#include <Eigen/Core>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace gfgomsf {
namespace gct {

// A calibrated IMU sample in the body frame. Its timestamp must use the same
// continuous seconds-based time axis as the spline knots (no GPS-week wrap).
// Accelerometers measure specific force, not reference-frame acceleration.
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
                                const Eigen::Vector3d& specific_force) {
        CtImuSample sample;
        sample.time_s = timestamp_s;
        sample.angular_rate_rad_s = angular_rate;
        sample.specific_force_m_s2 = specific_force;
        if (!sample.IsValid()) {
            throw std::invalid_argument("CT IMU sample must contain finite SI measurements and time");
        }
        return sample;
    }

    // delta_angle_rad and delta_velocity_m_s must already be calibrated and
    // expressed in the body frame. This conversion does not change axes or time.
    static CtImuSample FromIncrements(double timestamp_s, double interval_s,
                                     const Eigen::Vector3d& delta_angle_rad,
                                     const Eigen::Vector3d& delta_velocity_m_s) {
        if (!std::isfinite(interval_s) || interval_s <= 0.0) {
            throw std::invalid_argument("CT IMU integration interval must be finite and positive");
        }
        return FromRates(timestamp_s, delta_angle_rad / interval_s,
                         delta_velocity_m_s / interval_s);
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}  // namespace gct
}  // namespace gfgomsf

#endif  // GREAT_GCT_CT_TYPES_H
