#ifndef GREAT_GCT_CT_TYPES_H
#define GREAT_GCT_CT_TYPES_H
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace gfgomsf {
namespace gct {

// 机体系角速度（rad/s）与比力（m/s^2）。
// 与 GNSS、样条共用不跨周回绕的连续秒时间轴；标定和轴转换由调用方完成。
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

    // interval_s 必须对应本条增量的积分时长；不转换坐标轴和时间基准。
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

// ECEF 下的 GNSS 天线绝对位置（m）与协方差（m^2），时间基准与 IMU、样条一致。
struct CtGnssPosition {
    double time_s = std::numeric_limits<double>::quiet_NaN();
    Eigen::Vector3d position_ecef_m = Eigen::Vector3d::Zero();
    Eigen::Matrix3d position_covariance_ecef_m2 = Eigen::Matrix3d::Zero();
    // 有效协方差为 scale*原始协方差；保留原矩阵，避免滑窗重复放大。
    double covariance_scale = 1.0;

    bool IsValid() const 
    {
        if (!std::isfinite(time_s) || !position_ecef_m.allFinite()
            || !position_covariance_ecef_m2.allFinite()
            || !std::isfinite(covariance_scale) || covariance_scale < 1.0
            || !position_covariance_ecef_m2.isApprox(
                position_covariance_ecef_m2.transpose(), 1e-12)) 
        {
            return false;
        }
        // 协方差须正定，才能通过 Cholesky 分解白化残差。
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

    // t_gposdata::data_pos::Rpos 存放 ECEF 各轴方差（m^2），不是标准差。
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
