#ifndef GCT_GINS_BACKEND_H
#define GCT_GINS_BACKEND_H

#include "gct/ct_types.h"
#include "gct/gspline/rd_spline.h"
#include "gct/gspline/so3_spline.h"
#include <Eigen/Geometry>
#include <vector>

namespace gfgomsf
{
namespace gct
{

// State at the IMU centre; rotation maps body vectors to ECEF.
struct CtGinsState
{
    double time_s = 0.0;
    Eigen::Vector3d position_ecef_m = Eigen::Vector3d::Zero();
    Eigen::Vector3d velocity_ecef_m_s = Eigen::Vector3d::Zero();
    Eigen::Quaterniond rotation_body_to_ecef = Eigen::Quaterniond::Identity();

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// Observation storage and initial trajectory. Factor assembly/solving follows later.
class CtGinsBackend
{
public:
    static constexpr int SplineOrder = 4;

    CtGinsBackend();
    ~CtGinsBackend();

    // Lever arm: IMU -> antenna in body metres. Gravity is a fixed ECEF vector.
    void SetParameters(double knot_interval_s, const Eigen::Vector3d& lever_arm_body_m,const Eigen::Vector3d& gravity_ecef_m_s2);
    // Times must share one seconds-based axis and increase within each sensor stream.
    // Adding observations or changing parameters requires reinitializing the trajectory.
    void AddImuSample(const CtImuSample& sample);
    void AddGnssPosition(const CtGnssPosition& observation);
    void ClearData();

    // First GNSS position is the fixed ECEF origin. Positions are interpolated
    // for an initial guess; all attitude knots start at the supplied rotation.
    void Initialize(const Eigen::Quaterniond& rotation_body_to_ecef);
    bool IsInitialized() const;
    // Spline query interval is [MinTime(), MaxTime()); includes the last observation.
    double MinTime() const;
    double MaxTime() const;
    CtGinsState GetState(double time_s) const;

    const std::vector<CtImuSample>& GetImuSamples() const;
    const std::vector<CtGnssPosition>& GetGnssPositions() const;
    const Eigen::Vector3d& GetPositionOrigin() const;
    const Eigen::Vector3d& GetLeverArm() const;
    const Eigen::Vector3d& GetGravity() const;
    const RdSpline<3, SplineOrder>& GetPositionSpline() const;
    const So3Spline<SplineOrder>& GetAttitudeSpline() const;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    double _knot_interval;
    bool _initialized;
    Eigen::Vector3d _position_origin;
    Eigen::Vector3d _lever_arm;
    Eigen::Vector3d _gravity;
    std::vector<CtImuSample> _imu_data;
    std::vector<CtGnssPosition> _gnss_data;
    RdSpline<3, SplineOrder> _position_spline;
    So3Spline<SplineOrder> _attitude_spline;
};

}  // namespace gct
}  // namespace gfgomsf

#endif  // GCT_GINS_BACKEND_H
