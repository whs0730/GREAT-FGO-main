#ifndef GCT_GINS_BACKEND_H
#define GCT_GINS_BACKEND_H

#include "gct/ct_types.h"
#include "gct/ct_marginalization.h"
#include "gct/gspline/rd_spline.h"
#include "gct/gspline/so3_spline.h"
#include <Eigen/Geometry>
#include <string>
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

struct CtGinsSolveOptions
{
    // Standard deviations, not variances. IMU weights are 1/sigma per sample.
    double gyro_std_rad_s = 0.01;
    double accel_std_m_s2 = 0.1;
    double gyro_bias_prior_std_rad_s = 0.01;
    double accel_bias_prior_std_m_s2 = 0.1;
    double initial_attitude_std_rad = 0.05;
    int max_iterations = 100;
    int num_threads = 1;
};

struct CtGinsSolveResult
{
    // A usable result may reach the iteration limit without convergence.
    bool usable = false;
    bool converged = false;
    std::size_t imu_samples_used = 0;
    std::size_t gnss_positions_used = 0;
    std::size_t imu_samples_skipped = 0;
    std::size_t gnss_positions_skipped = 0;
    int iterations = 0;
    // Ceres cost is half the sum of squared weighted residuals.
    double initial_cost = 0.0;
    double final_cost = 0.0;
    std::string message;
};

// Batch continuous-time optimization with constant body-frame IMU biases.
class CtGinsBackend
{
public:
    static constexpr int SplineOrder = 4;

    CtGinsBackend();
    ~CtGinsBackend();

    // Lever arm: IMU -> antenna in body metres. Gravity is a fixed ECEF vector.
    void SetParameters(double knot_interval_s, const Eigen::Vector3d& lever_arm_body_m,const Eigen::Vector3d& gravity_ecef_m_s2);
    // Each sensor stream requires increasing timestamps on the shared seconds axis.
    // In window mode, appended data keep the last trajectory available until SolveWindow().
    // In batch mode, new observations invalidate the initialized trajectory.
    void AddImuSample(const CtImuSample& sample);
    void AddGnssPosition(const CtGnssPosition& observation);
    void ClearData();

    // Uses the first GNSS antenna position as the ECEF origin. Position knots get
    // GNSS-based guesses; all attitude knots start at the supplied rotation.
    void Initialize(const Eigen::Quaterniond& rotation_body_to_ecef);
    bool IsInitialized() const;
    // Sets starting biases and prior centres in body-frame rad/s and m/s^2.
    void SetInitialBiases(const Eigen::Vector3d& gyro_bias_rad_s,const Eigen::Vector3d& accel_bias_m_s2);
    // Requires SetParameters() and Initialize(); gravity and lever arm stay fixed.
    // Only a usable result replaces the stored trajectory and biases.
    CtGinsSolveResult Solve(const CtGinsSolveOptions& options = CtGinsSolveOptions());
    // Advances to the latest IMU time, retaining a knot-aligned interval of this length.
    // Call the first time within duration+dt of the first common observation.
    // Retained knots and biases warm-start the solve. Without marginalization enabled,
    // the first three retained knots are fixed after sliding.
    // An unusable solve preserves the previous trajectory and buffered observations.
    CtGinsSolveResult SolveWindow(double duration_s, const Eigen::Quaterniond& initial_rotation,
        const CtGinsSolveOptions& options = CtGinsSolveOptions());
    // Enable before the first window. The caller must supply a prior before each slide.
    void EnableMarginalization(bool enabled);
    bool LinearizeMarginalization(double duration_s, const CtGinsSolveOptions& options,
        CtMarginalizationSystem& system) const;
    void SetPendingMarginalization(const CtMarginalizationPrior& prior);
    const CtMarginalizationPrior& GetMarginalizationPrior() const;
    // Tracks an accepted solution, not the success of the latest Solve() call.
    bool IsSolved() const;
    // Query interval is [MinTime(), MaxTime()); covers the observation overlap end.
    double MinTime() const;
    double MaxTime() const;
    CtGinsState GetState(double time_s) const;

    const std::vector<CtImuSample>& GetImuSamples() const;
    const std::vector<CtGnssPosition>& GetGnssPositions() const;
    const Eigen::Vector3d& GetPositionOrigin() const;
    const Eigen::Vector3d& GetLeverArm() const;
    const Eigen::Vector3d& GetGravity() const;
    const Eigen::Vector3d& GetGyroBias() const;
    const Eigen::Vector3d& GetAccelBias() const;
    const RdSpline<3, SplineOrder>& GetPositionSpline() const;
    const So3Spline<SplineOrder>& GetAttitudeSpline() const;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
    double _knot_interval;
    bool _initialized;
    bool _parameters_set;
    bool _solved;
    bool _window_started;
    int _fixed_prefix_knots;
    bool _marginalization_enabled;
    double _window_end_time;
    CtMarginalizationPrior _marginalization_prior;
    CtMarginalizationPrior _pending_prior;
    Eigen::Vector3d _position_origin;
    Eigen::Vector3d _lever_arm;
    Eigen::Vector3d _gravity;
    Eigen::Vector3d _gyro_bias;
    Eigen::Vector3d _accel_bias;
    Eigen::Vector3d _initial_gyro_bias;
    Eigen::Vector3d _initial_accel_bias;
    Eigen::Quaterniond _initial_rotation;
    std::vector<CtImuSample> _imu_data;
    std::vector<CtGnssPosition> _gnss_data;
    RdSpline<3, SplineOrder> _position_spline;
    So3Spline<SplineOrder> _attitude_spline;
};

}  // namespace gct
}  // namespace gfgomsf

#endif  // GCT_GINS_BACKEND_H
