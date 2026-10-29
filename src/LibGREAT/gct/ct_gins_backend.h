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

// 状态对应 IMU 中心，姿态表示机体系到 ECEF 的旋转。
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
    // 以下为标准差，IMU 每条观测的权重为 1/sigma。
    double gyro_std_rad_s = 0.01;
    double accel_std_m_s2 = 0.1;
    double gyro_bias_prior_std_rad_s = 0.01;
    double accel_bias_prior_std_m_s2 = 0.1;
    double initial_attitude_std_rad = 0.05;
    int max_iterations = 100;
    int num_threads = 1;
    bool adaptive_gnss = false;
    double gnss_scale_lambda = 1.0;
    double gnss_scale_max = 100.0;
};

struct CtGnssWeightUpdate
{
    double time_s = 0.0;
    double squared_error = 0.0;
    double previous_scale = 1.0;
    double covariance_scale = 1.0;
};

struct CtGinsSolveResult
{
    // 达到迭代上限但未收敛的结果仍可能可用。
    bool usable = false;
    bool converged = false;
    std::size_t imu_samples_used = 0;
    std::size_t gnss_positions_used = 0;
    std::size_t imu_samples_skipped = 0;
    std::size_t gnss_positions_skipped = 0;
    // 自适应模式累计两轮迭代；代价和 message 仅描述最后一轮。
    int iterations = 0;
    int preliminary_iterations = 0;
    // 更新统计使用初解残差；初解和二次求解的轨迹代价使用不同权重，不直接作降幅比较。
    std::vector<CtGnssWeightUpdate> gnss_weight_updates;
    // Ceres 代价为加权残差平方和的一半。
    double initial_cost = 0.0;
    double final_cost = 0.0;
    // 窗口输出不超过最后一个完整受观测约束的结点区间。
    double optimized_until_s = 0.0;
    std::string message;
};

// 连续时间批量与滑窗优化，机体系下的 IMU 零偏在单次求解内保持常量。
class CtGinsBackend
{
public:
    static constexpr int SplineOrder = 4;

    CtGinsBackend();
    ~CtGinsBackend();

    // 杆臂为机体系下 IMU 指向天线的向量（m），重力为固定 ECEF 向量。
    void SetParameters(double knot_interval_s, const Eigen::Vector3d& lever_arm_body_m,const Eigen::Vector3d& gravity_ecef_m_s2);
    // 两路观测使用同一连续秒时间轴，各自严格递增。
    // 滑窗模式追加观测后仍可查询旧轨迹；批量模式追加观测会使轨迹需要重新初始化。
    void AddImuSample(const CtImuSample& sample);
    void AddGnssPosition(const CtGnssPosition& observation);
    void ClearData();

    // 首条 GNSS 天线位置作为 ECEF 原点；位置控制点按 GNSS 插值或外推，姿态控制点均取给定初始旋转。
    void Initialize(const Eigen::Quaterniond& rotation_body_to_ecef);
    bool IsInitialized() const;
    // 设置机体系零偏初值与先验中心，单位为 rad/s 和 m/s^2。
    void SetInitialBiases(const Eigen::Vector3d& gyro_bias_rad_s,const Eigen::Vector3d& accel_bias_m_s2);
    // 调用前须完成 SetParameters() 与 Initialize()；求解中固定重力和杆臂，仅接受可用解。
    CtGinsSolveResult Solve(const CtGinsSolveOptions& options = CtGinsSolveOptions());
    // 窗口推进至最新 IMU 时刻，并按结点间隔裁剪到给定时长。
    // 首次求解须在首个共同观测后的 duration+dt 内进行。
    // 保留控制点和零偏作为初值；未启用边缘化时固定滑窗后的前三个控制点。
    // 求解失败时保留原轨迹和观测缓存。
    CtGinsSolveResult SolveWindow(double duration_s, const Eigen::Quaterniond& initial_rotation,
        const CtGinsSolveOptions& options = CtGinsSolveOptions());
    // 首窗前启用边缘化，每次裁窗前由调用方提供对应先验。
    void EnableMarginalization(bool enabled);
    bool LinearizeMarginalization(double duration_s, const CtGinsSolveOptions& options,
        CtMarginalizationSystem& system) const;
    void SetPendingMarginalization(const CtMarginalizationPrior& prior);
    const CtMarginalizationPrior& GetMarginalizationPrior() const;
    // 表示当前是否保留已接受的解；最近一次求解失败不会清除旧解。
    bool IsSolved() const;
    // 查询区间为 [MinTime(),MaxTime())，包含观测重叠区间的末条观测。
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
    CtGinsSolveResult SolveFixedWeights(const CtGinsSolveOptions& options);
    double ObservationEndTime() const;
    std::vector<CtGnssWeightUpdate> UpdateGnssCovarianceScales(const CtGinsSolveOptions& options);
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
