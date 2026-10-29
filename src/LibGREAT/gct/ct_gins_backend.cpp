#include "gct/ct_gins_backend.h"
#include "gfactor/imu_gyro_factor.hpp"
#include "gct/ct_gnss_weighting.h"
#include "gfactor/imu_acce_factor.hpp"
#include "gfactor/ct_gnss_position_factor.h"
#include "gfactor/ct_marginalization_factor.h"
#include <ceres/ceres.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace gfgomsf
{
namespace gct
{

namespace
{

double WindowKnotShift(double latest_time, double duration, double start_time, double interval)
{
    double shift = (latest_time - duration - start_time) / interval;
    double nearest = std::round(shift);
    // 结点边界附近的浮点误差先吸附到整数，其余不足一段的偏移仍向下取整。
    if (std::abs(shift - nearest) < 1e-8)
    {
        shift = nearest;
    }
    return std::floor(shift);
}

class VectorPrior : public ceres::SizedCostFunction<3, 3>
{
public:
    VectorPrior(const Eigen::Vector3d& value, double std_dev)
        : _value(value), _weight(1.0 / std_dev)
    {
    }

    bool Evaluate(double const* const* parameters, double* residuals,
                  double** jacobians) const override
    {
        Eigen::Map<const Eigen::Vector3d> value(parameters[0]);
        Eigen::Map<Eigen::Vector3d> error(residuals);
        error = _weight * (value - _value);
        if (jacobians && jacobians[0])
        {
            Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> jacobian(jacobians[0]);
            jacobian = _weight * Eigen::Matrix3d::Identity();
        }
        return true;
    }

private:
    Eigen::Vector3d _value;
    double _weight;
};

// 先验约束起始时刻的样条姿态 R(t_start)，而非第一个四元数控制点。
struct InitialAttitudePrior
{
    InitialAttitudePrior(const Eigen::Quaterniond& rotation, double interval, double std_dev)
        : initial_rotation(rotation), inverse_dt(1.0 / interval), weight(1.0 / std_dev)
    {
    }

    template <class T>
    bool operator()(T const* const* blocks, T* residuals) const
    {
        Sophus::SO3<T> rotation;
        CeresSplineHelper<CtGinsBackend::SplineOrder>::template EvaluateLie<T, Sophus::SO3>(
            blocks, 0.0, inverse_dt, &rotation);
        Sophus::SO3<T> initial(initial_rotation.template cast<T>());
        Eigen::Map<Eigen::Matrix<T, 3, 1>> error(residuals);
        error = T(weight) * (initial.inverse() * rotation).log();
        return true;
    }

    Eigen::Quaterniond initial_rotation;
    double inverse_dt;
    double weight;
};

// 局部参数先放四个姿态控制点 [qx,qy,qz,qw]，再放四个位置控制点。
// 两组样条必须具有相同的起始时刻和结点间隔。
double GetLocalBlocks(RdSpline<3, CtGinsBackend::SplineOrder>& position_spline,
                    So3Spline<CtGinsBackend::SplineOrder>& attitude_spline, double time,
                    SplineMeta<CtGinsBackend::SplineOrder>& meta,
                    std::vector<double*>& blocks)
{
    std::pair<double, std::size_t> location = position_spline.ComputeTIndex(time);
    double interval = position_spline.GetTimeInterval();
    // 因子使用段内时间 [0,dt)，结点处的舍入误差可能使时间略小于 0 或达到 dt。
    double segment_time = location.first * interval;
    if (segment_time < 0.0)
    {
        segment_time = 0.0;
    }
    else if (segment_time >= interval)
    {
        segment_time = std::nextafter(interval, 0.0);
    }
    meta.segments.emplace_back(0.0, interval, CtGinsBackend::SplineOrder);
    int first = static_cast<int>(location.second);
    for (int i = 0; i < CtGinsBackend::SplineOrder; i++)
    {
        blocks.push_back(attitude_spline.GetKnot(first + i).data());
    }
    for (int i = 0; i < CtGinsBackend::SplineOrder; i++)
    {
        blocks.push_back(position_spline.GetKnot(first + i).data());
    }
    return segment_time;
}

void AddMarginalizationPrior(ceres::Problem& problem, const CtMarginalizationPrior& prior,
    RdSpline<3, CtGinsBackend::SplineOrder>& position_spline,
    So3Spline<CtGinsBackend::SplineOrder>& attitude_spline,
    Eigen::Vector3d& gyro_bias, Eigen::Vector3d& accel_bias)
{
    std::vector<double*> blocks;
    for (std::size_t i = 0; i < prior.blocks.size(); i++)
    {
        const CtMarginalizationBlock& block = prior.blocks[i];
        if (block.type == CtMarginalizationBlock::ROTATION)
        {
            blocks.push_back(attitude_spline.GetKnot(block.knot).data());
        }
        else if (block.type == CtMarginalizationBlock::POSITION)
        {
            blocks.push_back(position_spline.GetKnot(block.knot).data());
        }
        else if (block.type == CtMarginalizationBlock::GYRO_BIAS)
        {
            blocks.push_back(gyro_bias.data());
        }
        else
        {
            blocks.push_back(accel_bias.data());
        }
    }
    problem.AddResidualBlock(CtMarginalizationFactor::Create(prior), nullptr, blocks);
}

}  // namespace

CtGinsBackend::CtGinsBackend()
    : _knot_interval(0.1), _initialized(false), _parameters_set(false), _solved(false),
      _window_started(false), _fixed_prefix_knots(0), _marginalization_enabled(false), _window_end_time(0.0),
      _position_origin(Eigen::Vector3d::Zero()), _lever_arm(Eigen::Vector3d::Zero()),
      _gravity(Eigen::Vector3d::Zero()),
      _gyro_bias(Eigen::Vector3d::Zero()), _accel_bias(Eigen::Vector3d::Zero()),
      _initial_gyro_bias(Eigen::Vector3d::Zero()), _initial_accel_bias(Eigen::Vector3d::Zero()),
      _initial_rotation(Eigen::Quaterniond::Identity()),
      _position_spline(_knot_interval), _attitude_spline(_knot_interval)
{
}

CtGinsBackend::~CtGinsBackend() = default;

void CtGinsBackend::SetParameters(double knot_interval_s,
                                  const Eigen::Vector3d& lever_arm_body_m,
                                  const Eigen::Vector3d& gravity_ecef_m_s2)
{
    if (!std::isfinite(knot_interval_s) || knot_interval_s <= 0.0|| !lever_arm_body_m.allFinite() || !gravity_ecef_m_s2.allFinite())
    {
        throw std::invalid_argument("CT backend requires a positive interval and finite lever arm/gravity");
    }
    _knot_interval = knot_interval_s;
    _lever_arm = lever_arm_body_m;
    _gravity = gravity_ecef_m_s2;
    _parameters_set = true;
    _initialized = false;
    _solved = false;
    _window_started = false;
    _fixed_prefix_knots = 0;
    _window_end_time = 0.0;
    _marginalization_prior = CtMarginalizationPrior();
    _pending_prior = CtMarginalizationPrior();
}

void CtGinsBackend::AddImuSample(const CtImuSample& sample)
{
    if (!sample.IsValid())
    {
        throw std::invalid_argument("CT backend received an invalid IMU sample");
    }
    if (!_imu_data.empty() && sample.time_s <= _imu_data.back().time_s)
    {
        throw std::invalid_argument("CT IMU timestamps must strictly increase");
    }
    _imu_data.push_back(sample);
    if (!_window_started)
    {
        _initialized = false;
    }
    _solved = false;
}

void CtGinsBackend::AddGnssPosition(const CtGnssPosition& observation)
{
    if (!observation.IsValid())
    {
        throw std::invalid_argument("CT backend received an invalid GNSS position");
    }
    if (!_gnss_data.empty() && observation.time_s <= _gnss_data.back().time_s)
    {
        throw std::invalid_argument("CT GNSS timestamps must strictly increase");
    }
    _gnss_data.push_back(observation);
    if (!_window_started)
    {
        _initialized = false;
    }
    _solved = false;
}

void CtGinsBackend::ClearData()
{
    _imu_data.clear();
    _gnss_data.clear();
    _position_spline = RdSpline<3, SplineOrder>(_knot_interval);
    _attitude_spline = So3Spline<SplineOrder>(_knot_interval);
    _position_origin.setZero();
    _gyro_bias = _initial_gyro_bias;
    _accel_bias = _initial_accel_bias;
    _initial_rotation = Eigen::Quaterniond::Identity();
    _initialized = false;
    _solved = false;
    _window_started = false;
    _fixed_prefix_knots = 0;
    _window_end_time = 0.0;
    _marginalization_prior = CtMarginalizationPrior();
    _pending_prior = CtMarginalizationPrior();
}

void CtGinsBackend::Initialize(const Eigen::Quaterniond& rotation_body_to_ecef)
{
    if (_imu_data.size() < 2 || _gnss_data.size() < 2)
    {
        throw std::invalid_argument("CT initialization requires at least two IMU and two GNSS samples");
    }
    double rotation_norm = rotation_body_to_ecef.norm();
    if (!rotation_body_to_ecef.coeffs().allFinite()|| !std::isfinite(rotation_norm) || rotation_norm < 1e-12)
    {
        throw std::invalid_argument("CT initial rotation must be a finite nonzero quaternion");
    }
    double start_time = std::max(_imu_data.front().time_s, _gnss_data.front().time_s);
    double end_time = std::min(_imu_data.back().time_s, _gnss_data.back().time_s);
    double duration = end_time - start_time;
    if (!std::isfinite(duration) || duration <= 0.0|| start_time + _knot_interval <= start_time)
    {
        throw std::invalid_argument("CT initialization requires an overlapping time interval and representable knots");
    }

    // 多留一段，使末条观测仍位于样条的左闭右开区间内。
    double segment_count = std::floor(duration / _knot_interval) + 1.0;
    if (!std::isfinite(segment_count)|| segment_count > std::numeric_limits<int>::max() - SplineOrder + 1)
    {
        throw std::invalid_argument("CT initial trajectory has too many knots");
    }
    int knot_count = static_cast<int>(segment_count) + SplineOrder - 1;
    double max_time = start_time + segment_count * _knot_interval;
    // 舍入误差可能把末条观测放到不包含的右端点。
    if (max_time <= end_time)
    {
        if (knot_count == std::numeric_limits<int>::max())
        {
            throw std::invalid_argument("CT initial trajectory has too many knots");
        }
        knot_count++;
        max_time = start_time + (segment_count + 1.0) * _knot_interval;
    }
    if (!std::isfinite(max_time) || max_time <= end_time)
    {
        throw std::invalid_argument("CT spline end time cannot be represented");
    }

    Eigen::Quaterniond rotation = rotation_body_to_ecef.normalized();
    Sophus::SO3d attitude(rotation);
    Eigen::Vector3d origin = _gnss_data.front().position_ecef_m;
    Eigen::Vector3d lever_ecef = rotation * _lever_arm;
    RdSpline<3, SplineOrder> position_spline(_knot_interval, start_time);
    So3Spline<SplineOrder> attitude_spline(_knot_interval, start_time);

    std::size_t index = 0;
    for (int i = 0; i < knot_count; i++)
    {
        // 三次样条在 t0+(i-1)*dt 采样可重现匀速轨迹，边界控制点沿首尾 GNSS 线段外推。
        double knot_time = start_time + (static_cast<double>(i) - 1.0) * _knot_interval;
        while (index + 2 < _gnss_data.size()&& knot_time > _gnss_data[index + 1].time_s)
        {
            index++;
        }
        const CtGnssPosition& first = _gnss_data[index];
        const CtGnssPosition& second = _gnss_data[index + 1];
        double interval = second.time_s - first.time_s;
        double ratio = (knot_time - first.time_s) / interval;
        Eigen::Vector3d displacement = second.position_ecef_m - first.position_ecef_m;
        Eigen::Vector3d position = first.position_ecef_m - origin;
        position += ratio * displacement;
        position -= lever_ecef;
        if (!std::isfinite(knot_time) || !position.allFinite())
        {
            throw std::invalid_argument("CT initial position knot is not finite");
        }
        position_spline.KnotsPushBack(position);
        attitude_spline.KnotsPushBack(attitude);
    }

    _position_spline = std::move(position_spline);
    _attitude_spline = std::move(attitude_spline);
    _position_origin = origin;
    _window_started = false;
    _fixed_prefix_knots = 0;
    _window_end_time = 0.0;
    _marginalization_prior = CtMarginalizationPrior();
    _pending_prior = CtMarginalizationPrior();
    _initial_rotation = rotation;
    _gyro_bias = _initial_gyro_bias;
    _accel_bias = _initial_accel_bias;
    _initialized = true;
    _solved = false;
}

bool CtGinsBackend::IsInitialized() const
{
    return _initialized;
}

void CtGinsBackend::SetInitialBiases(const Eigen::Vector3d& gyro_bias_rad_s,
                                    const Eigen::Vector3d& accel_bias_m_s2)
{
    if (!gyro_bias_rad_s.allFinite() || !accel_bias_m_s2.allFinite())
    {
        throw std::invalid_argument("CT initial IMU biases must be finite");
    }
    _initial_gyro_bias = gyro_bias_rad_s;
    _initial_accel_bias = accel_bias_m_s2;
    _gyro_bias = gyro_bias_rad_s;
    _accel_bias = accel_bias_m_s2;
    _solved = false;
}

CtGinsSolveResult CtGinsBackend::SolveWindow(double duration_s,const Eigen::Quaterniond& initial_rotation, const CtGinsSolveOptions& options)
{
    if (!_parameters_set || !std::isfinite(duration_s) || duration_s < 2.0 * _knot_interval)
    {
        throw std::invalid_argument("CT window requires parameters and a duration of at least two knot intervals");
    }
    if (_imu_data.size() < 2 || _gnss_data.empty())
    {
        throw std::invalid_argument("CT window requires IMU data and an initial GNSS reference");
    }
    double end_time = _imu_data.back().time_s;
    // 在副本上更新窗口，求解失败时保留上一有效窗口。
    CtGinsBackend candidate = *this;
    if (!_window_started)
    {
        double first_time = std::max(_imu_data.front().time_s, _gnss_data.front().time_s);
        if (end_time - first_time > duration_s + _knot_interval)
        {
            throw std::invalid_argument("CT first window is too long; solve before collecting the next window");
        }
        candidate.Initialize(initial_rotation);
    }
    double start_time = candidate._position_spline.MinTime();
    Eigen::Vector3d tail_position = Eigen::Vector3d::Zero();
    Eigen::Vector3d tail_velocity = Eigen::Vector3d::Zero();
    Eigen::Vector3d tail_acceleration = Eigen::Vector3d::Zero();
    Eigen::Vector3d tail_gyro = Eigen::Vector3d::Zero();
    Sophus::SO3d tail_rotation;
    if (_window_started && _marginalization_enabled)
    {
        tail_position = _position_spline.Evaluate(_window_end_time);
        tail_velocity = _position_spline.Velocity(_window_end_time);
        tail_acceleration = _position_spline.Acceleration(_window_end_time);
        tail_rotation = _attitude_spline.Evaluate(_window_end_time);
        tail_gyro = _attitude_spline.VelocityBody(_window_end_time);
    }
    double shift = WindowKnotShift(end_time, duration_s, start_time, _knot_interval);
    if (shift > 0.0)
    {
        std::size_t old_count = candidate._position_spline.GetKnots().size();
        if (shift > static_cast<double>(old_count - (SplineOrder - 1)))
        {
            throw std::invalid_argument("CT update interval lost the three retained boundary knots");
        }
        std::size_t remove_count = static_cast<std::size_t>(shift);
        for (std::size_t i = 0; i < remove_count; i++)
        {
            candidate._position_spline.KnotsPopFront();
            candidate._attitude_spline.KnotsPopFront();
        }
        if (_marginalization_enabled)
        {
            if (_pending_prior.blocks.empty()
                || std::abs(_pending_prior.start_time - candidate._position_spline.MinTime()) > 1e-8)
            {
                throw std::logic_error("CT sliding requires a marginalization prior for the new window");
            }
            candidate._marginalization_prior = _pending_prior;
            candidate._pending_prior = CtMarginalizationPrior();
        }
        else
        {
            candidate._fixed_prefix_knots = SplineOrder - 1;
        }
    }
    while (candidate._position_spline.MaxTime() <= end_time)
    {
        int last = static_cast<int>(candidate._position_spline.GetKnots().size()) - 1;
        Eigen::Vector3d next_position = 2.0 * candidate._position_spline.GetKnot(last)
            - candidate._position_spline.GetKnot(last - 1);
        Sophus::SO3d last_rotation = candidate._attitude_spline.GetKnot(last);
        Sophus::SO3d delta_rotation = candidate._attitude_spline.GetKnot(last - 1).inverse() * last_rotation;
        Sophus::SO3d next_rotation = last_rotation * delta_rotation;
        if (_window_started && _marginalization_enabled)
        {
            // 从末个受观测约束的状态外推，避免使用未受约束的尾部控制点增量。
            double knot_time = candidate._position_spline.MinTime() + last * _knot_interval;
            double delta_time = knot_time - _window_end_time;
            next_position = tail_position + delta_time * tail_velocity
                + (0.5 * delta_time * delta_time - _knot_interval * _knot_interval / 6.0) * tail_acceleration;
            next_rotation = tail_rotation * Sophus::SO3d::exp(delta_time * tail_gyro);
        }
        candidate._position_spline.KnotsPushBack(next_position);
        candidate._attitude_spline.KnotsPushBack(next_rotation);
    }
    start_time = candidate._position_spline.MinTime();
    std::size_t imu_begin = 0;
    while (imu_begin < candidate._imu_data.size() && candidate._imu_data[imu_begin].time_s < start_time)
    {
        imu_begin++;
    }
    candidate._imu_data.erase(candidate._imu_data.begin(), candidate._imu_data.begin() + imu_begin);
    // 保留窗前最后一条 GNSS 记录，失锁期间仍可检查后续观测的时间顺序。
    std::size_t gnss_begin = 0;
    while (gnss_begin + 1 < candidate._gnss_data.size()
        && candidate._gnss_data[gnss_begin + 1].time_s < start_time)
    {
        gnss_begin++;
    }
    candidate._gnss_data.erase(candidate._gnss_data.begin(), candidate._gnss_data.begin() + gnss_begin);
    CtGinsSolveResult result = candidate.Solve(options);
    if (result.usable)
    {
        candidate._window_started = true;
        candidate._window_end_time = std::min(result.optimized_until_s,
            std::nextafter(candidate.MaxTime(), candidate.MinTime()));
        *this = std::move(candidate);
    }
    return result;
}

void CtGinsBackend::EnableMarginalization(bool enabled)
{
    if (_window_started)
    {
        throw std::logic_error("CT marginalization mode must be selected before the first window");
    }
    _marginalization_enabled = enabled;
}

const CtMarginalizationPrior& CtGinsBackend::GetMarginalizationPrior() const
{
    return _marginalization_prior;
}

void CtGinsBackend::SetPendingMarginalization(const CtMarginalizationPrior& prior)
{
    int columns = 3 * static_cast<int>(prior.blocks.size());
    if (!_marginalization_enabled || prior.blocks.empty() || !std::isfinite(prior.start_time)
        || prior.jacobian.cols() != columns || prior.jacobian.rows() != prior.residual.size()
        || prior.residual.size() == 0 || !prior.jacobian.allFinite() || !prior.residual.allFinite())
    {
        throw std::invalid_argument("Invalid CT marginalization prior dimensions or values");
    }
    for (std::size_t i = 0; i < prior.blocks.size(); i++)
    {
        const CtMarginalizationBlock& block = prior.blocks[i];
        if (!block.value.allFinite() || block.type < CtMarginalizationBlock::ROTATION
            || block.type > CtMarginalizationBlock::ACCEL_BIAS || block.knot < 0
            || ((block.type == CtMarginalizationBlock::ROTATION || block.type == CtMarginalizationBlock::POSITION)
                && static_cast<std::size_t>(block.knot) >= _position_spline.GetKnots().size()))
        {
            throw std::invalid_argument("Invalid CT marginalization block");
        }
        if (block.type == CtMarginalizationBlock::ROTATION && std::abs(block.value.norm() - 1.0) > 1e-8)
        {
            throw std::invalid_argument("CT marginalization quaternion must have unit norm");
        }
    }
    _pending_prior = prior;
}

bool CtGinsBackend::LinearizeMarginalization(double duration_s, const CtGinsSolveOptions& options,
    CtMarginalizationSystem& system) const
{
    system = CtMarginalizationSystem();
    if (!_marginalization_enabled || !_window_started)
    {
        return false;
    }
    if (!std::isfinite(duration_s) || duration_s < 2.0 * _knot_interval)
    {
        throw std::invalid_argument("Invalid CT marginalization window duration");
    }
    double start_time = _position_spline.MinTime();
    double shift = WindowKnotShift(_imu_data.back().time_s, duration_s, start_time, _knot_interval);
    if (shift <= 0.0)
    {
        return false;
    }
    std::size_t knot_count = _position_spline.GetKnots().size();
    if (shift > static_cast<double>(knot_count - (SplineOrder - 1)))
    {
        throw std::invalid_argument("CT marginalization update lost its overlap");
    }
    std::size_t remove_count = static_cast<std::size_t>(shift);
    double next_start = start_time;
    for (std::size_t i = 0; i < remove_count; i++)
    {
        next_start += _knot_interval;
    }
    RdSpline<3, SplineOrder> position_spline = _position_spline;
    So3Spline<SplineOrder> attitude_spline = _attitude_spline;
    position_spline.SetStartTime(0.0);
    attitude_spline.SetStartTime(0.0);
    Eigen::Vector3d gyro_bias = _gyro_bias;
    Eigen::Vector3d accel_bias = _accel_bias;
    Eigen::Vector3d gravity = _gravity;
    ceres::EigenQuaternionManifold manifold;
    ceres::Problem::Options problem_options;
    problem_options.manifold_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    ceres::Problem problem(problem_options);
    // 仅吸收出窗因子，重叠区观测仍以原始因子参与下一窗口。
    for (std::size_t i = 0; i < _imu_data.size(); i++)
    {
        CtImuSample sample = _imu_data[i];
        if (sample.time_s < start_time || sample.time_s >= next_start)
        {
            continue;
        }
        sample.time_s -= start_time;
        SplineMeta<SplineOrder> meta;
        std::vector<double*> blocks;
        sample.time_s = GetLocalBlocks(position_spline, attitude_spline, sample.time_s, meta, blocks);
        std::vector<double*> gyro_blocks(blocks.begin(), blocks.begin() + SplineOrder);
        gyro_blocks.push_back(gyro_bias.data());
        problem.AddResidualBlock(IMUGyroFactorWithConstBias<SplineOrder>::Create(
            meta, sample, 1.0 / options.gyro_std_rad_s), nullptr, gyro_blocks);
        blocks.push_back(accel_bias.data());
        blocks.push_back(gravity.data());
        problem.AddResidualBlock(IMUAcceFactorWithConstBias<SplineOrder>::Create(
            meta, meta, sample, 1.0 / options.accel_std_m_s2), nullptr, blocks);
        system.imu_samples++;
    }
    for (std::size_t i = 0; i < _gnss_data.size(); i++)
    {
        CtGnssPosition observation = _gnss_data[i];
        if (observation.time_s < start_time || observation.time_s >= next_start)
        {
            continue;
        }
        observation.time_s -= start_time;
        SplineMeta<SplineOrder> meta;
        std::vector<double*> blocks;
        observation.time_s = GetLocalBlocks(position_spline, attitude_spline, observation.time_s, meta, blocks);
        problem.AddResidualBlock(CtGnssPositionFactor<SplineOrder>::Create(
            meta, meta, observation, _lever_arm, _position_origin), nullptr, blocks);
        system.gnss_positions++;
    }
    if (!_marginalization_prior.blocks.empty())
    {
        AddMarginalizationPrior(problem, _marginalization_prior, position_spline, attitude_spline,
            gyro_bias, accel_bias);
    }
    else
    {
        SplineMeta<SplineOrder> meta;
        std::vector<double*> blocks;
        GetLocalBlocks(position_spline, attitude_spline, 0.0, meta, blocks);
        blocks.resize(SplineOrder);
        std::vector<int> sizes(SplineOrder, 4);
        problem.AddResidualBlock(ct_imu_detail::MakeCost(new InitialAttitudePrior(
            _initial_rotation, _knot_interval, options.initial_attitude_std_rad), sizes), nullptr, blocks);
    }
    // 零偏标定先验仍由 Solve() 添加，此处吸收会导致重复计权。
    if (problem.HasParameterBlock(gravity.data()))
    {
        problem.SetParameterBlockConstant(gravity.data());
    }
    for (std::size_t i = 0; i < knot_count; i++)
    {
        double* rotation = attitude_spline.GetKnot(static_cast<int>(i)).data();
        if (problem.HasParameterBlock(rotation))
        {
            problem.SetManifold(rotation, &manifold);
        }
    }
    ceres::Problem::EvaluateOptions evaluate_options;
    evaluate_options.num_threads = options.num_threads;
    evaluate_options.apply_loss_function = false;
    // 显式排列局部参数，并将保留控制点映射到裁窗后的索引。
    for (int pass = 0; pass < 2; pass++)
    {
        for (std::size_t i = 0; i < knot_count; i++)
        {
            bool dropped = i < remove_count;
            if (dropped != (pass == 0))
            {
                continue;
            }
            for (int type = 0; type < 2; type++)
            {
                double* block = type == 0 ? attitude_spline.GetKnot(static_cast<int>(i)).data()
                    : position_spline.GetKnot(static_cast<int>(i)).data();
                if (!problem.HasParameterBlock(block))
                {
                    continue;
                }
                evaluate_options.parameter_blocks.push_back(block);
                if (dropped)
                {
                    system.marginalized_size += 3;
                }
                else
                {
                    CtMarginalizationBlock kept;
                    kept.type = type == 0 ? CtMarginalizationBlock::ROTATION : CtMarginalizationBlock::POSITION;
                    kept.knot = static_cast<int>(i - remove_count);
                    kept.value.head<3>() = Eigen::Map<Eigen::Vector3d>(block);
                    if (type == 0)
                    {
                        kept.value[3] = block[3];
                    }
                    system.prior.blocks.push_back(kept);
                }
            }
        }
    }
    for (int i = 0; i < 2; i++)
    {
        double* bias = i == 0 ? gyro_bias.data() : accel_bias.data();
        if (problem.HasParameterBlock(bias))
        {
            evaluate_options.parameter_blocks.push_back(bias);
            CtMarginalizationBlock kept;
            kept.type = i == 0 ? CtMarginalizationBlock::GYRO_BIAS : CtMarginalizationBlock::ACCEL_BIAS;
            kept.value.head<3>() = Eigen::Map<Eigen::Vector3d>(bias);
            system.prior.blocks.push_back(kept);
        }
    }
    std::vector<double> residuals;
    ceres::CRSMatrix sparse_jacobian;
    double cost = 0.0;
    if (!problem.Evaluate(evaluate_options, &cost, &residuals, nullptr, &sparse_jacobian)
        || !std::isfinite(cost) || sparse_jacobian.num_cols != 3 * evaluate_options.parameter_blocks.size())
    {
        throw std::runtime_error("CT marginalization linearization failed");
    }
    Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(sparse_jacobian.num_rows, sparse_jacobian.num_cols);
    for (int row = 0; row < sparse_jacobian.num_rows; row++)
    {
        for (int j = sparse_jacobian.rows[row]; j < sparse_jacobian.rows[row + 1]; j++)
        {
            jacobian(row, sparse_jacobian.cols[j]) = sparse_jacobian.values[j];
        }
    }
    Eigen::Map<const Eigen::VectorXd> error(residuals.data(), residuals.size());
    system.hessian = jacobian.transpose() * jacobian;
    system.gradient = jacobian.transpose() * error;
    system.prior.start_time = next_start;
    if (!system.hessian.allFinite() || !system.gradient.allFinite() || system.prior.blocks.empty())
    {
        throw std::runtime_error("CT marginalization produced an invalid normal system");
    }
    return true;
}

CtGinsSolveResult CtGinsBackend::Solve(const CtGinsSolveOptions& options)
{
    if (!options.adaptive_gnss)
    {
        return SolveFixedWeights(options);
    }
    if (!std::isfinite(options.gnss_scale_lambda) || options.gnss_scale_lambda < 0.0
        || !std::isfinite(options.gnss_scale_max) || options.gnss_scale_max < 1.0)
    {
        throw std::invalid_argument("CT GNSS scale requires finite lambda >= 0 and maximum >= 1");
    }
    // 初解、尺度更新和二次求解共同提交；任一步失败都保留原轨迹、零偏和权重。
    CtGinsBackend candidate = *this;
    CtGinsSolveResult preliminary = candidate.SolveFixedWeights(options);
    if (!preliminary.usable)
    {
        return preliminary;
    }
    std::vector<CtGnssWeightUpdate> updates = candidate.UpdateGnssCovarianceScales(options);
    CtGinsSolveResult result = candidate.SolveFixedWeights(options);
    result.preliminary_iterations = preliminary.iterations;
    result.iterations += preliminary.iterations;
    result.gnss_weight_updates = std::move(updates);
    if (result.usable)
    {
        *this = std::move(candidate);
    }
    return result;
}

double CtGinsBackend::ObservationEndTime() const
{
    double start_time = _position_spline.MinTime();
    double end_time = _position_spline.MaxTime();
    if (_marginalization_enabled)
    {
        // 尾段过短时，末个控制点近乎无约束；先保留这些观测，等完整一段后再参与求解。
        double complete_end = start_time + std::floor(
            (_imu_data.back().time_s - start_time) / _knot_interval) * _knot_interval;
        if (complete_end > start_time)
        {
            end_time = std::min(end_time,
                std::nextafter(complete_end, std::numeric_limits<double>::infinity()));
        }
    }
    return end_time;
}

std::vector<CtGnssWeightUpdate> CtGinsBackend::UpdateGnssCovarianceScales(const CtGinsSolveOptions& options)
{
    std::vector<CtGnssWeightUpdate> updates;
    double start_time = _position_spline.MinTime();
    double end_time = ObservationEndTime();
    double previous_scale = 1.0;
    for (std::size_t i = 0; i < _gnss_data.size(); i++)
    {
        CtGnssPosition& observation = _gnss_data[i];
        if (observation.time_s < start_time)
        {
            // 窗前保留的最后一条观测提供时间平滑的边界，不重估已进入先验的权重。
            previous_scale = observation.covariance_scale;
            continue;
        }
        if (observation.time_s >= end_time)
        {
            break;
        }
        Eigen::Vector3d prediction = _position_spline.Evaluate(observation.time_s)
            + _attitude_spline.Evaluate(observation.time_s) * _lever_arm;
        Eigen::Vector3d error = prediction - (observation.position_ecef_m - _position_origin);
        Eigen::LLT<Eigen::Matrix3d> decomposition(observation.position_covariance_ecef_m2);
        Eigen::Vector3d normalized_error = decomposition.matrixL().solve(error);
        double q = normalized_error.squaredNorm();
        if (!std::isfinite(q))
        {
            throw std::runtime_error("CT GNSS scale update produced a nonfinite normalized residual");
        }
        double scale = EstimateGnssCovarianceScale(q, previous_scale,
            options.gnss_scale_lambda, options.gnss_scale_max);
        updates.push_back(CtGnssWeightUpdate{observation.time_s, q, previous_scale, scale});
        observation.covariance_scale = scale;
        previous_scale = scale;
    }
    return updates;
}

CtGinsSolveResult CtGinsBackend::SolveFixedWeights(const CtGinsSolveOptions& options)
{
    if (!_initialized || !_parameters_set)
    {
        throw std::logic_error("CT solve requires explicit parameters and an initialized trajectory");
    }
    const double deviations[] = {options.gyro_std_rad_s, options.accel_std_m_s2,
        options.gyro_bias_prior_std_rad_s, options.accel_bias_prior_std_m_s2,
        options.initial_attitude_std_rad};
    for (int i = 0; i < 5; i++)
    {
        if (!std::isfinite(deviations[i]) || deviations[i] <= 0.0|| !std::isfinite(1.0 / deviations[i]))
        {
            throw std::invalid_argument("CT measurement and prior standard deviations must be finite and positive");
        }
    }
    if (options.max_iterations <= 0 || options.num_threads <= 0)
    {
        throw std::invalid_argument("CT solver iterations and thread count must be positive");
    }

    CtGinsSolveResult result;
    double start_time = _position_spline.MinTime();
    double end_time = ObservationEndTime();
    result.optimized_until_s = std::min(_imu_data.back().time_s,
        std::nextafter(end_time, start_time));
    // 在副本上求解，失败时保留已接受的轨迹和零偏。
    RdSpline<3, SplineOrder> position_spline = _position_spline;
    So3Spline<SplineOrder> attitude_spline = _attitude_spline;
    Eigen::Vector3d gyro_bias = _gyro_bias;
    Eigen::Vector3d accel_bias = _accel_bias;
    Eigen::Vector3d gravity = _gravity;
    // 组装因子时使用相对起点的时间，保存结果前恢复原时间基准。
    position_spline.SetStartTime(0.0);
    attitude_spline.SetStartTime(0.0);

    ceres::EigenQuaternionManifold quaternion_manifold;
    ceres::Problem::Options problem_options;
    problem_options.manifold_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    ceres::Problem problem(problem_options);
    for (std::size_t i = 0; i < _imu_data.size(); i++)
    {
        CtImuSample sample = _imu_data[i];
        if (sample.time_s < start_time || sample.time_s >= end_time)
        {
            result.imu_samples_skipped++;
            continue;
        }
        sample.time_s -= start_time;
        SplineMeta<SplineOrder> meta;
        std::vector<double*> blocks;
        sample.time_s = GetLocalBlocks(position_spline, attitude_spline, sample.time_s, meta, blocks);
        std::vector<double*> gyro_blocks;
        for (int j = 0; j < SplineOrder; j++)
        {
            gyro_blocks.push_back(blocks[j]);
        }
        gyro_blocks.push_back(gyro_bias.data());
        ceres::CostFunction* gyro_cost = IMUGyroFactorWithConstBias<SplineOrder>::Create(
            meta, sample, 1.0 / options.gyro_std_rad_s);
        problem.AddResidualBlock(gyro_cost, nullptr, gyro_blocks);
        blocks.push_back(accel_bias.data());
        blocks.push_back(gravity.data());
        ceres::CostFunction* accel_cost = IMUAcceFactorWithConstBias<SplineOrder>::Create(
            meta, meta, sample, 1.0 / options.accel_std_m_s2);
        problem.AddResidualBlock(accel_cost, nullptr, blocks);
        result.imu_samples_used++;
    }
    for (std::size_t i = 0; i < _gnss_data.size(); i++)
    {
        CtGnssPosition observation = _gnss_data[i];
        if (observation.time_s < start_time || observation.time_s >= end_time)
        {
            result.gnss_positions_skipped++;
            continue;
        }
        observation.time_s -= start_time;
        SplineMeta<SplineOrder> meta;
        std::vector<double*> blocks;
        observation.time_s = GetLocalBlocks(position_spline, attitude_spline, observation.time_s, meta, blocks);
        ceres::CostFunction* cost = CtGnssPositionFactor<SplineOrder>::Create(
            meta, meta, observation, _lever_arm, _position_origin);
        problem.AddResidualBlock(cost, nullptr, blocks);
        result.gnss_positions_used++;
    }
    bool have_prior = !_marginalization_prior.blocks.empty();
    if (result.imu_samples_used < 2 || (_fixed_prefix_knots == 0 && !have_prior && result.gnss_positions_used < 2))
    {
        throw std::invalid_argument("CT joint solve requires at least two IMU and two GNSS observations inside the spline interval");
    }

    // 反复求解时，零偏先验始终以标定或初始对准值为中心。
    if (have_prior)
    {
        AddMarginalizationPrior(problem, _marginalization_prior, position_spline, attitude_spline,
            gyro_bias, accel_bias);
    }
    else if (_fixed_prefix_knots == 0)
    {
        SplineMeta<SplineOrder> initial_meta;
        std::vector<double*> attitude_blocks;
        GetLocalBlocks(position_spline, attitude_spline, 0.0, initial_meta, attitude_blocks);
        attitude_blocks.resize(SplineOrder);
        std::vector<int> sizes(SplineOrder, 4);
        InitialAttitudePrior* attitude_prior = new InitialAttitudePrior(
            _initial_rotation, _knot_interval, options.initial_attitude_std_rad);
        ceres::CostFunction* attitude_cost = ct_imu_detail::MakeCost(attitude_prior, sizes);
        problem.AddResidualBlock(attitude_cost, nullptr, attitude_blocks);
    }
    problem.AddResidualBlock(new VectorPrior(_initial_gyro_bias,
        options.gyro_bias_prior_std_rad_s), nullptr, gyro_bias.data());
    problem.AddResidualBlock(new VectorPrior(_initial_accel_bias,
        options.accel_bias_prior_std_m_s2), nullptr, accel_bias.data());
    problem.SetParameterBlockConstant(gravity.data());
    for (std::size_t i = 0; i < attitude_spline.GetKnots().size(); i++)
    {
        double* knot = attitude_spline.GetKnot(static_cast<int>(i)).data();
        if (problem.HasParameterBlock(knot))
        {
            problem.SetManifold(knot, &quaternion_manifold);
            if (i < static_cast<std::size_t>(_fixed_prefix_knots))
            {
                problem.SetParameterBlockConstant(knot);
            }
        }
        double* position = position_spline.GetKnot(static_cast<int>(i)).data();
        if (i < static_cast<std::size_t>(_fixed_prefix_knots) && problem.HasParameterBlock(position))
        {
            problem.SetParameterBlockConstant(position);
        }
    }

    ceres::Solver::Options solver_options;
    solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    solver_options.max_num_iterations = options.max_iterations;
    solver_options.num_threads = options.num_threads;
    solver_options.logging_type = ceres::SILENT;
    ceres::Solver::Summary summary;
    ceres::Solve(solver_options, &problem, &summary);
    result.usable = summary.IsSolutionUsable() && std::isfinite(summary.final_cost);
    result.converged = summary.termination_type == ceres::CONVERGENCE;
    result.iterations = static_cast<int>(summary.iterations.size());
    result.initial_cost = summary.initial_cost;
    result.final_cost = summary.final_cost;
    result.message = summary.BriefReport();
    if (result.usable)
    {
        position_spline.SetStartTime(start_time);
        attitude_spline.SetStartTime(start_time);
        _position_spline = std::move(position_spline);
        _attitude_spline = std::move(attitude_spline);
        _gyro_bias = gyro_bias;
        _accel_bias = accel_bias;
        _solved = true;
    }
    return result;
}

bool CtGinsBackend::IsSolved() const
{
    return _solved;
}

double CtGinsBackend::MinTime() const
{
    return GetPositionSpline().MinTime();
}

double CtGinsBackend::MaxTime() const
{
    return GetPositionSpline().MaxTime();
}

CtGinsState CtGinsBackend::GetState(double time_s) const
{
    if (!_initialized)
    {
        throw std::logic_error("CT trajectory is not initialized");
    }
    if (!std::isfinite(time_s) || time_s < _position_spline.MinTime() || time_s >= _position_spline.MaxTime())
    {
        throw std::out_of_range("CT query time is outside the spline interval");
    }
    CtGinsState state;
    state.time_s = time_s;
    state.position_ecef_m = _position_spline.Evaluate(time_s) + _position_origin;
    state.velocity_ecef_m_s = _position_spline.Velocity(time_s);
    state.rotation_body_to_ecef = _attitude_spline.Evaluate(time_s).unit_quaternion();
    return state;
}

const std::vector<CtImuSample>& CtGinsBackend::GetImuSamples() const
{
    return _imu_data;
}

const std::vector<CtGnssPosition>& CtGinsBackend::GetGnssPositions() const
{
    return _gnss_data;
}

const Eigen::Vector3d& CtGinsBackend::GetPositionOrigin() const
{
    if (!_initialized)
    {
        throw std::logic_error("CT trajectory is not initialized");
    }
    return _position_origin;
}

const Eigen::Vector3d& CtGinsBackend::GetLeverArm() const
{
    return _lever_arm;
}

const Eigen::Vector3d& CtGinsBackend::GetGravity() const
{
    return _gravity;
}

const Eigen::Vector3d& CtGinsBackend::GetGyroBias() const
{
    return _gyro_bias;
}

const Eigen::Vector3d& CtGinsBackend::GetAccelBias() const
{
    return _accel_bias;
}

const RdSpline<3, CtGinsBackend::SplineOrder>& CtGinsBackend::GetPositionSpline() const
{
    if (!_initialized)
    {
        throw std::logic_error("CT trajectory is not initialized");
    }
    return _position_spline;
}

const So3Spline<CtGinsBackend::SplineOrder>& CtGinsBackend::GetAttitudeSpline() const
{
    if (!_initialized)
    {
        throw std::logic_error("CT trajectory is not initialized");
    }
    return _attitude_spline;
}

}  // namespace gct
}  // namespace gfgomsf
