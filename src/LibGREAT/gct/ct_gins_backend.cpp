#include "gct/ct_gins_backend.h"
#include "gfactor/imu_gyro_factor.hpp"
#include "gfactor/imu_acce_factor.hpp"
#include "gfactor/ct_gnss_position_factor.h"
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

// The prior constrains R(t_start), not the first quaternion control point.
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

// Local blocks: four [qx,qy,qz,qw] attitude knots, then four position knots.
// Both splines must use the same start time and knot interval.
double GetLocalBlocks(RdSpline<3, CtGinsBackend::SplineOrder>& position_spline,
                    So3Spline<CtGinsBackend::SplineOrder>& attitude_spline, double time,
                    SplineMeta<CtGinsBackend::SplineOrder>& meta,
                    std::vector<double*>& blocks)
{
    std::pair<double, std::size_t> location = position_spline.ComputeTIndex(time);
    double interval = position_spline.GetTimeInterval();
    // Factors use segment-relative seconds in [0, dt).
    // Clamp roundoff at knots; 3.4 - 17*0.2 can be slightly negative.
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

}  // namespace

CtGinsBackend::CtGinsBackend()
    : _knot_interval(0.1), _initialized(false), _parameters_set(false), _solved(false),
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
    _initialized = false;
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
    _initialized = false;
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

    // One extra segment keeps the final observation inside the half-open interval.
    double segment_count = std::floor(duration / _knot_interval) + 1.0;
    if (!std::isfinite(segment_count)|| segment_count > std::numeric_limits<int>::max() - SplineOrder + 1)
    {
        throw std::invalid_argument("CT initial trajectory has too many knots");
    }
    int knot_count = static_cast<int>(segment_count) + SplineOrder - 1;
    double max_time = start_time + segment_count * _knot_interval;
    // Roundoff can place the last observation on the excluded upper bound.
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
        // Samples at t0+(i-1)*dt reproduce constant velocity in a cubic spline.
        // Boundary knots extrapolate the first/last GNSS segment.
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

CtGinsSolveResult CtGinsBackend::Solve(const CtGinsSolveOptions& options)
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
    double start_time = MinTime();
    double end_time = MaxTime();
    // Solve on copies; an unusable result leaves the accepted state unchanged.
    RdSpline<3, SplineOrder> position_spline = _position_spline;
    So3Spline<SplineOrder> attitude_spline = _attitude_spline;
    Eigen::Vector3d gyro_bias = _gyro_bias;
    Eigen::Vector3d accel_bias = _accel_bias;
    Eigen::Vector3d gravity = _gravity;
    // Use t-start_time during assembly; restore the caller's time origin before saving.
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
    if (result.imu_samples_used < 2 || result.gnss_positions_used < 2)
    {
        throw std::invalid_argument("CT joint solve requires at least two IMU and two GNSS observations inside the spline interval");
    }

    // Repeated solves keep bias priors centred on the calibration/alignment values.
    SplineMeta<SplineOrder> initial_meta;
    std::vector<double*> attitude_blocks;
    GetLocalBlocks(position_spline, attitude_spline, 0.0, initial_meta, attitude_blocks);
    attitude_blocks.resize(SplineOrder);
    std::vector<int> sizes(SplineOrder, 4);
    InitialAttitudePrior* attitude_prior = new InitialAttitudePrior(
        _initial_rotation, _knot_interval, options.initial_attitude_std_rad);
    ceres::CostFunction* attitude_cost = ct_imu_detail::MakeCost(attitude_prior, sizes);
    problem.AddResidualBlock(attitude_cost, nullptr, attitude_blocks);
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
    if (!std::isfinite(time_s) || time_s < MinTime() || time_s >= MaxTime())
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
