#include "gct/ct_gins_backend.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace gfgomsf
{
namespace gct
{

CtGinsBackend::CtGinsBackend()
    : _knot_interval(0.1), _initialized(false),
      _position_origin(Eigen::Vector3d::Zero()), _lever_arm(Eigen::Vector3d::Zero()),
      _gravity(Eigen::Vector3d::Zero()),
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
    _initialized = false;
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
}

void CtGinsBackend::ClearData()
{
    _imu_data.clear();
    _gnss_data.clear();
    _position_spline = RdSpline<3, SplineOrder>(_knot_interval);
    _attitude_spline = So3Spline<SplineOrder>(_knot_interval);
    _position_origin.setZero();
    _initialized = false;
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
    // Floating point rounding must not put the final observation on MaxTime().
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
        // Cubic uniform splines reproduce a line when knot i samples t0+(i-1)*dt.
        // Boundary control points use the first/last GNSS segment for extrapolation.
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
    _initialized = true;
}

bool CtGinsBackend::IsInitialized() const
{
    return _initialized;
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
