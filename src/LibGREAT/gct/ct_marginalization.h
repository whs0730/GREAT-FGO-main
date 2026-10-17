// CTraj: Continuous-Time Trajectory (Time-Varying State) Representation and Estimation Library
// Copyright 2024, the School of Geodesy and Geomatics (SGG), Wuhan University, China
// https://github.com/Unsigned-Long/CTraj.git
//
// Author: Shuolong Chen (shlchen@whu.edu.cn)
// GitHub: https://github.com/Unsigned-Long
//  ORCID: 0000-0002-5283-9057
//
// Purpose: See .h/.hpp file.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * The names of its contributors can not be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef GCT_MARGINALIZATION_H
#define GCT_MARGINALIZATION_H

#include <Eigen/Dense>
#include <vector>

namespace gfgomsf
{
namespace gct
{

struct CtMarginalizationBlock
{
    enum Type { ROTATION, POSITION, GYRO_BIAS, ACCEL_BIAS };
    Type type = POSITION;
    int knot = 0;
    // Rotation: [qx,qy,qz,qw]; other blocks use the first three entries.
    Eigen::Vector4d value = Eigen::Vector4d::Zero();
};

struct CtMarginalizationPrior
{
    double start_time = 0.0;
    std::vector<CtMarginalizationBlock> blocks;
    Eigen::MatrixXd jacobian;
    Eigen::VectorXd residual;
};

struct CtMarginalizationSystem
{
    // Dropped variables precede retained variables; every local block is 3D.
    int marginalized_size = 0;
    Eigen::MatrixXd hessian;
    Eigen::VectorXd gradient;
    CtMarginalizationPrior prior;
    std::size_t imu_samples = 0;
    std::size_t gnss_positions = 0;
};

}  // namespace gct
}  // namespace gfgomsf

#endif
