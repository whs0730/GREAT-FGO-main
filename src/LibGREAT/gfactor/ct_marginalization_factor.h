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

#ifndef CT_MARGINALIZATION_FACTOR_H
#define CT_MARGINALIZATION_FACTOR_H

#include "gct/ct_marginalization.h"
#include <ceres/dynamic_autodiff_cost_function.h>
#include <sophus/so3.hpp>

namespace gfgomsf
{
namespace gct
{

struct CtMarginalizationFactor
{
    explicit CtMarginalizationFactor(const CtMarginalizationPrior& prior) : prior(prior)
    {
    }

    template <class T>
    bool operator()(T const* const* parameters, T* residuals) const
    {
        Eigen::Matrix<T, Eigen::Dynamic, 1> delta(3 * prior.blocks.size());
        for (std::size_t i = 0; i < prior.blocks.size(); i++)
        {
            const CtMarginalizationBlock& block = prior.blocks[i];
            if (block.type == CtMarginalizationBlock::ROTATION)
            {
                Eigen::Map<const Eigen::Quaternion<T>> rotation(parameters[i]);
                Eigen::Quaternion<T> reference(T(block.value[3]), T(block.value[0]),T(block.value[1]), T(block.value[2]));
                // EigenQuaternionManifold uses left increments with half-angle coordinates.
                Sophus::SO3<T> difference(rotation * reference.conjugate());
                delta.template segment<3>(3 * i) = T(0.5) * difference.log();
            }
            else
            {
                Eigen::Map<const Eigen::Matrix<T, 3, 1>> value(parameters[i]);
                delta.template segment<3>(3 * i) = value - block.value.head<3>().template cast<T>();
            }
        }
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> error(residuals, prior.residual.size());
        error = prior.residual.template cast<T>() + prior.jacobian.template cast<T>() * delta;
        return true;
    }

    static ceres::CostFunction* Create(const CtMarginalizationPrior& prior)
    {
        ceres::DynamicAutoDiffCostFunction<CtMarginalizationFactor>* cost =
            new ceres::DynamicAutoDiffCostFunction<CtMarginalizationFactor>(new CtMarginalizationFactor(prior));
        for (std::size_t i = 0; i < prior.blocks.size(); i++)
        {
            cost->AddParameterBlock(prior.blocks[i].type == CtMarginalizationBlock::ROTATION ? 4 : 3);
        }
        cost->SetNumResiduals(static_cast<int>(prior.residual.size()));
        return cost;
    }

    CtMarginalizationPrior prior;
};

}  // namespace gct
}  // namespace gfgomsf

#endif
