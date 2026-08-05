#ifndef P_DD_FACTOR
#define P_DD_FACTOR

/**
 * @file         pseudorange_DD_factor.h
 * @author       GREAT-WHU (https://github.com/GREAT-WHU)
 * @brief        construct Double-Different pseudo-range factor
 * @version      1.0
 * @date         2025-11-04
 *
 * @copyright Copyright (c) 2025, Wuhan University. All rights reserved.
 *
 */

#include"gfgo/gutility.h"
#include "gexport/ExportLibGREAT.h"
#include "gmodels/gbiasmodel.h"

using namespace great;

namespace gfgo
{
	class LibGREAT_LIBRARY_EXPORT PseudorangeDDFactor : public ceres::SizedCostFunction<1,3>
	{
	public:

		/**
		 * @brief Pseudorange double-difference factor constructor
		 * Initializes with time, site pairs, parameters, satellite data and frequency band
		 */
		PseudorangeDDFactor(const t_gtime &cur_time, const pair<string, string> &base_rover_site, const t_gallpar  &params, const vector<pair<t_gsatdata, t_gsatdata>> &DD_sat_data, t_gbiasmodel *bias_model, const pair<FREQ_SEQ, GOBSBAND> &freq_band);
		
		/**
		 * @brief Update rover coordinates in parameter set
		 * Replaces rover position parameters with current estimate values
		 */
		void updatePara(t_gallpar &params_tmp,const Eigen::Vector3d &Pi, const Eigen::Vector3d &Vi=Eigen::Vector3d::Identity()) const;
		
		/**
		 * @brief Transform pseudorange design matrices to Eigen format
		 * Converts sparse design matrix, weight matrix and residuals to dense Eigen matrices for pseudorange double-difference observations
		 */
		void trans2Eigen(const vector<vector<pair<int, double>>> &B, const vector<double> &P, const vector<double> &l, Eigen::Matrix<double, 2, 3> &B_new, Eigen::Matrix<double, 2, 2> &P_new, Eigen::Matrix<double, 2, 1> &l_new) const;
		
		/**
		 * @brief Evaluate pseudorange double-difference factor for optimization
		 *
		 * Computes residuals and Jacobians for pseudorange DD observations in factor graph.
		 * Handles coordinate updates and measurement weighting. Constructs DD equations from single-difference pseudorange observations with proper covariance scaling.
		 */
		virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;		
		
		/**
		 * @brief Debug function for factor verification
		 * Tests residual and Jacobian computation with current parameters
		 * Outputs numerical results for validation and debugging purposes
		 */
		void check(double **parameters);		
	protected:
		t_gtime _cur_time;		
		pair<string, string> _base_rover_site;
		t_gallpar _params;
		//shared_ptr<t_gbiasmodel> _gprecise_bias_model = nullptr;
		t_gbiasmodel *_gprecise_bias_model = nullptr;
		pair<FREQ_SEQ, GOBSBAND> _freq_band;
		vector<pair<t_gsatdata, t_gsatdata>> _DD_sat_data;	
	
	};

}
#endif