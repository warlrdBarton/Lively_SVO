#pragma once

#pragma diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
// Eigen 3.2.7 uses std::binder1st and std::binder2nd which are deprecated since c++11
// Fix is in 3.3 devel (http://eigen.tuxfamily.org/bz/show_bug.cgi?id=872).
#include <Eigen/Core>
#pragma diagnostic pop

#include "svo/ceres_backend/estimator_types.hpp"
#include "svo/ceres_backend/parameter_block.hpp"


namespace svo
{
    namespace ceres_backend
    {

        /// \brief Wraps the parameter block for a gravity
        class GravityParameterBlock : public ParameterBlock
        {
        public:
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            typedef Eigen::Vector3d estimate_t;

            static constexpr size_t c_dimension = 3;
            static constexpr size_t c_minimal_dimension = 3;

            /// \brief Default constructor (assumes not fixed).
            GravityParameterBlock();

            /// \brief Constructor with estimate and id.
            /// @param[in] speedAndBias The speed and bias estimate.
            /// @param[in] id The (unique) ID of this block.
            GravityParameterBlock(const Eigen::Vector3d &gravity, uint64_t id);

            virtual ~GravityParameterBlock() {}

            // ---------------------------------------------------------------------------
            // Setters

            virtual void setEstimate(const Eigen::Vector3d &gravity)
            {
                estimate_ = gravity;
            }

            // ---------------------------------------------------------------------------
            // Getters

            virtual const Eigen::Vector3d &estimate() const { return estimate_; }

            virtual double *parameters() { return estimate_.data(); }

            virtual const double *parameters() const { return estimate_.data(); }

            virtual size_t dimension() const { return c_dimension; }

            virtual size_t minimalDimension() const { return c_minimal_dimension; }

            Eigen::MatrixXd TangentBasis(const Eigen::Vector3d &g0) const
            {
                Eigen::Vector3d b, c;
                Eigen::Vector3d a = g0.normalized();
                Eigen::Vector3d tmp(0, 0, 1);
                if (a == tmp)
                    tmp << 1, 0, 0;
                b = (tmp - a * (a.transpose() * tmp)).normalized();
                c = a.cross(b);
                Eigen::MatrixXd bc(3, 2);
                bc.block<3, 1>(0, 0) = b;
                bc.block<3, 1>(0, 1) = c;
                return bc;
            }

            virtual void plus(const double *x0, const double *uv,
                              double *x0_plus_Delta) const
            {
                Eigen::Map<const Eigen::Matrix<double, 3, 1>> x0_(x0);
                Eigen::Map<const Eigen::Matrix<double, 2, 1>> Delta_uv_(uv);
                Eigen::Map<Eigen::Matrix<double, 3, 1>> x0_plus_Delta_(x0_plus_Delta);
                Eigen::MatrixXd basis = TangentBasis(x0_);
                x0_plus_Delta_ = x0_ + basis * Delta_uv_;
            }

            /// \brief The jacobian of Plus(x, delta) w.r.t delta at delta = 0.
            /// @param[in] x0 Variable.
            /// @param[out] jacobian The Jacobian.
            virtual void plusJacobian(const double * /*unused: x*/,
                                      double *jacobian) const
            {
                Eigen::Map<Eigen::Matrix<double, 3, 2, Eigen::RowMajor>> j_uv(jacobian);
                j_uv = TangentBasis(this->estimate_);
            }


            Eigen::MatrixXd pseudoInverse(const Eigen::MatrixXd &a, double epsilon = std::numeric_limits<double>::epsilon()) const
            {
                Eigen::JacobiSVD<Eigen::MatrixXd> svd(a, Eigen::ComputeThinU | Eigen::ComputeThinV);

                double tolerance = epsilon * std::max(a.cols(), a.rows()) * svd.singularValues().array().abs()(0);

                return svd.matrixV() * (svd.singularValues().array().abs() > tolerance).select(svd.singularValues().array().inverse(), 0).matrix().asDiagonal() * svd.matrixU().adjoint();
            }

            virtual void minus(const double *x0, const double *x0_plus_Delta,
                               double *Delta_Chi) const
            {
                Eigen::Map<const Eigen::Matrix<double, 3, 1>> x0_(x0);
                Eigen::Map<Eigen::Matrix<double, 2, 1>> Delta_Chi_(Delta_Chi);
                Eigen::Map<const Eigen::Matrix<double, 9, 1>> x0_plus_Delta_(x0_plus_Delta);
                Eigen::MatrixXd basis = TangentBasis(x0_);
                Delta_Chi_ = pseudoInverse(basis) * (x0_plus_Delta_ - x0_);
            }

 
            virtual void liftJacobian(const double * /*unused: x*/,
                                      double *jacobian) const
            {
                Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> j_uv(jacobian);
                j_uv = pseudoInverse(TangentBasis(this->estimate_));
            }

            /// @brief Return parameter block type as string
            virtual std::string typeInfo() const
            {
                return "GravityParameterBlock";
            }

        private:
            Eigen::Vector3d estimate_;
        };
    }
}