// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "colmap/estimators/solvers/essential_matrix.h"

#include "colmap/geometry/essential_matrix.h"
#include "colmap/math/polynomial.h"
#include "colmap/util/eigen_alignment.h"
#include "colmap/util/logging.h"

#include <Eigen/Geometry>
#include <Eigen/LU>
#include <Eigen/SVD>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace colmap {

void EssentialMatrixFivePointEstimator::Estimate(
    const std::vector<X_t>& cam_rays1,
    const std::vector<Y_t>& cam_rays2,
    std::vector<M_t>* models) {
  THROW_CHECK_EQ(cam_rays1.size(), cam_rays2.size());
  THROW_CHECK_GE(cam_rays1.size(), 5);
  THROW_CHECK(models != nullptr);

  models->clear();

  // Setup system of equations: [cam_rays2(i,:), 1]' * E * [cam_rays1(i,:), 1]'.

  Eigen::Matrix<double, Eigen::Dynamic, 9> Q(cam_rays1.size(), 9);
  for (size_t i = 0; i < cam_rays1.size(); ++i) {
    Q.row(i) << cam_rays2[i].x() * cam_rays1[i].transpose(),
        cam_rays2[i].y() * cam_rays1[i].transpose(),
        cam_rays2[i].z() * cam_rays1[i].transpose();
  }

  // Step 1: Extraction of the nullspace.

  Eigen::Matrix<double, 9, 4> E;
  if (cam_rays1.size() == 5) {
    E = Q.transpose().fullPivHouseholderQr().matrixQ().rightCols<4>();
  } else {
    const Eigen::JacobiSVD<Eigen::Matrix<double, Eigen::Dynamic, 9>> svd(
        Q, Eigen::ComputeFullV);
    E = svd.matrixV().rightCols<4>();
  }

  // Step 3: Gauss-Jordan elimination with partial pivoting on A.

  Eigen::Matrix<double, 10, 20> A;
#include "colmap/estimators/solvers/essential_matrix_poly.h"
  const Eigen::Matrix<double, 10, 10> AA =
      A.block<10, 10>(0, 0).partialPivLu().solve(A.block<10, 10>(0, 10));

  // Step 4: Expansion of the determinant polynomial of the 3x3 polynomial
  //         matrix B to obtain the tenth degree polynomial.

  Eigen::Matrix<double, 13, 3> B;
  for (size_t i = 0; i < 3; ++i) {
    B(0, i) = 0;
    B(4, i) = 0;
    B(8, i) = 0;
    B.block<3, 1>(1, i) = AA.block<1, 3>(i * 2 + 4, 0);
    B.block<3, 1>(5, i) = AA.block<1, 3>(i * 2 + 4, 3);
    B.block<4, 1>(9, i) = AA.block<1, 4>(i * 2 + 4, 6);
    B.block<3, 1>(0, i) -= AA.block<1, 3>(i * 2 + 5, 0);
    B.block<3, 1>(4, i) -= AA.block<1, 3>(i * 2 + 5, 3);
    B.block<4, 1>(8, i) -= AA.block<1, 4>(i * 2 + 5, 6);
  }

  // Step 5: Extraction of roots from the degree 10 polynomial.
  Eigen::Matrix<double, 11, 1> coeffs;
#include "colmap/estimators/solvers/essential_matrix_coeffs.h"

  Eigen::VectorXd roots_real;
  Eigen::VectorXd roots_imag;
  if (!FindPolynomialRootsCompanionMatrix(coeffs, &roots_real, &roots_imag)) {
    return;
  }

  const int num_roots = roots_real.size();
  models->reserve(num_roots);

  for (int i = 0; i < num_roots; ++i) {
    const double kMaxRootImag = 1e-10;
    if (std::abs(roots_imag(i)) > kMaxRootImag) {
      continue;
    }

    const double z1 = roots_real(i);
    const double z2 = z1 * z1;
    const double z3 = z2 * z1;
    const double z4 = z3 * z1;

    Eigen::Matrix3d Bz;
    for (int j = 0; j < 3; ++j) {
      Bz(j, 0) = B(0, j) * z3 + B(1, j) * z2 + B(2, j) * z1 + B(3, j);
      Bz(j, 1) = B(4, j) * z3 + B(5, j) * z2 + B(6, j) * z1 + B(7, j);
      Bz(j, 2) = B(8, j) * z4 + B(9, j) * z3 + B(10, j) * z2 + B(11, j) * z1 +
                 B(12, j);
    }

    const Eigen::JacobiSVD<Eigen::Matrix3d> svd(Bz, Eigen::ComputeFullV);
    const Eigen::Vector3d X = svd.matrixV().rightCols<1>();

    const double kMaxX3 = 1e-10;
    if (std::abs(X(2)) < kMaxX3) {
      continue;
    }

    const Eigen::Matrix<double, 9, 1> e =
        (E.col(0) * (X(0) / X(2)) + E.col(1) * (X(1) / X(2)) + E.col(2) * z1 +
         E.col(3))
            .normalized();

    models->push_back(
        Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(
            e.data()));
  }
}

void EssentialMatrixFivePointEstimator::Residuals(
    const std::vector<X_t>& cam_rays1,
    const std::vector<Y_t>& cam_rays2,
    const M_t& E,
    std::vector<double>* residuals) {
  ComputeSquaredSampsonError(cam_rays1, cam_rays2, E, residuals);
}

void EssentialMatrixEightPointEstimator::Estimate(
    const std::vector<X_t>& cam_rays1,
    const std::vector<Y_t>& cam_rays2,
    std::vector<M_t>* models) {
  THROW_CHECK_EQ(cam_rays1.size(), cam_rays2.size());
  THROW_CHECK_GE(cam_rays1.size(), 8);
  THROW_CHECK(models != nullptr);

  models->clear();

  // Setup homogeneous linear equation as x2' * E * x1 = 0.
  Eigen::Matrix<double, Eigen::Dynamic, 9> A(cam_rays1.size(), 9);
  for (size_t i = 0; i < cam_rays1.size(); ++i) {
    A.row(i) << cam_rays2[i].x() * cam_rays1[i].transpose(),
        cam_rays2[i].y() * cam_rays1[i].transpose(),
        cam_rays2[i].z() * cam_rays1[i].transpose();
  }

  // Solve for the nullspace of the constraint matrix.
  Eigen::Matrix3d Q;
  if (cam_rays1.size() == 8) {
    Eigen::Matrix<double, 9, 9> QQ =
        A.transpose().householderQr().householderQ();
    Q = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(
        QQ.col(8).data());
  } else {
    Eigen::JacobiSVD<Eigen::Matrix<double, Eigen::Dynamic, 9>> svd(
        A, Eigen::ComputeFullV);
    Q = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(
        svd.matrixV().col(8).data());
  }

  // Enforcing the internal constraint that two singular values must be non-zero
  // and one must be zero.
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      Q, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Vector3d singular_values = svd.singularValues();
  singular_values(2) = 0.0;
  const Eigen::Matrix3d E =
      svd.matrixU() * singular_values.asDiagonal() * svd.matrixV().transpose();

  models->resize(1);
  (*models)[0] = E;
}

void EssentialMatrixEightPointEstimator::Residuals(
    const std::vector<X_t>& cam_rays1,
    const std::vector<Y_t>& cam_rays2,
    const M_t& E,
    std::vector<double>* residuals) {
  ComputeSquaredSampsonError(cam_rays1, cam_rays2, E, residuals);
}

namespace {

double ComputeSquaredProjectedEpipolarError(const Eigen::Vector3d& ray1,
                                            const Eigen::Vector3d& ray2,
                                            const Eigen::Matrix3d& E) {
  const Eigen::Vector3d epipolar_plane1 = E.transpose() * ray2;
  const double numerator = ray2.dot(E * ray1);
  const double denominator = ray1.squaredNorm() * epipolar_plane1.squaredNorm();
  if (denominator == 0) {
    return std::numeric_limits<double>::max();
  }
  return numerator * numerator / denominator;
}

void ComputeSquaredProjectedEpipolarError(
    const std::vector<Eigen::Vector3d>& rays1,
    const std::vector<Eigen::Vector3d>& rays2,
    const Eigen::Matrix3d& E,
    std::vector<double>* residuals) {
  const size_t num_rays1 = rays1.size();
  THROW_CHECK_EQ(num_rays1, rays2.size());
  residuals->resize(num_rays1);
  for (size_t i = 0; i < num_rays1; ++i) {
    (*residuals)[i] =
        ComputeSquaredProjectedEpipolarError(rays1[i], rays2[i], E);
  }
}

struct SphericalNormalizationModel {
  double S = 1.0;
  double K = 1.0;
  double score = std::numeric_limits<double>::max();
  Eigen::Matrix3d E = Eigen::Matrix3d::Zero();
};

// Run the standard 8-point algorithm on normalized bearing vectors and
// denormalize the resulting essential matrix.
Eigen::Matrix3d EstimateAndDenormalize(
    const std::vector<Eigen::Vector3d>& rays1,
    const std::vector<Eigen::Vector3d>& rays2,
    const Eigen::DiagonalMatrix<double, 3>& N) {
  // Normalize bearing vectors.
  std::vector<Eigen::Vector3d> norm_rays1;
  std::vector<Eigen::Vector3d> norm_rays2;
  norm_rays1.reserve(rays1.size());
  norm_rays2.reserve(rays2.size());
  for (size_t i = 0; i < rays1.size(); ++i) {
    norm_rays1.push_back(N * rays1[i]);
    norm_rays2.push_back(N * rays2[i]);
  }

  // Standard 8-PA on normalized vectors.
  Eigen::Matrix<double, Eigen::Dynamic, 9> A(norm_rays1.size(), 9);
  for (size_t i = 0; i < norm_rays1.size(); ++i) {
    A.row(i) << norm_rays2[i].x() * norm_rays1[i].transpose(),
        norm_rays2[i].y() * norm_rays1[i].transpose(),
        norm_rays2[i].z() * norm_rays1[i].transpose();
  }

  Eigen::Matrix3d E_hat;
  if (norm_rays1.size() == 8) {
    Eigen::Matrix<double, 9, 9> QQ =
        A.transpose().householderQr().householderQ();
    E_hat = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(
        QQ.col(8).data());
  } else {
    Eigen::JacobiSVD<Eigen::Matrix<double, Eigen::Dynamic, 9>> svd(
        A, Eigen::ComputeFullV);
    E_hat = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(
        svd.matrixV().col(8).data());
  }

  // Enforce rank-2 constraint.
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      E_hat, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Vector3d singular_values = svd.singularValues();
  singular_values(2) = 0.0;
  E_hat = svd.matrixU() * singular_values.asDiagonal() *
          svd.matrixV().transpose();

  // Denormalize: E = N^T * E_hat * N. Since N is diagonal, N^T = N.
  return N * E_hat * N;
}

SphericalNormalizationModel EvaluateSphericalNormalization(
    const std::vector<Eigen::Vector3d>& rays1,
    const std::vector<Eigen::Vector3d>& rays2,
    const double S,
    const double K) {
  constexpr double kMinScale = 1e-3;
  constexpr double kMaxScale = 1e3;
  SphericalNormalizationModel model;
  if (S < kMinScale || K < kMinScale || S > kMaxScale || K > kMaxScale) {
    return model;
  }

  const Eigen::DiagonalMatrix<double, 3> N(Eigen::Vector3d(S, S, K));
  model.S = S;
  model.K = K;
  model.E = EstimateAndDenormalize(rays1, rays2, N);

  std::vector<double> residuals;
  ComputeSquaredProjectedEpipolarError(rays1, rays2, model.E, &residuals);
  model.score = 0.0;
  for (const double residual : residuals) {
    model.score += residual;
  }
  if (!std::isfinite(model.score)) {
    model.score = std::numeric_limits<double>::max();
  }
  return model;
}

void RefineSphericalNormalization(const std::vector<Eigen::Vector3d>& rays1,
                                  const std::vector<Eigen::Vector3d>& rays2,
                                  SphericalNormalizationModel* best_model) {
  double step_S = 0.5;
  double step_K = 0.5;
  constexpr double kMinStep = 1e-3;
  constexpr int kMaxNumIterations = 32;

  for (int iter = 0; iter < kMaxNumIterations; ++iter) {
    SphericalNormalizationModel iteration_best = *best_model;
    for (const double delta_S : {-step_S, 0.0, step_S}) {
      for (const double delta_K : {-step_K, 0.0, step_K}) {
        if (delta_S == 0.0 && delta_K == 0.0) {
          continue;
        }
        const SphericalNormalizationModel candidate =
            EvaluateSphericalNormalization(rays1,
                                           rays2,
                                           best_model->S + delta_S,
                                           best_model->K + delta_K);
        if (candidate.score < iteration_best.score) {
          iteration_best = candidate;
        }
      }
    }

    if (iteration_best.score < best_model->score) {
      *best_model = iteration_best;
    } else {
      step_S *= 0.5;
      step_K *= 0.5;
      if (std::max(step_S, step_K) < kMinStep) {
        break;
      }
    }
  }
}

}  // namespace

void EssentialMatrixSphericalEightPointEstimator::Estimate(
    const std::vector<X_t>& cam_rays1,
    const std::vector<Y_t>& cam_rays2,
    std::vector<M_t>* models) {
  THROW_CHECK_EQ(cam_rays1.size(), cam_rays2.size());
  THROW_CHECK_GE(cam_rays1.size(), 8);
  THROW_CHECK(models != nullptr);

  models->clear();

  // Initialize and refine spherical normalization parameters S and K.
  // S controls XY expansion, K controls Z expansion.
  const std::array<double, 3> kScaleCandidates = {0.5, 1.0, 2.0};

  SphericalNormalizationModel best_model =
      EvaluateSphericalNormalization(cam_rays1, cam_rays2, 1.0, 1.0);

  for (const double S : kScaleCandidates) {
    for (const double K : kScaleCandidates) {
      const SphericalNormalizationModel candidate =
          EvaluateSphericalNormalization(cam_rays1, cam_rays2, S, K);
      if (candidate.score < best_model.score) {
        best_model = candidate;
      }
    }
  }
  RefineSphericalNormalization(cam_rays1, cam_rays2, &best_model);

  models->resize(1);
  (*models)[0] = best_model.E;
}

void EssentialMatrixSphericalEightPointEstimator::Residuals(
    const std::vector<X_t>& cam_rays1,
    const std::vector<Y_t>& cam_rays2,
    const M_t& E,
    std::vector<double>* residuals) {
  ComputeSquaredProjectedEpipolarError(cam_rays1, cam_rays2, E, residuals);
}

}  // namespace colmap
