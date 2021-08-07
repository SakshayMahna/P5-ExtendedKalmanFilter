#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/* 
 * Please note that the Eigen library does not initialize 
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  // Calculations
  VectorXd y = z - H_ * x_;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K = P_ * Ht * Si;
  MatrixXd I = MatrixXd::Identity(2, 2);

  // Update state
  x_ = x_ + (K * y);
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  // Auxillary variables
  float px = x_(0), py = x_(1);
  float vx = x_(2), vy = x_(3);

  // Calculate h(x) and Jacobian
  VectorXd h << sqrt(px * px + py * py), atan2(py, px),
                (px * vx + py * vy) / sqrt(px * px + py * py);
  MatrixXd Hj = CalculateJacobian(x_);

  // Calculations
  VectorXd y = z - h;
  MatrixXd Hjt = Hj.transpose();
  MatrixXd S = Hj * P_ * Hjt + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K = P_ * Hjt * Si;
  MatrixXd I = MatrixXd::Identity(2, 2);

  // Update state
  x_ = x_ + (K * y);
  P_ = (I - K * Hj) * P_;
}
