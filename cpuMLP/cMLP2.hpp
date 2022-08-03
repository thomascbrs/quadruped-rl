///////////////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief This is the header for cMLP2 class
///
/// \details Planner that outputs current and future locations of footsteps, the reference
///          trajectory of the base and the position, velocity, acceleration commands for feet in
///          swing phase based on the reference velocity given by the user and the current
///          position/velocity of the base
///
//////////////////////////////////////////////////////////////////////////////////////////////////

#include "Types.h"
#include "cpuMLP.hpp"

class cMLP2 {
 public:
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Empty constructor
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  cMLP2();

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Destructor.
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ~cMLP2() {}

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Initializer
  ///
  /// \param[in] polDirName Name of directory that contains policy parameters
  /// \param[in] estFileName Name of file that contains estimation parameters
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void initialize(std::string polDirName, std::string estFileName);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief  Forward pass
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void forward();

 private:

  // Control policy
  MLP_2<132, 12> policy_;

  // Estimation policy
  MLP_2<123, 3> state_estimator_;

  // Normalization of the input of the control policy
  VectorN obs_mean_;
  VectorN obs_var_;

  // History buffers
  const int num_history_stack_ = 6;
  MatrixN q_pos_error_hist_;
  MatrixN qd_hist;
  Vector12 previous_action_;
  Vector12 preprevious_action_;

  // Misc
  Vector12 P_;
  Vector12 D_;
  Vector12 pTarget12_;
  Vector3 vel_command_;
  VectorN state_est_obs_;
  VectorN obs_;

};

cMLP2::cMLP2()
    : obs_mean_(VectorN::Zero(132)),
      obs_var_(VectorN::Zero(132)),
      q_pos_error_hist_(MatrixN::Zero(num_history_stack_, 12)),
      qd_hist(MatrixN::Zero(num_history_stack_, 12)),
      previous_action_(Vector12::Zero()),
      preprevious_action_(Vector12::Zero()),
      P_(Vector12::Zero()),
      D_(Vector12::Zero()),
      pTarget12_(Vector12::Zero()),
      vel_command_(Vector3::Zero()),
      state_est_obs_(VectorN::Zero(123)),
      obs_(VectorN::Zero(132)) {
  // Empty
}

void cMLP2::initialize(std::string polDirName, std::string estFileName) {

  // Control policy
  policy_.updateParamFromTxt(polDirName + "");

  // Estimation policy
  state_estimator_.updateParamFromTxt(estFileName);

  // Normalization
  std::string in_line;
  std::ifstream obsMean_file, obsVariance_file;
  obsMean_file.open("mean.csv");
  obsVariance_file.open("var.csv");
  if(obsMean_file.is_open()) {
    for(int i = 0; i < _obsMean.size(); i++){
      std::getline(obsMean_file, in_line);
      _obsMean(i) = std::stod(in_line);
    }   
    obsMean_file.close();
  }
  if(obsVariance_file.is_open()) {
    for(int i = 0; i < _obsVar.size(); i++){
      std::getline(obsVariance_file, in_line);
      _obsVar(i) = std::stod(in_line);
    }
    obsVariance_file.close();   
  }

  // Control gains
  P_ = (Vector3(3.0, 3.0, 3.0)).replicate<4, 1>();
  D_ = (Vector3(0.2, 0.2, 0.2)).replicate<4, 1>();

}

void cMLP2::forward() { 
  std::cout << "Test" << std::endl;
  std::cout << P_ << std::endl;
}
