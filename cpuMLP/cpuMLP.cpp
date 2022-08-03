#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "cMLP2.hpp"

namespace py = pybind11;

PYBIND11_MODULE(policy,  m) {
  py::class_<MLP_3<132, 12>>(m, "PolicyMLP3")
    .def(py::init<>())
    .def("load", &MLP_3<132, 12>::updateParamFromTxt)
    .def("forward", &MLP_3<132, 12>::forward)
    ;
}

PYBIND11_MODULE(stateEst, m) {
  py::class_<MLP_2<123, 11>>(m, "StateEstMLP2")
    .def(py::init<>())
    .def("load", &MLP_2<123, 11>::updateParamFromTxt)
    .def("forward", &MLP_2<123, 11>::forward)
    ;
}

PYBIND11_MODULE(stateEst3, m) {
  py::class_<MLP_2<123, 3>>(m, "StateEstMLP3")
    .def(py::init<>())
    .def("load", &MLP_2<123, 3>::updateParamFromTxt)
    .def("forward", &MLP_2<123, 3>::forward)
    ;
}

PYBIND11_MODULE(interface, m) {
  py::class_<cMLP2>(m, "Interface")
    .def(py::init<>())
    .def("initialize", &cMLP2::initialize)
    .def("forward", &cMLP2::forward)
    .def("update_observation", &cMLP2::update_observation)
    .def_readwrite("obs_mean_", &cMLP2::obs_mean_)
    .def_readwrite("obs_var_", &cMLP2::obs_var_)
    .def_readwrite("P", &cMLP2::P_)
    .def_readwrite("D", &cMLP2::D_)
    .def_readwrite("pTarget12", &cMLP2::pTarget12_)
    .def_readwrite("q_init_", &cMLP2::q_init_)
    .def_readwrite("vel_command", &cMLP2::vel_command_)
    .def_readwrite("state_est_obs_", &cMLP2::state_est_obs_)
    .def_readwrite("obs_", &cMLP2::obs_)
    .def_readwrite("bound_", &cMLP2::bound_)
    .def_readwrite("bound_pi_", &cMLP2::bound_pi_)
    ;
}
