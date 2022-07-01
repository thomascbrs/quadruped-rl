#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "cpuMLP.hpp"

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
