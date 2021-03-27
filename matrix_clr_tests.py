import unittest
from matrix_clr import *
import numpy as np


class TestMatrixCLRMethods(unittest.TestCase):

    def setUp(self):
        self.ex_1 = np.array([[1.0, 1.0], [1.0, 1.0]])
        self.ex_2 = np.array([[1.0, 1.0], [0, 0]])
        self.ex_3 = np.array([[1.0, 1.0], [1.0, 0]])
        self.trust_values = np.ones((2, 2))

    def test_pairwise_coord(self):
        # compare computed values to those by hand
        TOL = 0.001  # error tolerance
        pc_ex_1 = pairwise_coord(self.ex_1, 0, 1)
        err_1 = np.abs(pc_ex_1 - 1. / 3)
        self.assertTrue(err_1 < TOL)
        pc_ex_2 = pairwise_coord(self.ex_2, 0, 1)
        err_2 = np.abs(pc_ex_2 - 1. / 2)
        self.assertTrue(err_2 < TOL)
        pc_ex_3 = pairwise_coord(self.ex_3, 0, 1)
        err_3 = np.abs(pc_ex_3 - 1. / 2)
        self.assertTrue(err_3 < TOL)

    def test_calc_pairwise_coord_mat(self):
        TOL = 0.001  # error tolerance
        fcalc_mat1 = calc_pairwise_coord_mat(self.ex1)
        fcalc_mat1_entry = fcalc_mat1[0, 0]
        handcalc_mat1_entry = np.float(1 / 3)
        err1 = np.abs(fcalc_mat1_entry - handcalc_mat1_entry)
        self.assertTrue(err1 < TOL)
        self.assertEqual(fcalc_mat1[0, 1], 0)
        self.assertEqual(fcalc_mat1[1, 0], 0)
        self.assertEqual(fcalc_mat1[1, 1], 0)
        mat2 = calc_pairwise_coord_mat(self.ex2)
        fcalc_mat2_entry = mat2[0, 0]
        handcalc_mat2_entry = np.float(1.2)
        err2 = np.abs(fcalc_mat2_entry - handcalc_mat2_entry)
        self.assertTrue(err2 < TOL)
        self.assertEqual(mat2[0, 1], 0)
        self.assertEqual(mat2[1, 0], 0)
        self.assertEqual(mat2[1, 1], 0)
        mat3 = calc_pairwise_coord_mat(self.ex3)
        fcalc_mat2_entry = mat3[0, 0]
        handcalc_mat2_entry = np.float(1.2)
        err3 = np.abs(fcalc_mat2_entry - handcalc_mat2_entry)
        self.assertTrue(err3 < TOL)
        self.assertEqual(mat3[0, 1], 0)
        self.assertEqual(mat3[1, 0], 0)
        self.assertEqual(mat3[1, 1], 0)

    def test_pairwise_term(self):
        calc_1 = pairwise_term(self.ex_1, 0, 1)
        self.assertEqual(calc_1, self.ex_1[0] * self.ex_1[1])
        calc_2 = pairwise_term(self.ex_2, 0, 1)
        self.assertEqual(calc_2, self.ex_2[0] * self.ex_2[1])
        calc_3 = pairwise_term(self.ex_3, 0, 1)
        self.assertEqual(calc_3, self.ex_3[0] * self.ex_3[1])

    def test_calc_pairwise_term_mat(self):
        TOL = 0.001
        ex1_grant0 = calc_pairwise_term_mat(self.ex1, 0)
        err_ex1_grant0 = np.abs(ex1_grant0[0, 1] - 1.0)
        self.assertTrue(err_ex1_grant0 < TOL)
        ex1_grant1 = calc_pairwise_term_mat(self.ex1, 1)
        err_ex1_grant1 = np.abs(ex1_grant1[0, 1] - 1.0)
        self.assertTrue(err_ex1_grant1 < TOL)
        ex2_grant0 = calc_pairwise_term_mat(self.ex2, 0)
        err_ex2_grant0 = np.abs(ex2_grant0[0, 1] - 1.0)
        self.assertTrue(err_ex2_grant0 < TOL)
        ex2_grant1 = calc_pairwise_term_mat(self.ex2, 1)
        err_ex2_grant1 = np.abs(ex2_grant1[0, 1] - 0.0)
        self.assertTrue(err_ex2_grant1 < TOL)
        ex3_grant0 = calc_pairwise_term_mat(self.ex3, 0)
        err_ex3_grant0 = np.abs(ex3_grant0[0, 1] - 1.0)
        self.assertTrue(err_ex3_grant0 < TOL)
        ex3_grant1 = calc_pairwise_term_mat(self.ex3, 1)
        err_ex3_grant1 = np.abs(ex3_grant1[0, 1] - 1.0)
        self.assertTrue(err_ex3_grant1 < TOL)

    def test_pairwise_qf_calc(self):
        TOL = 0.001
        fcalc_ex1_gr0_val = pairwise_qf_calc(ex1, trust_values, 0)
        handcalc_ex1_gr0_val = np.float(1 / 3)
        err_ex1_grant0 = np.abs(fcalc_ex1_gr0_val - handcalc_ex1_gr0_val)
        self.assertTrue(err_ex1_grant0 < TOL)
        fcalc_ex1_gr1_val = pairwise_qf_calc(ex1, trust_values, 0)
        handcalc_ex1_gr1_val = np.float(1 / 3)
        err_ex1_grant1 = np.abs(fcalc_ex1_gr0_val - handcalc_ex1_gr1_val)
        self.assertTrue(err_ex1_grant1 < TOL)
        fcalc_ex2_gr0_val = pairwise_qf_calc(ex2, trust_values, 1)
        handcalc_ex2_gr0_val = np.float(1 / 2)
        err_ex2_grant0 = np.abs(fcalc_ex2_gr0_val - handcalc_ex2_gr0_val)
        self.assertTrue(err_ex2_grant0 < TOL)
        fcalc_ex2_gr1_val = pairwise_qf_calc(ex2, trust_values, 1)
        handcalc_ex2_gr0_val = 0.0
        err_ex2_grant1 = np.abs(fcalc_ex2_gr0_val - handcalc_ex2_gr0_val)
        self.assertTrue(err_ex2_grant1 < TOL)


if __name__ == '__main__':
    unittest.main()
