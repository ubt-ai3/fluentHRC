#include <iostream>
#include <Eigen/Dense>

int main() {
	Eigen::Matrix<double, 9, 4> A;
	A << -0.071305, 0.014916, 0.880532, 1,
		-0.004013, 0.012207, 0.877308, 1,
		0.000004, 0.015608, 1.001212, 1,

		0.547196, -0.029478, 0.110121, 1,
		0.549692, -0.033713, -0.010473, 1,
		0.465424, -0.032748, 0.033514, 1,
		
		-0.018503, -0.007714,	0.126539,
		-0.085473, -0.006732,	0.128023,
		-0.106615, -0.007659,	0.082798,
		;

	Eigen::Matrix<double, 9, 2> B;
	B << 619,	986,
		622,	917,
		503,	924,

		1255,	381,
		1354,	372,
		1328,	439,

		1331,	867,
		1341,	933,
		1386,	951
		;

	Eigen::MatrixXd M(2 * A.rows() + 2 * A.cols());
	M << A, Eigen::MatrixXd::Zero(A.rows(), A.cols()),
		Eigen::MatrixXd::Zero(A.rows(), A.cols()), A;

	Eigen::VectorXd b(2 * A.cols());
	b << B.col(0), B.col(1);

	std::cout << "Here is the matrix A:\n" << A << std::endl;
	std::cout << "Here is the right hand side b:\n" << b << std::endl;
	std::cout << "The least-squares solution is:\n"
		<< A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b) << std::endl;

}
