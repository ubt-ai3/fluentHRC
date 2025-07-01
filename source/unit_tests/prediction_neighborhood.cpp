#include "CppUnitTest.h"

#include "../intention_prediction/feature_space.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace state_observation;
using namespace prediction;

namespace unittests
{
	/**
	 * Indexing of neighbor cells (displayed layerwise from top to bottom):
	 *
	 *  4   3   2
	 *  5   0   1
	 *	6   7   8
	 *	---------
	 * 13  12  11
	 * 14   9  10
	 * 15  16  17
	 *	---------
	 * 22  21  20
	 * 23  18  19
	 * 24  25  26
	 *	---------
	 */
	TEST_CLASS(neighbor_segmentation)
	{
	public:
		const object_parameters object_params;
		const float cube_dim = 0.03f;
		const double epsilon = 0.0000001;

		std::tuple<pn_net::Ptr, pn_boxed_place::Ptr, pn_belief_marking> create_marking(const std::vector<Eigen::Vector3f>& boxes,
			const Eigen::Vector3f& center = Eigen::Vector3f({0,0,0}))
		{
			auto net = std::make_shared<pn_net>(object_params);

			auto center_place = std::make_shared<pn_boxed_place>(aabb(Eigen::Vector3f(cube_dim, cube_dim, cube_dim), center));
			net->add_place(center_place);

			for(const auto& c : boxes)
			{
				net->add_place(std::make_shared<pn_boxed_place>(aabb(Eigen::Vector3f(cube_dim, cube_dim, cube_dim), c)));
			}

			auto t = std::make_shared<pn_token>();
			std::set<pn_instance> instances;
			for (const auto& p : net->get_places())
				instances.emplace( std::make_pair(p, t) );

			return { net, center_place,pn_belief_marking(std::make_shared<pn_binary_marking>(net, std::move(instances))) };
		}

		void check_occupancy(int index, const std::vector<float>& calculated_neighbors)
		{
			std::wstringstream message;
			message << "Expected cell " << index << " to be non-zero but got ";

			bool correct = true;
			for(int i = 0; i < calculated_neighbors.size(); i++)
			{
				bool is_zero = std::abs(calculated_neighbors.at(i)) < epsilon;
				if (i == index && is_zero)
					correct = false;
				else if (i != index && !is_zero) 
				{
					message << i << " (" << std::setprecision(3) << calculated_neighbors.at(i) << ") ";
					correct = false;
				}
			}

			Assert::IsTrue(correct, message.str().c_str());
		}

		void test_cell(int index, int x_offset, int y_offset, int z_offset)
		{
			std::vector<float> neighbors;
			neighbors.resize(27);

			for (float dx : {0, -1, 1}) {
				for (float dy : {0, -1, 1}) {
					for (float dz : {0, -1, 1})
					{
						auto [net, center, marking] = create_marking({ Eigen::Vector3f((x_offset + 0.25 * dx) * cube_dim, (y_offset + 0.25 * dy) * cube_dim, (z_offset + 0.25 * dz) * cube_dim) });
						std::ranges::transform(neighbors, transition_context::get_neighbors(marking, center, { center }),  neighbors.begin(), std::plus<float>());
					}
				}
			}

			check_occupancy(index, neighbors);
		}

		// z up
		TEST_METHOD(c_00_ccu)
		{
			test_cell(0, 0, 0, 1);
		}

		TEST_METHOD(c_01_rcc)
		{
			test_cell(1, 1, 0, 1);
		}

		TEST_METHOD(c_02_rbc)
		{
			test_cell(2, 1, 1, 1);
		}

		TEST_METHOD(c_03_cbc)
		{
			test_cell(3, 0, 1, 1);
		}

		TEST_METHOD(c_04_lbc)
		{
			test_cell(4, -1, 1, 1);
		}

		TEST_METHOD(c_05_lcc)
		{
			test_cell(5, -1, 0, 1);
		}

		TEST_METHOD(c_06_lfc)
		{
			test_cell(6, -1, -1, 1);
		}

		TEST_METHOD(c_07_cfc)
		{
			test_cell(7, 0, -1, 1);
		}

		TEST_METHOD(c_08_rfc)
		{
			test_cell(8, 1, -1, 1);
		}

		// z center
		TEST_METHOD(c_09_ccc)
		{
			test_cell(9, 0, 0, 0);
			/*auto [net, center, marking] = create_marking({ Eigen::Vector3f(0,0,0) });
			check_occupancy(9, transition_context::get_neighbors(marking, center, { center }));*/
		}

		TEST_METHOD(c_10_rcc)
		{
            test_cell(10, 1,0,0);
		}

		TEST_METHOD(c_11_rbc)
		{
            test_cell(11, 1, 1,0);
		}

		TEST_METHOD(c_12_cbc)
		{
            test_cell(12, 0,1,0);
		}

		TEST_METHOD(c_13_lbc)
		{
            test_cell(13, -1,1,0);
		}

		TEST_METHOD(c_14_lcc)
		{
            test_cell(14, -1,0,0);
		}

		TEST_METHOD(c_15_lfc)
		{
            test_cell(15, -1,-1,0);
		}

		TEST_METHOD(c_16_cfc)
		{
            test_cell(16, 0,-1,0);
		}

		TEST_METHOD(c_17_rfc)
		{
            test_cell(17, 1,-1,0);
		}


		// z down
		TEST_METHOD(c_18_ccd)
		{
			test_cell(18, 0, 0, -1);
		}

		TEST_METHOD(c_19_rcd)
		{
			test_cell(19, 1, 0, -1);
		}

		TEST_METHOD(c_20_rbd)
		{
			test_cell(20, 1, 1, -1);
		}

		TEST_METHOD(c_21_cbd)
		{
			test_cell(21, 0, 1, -1);
		}

		TEST_METHOD(c_22_lbd)
		{
			test_cell(22, -1, 1, -1);
		}

		TEST_METHOD(c_23_lcd)
		{
			test_cell(23, -1, 0, -1);
		}

		TEST_METHOD(c_24_lfd)
		{
			test_cell(24, -1, -1, -1);
		}

		TEST_METHOD(c_25_cfd)
		{
			test_cell(25, 0, -1, -1);
		}

		TEST_METHOD(c_26_rfd)
		{
			test_cell(26, 1, -1, -1);
		}
	};
}