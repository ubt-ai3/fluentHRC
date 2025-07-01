#include "CppUnitTest.h"

#include <state_observation/pn_model.hpp>
#include <state_observation/workspace_objects.hpp>

#include "state_observation/pn_model_extension.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace state_observation;

namespace Microsoft
{
namespace VisualStudio
{
namespace CppUnitTestFramework
{
template<> std::wstring ToString<pn_instance>(const pn_instance& t) { return std::wstring(L"(place,token)"); }
}
}
}

namespace unittests
{
TEST_CLASS(box_overlap_test)
{
public:
	object_parameters object_params;


	TEST_METHOD(aa_horizontal_apart)
	{
		pn_net net(object_params);
		pn_boxed_place::Ptr p1 = std::make_shared<pn_boxed_place>(
			obb(Eigen::Vector3f(0.03f, 0.03f, 0.03f))
			);

		pn_boxed_place::Ptr p2 = std::make_shared<pn_boxed_place>(
			obb(Eigen::Vector3f(0.03f, 0.03f, 0.03f),
			Eigen::Vector3f(0.06f, 0.f, 0.f))
			);

		net.add_place(p1);
		net.add_place(p2);

		Assert::IsTrue(p1->overlapping_places.size() == 0);
	}

	TEST_METHOD(aa_vertical_apart)
	{
		pn_net net(object_params);
		pn_boxed_place::Ptr p1 = std::make_shared<pn_boxed_place>(
			obb(Eigen::Vector3f(0.03f, 0.03f, 0.03f))
			);

		pn_boxed_place::Ptr p2 = std::make_shared<pn_boxed_place>(
			obb(Eigen::Vector3f(0.03f, 0.03f, 0.03f),
			Eigen::Vector3f(0.f, 0.f, 0.06f))
			);

		net.add_place(p1);
		net.add_place(p2);

		Assert::IsTrue(p1->overlapping_places.size() == 0);
	}

	TEST_METHOD(aa_identical)
	{
		pn_net net(object_params);
		pn_boxed_place::Ptr p1 = std::make_shared<pn_boxed_place>(
			obb(Eigen::Vector3f(0.03f, 0.03f, 0.03f))
			);

		pn_boxed_place::Ptr p2 = std::make_shared<pn_boxed_place>(
			obb(Eigen::Vector3f(0.03f, 0.03f, 0.03f))
			);

		net.add_place(p1);
		net.add_place(p2);

		Assert::IsTrue(p1->overlapping_places.size() == 1);
	}

	TEST_METHOD(aa_stacked)
	{
		pn_net net(object_params);
		pn_boxed_place::Ptr p1 = std::make_shared<pn_boxed_place>(
			obb(Eigen::Vector3f(0.03f, 0.03f, 0.03f))
			);

		pn_boxed_place::Ptr p2 = std::make_shared<pn_boxed_place>(
			obb(Eigen::Vector3f(0.03f, 0.03f, 0.03f),
			Eigen::Vector3f(0.f, 0.f, 0.03f))
			);

		net.add_place(p1);
		net.add_place(p2);

		Assert::IsTrue(p1->overlapping_places.size() == 0);
	}

	TEST_METHOD(aa_sided)
	{
		pn_net net(object_params);
		pn_boxed_place::Ptr p1 = std::make_shared<pn_boxed_place>(
			obb(Eigen::Vector3f(0.03f, 0.03f, 0.03f))
			);

		pn_boxed_place::Ptr p2 = std::make_shared<pn_boxed_place>(
			obb(Eigen::Vector3f(0.03f, 0.03f, 0.03f),
			Eigen::Vector3f(0.f, 0.03f, 0.f))
			);

		net.add_place(p1);
		net.add_place(p2);

		Assert::IsTrue(p1->overlapping_places.size() == 0);
	}


	TEST_METHOD(aa_minor_overlap)
	{
		pn_net net(object_params);
		pn_boxed_place::Ptr p1 = std::make_shared<pn_boxed_place>(
			obb(Eigen::Vector3f(0.03f, 0.03f, 0.03f))
			);

		pn_boxed_place::Ptr p2 = std::make_shared<pn_boxed_place>(
			obb(Eigen::Vector3f(0.03f, 0.03f, 0.03f),
			Eigen::Vector3f(0.f, 0.029f, 0.f))
			);

		net.add_place(p1);
		net.add_place(p2);

		Assert::IsTrue(p1->overlapping_places.size() == 0);
	}



	TEST_METHOD(aa_major_overlap)
	{
		pn_net net(object_params);
		pn_boxed_place::Ptr p1 = std::make_shared<pn_boxed_place>(
			obb(Eigen::Vector3f(0.03f, 0.03f, 0.03f))
			);

		pn_boxed_place::Ptr p2 = std::make_shared<pn_boxed_place>(
			obb(Eigen::Vector3f(0.03f, 0.03f, 0.03f),
			Eigen::Vector3f(0.f, 0.015f, 0.f))
			);

		net.add_place(p1);
		net.add_place(p2);

		Assert::IsTrue(p1->overlapping_places.size() == 1);
	}


	TEST_METHOD(aa_touching_edge)
	{
		pn_net net(object_params);
		pn_boxed_place::Ptr p1 = std::make_shared<pn_boxed_place>(
			obb(Eigen::Vector3f(0.03f, 0.03f, 0.03f))
			);

		pn_boxed_place::Ptr p2 = std::make_shared<pn_boxed_place>(
			obb(Eigen::Vector3f(0.03f, 0.03f, 0.03f),
			Eigen::Vector3f(0.03f, 0.03f, 0.f))
			);

		net.add_place(p1);
		net.add_place(p2);

		Assert::IsTrue(p1->overlapping_places.size() == 0);
	}

	TEST_METHOD(aa_edge_overlap)
	{
		pn_net net(object_params);
		pn_boxed_place::Ptr p1 = std::make_shared<pn_boxed_place>(
			obb(Eigen::Vector3f(0.03f, 0.03f, 0.03f))
			);

		pn_boxed_place::Ptr p2 = std::make_shared<pn_boxed_place>(
			obb(Eigen::Vector3f(0.03f, 0.03f, 0.03f),
			Eigen::Vector3f(0.022f, 0.022f, 0.f))
			);

		net.add_place(p1);
		net.add_place(p2);

		Assert::IsTrue(p1->overlapping_places.size() == 1);
	}

	TEST_METHOD(aa_contain)
	{
		pn_net net(object_params);
		pn_boxed_place::Ptr p1 = std::make_shared<pn_boxed_place>(
			obb(Eigen::Vector3f(0.03f, 0.03f, 0.03f))
			);

		pn_boxed_place::Ptr p2 = std::make_shared<pn_boxed_place>(
			obb(Eigen::Vector3f(0.03f, 0.06f, 0.03f),
			Eigen::Vector3f(0.0f, 0.015f, 0.f))
			);

		net.add_place(p1);
		net.add_place(p2);

		Assert::IsTrue(p1->overlapping_places.size() == 1);
	}

	TEST_METHOD(aa_contain_2)
	{
		pn_net net(object_params);
		pn_boxed_place::Ptr p1 = std::make_shared<pn_boxed_place>(
			obb(Eigen::Vector3f(0.03f, 0.03f, 0.03f))
			);

		pn_boxed_place::Ptr p2 = std::make_shared<pn_boxed_place>(
			obb(Eigen::Vector3f(0.03f, 0.03f, 0.06f),
			Eigen::Vector3f(0.0f, 0.015f, 0.f),
			Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f(0.f, 0.f, 1.f), Eigen::Vector3f(0.f, 1.f, 0.f))
			));

		net.add_place(p1);
		net.add_place(p2);

		Assert::IsTrue(p1->overlapping_places.size() == 1);
	}




	TEST_METHOD(rot_horizontal_apart)
	{
		pn_net net(object_params);
		pn_boxed_place::Ptr p1 = std::make_shared<pn_boxed_place>(
			obb(Eigen::Vector3f(0.03f, 0.03f, 0.03f),
			Eigen::Vector3f::Zero(),
			Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f(1.f, 0.f, 0.f), Eigen::Vector3f(1.f, 1.f, 0.f))
			));

		pn_boxed_place::Ptr p2 = std::make_shared<pn_boxed_place>(
			obb(Eigen::Vector3f(0.03f, 0.03f, 0.03f),
			Eigen::Vector3f(0.03f, 0.03f, 0.f),
			Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f(1.f, 0.f, 0.f), Eigen::Vector3f(1.f, 1.f, 0.f))
			));

		net.add_place(p1);
		net.add_place(p2);

		Assert::IsTrue(p1->overlapping_places.size() == 0);
	}

	TEST_METHOD(rot_vertical_apart)
	{
		pn_net net(object_params);
		pn_boxed_place::Ptr p1 = std::make_shared<pn_boxed_place>(
			obb(Eigen::Vector3f(0.03f, 0.03f, 0.03f),
			Eigen::Vector3f::Zero(),
			Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f(1.f, 0.f, 0.f), Eigen::Vector3f(1.f, 1.f, 0.f)))
			);

		pn_boxed_place::Ptr p2 = std::make_shared<pn_boxed_place>(
			obb(Eigen::Vector3f(0.03f, 0.03f, 0.03f),
			Eigen::Vector3f(0.f, 0.f, 0.06f),
			Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f(1.f, 0.f, 0.f), Eigen::Vector3f(1.f, 1.f, 0.f)))
			);

		net.add_place(p1);
		net.add_place(p2);

		Assert::IsTrue(p1->overlapping_places.size() == 0);
	}

	TEST_METHOD(rot_identical)
	{
		pn_net net(object_params);
		pn_boxed_place::Ptr p1 = std::make_shared<pn_boxed_place>(
			obb(Eigen::Vector3f(0.03f, 0.03f, 0.03f),
			Eigen::Vector3f::Zero(),
			Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f(1.f, 0.f, 0.f), Eigen::Vector3f(1.f, 1.f, 0.f)))
			);

		pn_boxed_place::Ptr p2 = std::make_shared<pn_boxed_place>(
			obb(Eigen::Vector3f(0.03f, 0.03f, 0.03f),
			Eigen::Vector3f::Zero(),
			Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f(1.f, 0.f, 0.f), Eigen::Vector3f(1.f, 1.f, 0.f)))
			);

		net.add_place(p1);
		net.add_place(p2);

		Assert::IsTrue(p1->overlapping_places.size() == 1);
	}

	TEST_METHOD(rot_stacked)
	{
		pn_net net(object_params);
		pn_boxed_place::Ptr p1 = std::make_shared<pn_boxed_place>(
			obb(Eigen::Vector3f(0.03f, 0.03f, 0.03f),
			Eigen::Vector3f::Zero(),
			Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f(1.f, 0.f, 0.f), Eigen::Vector3f(1.f, 1.f, 0.f)))
			);

		pn_boxed_place::Ptr p2 = std::make_shared<pn_boxed_place>(
			obb(Eigen::Vector3f(0.03f, 0.03f, 0.03f),
			Eigen::Vector3f(0.f, 0.f, 0.03f),
			Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f(1.f, 0.f, 0.f), Eigen::Vector3f(1.f, 1.f, 0.f)))
			);

		net.add_place(p1);
		net.add_place(p2);

		Assert::IsTrue(p1->overlapping_places.size() == 0);
	}

	TEST_METHOD(rot_sided)
	{
		pn_net net(object_params);
		pn_boxed_place::Ptr p1 = std::make_shared<pn_boxed_place>(
			obb(Eigen::Vector3f(0.03f, 0.03f, 0.03f),
			Eigen::Vector3f::Zero(),
			Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f(1.f, 0.f, 0.f), Eigen::Vector3f(1.f, 1.f, 0.f)))
			);

		pn_boxed_place::Ptr p2 = std::make_shared<pn_boxed_place>(
			obb(Eigen::Vector3f(0.03f, 0.03f, 0.03f),
			Eigen::Vector3f(0.021213203f, 0.021213203f, 0.f),
			Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f(1.f, 0.f, 0.f), Eigen::Vector3f(1.f, 1.f, 0.f)))
			);

		net.add_place(p1);
		net.add_place(p2);

		Assert::IsTrue(p1->overlapping_places.size() == 0);
	}


	TEST_METHOD(rot_minor_overlap)
	{
		pn_net net(object_params);
		pn_boxed_place::Ptr p1 = std::make_shared<pn_boxed_place>(
			obb(Eigen::Vector3f(0.03f, 0.03f, 0.03f),
			Eigen::Vector3f::Zero(),
			Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f(1.f, 0.f, 0.f), Eigen::Vector3f(1.f, 1.f, 0.f)))
			);

		pn_boxed_place::Ptr p2 = std::make_shared<pn_boxed_place>(
			obb(Eigen::Vector3f(0.03f, 0.03f, 0.03f),
			Eigen::Vector3f(0.02f, 0.02f, 0.f),
			Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f(1.f, 0.f, 0.f), Eigen::Vector3f(1.f, 1.f, 0.f)))
			);

		net.add_place(p1);
		net.add_place(p2);

		Assert::IsTrue(p1->overlapping_places.size() == 0);
	}



	TEST_METHOD(rot_major_overlap)
	{
		pn_net net(object_params);
		pn_boxed_place::Ptr p1 = std::make_shared<pn_boxed_place>(
			obb(Eigen::Vector3f(0.03f, 0.03f, 0.03f),
			Eigen::Vector3f::Zero(),
			Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f(1.f, 0.f, 0.f), Eigen::Vector3f(1.f, 1.f, 0.f)))
			);

		pn_boxed_place::Ptr p2 = std::make_shared<pn_boxed_place>(
			obb(Eigen::Vector3f(0.03f, 0.03f, 0.03f),
			Eigen::Vector3f(0.01f, 0.01f, 0.f),
			Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f(1.f, 0.f, 0.f), Eigen::Vector3f(1.f, 1.f, 0.f)))
			);

		net.add_place(p1);
		net.add_place(p2);

		Assert::IsTrue(p1->overlapping_places.size() == 1);
	}


	TEST_METHOD(rot_touching_edge)
	{
		pn_net net(object_params);
		pn_boxed_place::Ptr p1 = std::make_shared<pn_boxed_place>(
			obb(Eigen::Vector3f(0.03f, 0.03f, 0.03f),
			Eigen::Vector3f::Zero(),
			Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f(1.f, 0.f, 0.f), Eigen::Vector3f(1.f, 1.f, 0.f)))
			);

		pn_boxed_place::Ptr p2 = std::make_shared<pn_boxed_place>(
			obb(Eigen::Vector3f(0.03f, 0.03f, 0.03f),
			Eigen::Vector3f(-0.042426f, 0.f, 0.f),
			Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f(1.f, 0.f, 0.f), Eigen::Vector3f(1.f, 1.f, 0.f)))
			);

		net.add_place(p1);
		net.add_place(p2);

		Assert::IsTrue(p1->overlapping_places.size() == 0);
	}

	TEST_METHOD(rot_edge_overlap)
	{
		pn_net net(object_params);
		pn_boxed_place::Ptr p1 = std::make_shared<pn_boxed_place>(
			obb(Eigen::Vector3f(0.03f, 0.03f, 0.03f),
			Eigen::Vector3f::Zero(),
			Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f(1.f, 0.f, 0.f), Eigen::Vector3f(1.f, 1.f, 0.f)))
			);

		pn_boxed_place::Ptr p2 = std::make_shared<pn_boxed_place>(
			obb(Eigen::Vector3f(0.03f, 0.03f, 0.03f),
			Eigen::Vector3f(0.021213f, 0.0f, 0.f),
			Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f(1.f, 0.f, 0.f), Eigen::Vector3f(1.f, 1.f, 0.f)))
			);

		net.add_place(p1);
		net.add_place(p2);

		Assert::IsTrue(p1->overlapping_places.size() == 1);
	}

	TEST_METHOD(rot_contain)
	{
		pn_net net(object_params);
		pn_boxed_place::Ptr p1 = std::make_shared<pn_boxed_place>(
			obb(Eigen::Vector3f(0.03f, 0.03f, 0.03f),
			Eigen::Vector3f::Zero(),
			Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f(1.f, 0.f, 0.f), Eigen::Vector3f(1.f, 1.f, 0.f)))
			);

		pn_boxed_place::Ptr p2 = std::make_shared<pn_boxed_place>(
			obb(Eigen::Vector3f(0.03f, 0.06f, 0.03f),
			Eigen::Vector3f(0.0106f, 0.0106f, 0.f))
			);

		net.add_place(p1);
		net.add_place(p2);

		Assert::IsTrue(p1->overlapping_places.size() == 1);
	}

	TEST_METHOD(rot_contain_2)
	{
		pn_net net(object_params);
		pn_boxed_place::Ptr p1 = std::make_shared<pn_boxed_place>(
			obb(Eigen::Vector3f(0.03f, 0.03f, 0.03f),
			Eigen::Vector3f::Zero(),
			Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f(1.f, 0.f, 0.f), Eigen::Vector3f(1.f, 1.f, 0.f)))
			);

		pn_boxed_place::Ptr p2 = std::make_shared<pn_boxed_place>(
			obb(Eigen::Vector3f(0.03f, 0.03f, 0.06f),
			Eigen::Vector3f(0.0106f, 0.0106f, 0.f),
			Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f(1.f, 0.f, 0.f), Eigen::Vector3f(1.f, 1.f, 0.f)) * Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f(0.f, 0.f, 1.f), Eigen::Vector3f(1.f, 0.f, 0.f))
			));

		net.add_place(p1);
		net.add_place(p2);

		Assert::IsTrue(p1->overlapping_places.size() == 1);
	}
};

TEST_CLASS(box_overlap_test_mogaze)
{
public:
	object_parameters object_params;

	Eigen::Vector3f plate = Eigen::Vector3f(.183f, 0.183f, 0.0071f);
	Eigen::Vector3f bowl = Eigen::Vector3f(0.283f, 0.375f, 0.106f);
	Eigen::Vector3f cup = Eigen::Vector3f(0.0826f, 0.0708f, 0.0801f);
	Eigen::Vector3f jug = Eigen::Vector3f(0.118f, 0.153f, 0.289f);


	TEST_METHOD(bowl_on_plate)
	{
		pn_net net(object_params);
		pn_boxed_place::Ptr p1 = std::make_shared<pn_boxed_place>(
			aabb(plate,
			Eigen::Vector3f(-1.1067572886293584, -0.1811052431267771, 1.5377498973499646)
			));

		pn_boxed_place::Ptr p2 = std::make_shared<pn_boxed_place>(
			aabb(bowl,
			Eigen::Vector3f(-1.105384111404419, -0.18190918862819672, 1.6023790836334229)
			));

		net.add_place(p1);
		net.add_place(p2);

		Assert::IsTrue(p1->overlapping_places.size() == 1);
	}

	TEST_METHOD(table_cups_no_overlap)
	{
		pn_net net(object_params);
		pn_boxed_place::Ptr p1 = std::make_shared<pn_boxed_place>(
			aabb(cup,
			Eigen::Vector3f(0.7192233895000658, 2.0295781880094292, 0.7473909509809393)
			));

		auto add = [&](float x, float y, float z) {
			net.add_place(std::make_shared<pn_boxed_place>(
				aabb(cup,
				Eigen::Vector3f(x,y,z)
				)));
		};


		net.add_place(p1);

		add(0.44692884188778, 2.045256331281842, 0.7494148949407181);
			add(0.4539528135643449, 2.32921119623406, 0.7473318174827931);
			add(0.7705549309330602, 2.302187165906352, 0.7488859084344679);
			add(0.6854606419801712, 2.3725943565368652, 0.7467580040295919);
			add(0.3980290989081065, 1.9552951256434123, 0.7494793633619944);
			add(0.645060624395098, 2.2887537479400635, 0.7474215541567121);
			add(0.43760519723097485, 2.183522860209147, 0.7491647402445475);
			add(0.5488172024488449, 2.2469115257263184, 0.7477723807096481);
			add(0.8833223581314087, 2.027923822402954, 0.7520107626914978);
			add(0.5809069275856018, 2.4291491508483887, 0.7471030354499817);
			add(0.42362236976623535, 2.350872755050659, 0.7915103435516357);

		Assert::AreEqual<size_t>(0,p1->overlapping_places.size());
	}

	TEST_METHOD(table_cups_and_bowl_no_overlap)
	{
		pn_net net(object_params);
		pn_boxed_place::Ptr p1 = std::make_shared<pn_boxed_place>(
			aabb(bowl,
			Eigen::Vector3f(0.6138976514339447, 2.1798502845423564, 0.7774295487574169)
			));

		auto add = [&](float x, float y, float z) {
			net.add_place(std::make_shared<pn_boxed_place>(
				aabb(cup,
				Eigen::Vector3f(x, y, z)
			)));
		};


		net.add_place(p1);

		add(0.73, 2.02, 0.7473909509809393);
		add(0.44692884188778, 2.045256331281842, 0.7494148949407181);
		add(0.4539528135643449, 2.32921119623406, 0.7473318174827931);
		add(0.7705549309330602, 2.302187165906352, 0.7488859084344679);
		add(0.6854606419801712, 2.3725943565368652, 0.7467580040295919);
		add(0.3980290989081065, 1.9552951256434123, 0.7494793633619944);
		add(0.645060624395098, 2.2887537479400635, 0.7474215541567121);
		add(0.43760519723097485, 2.183522860209147, 0.7491647402445475);
		add(0.5488172024488449, 2.2469115257263184, 0.7477723807096481);
		add(0.8833223581314087, 2.027923822402954, 0.7520107626914978);
		add(0.5809069275856018, 2.4291491508483887, 0.7471030354499817);
		add(0.42362236976623535, 2.350872755050659, 0.7915103435516357);

		Assert::AreEqual<size_t>(2,p1->overlapping_places.size());
	}

	TEST_METHOD(cup_on_plate)
	{
		pn_net net(object_params);
		pn_boxed_place::Ptr p1 = std::make_shared<pn_boxed_place>(
			aabb(plate,
			Eigen::Vector3f(0.6264356970787048, -0.2727602422237396, 0.9343172907829285)
			));

		pn_boxed_place::Ptr p2 = std::make_shared<pn_boxed_place>(
			aabb(cup,
			Eigen::Vector3f(0.6110515291573572, -0.2715011397834684, 0.9679508707562431)
			));

		net.add_place(p1);
		net.add_place(p2);

		Assert::IsTrue(p1->overlapping_places.size() == 1);
	}

	TEST_METHOD(jug_and_bowl_on_different_shelves)
	{
		pn_net net(object_params);
		pn_boxed_place::Ptr p1 = std::make_shared<pn_boxed_place>(
			aabb(bowl,
			Eigen::Vector3f(0.683334291, -0.267260849, 0.704191089)
			));

		pn_boxed_place::Ptr p2 = std::make_shared<pn_boxed_place>(
			aabb(jug,
			Eigen::Vector3f(0.683258235, -0.272722363, 1.08434725)
			));

		net.add_place(p1);
		net.add_place(p2);

		Assert::IsTrue(p1->overlapping_places.size() == 0);
	}

	TEST_METHOD(cup_in_bowl_1)
	{
		pn_net net(object_params);
		pn_boxed_place::Ptr p1 = std::make_shared<pn_boxed_place>(
			aabb(bowl,
			Eigen::Vector3f(0.683334291, -0.267260849, 0.704191089)
			));

		pn_boxed_place::Ptr p2 = std::make_shared<pn_boxed_place>(
			aabb(cup,
			Eigen::Vector3f(0.7558373527706794, -0.25462542502385266, 0.6710559208438082)
			));

		net.add_place(p1);
		net.add_place(p2);

		Assert::IsTrue(p1->overlapping_places.size() == 1);
	}

	TEST_METHOD(cup_in_bowl_2)
	{
		pn_net net(object_params);
		pn_boxed_place::Ptr p1 = std::make_shared<pn_boxed_place>(
			aabb(bowl,
			Eigen::Vector3f(0.683334291, -0.267260849, 0.704191089)
			));

		pn_boxed_place::Ptr p2 = std::make_shared<pn_boxed_place>(
			aabb(cup,
			Eigen::Vector3f(0.6243495979207627, -0.2604966249237669, 0.6713497029974106)
			));

		net.add_place(p1);
		net.add_place(p2);

		Assert::IsTrue(p1->overlapping_places.size() == 1);
	}
};

}

