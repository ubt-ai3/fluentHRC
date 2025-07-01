#include "CppUnitTest.h"


#include "pn_util.hpp"

#include <thread>
#include <string>

#include "state_observation/pn_reasoning.hpp"



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
	
	
TEST_CLASS(pn_reasoning_belief_test)
{
public:
	static constexpr double EPSILON = pn_feasible_transition_extractor::certainty_equal_threshold;

	const state_observation::object_parameters object_params;

	void assert_equal(const pn_belief_marking::ConstPtr& expected, const pn_belief_marking::ConstPtr& actual)
	{
#ifdef DEBUG_PN_ID
		auto to_string = [](const pn_binary_marking::Ptr& m) {
			std::stringstream s;
			for (const auto& instance : m->distribution)
				s << "(" << instance.first->id << "," << instance.second->id << "),";
			return s.str();
		};
#endif
		for (const auto& marking_expected : expected->distribution)
		{
			auto iter_actual = actual->distribution.find(marking_expected.first);

			if (iter_actual == actual->distribution.end()) {

#ifdef DEBUG_PN_ID
				std::stringstream s;
				s << to_string(marking_expected.first) << " not generated";
				std::string str(s.str());
				Assert::Fail(std::wstring(str.begin(), str.end()).c_str());
#else
				Assert::Fail();
#endif
			}
			else {
#ifdef DEBUG_PN_ID
				std::string str(to_string(marking_expected.first));
				Assert::AreEqual(marking_expected.second, iter_actual->second, EPSILON,
					std::wstring(str.begin(), str.end()).c_str());
#else
				Assert::AreEqual(marking_expected.second, iter_actual->second, EPSILON);
#endif
			}
		}

		for (const auto& marking_actual : actual->distribution)
		{
			auto iter_expected = expected->distribution.find(marking_actual.first);

			if (iter_expected == expected->distribution.end()) {

#ifdef DEBUG_PN_ID
				std::stringstream s;
				s << to_string(marking_actual.first) << " invalid";
				std::string str(s.str());
				Assert::Fail(std::wstring(str.begin(), str.end()).c_str());
#else
				Assert::Fail();
#endif
			}
			else {
#ifdef DEBUG_PN_ID
				std::string str(to_string(marking_actual.first));
				Assert::AreEqual(iter_expected->second, marking_actual.second, EPSILON,
					std::wstring(str.begin(), str.end()).c_str());
#else
				Assert::AreEqual(marking_expected.second, iter_actual->second, EPSILON);
#endif
			}
		}

	}

	void evaluate_and_check(const pn_net::Ptr& net,
		const pn_belief_marking::ConstPtr& initial_marking,
		const pn_belief_marking::ConstPtr& final_marking,
		const pn_emission::ConstPtr& emissions,
		pn_emission::ConstPtr prev_emissions = nullptr)
	{
		pn_feasible_transition_extractor diff(net, initial_marking->to_marking());
		if(prev_emissions)
		{
			diff.update(prev_emissions);
			diff.update(initial_marking->to_marking());
		}
		
		//// derive prev_emission from difference of markings
		//if(!prev_emissions)
		//{
		//	auto places = net->get_places();
		//	std::set<pn_place::Ptr> empty_places(places.begin(), places.end());
		//	std::set<pn_place::Ptr> unobserved_places;
		//	auto initial_instances = initial_marking->to_marking()->distribution;
		//	std::map<pn_instance, double> common_instances;

		//	for (const auto& entry : initial_instances) {
		//		empty_places.erase(entry.first.first);
		//		unobserved_places.emplace(entry.first.first);
		//	}

		//	for (const auto& entry : final_marking->to_marking()->distribution)
		//	{
		//		empty_places.erase(entry.first.first);
		//		if(initial_instances.find(entry.first) == initial_instances.end() )
		//		{
		//			unobserved_places.emplace(entry.first.first);
		//		} else
		//		{
		//			unobserved_places.erase(entry.first.first);
		//			common_instances.emplace(entry);
		//		}
		//	}

		//	prev_emissions = std::make_shared<pn_emission>(std::move(empty_places), std::move(unobserved_places), std::move(common_instances));
		//}

		diff.update(emissions);
		auto dbn = diff.extract();
		
		sampling_optimizer_belief optimizer(dbn, initial_marking, emissions);
		auto generated_marking = optimizer.update(5000);

		assert_equal(final_marking, generated_marking);
	}

	pn_emission::ConstPtr generate_emission(const pn_net& net,
		const std::map<pn_instance, double>& observed_tokens,
		const std::set<pn_place::Ptr>& empty_places = std::set<pn_place::Ptr>())
	{
		std::map<pn_instance, double> token_distribution;
		std::set<pn_place::Ptr> unobserved_places(net.get_places().begin(), net.get_places().end());
		std::map<pn_place::Ptr, double> max_probabilities;

		for (const pn_place::Ptr& empty_place : empty_places)
			unobserved_places.erase(empty_place);

		for (const auto& entry : observed_tokens)
		{
			auto peak = max_probabilities.find(entry.first.first);

			if (peak == max_probabilities.end())
				max_probabilities.emplace(entry.first.first, entry.second);
			else
				peak->second = std::max(peak->second, entry.second);

			unobserved_places.erase(entry.first.first);

			for (const auto& token : net.get_tokens())
				token_distribution.emplace(std::make_pair(entry.first.first, token), 0);

			token_distribution.insert_or_assign(entry.first, entry.second);
		}

		return std::make_shared<pn_emission>(std::set<pn_place::Ptr>(empty_places), std::move(unobserved_places), std::move(token_distribution), std::move(max_probabilities));
	}

	TEST_METHOD(no_change)
	{
		auto net = pn_factory::two_block_stack(object_params);

		pn_binary_marking::Ptr initial_dist = std::make_shared<pn_binary_marking>(net, std::set<pn_instance>({ std::make_pair(net->get_place(0), net->get_token(0)),
			std::make_pair(net->get_place(1), net->get_token(1)) }));
		pn_belief_marking::ConstPtr initial_marking = std::make_shared<pn_belief_marking>(initial_dist);

		evaluate_and_check(net, initial_marking, initial_marking,
			generate_emission(*net, { std::make_pair(std::make_pair(net->get_place(0),  net->get_token(0)), 1.) }));
	}

	TEST_METHOD(one_step_certain)
	{
		auto net = pn_factory::cycle(object_params);

		pn_binary_marking::Ptr initial_dist = std::make_shared<pn_binary_marking>(net, std::set<pn_instance>({ std::make_pair(net->get_place(0), net->get_token(0)) }));
		pn_belief_marking::ConstPtr initial_marking = std::make_shared<pn_belief_marking>(initial_dist);

		pn_binary_marking::Ptr final_dist = std::make_shared<pn_binary_marking>(net, std::set<pn_instance>({ std::make_pair(net->get_place(1), net->get_token(0)) }));
		pn_belief_marking::ConstPtr final_marking = std::make_shared<pn_belief_marking>(final_dist);

		evaluate_and_check(net, initial_marking, final_marking,
			generate_emission(*net, { std::make_pair(std::make_pair(net->get_place(1),  net->get_token(0)), 1.) }, { net->get_place(0) }));
	}

	TEST_METHOD(one_step_uncertain)
	{
		auto net = pn_factory::cycle(object_params);

		pn_binary_marking::Ptr initial_dist = std::make_shared<pn_binary_marking>(net, std::set<pn_instance>({ std::make_pair(net->get_place(0), net->get_token(0)) }));
		pn_belief_marking::ConstPtr initial_marking = std::make_shared<pn_belief_marking>(initial_dist);

		pn_binary_marking::Ptr final_dist = std::make_shared<pn_binary_marking>(net, std::set<pn_instance>({ std::make_pair(net->get_place(1), net->get_token(0)) }));
		pn_belief_marking::ConstPtr final_marking = std::make_shared<pn_belief_marking>(final_dist);


		evaluate_and_check(net, initial_marking, final_marking,
			generate_emission(*net, {  }, { net->get_place(0), net->get_place(3), net->get_place(4) }));

	}

	TEST_METHOD(two_steps_certain)
	{
		auto net = pn_factory::cycle(object_params);

		pn_binary_marking::Ptr initial_dist = std::make_shared<pn_binary_marking>(net, std::set<pn_instance>({ std::make_pair(net->get_place(0), net->get_token(0)) }));
		pn_belief_marking::ConstPtr initial_marking = std::make_shared<pn_belief_marking>(initial_dist);

		pn_binary_marking::Ptr final_dist = std::make_shared<pn_binary_marking>(net, std::set<pn_instance>({ std::make_pair(net->get_place(1), net->get_token(0)) }));
		pn_belief_marking::ConstPtr final_marking = std::make_shared<pn_belief_marking>(final_dist);

		evaluate_and_check(net, initial_marking, final_marking,
			generate_emission(*net, { std::make_pair(std::make_pair(net->get_place(1),  net->get_token(0)), 1.) }, { net->get_place(0) }));
	}

	TEST_METHOD(ambigious_paths_certain)
	{
		auto net = pn_factory::two_paths(object_params);

		pn_binary_marking::Ptr initial_dist = std::make_shared<pn_binary_marking>(net, std::set<pn_instance>({ std::make_pair(net->get_place(0), net->get_token(0)) }));
		pn_belief_marking::ConstPtr initial_marking = std::make_shared<pn_belief_marking>(initial_dist);

		pn_binary_marking::Ptr final_dist = std::make_shared<pn_binary_marking>(net, std::set<pn_instance>({ std::make_pair(net->get_place(3), net->get_token(0)) }));
		pn_belief_marking::ConstPtr final_marking = std::make_shared<pn_belief_marking>(final_dist);

		evaluate_and_check(net, initial_marking, final_marking,
			generate_emission(*net, { std::make_pair(std::make_pair(net->get_place(3),  net->get_token(0)), 1.) }));
	}

	TEST_METHOD(ambigious_paths_uncertain)
	{
		auto net = pn_factory::two_paths(object_params);

		pn_binary_marking::Ptr initial_dist = std::make_shared<pn_binary_marking>(net, std::set<pn_instance>({ std::make_pair(net->get_place(0), net->get_token(0)) }));
		pn_belief_marking::ConstPtr initial_marking = std::make_shared<pn_belief_marking>(initial_dist);

		pn_binary_marking::Ptr v1 = std::make_shared<pn_binary_marking>(net, std::set<pn_instance>({ std::make_pair(net->get_place(1), net->get_token(0)) }));
		pn_binary_marking::Ptr v2 = std::make_shared<pn_binary_marking>(net, std::set<pn_instance>({ std::make_pair(net->get_place(2), net->get_token(0)) }));
		pn_belief_marking::marking_dist_t final_dist;
		final_dist.emplace(v1, 0.5);
		final_dist.emplace(v2, 0.5);
		pn_belief_marking::ConstPtr final_marking = std::make_shared<pn_belief_marking>(net, std::move(final_dist));

		evaluate_and_check(net, initial_marking, final_marking,
			generate_emission(*net, {}, { net->get_place(0) }));
	}

	TEST_METHOD(collect_uncertain)
	{
		auto net = pn_factory::cycle(object_params);

		pn_binary_marking::Ptr v1 = std::make_shared<pn_binary_marking>(net, std::set<pn_instance>({ std::make_pair(net->get_place(0), net->get_token(0)) }));
		pn_binary_marking::Ptr v2 = std::make_shared<pn_binary_marking>(net, std::set<pn_instance>({ std::make_pair(net->get_place(2), net->get_token(0)) }));
		pn_belief_marking::marking_dist_t initial_dist;
		initial_dist.emplace(v1, 0.5);
		initial_dist.emplace(v2, 0.5);
		pn_belief_marking::ConstPtr initial_marking = std::make_shared<pn_belief_marking>(net, std::move(initial_dist));

		pn_binary_marking::Ptr final_dist = std::make_shared<pn_binary_marking>(net, std::set<pn_instance>({ std::make_pair(net->get_place(4), net->get_token(0)) }));
		pn_belief_marking::ConstPtr final_marking = std::make_shared<pn_belief_marking>(final_dist);

		evaluate_and_check(net, initial_marking, final_marking,
			generate_emission(*net, { std::make_pair(std::make_pair(net->get_place(4),  net->get_token(0)), 1.) }, { net->get_place(0) }));

	}

	TEST_METHOD(pick)
	{
		auto net = pn_factory::pick_and_place(object_params,2, 1, 2);

		pn_binary_marking::Ptr initial_dist = std::make_shared<pn_binary_marking>(net, std::set<pn_instance>({ std::make_pair(net->get_place(2), net->get_token(0)) }));
		pn_belief_marking::ConstPtr initial_marking = std::make_shared<pn_belief_marking>(initial_dist);

		pn_binary_marking::Ptr v1 = std::make_shared<pn_binary_marking>(net, std::set<pn_instance>({ std::make_pair(net->get_place(0), net->get_token(0)) }));
		pn_binary_marking::Ptr v2 = std::make_shared<pn_binary_marking>(net, std::set<pn_instance>({ std::make_pair(net->get_place(1), net->get_token(0))}));
		pn_belief_marking::marking_dist_t final_dist;
		final_dist.emplace(v1, 0.5);
		final_dist.emplace(v2, 0.5);
		pn_belief_marking::ConstPtr final_marking = std::make_shared<pn_belief_marking>(net, std::move(final_dist));

		evaluate_and_check(net, initial_marking, final_marking,
			generate_emission(*net, {}, { net->get_place(2) }));

	}

	TEST_METHOD(swap_starting_in_hand)
	{
		auto net = pn_factory::pick_and_place(object_params,2,2,2);

		pn_binary_marking::Ptr v1 = std::make_shared<pn_binary_marking>(net, std::set<pn_instance>({ std::make_pair(net->get_place(0), net->get_token(0)), std::make_pair(net->get_place(2), net->get_token(1)) }));
//		pn_binary_marking::Ptr v2 = std::make_shared<pn_binary_marking>(net, std::set<pn_instance>({ std::make_pair(net->get_place(1), net->get_token(0)), std::make_pair(net->get_place(2), net->get_token(1)) }));
		pn_belief_marking::marking_dist_t initial_dist;
		initial_dist.emplace(v1, 1);
//		initial_dist.emplace(v2, 0.5);
		pn_belief_marking::ConstPtr initial_marking = std::make_shared<pn_belief_marking>(net, std::move(initial_dist));

		pn_binary_marking::Ptr final_dist = std::make_shared<pn_binary_marking>(net, std::set<pn_instance>({ std::make_pair(net->get_place(2), net->get_token(0)), std::make_pair(net->get_place(3), net->get_token(1)) }));
		pn_belief_marking::ConstPtr final_marking = std::make_shared<pn_belief_marking>(final_dist);

		evaluate_and_check(net, initial_marking, final_marking,
			generate_emission(*net, {
			std::make_pair(std::make_pair(net->get_place(2), net->get_token(0)), 1.),
			std::make_pair(std::make_pair(net->get_place(3), net->get_token(1)), 1.),
			}),
			std::make_shared<pn_emission>(std::set<pn_place::Ptr>(), std::set<pn_place::Ptr>(net->get_places().begin(), net->get_places().end()), std::map<pn_instance, double>(), 
				std::map<pn_place::Ptr, double>({
				std::make_pair(net->get_place(2), 1.),
				std::make_pair(net->get_place(3), 1.),
				})));

	}

	TEST_METHOD(multiple_sources)
	{
		auto net = pn_factory::pick_and_place(object_params, 4, 1, 4);
		// destination place: 4
		// source places: 5-7
		pn_binary_marking::Ptr initial_dist = std::make_shared<pn_binary_marking>(net, std::set<pn_instance>({ 
			std::make_pair(net->get_place(5), net->get_token(0)),
			std::make_pair(net->get_place(6), net->get_token(0)),
			std::make_pair(net->get_place(7), net->get_token(0))
			}));
		pn_belief_marking::ConstPtr initial_marking = std::make_shared<pn_belief_marking>(initial_dist);
		
		pn_belief_marking::marking_dist_t final_dist;
		for (int i = 5; i < 8; i++)
		{
			std::set<pn_instance> instances({ std::make_pair(net->get_place(4), net->get_token(0)) });
			for (int j = 5; j < 8; j++)
				if (i != j)
					instances.emplace(net->get_place(j), net->get_token(0));

			final_dist.emplace(std::make_shared<pn_binary_marking>(net, instances), 1./3);
		}
		pn_belief_marking::ConstPtr final_marking = std::make_shared<pn_belief_marking>(net, std::move(final_dist));

		evaluate_and_check(net, initial_marking, final_marking,
			generate_emission(*net, { std::make_pair(std::make_pair(net->get_place(4), net->get_token(0)), 1.)}));

	}

	/*
	// violates assumption that objects are exchanged between two observations

	TEST_METHOD(swap)
	{
		auto net = pn_factory::pick_and_place(2, 2, 2);

		pn_binary_marking::Ptr v = std::make_shared<pn_binary_marking>(net, std::set<pn_instance>({ std::make_pair(net->get_place(3), net->get_token(0)), std::make_pair(net->get_place(2), net->get_token(1)) }));
		pn_belief_marking::marking_dist_t initial_dist;
		initial_dist.emplace(v, 1);
		pn_belief_marking::ConstPtr initial_marking = std::make_shared<pn_belief_marking>(net, std::move(initial_dist));

		pn_binary_marking::Ptr final_dist = std::make_shared<pn_binary_marking>(net, std::set<pn_instance>({ std::make_pair(net->get_place(2), net->get_token(0)), std::make_pair(net->get_place(3), net->get_token(1)) }));
		pn_belief_marking::ConstPtr final_marking = std::make_shared<pn_belief_marking>(final_dist);

		evaluate_and_check(net, initial_marking, final_marking,
			generate_emission(*net, {
			std::make_pair(std::make_pair(net->get_place(2), net->get_token(0)), 1.),
			std::make_pair(std::make_pair(net->get_place(3), net->get_token(1)), 1.)
			}));

	}
		*/

	/*
	TEST_METHOD(side_condition_certain)
	{
		auto net = pn_factory::two_block_stack();

		std::map<pn_instance, double> initial_dist({
			std::make_pair(std::make_pair(net->get_place(3), net->get_token(0)), 1.),
			std::make_pair(std::make_pair(net->get_place(1), net->get_token(1)), 1.),
			});
		pn_marking::ConstPtr initial_marking = std::make_shared<pn_marking>(net, std::move(initial_dist));

		std::map<pn_instance, double> final_dist({
			std::make_pair(std::make_pair(net->get_place(1), net->get_token(1)), 0.),
			std::make_pair(std::make_pair(net->get_place(3), net->get_token(0)), 1.),
			std::make_pair(std::make_pair(net->get_place(4), net->get_token(1)), 1.),
			});
		pn_marking::ConstPtr final_marking = std::make_shared<pn_marking>(net, std::move(final_dist));

		evaluate_and_check(net, initial_marking, final_marking,
			{
			std::make_pair(std::make_pair(net->get_place(3),  net->get_token(0)), 1.),
			std::make_pair(std::make_pair(net->get_place(4),  net->get_token(1)), 1.),
			std::make_pair(std::make_pair(net->get_place(1),  net->get_token(1)), 0.)
			});

	}

	TEST_METHOD(stacking_complete_certain)
	{
		//		std::this_thread::sleep_for(std::chrono::seconds(5));

		auto net = pn_factory::two_block_stack();

		std::map<pn_instance, double> initial_dist({
			std::make_pair(std::make_pair(net->get_place(0), net->get_token(0)), 1.),
			std::make_pair(std::make_pair(net->get_place(1), net->get_token(1)), 1.),
			});
		pn_marking::ConstPtr initial_marking = std::make_shared<pn_marking>(net, std::move(initial_dist));

		std::map<pn_instance, double> final_dist({
			std::make_pair(std::make_pair(net->get_place(3), net->get_token(0)), 1.),
			std::make_pair(std::make_pair(net->get_place(4), net->get_token(1)), 1.),
			});
		pn_marking::ConstPtr final_marking = std::make_shared<pn_marking>(net, std::move(final_dist));

		std::map<pn_instance, double> emissions = {
			std::make_pair(std::make_pair(net->get_place(0),  net->get_token(1)), 0.),
			std::make_pair(std::make_pair(net->get_place(1),  net->get_token(1)), 0.),
			std::make_pair(std::make_pair(net->get_place(4),  net->get_token(1)), 1.)
		};

		evaluate_and_check(net, initial_marking, final_marking, emissions);

	}

	TEST_METHOD(stacking_complete_uncertain)
	{
		auto net = pn_factory::two_block_stack();

		std::map<pn_instance, double> initial_dist({
			std::make_pair(std::make_pair(net->get_place(0), net->get_token(0)), 0.5),
			std::make_pair(std::make_pair(net->get_place(3), net->get_token(0)), 0.5),
			std::make_pair(std::make_pair(net->get_place(1), net->get_token(1)), 0.5),
			std::make_pair(std::make_pair(net->get_place(2), net->get_token(1)), 0.5),
			});
		pn_marking::ConstPtr initial_marking = std::make_shared<pn_marking>(net, std::move(initial_dist));

		std::map<pn_instance, double> final_dist({
			std::make_pair(std::make_pair(net->get_place(3), net->get_token(0)), 1.),
			std::make_pair(std::make_pair(net->get_place(4), net->get_token(1)), 1.),
			});
		pn_marking::ConstPtr final_marking = std::make_shared<pn_marking>(net, std::move(final_dist));

		evaluate_and_check(net, initial_marking, final_marking,
			{
			std::make_pair(std::make_pair(net->get_place(4),  net->get_token(1)), 1.)
			});
	}

	TEST_METHOD(omega_network_2_certain)
	{
		auto net = pn_factory::omega_network(1);

		std::map<pn_instance, double> initial_dist({
			std::make_pair(std::make_pair(net->get_place(0), net->get_token(0)), 1),
			std::make_pair(std::make_pair(net->get_place(1), net->get_token(1)), 1),
			});
		pn_marking::ConstPtr initial_marking = std::make_shared<pn_marking>(net, std::move(initial_dist));


		std::map<pn_instance, double> final_dist({
			std::make_pair(std::make_pair(net->get_place(2), net->get_token(1)), 1.),
			std::make_pair(std::make_pair(net->get_place(3), net->get_token(0)), 1.),
			});
		for (int i = 4; i < net->get_places().size(); i++)
		{
			for (const pn_token::Ptr token : net->get_tokens())
				final_dist.emplace(std::make_pair(net->get_place(i), token), 0.);
		}
		pn_marking::ConstPtr final_marking = std::make_shared<pn_marking>(net, std::move(final_dist));


		evaluate_and_check(net, initial_marking, final_marking, final_dist);
	}

	TEST_METHOD(omega_network_4_certain)
	{
		auto net = pn_factory::omega_network(2);

		std::map<pn_instance, double> initial_dist({
			std::make_pair(std::make_pair(net->get_place(0), net->get_token(0)), 1.),
			std::make_pair(std::make_pair(net->get_place(1), net->get_token(1)), 1.),
			std::make_pair(std::make_pair(net->get_place(2), net->get_token(2)), 1.),
			std::make_pair(std::make_pair(net->get_place(3), net->get_token(3)), 1.),
			});
		pn_marking::ConstPtr initial_marking = std::make_shared<pn_marking>(net, std::move(initial_dist));


		std::map<pn_instance, double> final_dist({
			std::make_pair(std::make_pair(net->get_place(4), net->get_token(1)), 1.),
			std::make_pair(std::make_pair(net->get_place(5), net->get_token(3)), 1.),
			std::make_pair(std::make_pair(net->get_place(6), net->get_token(2)), 1.),
			std::make_pair(std::make_pair(net->get_place(7), net->get_token(0)), 1.),
			});
		for (int i = 8; i < net->get_places().size(); i++)
		{
			for (const pn_token::Ptr token : net->get_tokens())
				final_dist.emplace(std::make_pair(net->get_place(i), token), 0.);
		}


		evaluate_and_check(net, initial_marking,
			std::make_shared<pn_marking>(net, std::map<pn_instance, double>(final_dist)), final_dist);
	}

	TEST_METHOD(omega_network_4_certain_partial)
	{
		auto net = pn_factory::omega_network(2);

		std::map<pn_instance, double> initial_dist({
			std::make_pair(std::make_pair(net->get_place(0), net->get_token(0)), 1.),
			std::make_pair(std::make_pair(net->get_place(1), net->get_token(1)), 1.),
			std::make_pair(std::make_pair(net->get_place(2), net->get_token(2)), 1.),
			std::make_pair(std::make_pair(net->get_place(3), net->get_token(3)), 1.),
			});
		pn_marking::ConstPtr initial_marking = std::make_shared<pn_marking>(net, std::move(initial_dist));


		std::map<pn_instance, double> final_dist({
			std::make_pair(std::make_pair(net->get_place(4), net->get_token(1)), 1.),
			std::make_pair(std::make_pair(net->get_place(5), net->get_token(3)), 1.),
			std::make_pair(std::make_pair(net->get_place(6), net->get_token(2)), 1.),
			std::make_pair(std::make_pair(net->get_place(7), net->get_token(0)), 1.),
			});
		for (int i = 8; i < net->get_places().size(); i++)
		{
			for (const pn_token::Ptr token : net->get_tokens())
				final_dist.emplace(std::make_pair(net->get_place(i), token), 0.);
		}
		pn_marking::ConstPtr final_marking = std::make_shared<pn_marking>(net, std::move(final_dist));


		std::map<pn_instance, double> observed_dist({
			std::make_pair(std::make_pair(net->get_place(0), net->get_token(0)), 0.),
			std::make_pair(std::make_pair(net->get_place(1), net->get_token(1)), 0.),
			std::make_pair(std::make_pair(net->get_place(2), net->get_token(2)), 0.),
			std::make_pair(std::make_pair(net->get_place(3), net->get_token(3)), 0.),
			std::make_pair(std::make_pair(net->get_place(5), net->get_token(3)), 1.),
			std::make_pair(std::make_pair(net->get_place(7), net->get_token(0)), 1.),
			});
		for (int i = 8; i < net->get_places().size(); i++)
		{
			for (const pn_token::Ptr token : net->get_tokens())
				observed_dist.emplace(std::make_pair(net->get_place(i), token), 0.);
		}

		evaluate_and_check(net, initial_marking, final_marking, observed_dist);
	}

	TEST_METHOD(omega_network_8_certain)
	{
		unsigned int n = 3;
		auto net = pn_factory::omega_network(n);

		std::map<pn_instance, double> initial_dist;
		for (int i = 0; i < std::pow(2, n); i++)
			initial_dist.emplace(std::make_pair(net->get_place(i), net->get_token(i)), 1.);

		pn_marking::ConstPtr initial_marking = std::make_shared<pn_marking>(net, std::move(initial_dist));


		std::map<pn_instance, double> final_dist({
			std::make_pair(std::make_pair(net->get_place(8), net->get_token(2)), 1.),
			std::make_pair(std::make_pair(net->get_place(9), net->get_token(3)), 1.),
			std::make_pair(std::make_pair(net->get_place(10), net->get_token(4)), 1.),
			std::make_pair(std::make_pair(net->get_place(11), net->get_token(1)), 1.),
			std::make_pair(std::make_pair(net->get_place(12), net->get_token(6)), 1.),
			std::make_pair(std::make_pair(net->get_place(13), net->get_token(5)), 1.),
			std::make_pair(std::make_pair(net->get_place(14), net->get_token(7)), 1.),
			std::make_pair(std::make_pair(net->get_place(15), net->get_token(0)), 1.),
			});
		for (int i = 8; i < net->get_places().size(); i++)
		{
			for (const pn_token::Ptr token : net->get_tokens())
				final_dist.emplace(std::make_pair(net->get_place(i), token), 0.);
		}


		evaluate_and_check(net, initial_marking,
			std::make_shared<pn_marking>(net, std::map<pn_instance, double>(final_dist)), final_dist);
	}

	*/
};
}