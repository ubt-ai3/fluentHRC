#include "CppUnitTest.h"

#include "state_observation/pn_reasoning.hpp"
#include "pn_util.hpp"

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
	TEST_CLASS(pn_differ_test)
	{
	public:
		const state_observation::object_parameters object_params;



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
		
		TEST_METHOD(single_node)
		{
			object_parameters object_params;

			pn_net::Ptr net(new pn_net(object_params));
			auto place = net->create_place();
			pn_token::Ptr token(new pn_token);
			std::map<pn_instance, double> dist({ std::make_pair(std::make_pair(place,token), 0.) });
			pn_marking::ConstPtr marking = std::make_shared<pn_marking>(net, std::move(dist));

			pn_feasible_transition_extractor diff(net, marking);
			diff.update(generate_emission(*net, { std::make_pair(std::make_pair(place, token), 1.) }));
			auto transitions = diff.extract();

			Assert::AreEqual(0, (int) transitions.size());
		}

		TEST_METHOD(ambigious_paths)
		{
			auto net = pn_factory::two_paths(object_params);
			
			std::map<pn_instance, double> dist({ std::make_pair(std::make_pair(net->get_place(0),net->get_token(0)), 1.)});
			pn_marking::ConstPtr marking = std::make_shared<pn_marking>(net, std::move(dist));

			pn_feasible_transition_extractor diff(net, marking);
			diff.update(generate_emission(*net, { std::make_pair(std::make_pair(net->get_place(3), net->get_token(0)), 1.) }));
			auto transitions = diff.extract();

			Assert::AreEqual(4, (int)transitions.size());
		}

		TEST_METHOD(cycle_segment)
		{
			auto net = pn_factory::cycle(object_params);
			
			std::map<pn_instance, double> dist({ std::make_pair(std::make_pair(net->get_place(0), net->get_token(0)), 1.)});
			pn_marking::ConstPtr marking = std::make_shared<pn_marking>(net, std::move(dist));

			pn_feasible_transition_extractor diff(net, marking);
			diff.update(generate_emission(*net, 
				{std::make_pair(std::make_pair(net->get_place(2),  net->get_token(0)), 1.) },
				{ net->get_place(0) }
			));
			auto transitions = diff.extract();

			Assert::AreEqual(2, (int)transitions.size());
		}


		TEST_METHOD(ensure_termination)
		{
			auto net = pn_factory::cycle(object_params);

			std::map<pn_instance, double> dist({  });
			pn_marking::ConstPtr marking = std::make_shared<pn_marking>(net, std::move(dist));

			pn_feasible_transition_extractor diff(net, marking);
			diff.update(generate_emission(*net, { std::make_pair(std::make_pair(net->get_place(0),  net->get_token(0)), 1) }));
			auto transitions = diff.extract();

			Assert::IsTrue((int)transitions.size() <= 5);
		}


		TEST_METHOD(collect_uncertain)
		{
			auto net = pn_factory::cycle(object_params);

			std::map<pn_instance, double> initial_dist({
				std::make_pair(std::make_pair(net->get_place(0), net->get_token(0)),0.5),
				std::make_pair(std::make_pair(net->get_place(2), net->get_token(0)), 0.5),
				});
			pn_marking::ConstPtr initial_marking = std::make_shared<pn_marking>(net, std::move(initial_dist));


			pn_feasible_transition_extractor diff(net, initial_marking);
			diff.update(generate_emission(*net, {
				std::make_pair(std::make_pair(net->get_place(4),  net->get_token(0)), 1.)
				}));
			auto transitions = diff.extract();

			Assert::AreEqual(4, (int)transitions.size());
		}

		TEST_METHOD(generate_unobserved)
		{
			auto net = pn_factory::two_paths(object_params);

			std::map<pn_instance, double> initial_dist({
				std::make_pair(std::make_pair(net->get_place(0), net->get_token(0)), 1.)
				});
			pn_marking::ConstPtr initial_marking = std::make_shared<pn_marking>(net, std::move(initial_dist));


			pn_feasible_transition_extractor diff(net, initial_marking);
			diff.update(generate_emission(*net, {}, { net->get_place(0) }));

			auto transitions = diff.extract();

			Assert::IsTrue((int)transitions.size() >= 2);
		}

		TEST_METHOD(stacking_uncertain)
		{
			auto net = pn_factory::two_block_stack(object_params);

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

			pn_feasible_transition_extractor diff(net, initial_marking);
			diff.update(generate_emission(*net, {
				std::make_pair(std::make_pair(net->get_place(4),  net->get_token(1)), 1.)
				}));
			auto transitions = diff.extract();

			Assert::AreEqual(4, (int)transitions.size());
			
		}
	};

	TEST_CLASS(pn_execution_test)
	{
	public:
		static constexpr double EPSILON = 0.0001;

		const state_observation::object_parameters object_params;
		
		TEST_METHOD(one_certain_step)
		{
			auto net = pn_factory::two_paths(object_params);

			std::map<pn_instance, double> dist({ std::make_pair(std::make_pair(net->get_place(0),net->get_token(0)), 1.) });
			pn_marking::ConstPtr marking = std::make_shared<pn_marking>(net, std::move(dist));

			Assert::AreEqual(1., marking->is_enabled(net->get_transition(0)), 0.001, L"Transition enabled.");
			marking = marking->fire(net->get_transition(0));

			Assert::AreEqual(1., marking->get_probability(net->get_place(1), net->get_token(0)), EPSILON, L"Token produced.");
			Assert::AreEqual(0., marking->get_probability(net->get_place(0), net->get_token(0)), EPSILON, L"Token consumed.");
		}

		TEST_METHOD(one_uncertain_step)
		{
			auto net = pn_factory::two_paths(object_params);

			std::map<pn_instance, double> dist({ std::make_pair(std::make_pair(net->get_place(0),net->get_token(0)), 0.5) });
			pn_marking::ConstPtr marking = std::make_shared<pn_marking>(net, std::move(dist));

			marking = marking->fire(net->get_transition(0));

			Assert::AreEqual(0.5, marking->get_probability(net->get_place(1), net->get_token(0)), EPSILON);
			Assert::AreEqual(0., marking->get_probability(net->get_place(0), net->get_token(0)), EPSILON);
		}

		TEST_METHOD(no_transition)
		{
			auto net = pn_factory::two_paths(object_params);

			std::map<pn_instance, double> dist({ std::make_pair(std::make_pair(net->get_place(1),net->get_token(0)), 1.) });
			pn_marking::ConstPtr marking = std::make_shared<pn_marking>(net, std::move(dist));

			try {
				std::ignore = marking->fire(net->get_transition(1));
				Assert::Fail();
			} catch(std::invalid_argument &)
			{	}

		}

		TEST_METHOD(stack_certain)
		{
			auto net = pn_factory::two_block_stack(object_params);

			const auto m0 = net->get_token(0);
			const auto m1 = net->get_token(1);

			std::map<pn_instance, double> dist({ std::make_pair(std::make_pair(net->get_place(0),m0), 1.),
				std::make_pair(std::make_pair(net->get_place(1),m1), 1.) });
			pn_marking::ConstPtr marking = std::make_shared<pn_marking>(net, std::move(dist));

			marking = marking->fire(net->get_transition(0));
			marking = marking->fire(net->get_transition(1));
			marking = marking->fire(net->get_transition(2));
			marking = marking->fire(net->get_transition(3));

			const auto p3 = net->get_place(3);
			const auto p4 = net->get_place(4);
			
			for(const auto& p : net->get_places())
				for(const auto& m : net->get_tokens() )
				{
					if(p == p3 && m == m0)
						Assert::AreEqual(1., marking->get_probability(p, m), EPSILON);
					else if(p == p4 && m == m1)
						Assert::AreEqual(1., marking->get_probability(p, m), EPSILON);
					else
						Assert::AreEqual(0., marking->get_probability(p, m), EPSILON);
				}
			
		}

		TEST_METHOD(stack_uncertain)
		{
			auto net = pn_factory::two_block_stack(object_params);

			const auto m0 = net->get_token(0);
			const auto m1 = net->get_token(1);

			std::map<pn_instance, double> dist({ std::make_pair(std::make_pair(net->get_place(0),m0), 0.5),
				std::make_pair(std::make_pair(net->get_place(1),m1), 0.8) });
			pn_marking::ConstPtr marking = std::make_shared<pn_marking>(net, std::move(dist));

			marking = marking->fire(net->get_transition(0));
			marking = marking->fire(net->get_transition(1));
			marking = marking->fire(net->get_transition(2));
			marking = marking->fire(net->get_transition(3));

			const auto p2 = net->get_place(2);
			const auto p3 = net->get_place(3);
			const auto p4 = net->get_place(4);

			for (const auto& p : net->get_places())
				for (const auto& m : net->get_tokens())
				{
					if (p == p3 && m == m0)
						Assert::AreEqual(0.5, marking->get_probability(p, m), EPSILON);
					else if (p == p4 && m == m1)
						Assert::AreEqual(0.5, marking->get_probability(p, m), EPSILON);
					else if (p == p2 && m == m1)
						Assert::AreEqual(0.3, marking->get_probability(p, m), EPSILON);
					else
						Assert::AreEqual(0., marking->get_probability(p, m), EPSILON);
				}

		}

		
	};

	TEST_CLASS(pn_binary_execution_test)
	{
	public:
		const state_observation::object_parameters object_params;

		TEST_METHOD(hashing) 
		{
			auto net = pn_factory::two_block_stack(object_params);

			pn_binary_marking::Ptr v1 = std::make_shared<pn_binary_marking>(net, std::set<pn_instance>({ std::make_pair(net->get_place(0), net->get_token(0)) }));
			pn_binary_marking::Ptr u1 = std::make_shared<pn_binary_marking>(net, std::set<pn_instance>({ std::make_pair(net->get_place(0), net->get_token(0)) }));
		
			Assert::AreEqual(v1->hash(), u1->hash());

			pn_binary_marking::Ptr v2 = std::make_shared<pn_binary_marking>(net, std::set<pn_instance>({ std::make_pair(net->get_place(0), net->get_token(0)), std::make_pair(net->get_place(1), net->get_token(1)) }));
			pn_binary_marking::Ptr u2 = std::make_shared<pn_binary_marking>(net, std::set<pn_instance>({ std::make_pair(net->get_place(0), net->get_token(0)), std::make_pair(net->get_place(1), net->get_token(1)) }));

			Assert::AreEqual(v2->hash(), u2->hash());

			pn_binary_marking::Ptr v3 = std::make_shared<pn_binary_marking>(net, std::set<pn_instance>({ std::make_pair(net->get_place(1), net->get_token(1)), std::make_pair(net->get_place(0), net->get_token(0)) }));
			pn_binary_marking::Ptr u3 = std::make_shared<pn_binary_marking>(net, std::set<pn_instance>({ std::make_pair(net->get_place(0), net->get_token(0)), std::make_pair(net->get_place(1), net->get_token(1)) }));

			Assert::AreEqual(v3->hash(), u3->hash());
		}

		TEST_METHOD(equality_check)
		{
			auto net = pn_factory::two_block_stack(object_params);

			pn_binary_marking::Ptr v1 = std::make_shared<pn_binary_marking>(net, std::set<pn_instance>({ std::make_pair(net->get_place(0), net->get_token(0)) }));
			pn_binary_marking::Ptr u1 = std::make_shared<pn_binary_marking>(net, std::set<pn_instance>({ std::make_pair(net->get_place(0), net->get_token(0)) }));

			Assert::IsTrue(pn_binary_marking::eq(v1,u1), L"not equal");

			pn_binary_marking::Ptr v2 = std::make_shared<pn_binary_marking>(net, std::set<pn_instance>({ std::make_pair(net->get_place(0), net->get_token(0)), std::make_pair(net->get_place(1), net->get_token(1)) }));
			pn_binary_marking::Ptr u2 = std::make_shared<pn_binary_marking>(net, std::set<pn_instance>({ std::make_pair(net->get_place(0), net->get_token(0)), std::make_pair(net->get_place(1), net->get_token(1)) }));

			Assert::IsTrue(pn_binary_marking::eq(v2 , u2), L"not equal");

			pn_binary_marking::Ptr v3 = std::make_shared<pn_binary_marking>(net, std::set<pn_instance>({ std::make_pair(net->get_place(1), net->get_token(1)), std::make_pair(net->get_place(0), net->get_token(0)) }));
			pn_binary_marking::Ptr u3 = std::make_shared<pn_binary_marking>(net, std::set<pn_instance>({ std::make_pair(net->get_place(0), net->get_token(0)), std::make_pair(net->get_place(1), net->get_token(1)) }));

			Assert::IsTrue(pn_binary_marking::eq(v3 , u3), L"not equal");
		}

		TEST_METHOD(one_certain_step)
		{
			auto net = pn_factory::two_paths(object_params);

			std::set<pn_instance> dist({ std::make_pair(net->get_place(0),net->get_token(0)) });
			pn_binary_marking::Ptr marking = std::make_shared<pn_binary_marking>(net, std::move(dist));

			Assert::AreEqual(1., marking->is_enabled(net->get_transition(0)), 0.001, L"Transition enabled.");
			marking = marking->fire(net->get_transition(0));

			Assert::IsTrue(marking->has(net->get_place(1), net->get_token(0)), L"Token produced.");
			Assert::IsFalse(marking->has(net->get_place(0), net->get_token(0)), L"Token consumed.");
		}

		TEST_METHOD(no_transition)
		{
			auto net = pn_factory::two_paths(object_params);

			std::set<pn_instance> dist({ std::make_pair(net->get_place(1),net->get_token(0)) });
			pn_binary_marking::Ptr marking = std::make_shared<pn_binary_marking>(net, std::move(dist));

			try {
				marking->fire(net->get_transition(1));
				Assert::Fail();
			}
			catch (std::invalid_argument&)
			{
			}

		}

		TEST_METHOD(stack_certain)
		{
			auto net = pn_factory::two_block_stack(object_params);

			const auto m0 = net->get_token(0);
			const auto m1 = net->get_token(1);

			std::set<pn_instance> dist({ std::make_pair(net->get_place(0),m0),
				std::make_pair(net->get_place(1),m1) });
			pn_binary_marking::Ptr marking = std::make_shared<pn_binary_marking>(net, std::move(dist));

			marking = marking->fire(net->get_transition(0));
			marking = marking->fire(net->get_transition(2));
			marking = marking->fire(net->get_transition(1));
			marking = marking->fire(net->get_transition(3));

			const auto p3 = net->get_place(3);
			const auto p4 = net->get_place(4);

			for (const auto& p : net->get_places())
				for (const auto& m : net->get_tokens())
				{
					if (p == p3 && m == m0)
						Assert::IsTrue(marking->has(p, m));
					else if (p == p4 && m == m1)
						Assert::IsTrue(marking->has(p, m));
					else
						Assert::IsFalse(marking->has(p, m));
				}

		}

		


	};
}

