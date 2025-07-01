#include "pn_util.hpp"
#include <cmath>

namespace unittests
{

using namespace state_observation;


state_observation::pn_net::Ptr pn_factory::two_paths(const state_observation::object_parameters& object_params)
{
	state_observation::pn_net::Ptr net = std::make_shared<state_observation::pn_net>(object_params);
	auto p0 = net->create_place();
	auto p1 = net->create_place();
	auto p2 = net->create_place();
	auto p3 = net->create_place();
	state_observation::pn_token::Ptr t = std::make_shared<state_observation::pn_token>();

	net->create_transition({std::make_pair(p0, t)}, {std::make_pair(p1, t)});
	net->create_transition({std::make_pair(p0, t)}, {std::make_pair(p2, t)});
	net->create_transition({std::make_pair(p1, t)}, {std::make_pair(p3, t)});
	net->create_transition({std::make_pair(p2, t)}, {std::make_pair(p3, t)});

	return net;
}
state_observation::pn_net::Ptr pn_factory::cycle(const state_observation::object_parameters& object_params)
{
	state_observation::pn_net::Ptr net = std::make_shared<state_observation::pn_net>(object_params);
	auto p0 = net->create_place();
	auto p1 = net->create_place();
	auto p2 = net->create_place();
	auto p3 = net->create_place();
	auto p4 = net->create_place();
	state_observation::pn_token::Ptr t = std::make_shared<state_observation::pn_token>();

	net->create_transition({ std::make_pair(p0,t) }, { std::make_pair(p1,t) });
	net->create_transition({ std::make_pair(p1,t) }, { std::make_pair(p2,t) });
	net->create_transition({ std::make_pair(p2,t) }, { std::make_pair(p3,t) });
	net->create_transition({ std::make_pair(p3,t) }, { std::make_pair(p4,t) });
	net->create_transition({ std::make_pair(p4,t) }, { std::make_pair(p0,t) });

	return net;
}
state_observation::pn_net::Ptr pn_factory::two_block_stack(const state_observation::object_parameters& object_params)
{
	state_observation::pn_net::Ptr net = std::make_shared<state_observation::pn_net>(object_params);
	
	auto p0 = net->create_place();
	auto p1 = net->create_place();
	auto p2 = net->create_place(true);
	auto p3 = net->create_place();
	auto p4 = net->create_place();
	
	state_observation::pn_token::Ptr m0 = std::make_shared<state_observation::pn_token>();
	state_observation::pn_token::Ptr m1 = std::make_shared<state_observation::pn_token>();

	net->create_transition({ std::make_pair(p0,m0) }, { std::make_pair(p2,m0) });
	net->create_transition({ std::make_pair(p1,m1) }, { std::make_pair(p2,m1) });
	net->create_transition({ std::make_pair(p2,m0) }, { std::make_pair(p3,m0) });
	net->create_transition({ std::make_pair(p2,m1), std::make_pair(p3, m0) }, { std::make_pair(p4,m1), std::make_pair(p3, m0) });

	return net;
}

	
state_observation::pn_net::Ptr pn_factory::omega_network(const state_observation::object_parameters& object_params, unsigned int n)
{
	const unsigned int count_nodes = static_cast<unsigned int>(std::pow(2, n));
	state_observation::pn_net::Ptr net = std::make_shared<state_observation::pn_net>(object_params);

	auto rotate_right = [n](unsigned int x)
	{
		unsigned int lowest_bit = x % 2;
		unsigned int shifted = x >> 1;
		shifted |= lowest_bit << (n - 1);
		return shifted;
	};
	
	std::vector<pn_place::Ptr> inputs;
	std::vector<pn_place::Ptr> outputs;
	std::vector<pn_token::Ptr> tokens;
	
	// has 2 * count_nodes entries, the lowest bit distinguishes
	// the two outputs of a switch
	std::vector<pn_place::Ptr> prev_stage;

	for(unsigned int i = 0; i < count_nodes; i++)
	{
		inputs.push_back(net->create_place());
		// each of the count_nodes switches has 2 inputs and outputs
		// named low and high
		prev_stage.push_back(inputs.back());
		prev_stage.push_back(inputs.back());
		tokens.push_back(std::make_shared<pn_token>());
	}

	for (int i = 0; i < std::pow(2, n); i++)
	{
		outputs.push_back(net->create_place());
	}

	auto create_instances = [&tokens](const pn_place::Ptr& place)
	{
		std::vector<pn_instance> instances;
		for (const pn_token::Ptr& token : tokens)
		{
			instances.emplace_back(place, token);
		}

		return instances;
	};
	
	for(unsigned int step = 0; step < n - 1; step++)
	{
		std::vector<pn_place::Ptr> next_stage;
		for(unsigned int i = 0; i < count_nodes; i++)
		{
			pn_place::Ptr in_place_low = prev_stage[2 * rotate_right(i ^ 1)];
			pn_place::Ptr in_place_high = prev_stage[2 * rotate_right(i) + 1];

			pn_place::Ptr out_place_low = net->create_place();
			pn_place::Ptr out_place_high = net->create_place();

			next_stage.push_back(out_place_low);
			next_stage.push_back(out_place_high);

			for(const pn_token::Ptr& token1 : tokens)
			{
				for (const pn_token::Ptr& token2 : tokens)
				{
					// = switch
					net->create_transition(
						{ std::make_pair(in_place_low, token1), std::make_pair(in_place_high, token2) },
						{ std::make_pair(out_place_low, token1), std::make_pair(out_place_high, token2) }
					);

					// x switch
					net->create_transition(
						{ std::make_pair(in_place_low, token1), std::make_pair(in_place_high, token2) },
						{ std::make_pair(out_place_low, token2), std::make_pair(out_place_high, token1) }
					);
				}
			}

		}
		prev_stage = std::move(next_stage);
	}

	// connect to outputs
	for (unsigned int i = 0; i < count_nodes; i++)
	{
		pn_place::Ptr in_place_low = prev_stage[2 * rotate_right(i ^ 1)];
		pn_place::Ptr in_place_high = prev_stage[2 * rotate_right(i) + 1];

		pn_place::Ptr out_place = outputs[i];

		for (const pn_token::Ptr& token : tokens)
		{
			net->create_transition(
				{ std::make_pair(in_place_low, token) },
				{ std::make_pair(out_place, token) }
			);

			net->create_transition(
				{ std::make_pair(in_place_high, token) },
				{ std::make_pair(out_place, token) }
			);
			
		}

	}

	return net;
}


state_observation::pn_net::Ptr pn_factory::pick_and_place(const state_observation::object_parameters& object_params,unsigned int places, unsigned int tokens, unsigned int agents)
{
	state_observation::pn_net::Ptr net = std::make_shared<state_observation::pn_net>(object_params);

	for (unsigned int i = 0; i < agents; i++)
		net->create_place(true);

	for (unsigned int i = 0; i < places; i++)
		net->create_place();

	for (unsigned int t = 0; t < tokens; t++)
	{
		auto token = std::make_shared<pn_token>();

		for (unsigned int p = 0; p < places; p++)
			for (unsigned int a = 0; a < agents; a++)
			{
				net->create_transition(
					{ std::make_pair(net->get_place(agents + p), token) },
					{ std::make_pair(net->get_place(a), token) }
				);

				net->create_transition(
					{ std::make_pair(net->get_place(a), token) },
					{ std::make_pair(net->get_place(agents + p), token) }
				);
			}
	}

	return net;
}
}
