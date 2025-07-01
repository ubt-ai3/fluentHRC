//#include <enact_core/world.hpp>

#include <state_observation/building_estimation.hpp>
#include <simulation/benchmark.hpp>


using namespace state_observation;

//adapted from task.cpp
std::vector<state_observation::pn_transition::Ptr> generate_feasible_actions(const pn_belief_marking::Ptr& marking, const pn_place::Ptr& agent_place)
{
	std::vector<state_observation::pn_transition::Ptr> result;

	const auto& forward_transitions = marking->net.lock()->get_forward_transitions(agent_place);

	for (const pn_transition::Ptr& transition : forward_transitions)
	{

		if (!marking->is_enabled(transition))
			continue;

		if (auto picking = std::dynamic_pointer_cast<pick_action>(transition))
		{
			const auto next_marking = marking->fire(transition);

			bool is_progress = false;
			for (const auto& next : forward_transitions) {
				if (next_marking->is_enabled(next)) {
					is_progress = true;
					break;
				}
			}

			if (!is_progress)
				continue;
		}


		result.push_back(transition);
	}
	//std::cout << "Amount: " << result.size() << std::endl << std::endl << std::endl;
	//if (result.empty())
		//std::cout << "No feasible actions generated" << std::endl;

	return result;
}

/*
 * Examples:
 *
 * 1 right "" "72469,0,1451539403653235655,1,8 1,5 0,6 0,10 1,4 0,3 0,58 9,67 9,39 2,48 3,38 1,41 1,59 9,50 9,42 1,46 1,55 9,37 2,57 9,60 9,62 9,68 9,51 9,53 9,56 9,44 0,63 9,34 6,65 9,66 9,69 9,61 9,64 9,52 9,70 9,72 9,40 3,45 9,71 9,43 1,47 9,49 9,35 7,32 6,33 6,13 1,23 3,14 1,25 4,27 4,28 5,26 4,29 5,17 2,18 2,24 4,30 5,31 5"
 * output: pick_17_2,pick_18_2,pick_26_4,pick_24_4,pick_25_4,pick_28_5,pick_27_4,pick_29_5,pick_31_5,pick_30_5
 * structure:
 *
 *  x xx
 * xxxxx
 * xxxxx
 *
 * "1" "right" "" "15893,0,145930431744552639,1,8 1,9 1,11 1,5 0,6 0,7 0,10 1,4 0,3 0,58 9,67 9,39 9,48 9,38 9,41 9,59 9,50 9,42 9,46 9,55 9,37 9,57 9,60 9,62 9,68 9,51 9,53 9,56 9,44 9,63 9,34 6,65 9,66 9,69 9,61 9,64 9,52 9,70 9,72 9,40 9,45 9,71 9,43 9,47 9,49 9,35 7,32 6,33 6,12 1,13 1,23 3,15 1,21 3,14 1,16 1,22 3,25 4,27 4,28 5,26 4,29 5,17 2,18 2,19 2,24 4,20 2,30 5,31 5"
 * output: pick_11_1,pick_10_1,pick_9_1,pick_8_1,pick_13_1,pick_20_2,pick_12_1,pick_19_2,pick_17_2,pick_16_1,pick_18_2,pick_15_1,pick_14_1,pick_22_3,pick_23_3,pick_21_3
 * structure not started
 */
int main(int argc, char** argv)
{
	const int num_arguments = 4;

	if (argc == 1)
	{
		std::cout << "Usage: app_util.exe task hand grabbed marking\n"
			<< "\ttask\t0,1,2 - 0 is decompose\n"
			<< "\thand\tleft,right\n"
			<< "\tgrabbed\ttoken_id or \"\" if nothing is grabbed\n"
			<< "\tmarking\tone or more lines from marking.csv, format: time,id,hash,probability,list of (placeId space tokenId)\n"
			<< "Output is a comma seperated list of action ids. An action id is (pick|place)_placeID_tokenID\n"
			<< "In case of invalid parameters the exit code indicates which argument is incorrect and a message is printed to cerr" << std::endl;
		exit(0);
	}

	if (argc <= num_arguments)
	{
		std::cerr << "Too few arguments. See usage by calling the program without arguments.";
		exit(argc - 1);
	}

	if (argc > num_arguments + 1) {
		std::cerr << "Too many arguments. Did you properly enclose markings in \"\"?";
		exit(num_arguments);
	}

	bool verbose = false;

	computed_workspace_parameters workspace_params{true};
	object_parameters object_params;
	object_prototype_loader loader;
	enact_core::world_context world;

	std::string task = argv[1];
	std::string hand = argv[2];
	std::string grabbed = argv[3];
	std::string marking_str = argv[4]; //"243593,0,2278688417392361301,1,8 1,9 1,5 0,6 0,4 0,3 0,58 9,67 9,39 2,48 3,38 1,41 1,59 9,50 9,42 1,46 1,55 9,37 2,57 9,60 9,62 9,68 9,51 9,53 9,56 9,44 0,63 9,34 6,65 9,66 9,69 9,61 9,64 9,52 9,70 9,72 9,40 3,45 2,71 9,43 1,47 2,49 9,35 7,32 6,33 6,13 1,23 3,14 1,25 4,27 4,28 5,26 4,29 5,24 4,30 5,31 5";

	//relevant code from task_manager, all buildings must be initilized so that place ids match
	auto net = benchmark::init_net(object_params, 3);
	auto resource_pool = benchmark::init_resource_pool(net, loader);
	auto name_to_token = benchmark::named_tokens(resource_pool);
	auto decomposition_goal = benchmark::decompose(net, resource_pool);
	auto buildings = std::map<std::string, building::Ptr>({
			{ "1", benchmark::flag_denmark(net, name_to_token)},
			{ "2", benchmark::building_2(net, name_to_token)}
		});
	auto initial_marking = benchmark::to_marking(net, resource_pool, buildings);

	std::map<int, pn_place::Ptr> places_by_id;
	for (const auto& p : net->get_places())
		places_by_id.emplace(p->id, p);

	std::map<int, pn_token::Ptr> tokens_by_id;
	for (const auto& t : net->get_tokens())
		tokens_by_id.emplace(t->id, t);

	auto empty_token = pn_empty_token::find_or_create(net);

	// parse parameters
	auto print_error_and_exit = [&argv](int argument_number)
	{
		std::cerr << "Invalid argument " << argument_number << " got: " << argv[argument_number] << std::endl;
		exit(argument_number);
	};

	auto iter = net->get_agent_places().begin()++;

	if(hand.compare("right") == 0)
		iter++;
	else if (hand.compare("left") != 0)
		print_error_and_exit(2);


	auto agent_place = *iter;

	if (task.compare("0") == 0)
		net->set_goal(decomposition_goal.first);
	else {
		auto iter = buildings.find(task);

		if (iter == buildings.end())
			print_error_and_exit(1);

		net->set_goal(iter->second->get_goal().first);
	}

	pn_token::Ptr grabbed_token;
	try
	{
		if(!grabbed.empty())
			grabbed_token = tokens_by_id.at(std::stoi(grabbed));
	}
	catch (...)
	{
		print_error_and_exit(3);
	}

	//process marking
	auto print_marking_error_and_exit = [&](const std::exception& e, std::string substring)
	{
		std::cerr << "Error while parsing \"" << substring << "\" of marking: " << e.what() << std::endl;
		exit(num_arguments);
	};

	std::stringstream markings_ss(marking_str);
	std::string line;
	pn_belief_marking::marking_dist_t marking_dist;
	double sum_probability = 0;
	while (getline(markings_ss, line))
	{
		try {

			std::stringstream marking_ss(line);
			std::string entry;

			getline(marking_ss, entry, ','); // int time;
			getline(marking_ss, entry, ','); // int id;
			getline(marking_ss, entry, ',');  //std::string hash;
			getline(marking_ss, entry, ',');  //float probability;

			double probability = std::stod(entry);
			sum_probability += probability;
			std::map<pn_place::Ptr, pn_token::Ptr> token_distribution;

			for (const auto& instance : initial_marking->distribution)
				if (std::dynamic_pointer_cast<pn_empty_token>(instance.second))
					token_distribution.emplace(instance);

			while (getline(marking_ss, entry, ','))
			{
				try {
					std::string place_str;
					std::string token_str;
					std::stringstream ss_entry(entry);

					getline(ss_entry, place_str, ' ');
					getline(ss_entry, token_str, ' ');
					auto place = places_by_id.at(std::stoi(place_str));
					auto token = tokens_by_id.at(std::stoi(token_str));
					token_distribution.insert_or_assign(place, token);
				}
				catch (const std::exception& e)
				{
					print_marking_error_and_exit(e, entry);
				}
			}

			if (grabbed_token)
				token_distribution.insert_or_assign(agent_place, grabbed_token);
			else
				token_distribution.erase(agent_place);

			auto binary_marking = std::make_shared<pn_binary_marking>(net, std::set<pn_instance>(token_distribution.begin(), token_distribution.end()));
			marking_dist.insert_or_assign(binary_marking, probability);

		}
		catch (const std::exception& e)
		{
			print_marking_error_and_exit(e, line);
		}
	}

	if (std::abs(sum_probability - 1) > 0.01) {
		std::cerr << "Probabilities do not sum to 1 but " << sum_probability << std::endl;
		exit(num_arguments);
	}

	auto marking = std::make_shared<pn_belief_marking>(net, std::move(marking_dist));
	auto actions = generate_feasible_actions(marking, agent_place);

	pn_boxed_place::Ptr destination;
	std::stringstream output;
	auto print_pos = [](const pn_boxed_place::Ptr& p)
	{
		auto vec = p->box.translation;
		std::cout << "\t\tplace " << p->id << " " << std::setprecision(3) << std::setfill('0') << "{ " << vec.x() << ", " << vec.y() << ", " << vec.z() << " }" << std::endl;
	};

	bool first = true;
	for (const auto& a : actions) {
		if (!first)
			output << ",";
		first = false;

		if (const auto dynPIAction = std::dynamic_pointer_cast<pick_action>(a))
		{
			destination = dynPIAction->from;

			if (verbose) {
				std::cout << "Picking " << dynPIAction->token->object->get_name() << " from ";
				print_pos(destination);
			}
			else
				output << "pick_" << destination->id << "_" << dynPIAction->token->id;
		}
		else if (const auto dynPAction = std::dynamic_pointer_cast<place_action>(a); dynPAction)
		{
			destination = dynPAction->to;
			if (verbose) {
				std::cout << "Placing " << dynPAction->token->object->get_name() << " at ";
				print_pos(destination);
			}
			else
				output << "place_" << destination->id << "_" << dynPAction->token->id;
		}
		else if (const auto dynSAction = std::dynamic_pointer_cast<stack_action>(a); dynSAction)
		{
			destination = dynSAction->to.first;
			if (verbose) {
				std::cout << "Stacking " << dynSAction->top_object->get_name() << " at ";
				print_pos(destination);
			}
			else
				output << "place_" << destination->id << "_" << dynSAction->to.second->id;
		}
		else if (const auto dynRSAction = std::dynamic_pointer_cast<reverse_stack_action>(a); dynRSAction)
		{
			destination = dynRSAction->from.first;
			if (verbose) {
				std::cout << "Picking " << dynRSAction->from.second->object->get_name() << " from stack at ";
				print_pos(destination);
			}
			else
				output << "pick_" << destination->id << "_" << dynRSAction->from.second->id;
		}

	}

	std::cout << output.str();
}