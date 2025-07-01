#include "observed_agent.hpp"

#include <filesystem>
#include <fstream>
#include <ranges>

#include <boost/date_time/posix_time/posix_time_types.hpp>
#include <boost/date_time/posix_time/ptime.hpp>

#pragma warning( push )
#pragma warning( disable : 4996 )
#include <caffe/util/upgrade_proto.hpp>
#include <caffe/layers/memory_data_layer.hpp>
#pragma warning( pop )

#include <google/protobuf/text_format.h>

#include <Eigen/src/Geometry/AngleAxis.h>
#include <pcl/common/impl/accumulators.hpp>


#include "enact_core/access.hpp"
#include "enact_core/data.hpp"
#include "enact_core/lock.hpp"

#include "state_observation/pn_model_extension.hpp"

namespace prediction
{

using namespace hand_pose_estimation;
using namespace state_observation;
using namespace std::chrono;

float transition_context::bell_curve(float x, float stdev)
{
	return std::expf(-x * x / (2 * stdev * stdev));
}



std::vector<float> transition_context::get_neighbors(const pn_belief_marking& marking,
	const pn_boxed_place::Ptr& place,
	const std::set<pn_place::Ptr>& excluded_places)
{
	std::vector<float> neighbors(27, 0.f);
	float diag = place->box.diagonal.norm();

	const auto net = marking.net.lock();
	for (const auto& other : net->get_places())
	{
		if (excluded_places.contains(other))
			continue;

		auto boxed_other = std::dynamic_pointer_cast<pn_boxed_place>(other);

		if (!boxed_other)
			continue;

		if (marking.get_summed_probability(other) <= 0.5f)
			continue;

		Eigen::Vector3f vec = place->box.translation - boxed_other->box.translation;

		float z_dir = vec.dot(Eigen::Vector3f::UnitZ());
		float proximity = bell_curve(vec.norm(), neighbor_distance_threshold);
		if (std::abs(z_dir) >= M_PI / 8.f)
		{
			size_t index = z_dir > 0 ? 0 : 18;
			neighbors[index] = std::max(neighbors[index], proximity);
			continue;
		}

		int row = 1 - static_cast<int>(std::roundf(z_dir * M_PI_4));

		float plane_angle = std::atan2(vec.y(), vec.x());
		if (plane_angle < -M_PI / 8.f)
			plane_angle += M_2_PI;

		int cell = 1 + static_cast<int>(std::roundf(plane_angle * M_PI_4));
		size_t index = row * 9 + cell;
		neighbors[index] = std::max(neighbors[index], proximity);

	}

	return neighbors;
}

float transition_context::compare_neighbors(const std::vector<float>& lhs, const std::vector<float>& rhs)
{
	double error = 0;

	float sum = 0;
	for (int i = 0; i < std::min(lhs.size(), rhs.size()); i++)
		error += std::abs(lhs[i] - rhs[i]);

	return bell_curve(error, 1.f);
}

transition_context::transition_context(const transition_context& other)
	:
	workspace_params(other.workspace_params),
	transition(other.transition),
	neighbors(other.neighbors),
	center(other.center),
	center_xy(other.center_xy),
	box(other.box),
	color(other.color),
	volume(other.volume),
	action_type(other.action_type),
	timestamp(other.timestamp)
{}

transition_context::transition_context(transition_context&& other)
	:
	workspace_params(other.workspace_params),
	transition(std::move(other.transition)),
	neighbors(std::move(other.neighbors)),
	center(std::move(other.center)),
	center_xy(std::move(other.center_xy)),
	box(std::move(other.box)),
	color(std::move(other.color)),
	volume(other.volume),
	action_type(other.action_type),
	timestamp(other.timestamp)
{}

transition_context::transition_context(const computed_workspace_parameters& workspace_params,
	const state_observation::pn_transition::Ptr& transition,
	const state_observation::pn_belief_marking& marking,
	const state_observation::pn_place::Ptr& hand,
	std::chrono::duration<float> timestamp)
	:
	workspace_params(workspace_params),
	transition(transition),
	center(std::numeric_limits<float>::quiet_NaN()* Eigen::Vector3f::Ones()),
	center_xy(std::numeric_limits<float>::quiet_NaN()* Eigen::Vector2f::Ones()),
	color(0),
	volume(0),
	action_type(0),
	timestamp(timestamp)
{
	pn_boxed_place::Ptr center_place = nullptr;
	std::set<pn_place::Ptr> involved_places;

	pn_object_token::ConstPtr token;

	for (const pn_instance& input : transition->inputs)
	{
		involved_places.emplace(input.first);

		if (!transition->is_side_condition(input))
		{
			if (input.first == hand)
				action_type -= 1;

			auto boxed_place = std::dynamic_pointer_cast<pn_boxed_place>(input.first);

			if (boxed_place)
			{
				center_place = boxed_place;
				token = std::dynamic_pointer_cast<pn_object_token>(input.second);
			}
		}
	}

	for (const pn_instance& output : transition->outputs)
	{
		involved_places.emplace(output.first);

		if (!transition->is_side_condition(output))
		{
			if (output.first == hand)
				action_type += 1;

			if (auto boxed_place = std::dynamic_pointer_cast<pn_boxed_place>(output.first))
			{
				center_place = boxed_place;
				token = std::dynamic_pointer_cast<pn_object_token>(output.second);
			}
		}
	}

	if (!center_place)
		return;

	if (token)
	{
		volume = token->object->get_bounding_box().diagonal.prod();
		const auto& col = token->object->get_mean_color();

		color = (0.299f * col.r + 0.587f * col.g + 0.114f * col.b) / 255.f;
	}

	center = center_place->box.translation;
	center_xy = center.head(2);
	box = center_place->box;
	neighbors = get_neighbors(marking, center_place, involved_places);
}


transition_context& transition_context::operator=(const transition_context& other)
{
	transition = other.transition;
	neighbors = other.neighbors;
	center = other.center;
	center_xy = other.center_xy;
	box = other.box;
	color = other.color;
	volume = other.volume;
	timestamp = other.timestamp;
	action_type = other.action_type;

	return *this;
}

float transition_context::compare(const transition_context& other) const
{
	if (other.transition == transition)
		return 1.f;

	return .125f * bell_curve(other.color - color, color_similarity) +
		.125f * bell_curve(other.volume / volume, volume_similarity) +
		.375f * bell_curve((other.center - center).norm(), neighbor_distance_threshold) +
		.375f * compare_neighbors(other.neighbors, neighbors);
}

std::vector<float> transition_context::to_feature_vector() const
{
	std::vector<float> result;
	result.reserve(feature_vector_size);
	//entries 0 - 26
	std::ranges::copy(neighbors, std::back_inserter(result));

	// 27
	result.push_back(color);

	Eigen::Vector3f n_center = (center - workspace_params.crop_box_min).cwiseQuotient(workspace_params.crop_box_max - workspace_params.crop_box_min);

	// 28 - 30
	result.push_back(n_center.x());
	result.push_back(n_center.y());
	result.push_back(n_center.z());

	// 31
	result.push_back(volume / std::powf(workspace_params.max_object_dimension, 3.f));

	for (float& entry : result)
		entry = 2.f * entry - 1.f;

	// 32
	result.push_back(action_type);

	return result;
}


std::mutex net_predictor::net_read_mutex = std::mutex();

net_predictor::net_predictor(const computed_workspace_parameters& workspace_params,
	const std::string& path,
	std::chrono::high_resolution_clock::time_point start_time)
	:
	workspace_params(workspace_params),
	epoch(max_epoch),
	discriminative_features(transition_context::feature_vector_size, 0.f),
	path(path)
{
	//Workaround for simulations with multiple agents
	while (true)
	{
		try
		{
			testnet.reset(new caffe::Net<float>(readParams(net_path, caffe::TEST)));
			break;
		}
		catch (const std::exception& e)
		{
			std::cerr << e.what() << "\t" << net_path << std::endl;
			std::this_thread::sleep_for(std::chrono::milliseconds(500));
		}
	}

	std::filesystem::create_directories(path);

	file.open(path + "vector_lengths.csv");
	file << "time (ms),data_history,indices_history,data_candidates,indices_candidates,labels" << std::endl;

	schedule([this]() { init_weights(); });
}

net_predictor::~net_predictor()
{
	stop_thread();
}

caffe::NetParameter net_predictor::readParams(const std::string& param_file,
	caffe::Phase phase, const int level,
	const std::vector<std::string>* stages)
{
	std::filesystem::path filePath = param_file;
	if (!std::filesystem::exists(filePath))
		throw std::exception("[Net]: Param file not found");

	std::stringstream buffer;
	{
		auto file = std::ifstream(filePath);
		if (!file)
			throw std::exception("[Net]: File not open");
		buffer << file.rdbuf();
	}
	caffe::NetParameter param;

	if (!google::protobuf::TextFormat::ParseFromString(buffer.str(), &param))
		throw std::exception("[Net]: Invalid params file");
	caffe::UpgradeNetAsNeeded(param_file, &param);

	// Set phase, stages and level
	param.mutable_state()->set_phase(phase);
	if (stages != nullptr)
	{
		for (const auto& stage : *stages)
		{
			param.mutable_state()->add_stage(stage);
		}
	}
	param.mutable_state()->set_level(level);

	return param;
}

void net_predictor::init_weights()
{
	//caffe::Caffe::set_mode(caffe::Caffe::GPU);

	//caffe::Caffe::SetDevice(0);
	//std::cout << "Caffe runs on (0 = CPU, 1 = GPU): " << caffe::Caffe::mode() << " memory location: " << (size_t)&caffe::Caffe::Get() << std::endl;

	{
		std::lock_guard<std::mutex> lock(net_read_mutex);

		// file paths in solver.prototxt are relative to working directory
		//auto current_path = boost::filesystem::current_path();

		auto full_solver_file_path = std::filesystem::current_path() / std::filesystem::path{solver_path};
//		auto solver_file_name = full_solver_file_path.filename();
	//	auto full_solver_path = full_solver_file_path.remove_filename();
		//boost::filesystem::current_path(full_solver_path);


		caffe::SolverParameter solver_param;
		caffe::ReadSolverParamsFromTextFileOrDie(full_solver_file_path.string(), &solver_param);

		while (true)
			try
		{
			solver_param.set_net((full_solver_file_path.parent_path() / solver_param.net()).string());
			solver = std::unique_ptr<caffe::Solver<float>>(caffe::SolverRegistry<float>::CreateSolver(solver_param));
			break;
		}
		catch (const std::exception& e)
		{
			std::cerr << e.what();
		}

		//boost::filesystem::current_path(current_path);
	}

	{
		auto& blob = solver->net()->
			layer_by_name("history_weights")->blobs()[0];

		auto params = blob->mutable_cpu_data();
		params[0] = 0.1f;
		params[1] = 0.5f;
		params[2] = 0.1f;
		params[3] = 0.3f;
	}

	{
		auto& blob = solver->net()->
			layer_by_name("combine_neighbors")->blobs()[0];

		std::vector<float> means({ 5, 11, 15,21 });
		const float stdev = 1.f;
		const float variance = stdev * stdev;

		auto params = blob->mutable_cpu_data();
		for (int row = 0; row < blob->shape(0); row++)
			for (int col = 0; col < blob->shape(1); col++)
			{
				params[row * blob->shape(1) + col] = 1 / std::sqrt(2 * M_PI * variance) *
					std::exp(-(col - means[row]) * (col - means[row]) / (2 * variance));
			}
	}
}

void net_predictor::log_shape(const caffe::Net<float>& net)
{
	const auto& top_vecs_ = net.top_vecs();

	for (int layer_id = 0; layer_id < net.layers().size(); ++layer_id)
	{
		LOG_IF(INFO, caffe::Caffe::root_solver())
			<< "Layer " << net.layer_names()[layer_id];

		for (int top_id = 0; top_id < top_vecs_[layer_id].size(); ++top_id)
		{
			LOG_IF(INFO, caffe::Caffe::root_solver())
				<< "Top shape: " << top_vecs_[layer_id][top_id]->shape_string();
			if (net.layers()[layer_id]->loss(top_id))
			{
				LOG_IF(INFO, caffe::Caffe::root_solver())
					<< "    with loss weight " << net.layers()[layer_id]->loss(top_id);
			}
		}
	}
}

void net_predictor::print_weights()
{
	const auto& net = *testnet;
	auto precision = std::cout.precision();

	std::cout << std::setprecision(23);

	for (size_t index = 0; index < net.layers().size(); ++index)
	{
		const auto& layer = net.layers()[index];

		for (size_t b_index = 0; b_index < layer->blobs().size(); ++b_index)
		{
			std::cout << "m_" << net.layer_names()[index];

			if (b_index > 0)
				std::cout << "_bias";

			std::cout << " = np.array([";

			const auto& blob = *layer->blobs()[b_index];
			bool print_comma = false;

			for (auto iter = blob.cpu_data(); iter != blob.cpu_data() + blob.count(); ++iter)
			{
				if (print_comma)
					std::cout << ", ";
				else
					print_comma = true;
				std::cout << *iter;
			}

			std::cout << "])";
			if (blob.shape().size() >= 2)
				std::cout << ".reshape(" << blob.shape()[0] << ", " << blob.shape()[1] << ")";

			std::cout << std::endl;
		}
	}

	std::cout.precision(precision);
}

void net_predictor::enforce_discriminative_features()
{
	auto apply_weights = [&](caffe::Blob<float>& blob, const std::vector<float>& weights)
	{
		auto params = blob.mutable_cpu_data();
		for (int row = 0; row < blob.shape(0); row++)
		{
			float oldSum = 0.f;
			float newSum = 0.f;
			for (int col = 0; col < blob.shape(1); col++)
			{
				float& param = params[row * blob.shape(1) + col];
				oldSum += param;
				param *= weights[col];
				newSum += param;
			}

			for (int col = 0; col < blob.shape(1); col++)
				params[row * blob.shape(1) + col] *= 1.f / newSum;
		}
	};

	{
		constexpr float factor = 1.125f;
		auto& blob = *solver->net()->
		                      layer_by_name("combine_neighbors")->blobs()[0];

		std::vector<float> weights(discriminative_features.begin(), discriminative_features.begin() + 27);
		//size_t neighbor_size = blob.shape(0);

		for (float& w : weights)
			w = std::abs(w) > epsilon ? factor : 1.f / factor;

		apply_weights(blob, weights);
	}

	//{
	//	auto& blob = *solver->net()->
	//		layer_by_name("fc1").get()->blobs()[0];

	//	std::vector<float> weights(discriminative_features.begin() + 27 - neighbor_size, discriminative_features.end());


	//	for (float& w : weights)
	//		w = std::abs(w) > epsilon ? factor : 1.f / factor;

	//	for (size_t i = 0; i < neighbor_size; i++)
	//		weights[i] = factor;

	//	apply_weights(blob, weights);
	//}
}

void net_predictor::train()
{
	if (!solver)
	{
		std::cerr << "Solver not ready yet" << std::endl;
		return;
	}

	const size_t dim = transition_context::feature_vector_size;

	if (data_history.size() < dim * (time_horizon + 1))
		return;

	auto& net = *solver->net();
	auto reshape = [&](const std::string& name, std::vector<float>& container, size_t batch_size)
	{
		//auto& blob = *net.blob_by_name(name);
		//auto shape = blob.shape();
		//shape[0] = batch_size;
		//net.blob_by_name(name)->Reshape(std::move(shape));

		const auto layer = std::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(net.layer_by_name(name));
		layer->set_batch_size(batch_size);
		layer->Reset(container.data(), container.data(), batch_size);
	};

	reshape("data_history", data_history, data_history.size() / dim);
	reshape("data_candidates", data_candidates, data_candidates.size() / dim);
	reshape("indices_history", indices_history, indices_history.size());
	reshape("indices_candidates", indices_candidates, indices_candidates.size());
	reshape("labels", labels, labels.size());

	net.Reshape();
	//log_shape(net);

	auto print_vec = [dim](const std::vector<float>& vec)
	{
		for (const float& entry : vec)
			std::cout << entry << " ";
	};

	size_t candidates_begin = *(indices_candidates.cend() - 5);

	//std::cout << "!before training: ";
	//print_vec(predict(&*(data_history.end() - (time_horizon + 1) * dim), &data_candidates[candidates_begin * dim], data_candidates.size() / dim - candidates_begin));
	//std::cout << std::endl;

	//enforce_discriminative_features();
	//std::cout << "Caffe runs on (0 = CPU, 1 = GPU): " << caffe::Caffe::mode() << " memory location: " << (size_t)&caffe::Caffe::Get() << std::endl;
	solver->Step(solver->param().max_iter());

	{
		std::lock_guard<std::mutex> lock(testnet_mutex);
		testnet->ShareTrainedLayersWith(&*solver->net());
	}

	//if(epoch == 2)
	//	print_weights();

	//std::cout << "training finished, epoch " << epoch << ", total history " << data_history.size() / dim << std::endl;

	//std::cout << "!after training: ";
	//print_vec(predict(&*(data_history.end() - (time_horizon + 1) * dim), &data_candidates[candidates_begin * dim], data_candidates.size() / dim - candidates_begin));
	//std::cout << std::endl;

	if (++epoch < max_epoch)
	{
		//schedule([this]() {train(); });
		train();
	}
}

/*
void net_predictor::finished()
{
	schedule([this]()
	{
		int start = 0;
		const size_t dim = transition_context::feature_vector_size;
		while (true)
		{
			int end = start;
			float pos_candidate = indices_candidates[start];

			while (end < indices_candidates.size() && indices_candidates[end] == pos_candidate)
				end += 2 * time_horizon;

			auto vec = predict(&data_history[dim * indices_history[start]], &data_candidates[dim * indices_candidates[start]], indices_candidates[end - 1] + 1 - indices_candidates[start]);

			std::cout << "!completed training: ";
			for (const float& entry : vec)
				std::cout << entry << " ";
			std::cout << std::endl;

			if (end >= indices_candidates.size())
				break;

			start = end;
		}

		auto print_vec = [&](const std::vector<float>& vec, size_t second_dimension = 0)
		{
			std::cout << " = np.array([";

			bool print_comma = false;

			for (const auto& entry : vec)
			{
				if (print_comma)
					std::cout << ", ";
				else
					print_comma = true;
				std::cout << entry;
			}

			std::cout << "])";
			if (second_dimension)
				std::cout << ".reshape(-1," << second_dimension << ")";

			std::cout << std::endl;
		};

		auto precision = std::cout.precision();

		std::cout << std::setprecision(23);
		std::cout << "data_history"; print_vec(data_history, 33);
		std::cout << "data_candidates"; print_vec(data_candidates, 33);
		std::cout << "indices_history"; print_vec(indices_history);
		std::cout << "indices_candidates"; print_vec(indices_candidates);
		std::cout << "labels"; print_vec(labels);
		std::cout.precision(precision);
	});
}
*/

void net_predictor::finished()
{

	std::filesystem::create_directories(path);

	try{
	{
		try
		{
			std::lock_guard<std::mutex> lock(testnet_mutex);
			testnet->ToHDF5(path + "weights.hdf5");
		}
		catch (...) {}
	}

	auto log_vec = [&](const std::vector<float>& vec, const std::string& path)
	{
		try
		{
			std::ofstream file(path, std::ios::out | std::ios::binary);
			file.write(reinterpret_cast<const char*>(vec.data()), vec.size() * sizeof(float));
		}
		catch (...) {}
	};

	std::lock_guard<std::mutex> lock(local_data_mutex);


	log_vec(data_history, path + "data_history.bin");
	log_vec(data_history, path + "indices_history.bin");
	log_vec(data_history, path + "data_candidates.bin");
	log_vec(data_history, path + "indices_candidates.bin");
	log_vec(data_history, path + "labels.bin");

	file.close();
	}
	catch (...) {}
}

net_predictor::transition_probability net_predictor::predict(const std::vector<transition_context>& candidates)
{
	const size_t dim = transition_context::feature_vector_size;

	if (data_history.size() < time_horizon * dim)
		throw std::exception("Input to neural network for action prediction has invalid size");

	std::vector<float> data_candidates;
	data_candidates.reserve(candidates.size() * dim);
	for (const auto& candidate : candidates)
	{
		const auto& vec = candidate.to_feature_vector();
		std::ranges::copy(vec, std::back_inserter(data_candidates));
	}

	std::unique_lock<std::mutex> lock(local_data_mutex);
	auto probabilities = predict(&*(data_history.end() - time_horizon * dim), data_candidates.data(), candidates.size());
	lock.release();

	transition_probability result;
	size_t i = 0;
	for (const auto& candidate : candidates)
	{
		result.emplace(candidate.transition, probabilities.at(i));
		i++;
	}

	return result;
}

net_predictor::transition_probability net_predictor::predict(const std::vector<transition_context>& history, const std::vector<transition_context>& candidates)
{
	const size_t dim = transition_context::feature_vector_size;

	if (history.size() < time_horizon)
		throw std::exception("Input to neural network for action prediction has invalid size");

	std::vector<float> data_candidates;
	data_candidates.reserve(candidates.size() * dim);
	for (const auto& candidate : candidates)
	{
		const auto& vec = candidate.to_feature_vector();
		std::ranges::copy(vec, std::back_inserter(data_candidates));
	}

	std::vector<float> data_history;
	data_history.reserve(history.size() * dim);
	for (const auto& hist : history)
	{
		const auto& vec = hist.to_feature_vector();
		std::ranges::copy(vec, std::back_inserter(data_history));
	}

	auto probabilities = predict(data_history.data(), data_candidates.data(), candidates.size());

	transition_probability result;
	size_t i = 0;
	for (const auto& candidate : candidates)
	{
		result.emplace(candidate.transition, probabilities.at(i));
		i++;
	}

	return result;
}

std::vector<float> net_predictor::predict(const float* data_history, const float* data_candidates, size_t count_candidates)
{
	const size_t dim = transition_context::feature_vector_size;

	std::vector<float> indices_history;
	std::vector<float> indices_candidates;

	indices_history.reserve(count_candidates * time_horizon);
	indices_candidates.reserve(count_candidates * time_horizon);

	for (size_t i = 0; i < count_candidates; i++)
	{
		indices_history.push_back(0);
		indices_history.push_back(1);
		indices_history.push_back(2);
		indices_history.push_back(3);

		indices_candidates.push_back(i);
		indices_candidates.push_back(i);
		indices_candidates.push_back(i);
		indices_candidates.push_back(i);
	}

	std::unique_lock<std::mutex> lock(testnet_mutex);
	auto& net = *testnet;
	auto reshape = [&](const std::string& name, const float* container, size_t batch_size)
	{
		const auto layer = std::dynamic_pointer_cast<caffe::MemoryDataLayer<float>>(net.layer_by_name(name));
		layer->set_batch_size(batch_size);
		layer->Reset(const_cast<float*>(container), const_cast<float*>(container), batch_size);
	};

	reshape("data_history", data_history, time_horizon);
	reshape("data_candidates", data_candidates, count_candidates);
	reshape("indices_history", indices_history.data(), indices_history.size());
	reshape("indices_candidates", indices_candidates.data(), indices_candidates.size());

	net.Reshape();
	//log_shape(net);

	net.Forward();

	auto print_output = [&](const std::string& layer_name)
	{
		const auto& output = net.blob_by_name(layer_name);

		if (output)
		{
			std::cout << layer_name << ": ";
			for (int i = 0; i < output->count(); i++)
				std::cout << output->mutable_cpu_data()[i] << " ";
			std::cout << std::endl;
		}
	};

	const auto& output_blob = net.blob_by_name("output");
	return { output_blob->mutable_cpu_data(), output_blob->mutable_cpu_data() + output_blob->count() };
}

void net_predictor::add_data(const transition_context& executed, const std::vector<transition_context>& non_executed)
{
	//schedule([this, executed(executed), non_executed(non_executed)]()
	//{
	const size_t dim = transition_context::feature_vector_size;


	const auto& executed_vec = executed.to_feature_vector();
	std::ranges::copy(executed_vec, std::back_inserter(data_history));

	size_t h_len = data_history.size() / dim;

	if (non_executed.empty() || h_len <= time_horizon) // do not add training instance, if we have no negative examples
		return;

	std::vector<float> indices_history_new;
	for (size_t i = h_len - time_horizon - 1; i < h_len - 1; i++)
		indices_history_new.push_back(static_cast<float>(i));

	std::lock_guard<std::mutex> lock(local_data_mutex);

	size_t count_new_data = non_executed.size() * 2;
	data_candidates.reserve(data_candidates.size() + (non_executed.size() + 1) * dim);
	indices_candidates.reserve(indices_candidates.size() + count_new_data * time_horizon);
	indices_history.reserve(indices_history.size() + count_new_data * time_horizon);
	labels.reserve(labels.size() + count_new_data);

	size_t executed_candidate_index = data_candidates.size() / dim;
	std::ranges::copy(executed_vec, std::back_inserter(data_candidates));

	if (file.is_open())
		file << duration_cast<milliseconds>(high_resolution_clock::now() - start_time).count()
			<< "," << data_history.size()
			<< "," << indices_history.size()
			<< "," << data_candidates.size()
			<< "," << indices_candidates.size()
			<< "," << labels.size()
			<< std::endl;

	size_t non_executed_candidate_index = data_candidates.size() / dim;
	for (const auto& non_exec : non_executed)
	{
		for (int i = 0; i < time_horizon; i++)
			indices_candidates.push_back(executed_candidate_index);
		std::ranges::copy(indices_history_new, std::back_inserter(indices_history));
		labels.push_back(1);

		const auto& vec = non_exec.to_feature_vector();
		std::ranges::copy(vec, std::back_inserter(data_candidates));

		for (int i = 0; i < time_horizon; i++)
			indices_candidates.push_back(non_executed_candidate_index);
		std::ranges::copy(indices_history_new, std::back_inserter(indices_history));
		labels.push_back(-1);

		for (size_t i = 0; i < discriminative_features.size(); i++)
		{
			discriminative_features[i] += (data_candidates[dim * executed_candidate_index + i] - data_candidates[dim * non_executed_candidate_index + i]) / non_executed.size();
		}

		non_executed_candidate_index++;
	}

	if (!data_candidates.empty())
	{
		if (epoch >= max_epoch)
			//schedule([this]() {train(); });
			train();

		epoch = 0;
	}
	return;

	// for debugging
#ifdef DEBUG_PN_ID
	std::cout << "action: ";
	for (const auto& instance : executed.transition->inputs)
		if (!executed.transition->is_side_condition(instance))
		{
			std::cout << "(" << instance.first->id << ", " << instance.second->id << ")";
			break;
		}

	std::cout << " -> ";

	for (const auto& instance : executed.transition->outputs)
		if (!executed.transition->is_side_condition(instance))
		{
			std::cout << "(" << instance.first->id << ", " << instance.second->id << ") ";
			break;
		}
	std::cout << std::endl;
#endif
	train();
	//});
}

void net_predictor::replace_last(const transition_context& executed)
{
	const size_t dim = transition_context::feature_vector_size;

	if (data_history.size() < dim)
		return;

	const auto& executed_vec = executed.to_feature_vector();
	std::ranges::copy(executed_vec, data_history.end() - dim);
}

void observed_agent::normalize(transition_probability& distribution)
{
	double sum = 0.f;
	for (const auto& probability : distribution | std::views::values)
		sum += probability;

	if (sum < min_probability)
		return;

	for (auto& probability : distribution | std::views::values)
		probability /= sum;
}

observed_agent::observed_agent(enact_core::world_context& world,
	const pn_net::Ptr& net,
	const state_observation::computed_workspace_parameters& workspace_params,
	entity_id tracked_hand,
	state_observation::pn_place::Ptr model_hand,
	std::chrono::high_resolution_clock::time_point start_time)
	:
	net(net),
	tracked_hand(std::move(tracked_hand)),
	model_hand(std::move(model_hand)),
	world(world),
	workspace_params(workspace_params),
	front(0.f, 1.f, 0.f),
	start_time(start_time)
{
	finished(start_time);

	get_predictor();
}

#ifdef DEBUG_PN_ID
observed_agent::~observed_agent()
{
	std::ofstream file (path + "actions.csv");
	file << "timestamp,action type,object,place id" << std::endl;

	for (const auto& entry : executed_actions)
		file << duration_cast<milliseconds>(entry.timestamp).count() << "," << entry.transition->to_string();
}
#endif



void observed_agent::update_action_candidates(const std::set<pn_place::Ptr>& occluded_places)
{
	enact_core::lock l(world, enact_core::lock_request(tracked_hand, hand_trajectory::aspect_id, enact_core::lock_request::read));
	const enact_core::const_access<hand_trajectory_data> access_object(l.at(tracked_hand, hand_trajectory::aspect_id));
	auto& obj = access_object->payload;

	auto iter = obj.poses.rbegin();
	if (iter == obj.poses.rend() || iter->first <= latest_update)
		return;

	std::chrono::duration<float> current_timestamp = iter->first;
	//front = iter->second.wrist_pose.matrix().col(1).head(3);

	std::unique_lock<std::mutex> lock(net->mutex);
	auto places = net->get_places();
	lock.unlock();

	do
	{
		for (auto place : places)
		{
			auto boxed_place = std::dynamic_pointer_cast<pn_boxed_place>(place);

			if (boxed_place)
			{
				float distance = get_distance(iter->second, *boxed_place);
				auto action_iter = proxy_places_and_timestamp.find(boxed_place);

				if (distance > boxed_place->box.diagonal.norm() + max_hand_distance)
				{
					if (action_iter != proxy_places_and_timestamp.end() &&
						occluded_places.contains(place))
						action_iter->second = std::max(action_iter->second, current_timestamp);

					continue;
				}


				if (action_iter == proxy_places_and_timestamp.end())
				{
					//std::cout << "proxy add " << boxed_place->id << " " << current_timestamp << std::endl;
					proxy_places_and_timestamp.emplace(boxed_place, current_timestamp);
				}
				else
					action_iter->second = std::max(action_iter->second, current_timestamp);

			}
		}

	} while (++iter != obj.poses.rend() && iter->first > latest_update);

	latest_update = current_timestamp;
	certainty = obj.certainty_score;
}

std::set<pn_transition::Ptr> observed_agent::get_action_candidates()
{
	std::set<pn_transition::Ptr> result;
	for (auto iter = proxy_places_and_timestamp.begin(); iter != proxy_places_and_timestamp.end(); )
	{
		if (iter->second < latest_update - proximity_forget_duration)
		{
			//std::cout << "proxy remove " << iter->first->id << " " << latest_update << std::endl;
			iter = proxy_places_and_timestamp.erase(iter);
			continue;
		}

		// add transitions that go from @param{model_hand} to iter->first
		for (const auto& w_action : iter->first->get_incoming_transitions())
		{
			auto action = w_action.lock();

			for (const auto& place : action->get_inputs())
				if (place == model_hand)
				{
					result.emplace(action);
					break;
				}
		}

		// add transitions that go from iter->first to @param{model_hand}
		for (const auto& w_action : iter->first->get_outgoing_transitions())
		{
			auto action = w_action.lock();

			for (const auto& place : action->get_outputs())
				if (place == model_hand)
				{
					result.emplace(action);
					break;
				}
		}

		++iter;
	}

	return result;
}

std::vector<transition_context> observed_agent::get_executable_actions(
	const state_observation::pn_belief_marking& marking)
{
	const auto net = marking.net.lock();
	if (goal != net->get_goal())
	{
		forward_transitions = net->get_forward_transitions(model_hand);
		goal = net->get_goal();
	}

	std::vector< transition_context> candidates;
	for (const auto& w_t : model_hand->get_incoming_transitions())
	{
		auto t = w_t.lock();
		if (forward_transitions.contains(t) && marking.is_enabled(t) > enable_threshold)
			candidates.emplace_back(workspace_params, t, marking, model_hand);
	}

	for (const auto& w_t : model_hand->get_outgoing_transitions())
	{
		auto t = w_t.lock();
		if (forward_transitions.contains(t) && marking.is_enabled(t) > enable_threshold)
			candidates.emplace_back(workspace_params, t, marking, model_hand);
	}

	return candidates;
}

const std::vector<transition_context>& observed_agent::get_executed_actions() const
{
	return executed_actions;
}

bool observed_agent::has_object_grabbed() const noexcept
{
	if(executed_actions.empty())
		return false;

	return get_postcondition(*executed_actions.back().transition) != nullptr;
}

bool observed_agent::is_forward_transition(const pn_transition::Ptr& action) const
{
	const auto goal_conditions = net->get_goal_instances();

	for (const auto& arc : action->inputs)
	{
		if (action->is_side_condition(arc))
			continue;

		if (goal_conditions.contains(arc))
			return false;
	}

	for (const auto& arc : action->outputs)
	{
		if (action->is_side_condition(arc))
			continue;

		if (goal_conditions.contains(arc))
			return true;
	}

	return std::dynamic_pointer_cast<pick_action>(action) || std::dynamic_pointer_cast<reverse_stack_action>(action);
}

float observed_agent::get_distance(const hand_pose_18DoF& pose, const pn_boxed_place& place) const
{
	return (place.box.translation - pose.get_centroid()).norm();
}

void observed_agent::add_transition(const pn_transition::Ptr& transition,
	const state_observation::pn_belief_marking& prev_marking,
	std::chrono::duration<float> timestamp)
{


	auto empty_token = pn_empty_token::find_or_create(prev_marking.net.lock());

	//if (!executed_actions.empty() && get_postcondition(*executed_actions.back().transition) != get_precondition(*transition))
	//{
	//	std::cout << "Discontinouse action history for agent " << model_hand->id << std::endl;
	//	std::cout << "prev: " << executed_actions.back().transition->to_string() << std::endl;
	//	std::cout << "current: " << transition->to_string() << std::endl;

	//	const auto marking_dist = prev_marking.to_marking()->distribution;
	//	std::cout << "prev marking (" << marking_dist.size() << "): " << std::setprecision(2);
	//	for (const auto& entry : marking_dist)
	//	{
	//		std::cout << "(" << entry.first.first->id << ", ";
	//		auto obj_token = std::dynamic_pointer_cast<pn_object_token>(entry.first.second);

	//		if (obj_token)
	//			std::cout << obj_token->object->get_name()[0];
	//		else if (entry.first.second == empty_token)
	//			std::cout << "e";
	//		else
	//			std::cout << "b";

	//		std::cout << ") " << entry.second << ", ";
	//	}
	//}



	//for (const auto& arcs : { transition->inputs, transition->outputs })
	//	for (const auto& instance : arcs)
	//	{
	//		auto boxed_place = std::dynamic_pointer_cast<pn_boxed_place>(instance.first);
	//		if (!boxed_place || instance.second == empty_token)
	//			continue;

	//		if (transition->is_side_condition(instance))
	//			continue;

	//		std::cout << "proxy remove " << boxed_place->id << " " <<timestamp << std::endl;
	//		proxy_places_and_timestamp.erase(boxed_place);
	//	}



	std::vector<transition_context> non_executed;

	for (const auto& w_t : model_hand->get_incoming_transitions())
	{
		auto t = w_t.lock();
		if (t != transition && prev_marking.is_enabled(t) > enable_threshold)
			non_executed.emplace_back(workspace_params, t, prev_marking, model_hand, timestamp);
	}

	for (const auto& w_t : model_hand->get_outgoing_transitions())
	{
		auto t = w_t.lock();
		if (t != transition && prev_marking.is_enabled(t) > enable_threshold)
			non_executed.emplace_back(workspace_params, t, prev_marking, model_hand, timestamp);
	}

	add_transition(transition_context(workspace_params, transition, prev_marking, model_hand, timestamp), non_executed);
}

void observed_agent::add_transition(const transition_context& executed,
	const std::vector<transition_context>& non_executed)
{
	if (pending_action && executed.transition->reverses(*pending_action->first.transition))
	{
		pending_action = nullptr;
		return;
	}

	pn_token::Ptr start_condition = nullptr;
	if (!executed_actions.empty())
		start_condition = get_postcondition(*executed_actions.back().transition);

	auto add = [&](const transition_context& executed, const std::vector<transition_context>& non_executed)
	{
		executed_actions.emplace_back(executed);
		get_predictor().add_data(executed_actions.back(), non_executed);
	};

	if (pending_action)
	{
		if (start_condition == get_precondition(*executed.transition) && 
			get_precondition(*pending_action->first.transition) == get_postcondition(*executed.transition))
		{
			add(executed, non_executed);
			add(pending_action->first, pending_action->second);
			pending_action = nullptr;

			return;
		}
		else if (start_condition == get_precondition(*pending_action->first.transition) && 
			get_precondition(*executed.transition) == get_postcondition(*pending_action->first.transition))
		{
			add(pending_action->first, pending_action->second);
			executed_actions.emplace_back(std::move(executed));
			pending_action = nullptr;

			return;
		}
		else
		{
			auto prev_condition = executed_actions.size() >= 2 ? get_postcondition(*(++executed_actions.rbegin())->transition) : nullptr;

			if (is_forward_transition(pending_action->first.transition) && is_forward_transition(executed.transition) &&
				prev_condition == get_precondition(*pending_action->first.transition) && get_precondition(*executed.transition) == get_postcondition(*pending_action->first.transition))
			{
				executed_actions.back() = pending_action->first;
				get_predictor().replace_last(executed_actions.back());
				add(executed, non_executed);
				pending_action = nullptr;

				return;
			}

			if (!is_forward_transition(pending_action->first.transition))
				pending_action = nullptr;

		}
	}

	if (start_condition == get_precondition(*executed.transition) && is_forward_transition(executed.transition))
		add(executed, non_executed);
	else if (pending_action == nullptr || is_forward_transition(executed.transition))
		pending_action = std::make_unique<std::pair<transition_context, std::vector<transition_context>>>(std::make_pair(executed, non_executed));

	
}

void observed_agent::add_transitions(std::set<pn_transition::Ptr> transitions,
	const pn_belief_marking& prev_marking,
	const pn_belief_marking& current_marking,
	std::chrono::duration<float> timestamp)
{
	pn_token::Ptr token_in_hand = nullptr;

	if (!executed_actions.empty())
	{
		token_in_hand = get_postcondition(*executed_actions.back().transition);

		remove_double_detections(transitions);
	}

	while (!transitions.empty())
	{
		auto iter = transitions.begin();
		while (iter != transitions.end() && get_precondition(**iter) != token_in_hand)
			++iter;

		if (iter == transitions.end())
			iter = transitions.begin();

		token_in_hand = get_postcondition(**iter);
		add_transition(*iter, token_in_hand ? prev_marking : current_marking, timestamp);
		transitions.erase(iter);
	}

	start_training();
}

void observed_agent::start_training()
{
	get_predictor().train();
}

void observed_agent::finished(std::chrono::high_resolution_clock::time_point start)
{
	for (const auto& predictor : predictors | std::views::values)
		predictor->finished();

	predictors.clear();
	start_time = start;

	const boost::posix_time::ptime time = boost::posix_time::second_clock::local_time();
	std::stringstream stream;
	stream << time.date().year()
		<< "-" << time.date().month().as_number()
		<< "-" << time.date().day()
		<< "-" << time.time_of_day().hours()
		<< "-" << time.time_of_day().minutes()
		<< "/";

	path = stream.str();
}

pn_token::Ptr observed_agent::get_precondition(const pn_transition& transition) const noexcept
{
	for (const pn_instance& instance : transition.inputs)
		if (instance.first == model_hand)
			return instance.second;

	return nullptr;
}

pn_token::Ptr observed_agent::get_postcondition(const pn_transition& transition) const noexcept
{
	for (const pn_instance& instance : transition.outputs)
		if (instance.first == model_hand)
			return instance.second;

	return nullptr;
}

const transition_context& observed_agent::get_top_left(const std::vector<transition_context>& contexts) const
{
	Eigen::Vector3f top_left_dir = front;
	top_left_dir.z() = 0.f;
	top_left_dir = Eigen::AngleAxisf(M_PI_4, Eigen::Vector3f::UnitZ()) * top_left_dir;
	top_left_dir.normalize();

	auto top_left_ctx = std::ranges::max_element(contexts,
	    [&](const transition_context& lhs, const transition_context& rhs) { return lhs.center.dot(top_left_dir) < rhs.center.dot(top_left_dir); });

	return *top_left_ctx;
}

void observed_agent::remove_double_detections(std::set<pn_transition::Ptr>& transitions) const
{
	if (!executed_actions.empty())
	{
		std::chrono::duration<float> last_timestamp = executed_actions.back().timestamp;
		for (const auto& executed_action : std::ranges::reverse_view(executed_actions))
		{
			if (executed_action.timestamp != last_timestamp)
				break;

			transitions.erase(executed_action.transition);
		}
	}
}

observed_agent::transition_probability observed_agent::get_similars(const state_observation::pn_transition::Ptr& transition,
	const pn_belief_marking& marking) const
{
	transition_probability result;
	transition_context transition_ctx(workspace_params, transition, marking);

	for (const auto& t : marking.net.lock()->get_transitions())
	{
		float similarity = transition_context(workspace_params, t, marking, model_hand).compare(transition_ctx);
		if (similarity > min_probability)
			result.emplace(t, similarity);
	}

	return result;
}

observed_agent::transition_probability observed_agent::apply_general_prediction_rules(const std::vector<transition_context>& contexts) const
{
	const transition_context& top_left_ctx = get_top_left(contexts);

	auto lowest = std::ranges::min_element(contexts,
	                                       [&](const transition_context& lhs, const transition_context& rhs) { return lhs.box.bottom_z() < rhs.box.bottom_z(); });

	float min_z = lowest->box.bottom_z();
	Eigen::Vector3f top_left = top_left_ctx.center;

	transition_probability result;

	auto max_dist_ctx = std::ranges::max_element(contexts,
	                                             [&](const transition_context& lhs, const transition_context& rhs) { return (lhs.center - top_left).squaredNorm() < (rhs.center - top_left).squaredNorm(); });

	float stdev = std::max(transition_context::neighbor_distance_threshold, 0.5f * (max_dist_ctx->center - top_left).norm());

	for (const transition_context& ctx : contexts)
	{
		double prob = transition_context::bell_curve((top_left - ctx.center).norm(), stdev) *
			std::pow(0.5f, ctx.box.bottom_z() - min_z);

		if (prob >= min_probability)
			result.emplace(ctx.transition, prob);
	}

	return result;

}

observed_agent::transition_probability observed_agent::apply_general_prediction_rules(const std::vector<transition_context>& contexts,
	const transition_context& prev_action) const
{
	auto to_weight = [&](float best_diff, float diff)
	{
		return 0.5f * transition_context::bell_curve(std::max(0.f, diff - best_diff), best_diff) + 0.5f;
	};



	Eigen::Vector2f br = -front.head(2);
	br = Eigen::Rotation2D<float>(0.3f * M_PI) * br;
	br.normalize();

	auto br_distance = [&](const Eigen::Vector2f& v)
	{
		return v.norm() / std::max(0.f, br.dot(v.normalized()));
	};

	auto closest_br = std::ranges::min_element(contexts,
	                                           [&](const transition_context& lhs, const transition_context& rhs) { return br_distance(lhs.center_xy - prev_action.center_xy) < br_distance(rhs.center_xy - prev_action.center_xy); });

	auto closest = std::ranges::min_element(contexts,
	                                        [&](const transition_context& lhs, const transition_context& rhs) { return (lhs.center_xy - prev_action.center_xy).norm() < (rhs.center_xy - prev_action.center_xy).norm(); });

	auto closest_br_distance = std::isfinite(br_distance(closest_br->center_xy)) ? br_distance(closest_br->center_xy) : 0.00000001f;

	auto lowest = std::ranges::min_element(contexts,
	                                       [&](const transition_context& lhs, const transition_context& rhs) { return lhs.box.bottom_z() < rhs.box.bottom_z(); });

	transition_probability result;
	double min = 1.f;
	for (const transition_context& ctx : contexts)
	{
		// Rules according to Mayer12
		double prob = to_weight(closest_br_distance, br_distance(ctx.center_xy - prev_action.center_xy)) * // R1: Work from (top) left to right
			to_weight((closest->center_xy - prev_action.center_xy).norm(), (ctx.center_xy - prev_action.center_xy).norm()) * // R2: Prefer vicinity 
			to_weight(prev_action.box.diagonal.z() / 2, ctx.box.bottom_z() - prev_action.box.bottom_z()) * // R3: Complete layers, i.e. prefer same layer
			(0.5f + 0.25f * transition_context::bell_curve(ctx.color - prev_action.color, .02f) +
				0.25f * transition_context::bell_curve(ctx.volume - prev_action.volume, .02f)) * // R4: Prefer identical objects
			std::pow(2, ctx.action_type * prev_action.action_type - 1);

		min = std::min(min, prob);

		if (prob >= min_probability)
			result.emplace(ctx.transition, prob);
	}

	for (auto& entry : result)
		entry.second -= min;

	return result;
}

observed_agent::transition_probability observed_agent::apply_behavior_based_prediction_rules(const std::vector<transition_context>& candidates,
	const std::vector<transition_context>& time_horizon_history) const
{
	auto iter = predictors.find(net->get_goal());
	if (iter == predictors.end())
	{
		observed_agent::transition_probability result;
		float weight = 1.f / candidates.size();
		for (const auto& candidate : candidates)
			result.emplace(candidate.transition, weight);

		return result;
	}

	auto result = iter->second->predict(time_horizon_history, candidates);
	double max = -1;
	for (const auto& probability : result | std::views::values)
		max = std::max(max, probability);

	for (auto& probability : result | std::views::values)
		probability = transition_context::bell_curve(max - probability, 0.3f);

	normalize(result);

	return result;
}

net_predictor& observed_agent::get_predictor()
{
	auto iter = predictors.find(net->get_goal());

	if (iter != predictors.end())
		return *iter->second;

	std::stringstream stream;
	stream << path << net->get_goal()->id << "/" << model_hand->id << "/";

	return *predictors.emplace(net->get_goal(), std::make_unique<net_predictor>(workspace_params, stream.str(), start_time)).first->second;
}


std::map<pn_transition::Ptr, double> observed_agent::predict(const std::vector<transition_context>& candidate_contexts,
	const std::vector<pn_transition::Ptr>& future_transitions,
	const state_observation::pn_belief_marking& marking) const
{
	std::vector< transition_context> contexts;
	contexts.reserve(future_transitions.size());

	for (const auto& t : future_transitions)
		contexts.emplace_back(workspace_params, t, marking, model_hand);

	return predict(candidate_contexts, contexts);
}

std::map<pn_transition::Ptr, double> observed_agent::predict(const std::vector<transition_context>& candidate_contexts,
	const std::vector< transition_context>& future_transitions) const
{
	// handle case candidates.size() <= 1 or no history
	transition_probability result;
	auto add_to_result = [&result](const pn_transition::Ptr& t, double val)
	{
		auto iter = result.find(t);
		if (iter == result.end())
			result.emplace(t, val);
		else
			iter->second += val;
	};

	if (candidate_contexts.size() <= 1)
	{
		for (const auto& ctx : candidate_contexts)
			add_to_result(ctx.transition, 1.);

		return result;
	}

	if (executed_actions.empty())
	{
		transition_probability prob = apply_general_prediction_rules(candidate_contexts);
		normalize(prob);
		return prob;
	}

	// construct history
	std::vector< transition_context> history;
	history.reserve(net_predictor::time_horizon);

	auto iter = future_transitions.rbegin();
	while (history.size() < net_predictor::time_horizon && iter != future_transitions.rend())
		history.emplace(history.begin(), *iter++);

	iter = executed_actions.rbegin();
	while (history.size() < net_predictor::time_horizon && iter != executed_actions.rend())
		history.emplace(history.begin(), *iter++);


	// apply neural net
	float weight = 1.f;
	if (history.size() >= net_predictor::time_horizon)
	{
		result = apply_behavior_based_prediction_rules(candidate_contexts, history);

		float max = 0.f;
		float prev_max = 0.f;
		float min = 1.f;

		for (const auto& probability : result | std::views::values)
		{
			if (probability < min)
				min = probability;
			if (probability > prev_max)
				prev_max = probability;
			if (probability > max)
			{
				prev_max = max;
				max = probability;
			}
		}

		constexpr float threshold = 0.25f;
		if (max - prev_max > threshold)
			return result;


		if (max - min < 0.01f)
			result.clear();
		else
			weight = std::min(1 - (max - prev_max) / threshold, std::powf(2, net_predictor::time_horizon - executed_actions.size()));
	}


	// use rules if net cannot be applied or is not discriminative
	int i = 0;
	for (auto& h_iter : std::ranges::reverse_view(history))
	{
		for (auto [transition, probability] : apply_general_prediction_rules(candidate_contexts, h_iter))
			add_to_result(transition, probability * weight);

		if (++i > 2)
			break;
	}

	normalize(result);

	return result;
}

};