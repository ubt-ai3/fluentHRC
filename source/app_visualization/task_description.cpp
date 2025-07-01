#include "task_description.hpp"

#include <opencv2/highgui.hpp>

#include "enact_core/world.hpp"
#include "enact_core/access.hpp"
#include "enact_core/data.hpp"
#include "enact_core/lock.hpp"
#include "enact_core/id.hpp"

// avoid duplicate definition of cv::Point<int> leading to linking errors
#include "hand_pose_estimation/hand_model.hpp"

namespace state_observation
{

/////////////////////////////////////////////////////////////
//
//
//  Class: task_description
//
//
/////////////////////////////////////////////////////////////



const std::map<std::string, int> task_description::resources = {
	{"wooden small cylinder", 4},
	{"wooden semicylinder", 4},
	{"wooden cylinder", 4},
	{"wooden cube", 4},
	{"wooden block", 10},
	{"wooden triangular prism", 1},
	{"wooden bridge", 4},
	{"purple bridge", 2},
	{"magenta bridge", 2},
	{"wooden flat block", 2},
	{"wooden plank", 4},
	{"cyan plank", 2},
	{"red plank", 1},
	{"yellow plank", 2},
	{"red block", 3},
	{"blue block", 3},
	{"yellow block", 2},
	{"red small cylinder", 4},
	{"magenta cylinder", 1},
	{"red flat block", 1},
	{"yellow flat block", 1},
	{"purple flat block", 1},
	{"purple semicylinder", 4},
	{"magenta semicylinder", 1},
	{"cyan triangular prism", 2},
	{"red cube", 3},
	{"purple cube", 1},
	{"cyan cube", 1},
	{"yellow cube", 1}
};

const std::string task_description::window_name = "workplace";


task_description::task_description(enact_core::world_context& world, 
	pn_world_tracer& tracing,
	const std::vector<object_prototype::ConstPtr>& prototypes,
	const Eigen::Matrix<float, 3, 4>& projection)
	:
	world(world),
	projection(projection),
	drawing_net_overlay(1080, 1920, CV_8UC4, cv::Scalar(0,0,0,0)),
	net(std::make_shared<pn_net>(*tracing.get_net()->object_params)),
	tracing(tracing)
{
	const auto& token_traces = tracing.get_token_traces();
	std::set<pn_instance> distribution;

	for(const auto& entry : token_traces)
	{
		auto iter = resources.find(entry.first->get_name());
		if (iter != resources.end())
		{
			pn_place::Ptr place = net->create_place();
			resource_pools.emplace(entry.second, place);
			token_to_prototype.emplace(entry.second, entry.first);
			distribution.emplace(std::make_pair(place, entry.second));
		}
	}
	
	for (int i = 0; i < 4; i++)
	{
		hands.push_back(net->create_place(true));
		for (const auto& pool : resource_pools)
			net->create_transition({ std::make_pair(pool.second, pool.first) }, { std::make_pair(hands.back(), pool.first) });
	}

	marking = std::make_shared<pn_belief_marking>(std::make_shared<pn_binary_marking>(net,std::move(distribution)));
	differ = std::make_shared<pn_feasible_transition_extractor>(net, marking->to_marking());
}

task_description::~task_description()
{
	stop_thread();
}

void task_description::update(const cv::Mat& img)
{
	//drawing_net_overlay.resize(img.size);
}
	
void task_description::evaluate_net(std::chrono::duration<float> timestamp)
{
	auto emissions = tracing.generate_emissions(timestamp);
	differ->update(marking->to_marking());
	differ->update(emissions);
	reasoning = std::make_shared<sampling_optimizer_belief>(differ->extract(), marking, emissions);

if (!initial_recognition_done)
	{
		if (reasoning->emission_consistency(marking->distribution.begin()->first) > 0.)
			initial_recognition_done = true;
		else
			return;
	}

	marking = reasoning->update(2000);

	draw_net();
}

void task_description::update(const strong_id& id, enact_priority::operation op)
{
	schedule([id, op, this]()
		{
			enact_core::lock l(world, enact_core::lock_request(id, object_instance::aspect_id, enact_core::lock_request::read));
			enact_core::const_access<object_instance_data> access_object(l.at(id, object_instance::aspect_id));
			const object_instance& obj = access_object->payload;

			if(op==enact_priority::operation::DELETED || obj.observation_history.empty())
			{
				/*auto iter = instances.find(id);
				if(iter != instances.end())
				{
					auto iter_circle = circle_centers.find(iter->second.first);
					if(iter_circle != circle_centers.end())
					{
						cv::circle(drawing_net_overlay, iter_circle->second, place_radius, cv::Scalar(0, 0, 0, 0), cv::FILLED);
					}
				}

					(*emitter)(drawing_net_overlay, enact_priority::operation::UPDATE);*/
				return;
			}

			if(op == enact_priority::operation::UPDATE && 
				obj.observation_history.back()->classification_results.size() >= 3 &&
				obj.get_classified_segment()->classification_results.front().local_certainty_score > 0.5)
			{
				Eigen::Vector4f centroid(obj.observation_history.back()->centroid.getArray4fMap());
				Eigen::Vector2f center_2d((projection * centroid).hnormalized());
				cv::Point2i center(center_2d.x(), center_2d.y());

				pn_place::Ptr place = net->create_place();
				for(const auto& token_trace : tracing.get_token_traces())
					for(const pn_place::Ptr& hand : hands)
					{
						net->create_transition({ std::make_pair(hand, token_trace.second) }, { std::make_pair(place, token_trace.second) });
						net->create_transition({ std::make_pair(place, token_trace.second) }, { std::make_pair(hand, token_trace.second) });
					}

				circle_centers.emplace(place, center);
				
				cv::circle(drawing_net_overlay, center, place_radius, cv::Scalar(255, 255, 255, 255));

				(*emitter)(drawing_net_overlay, enact_priority::operation::UPDATE);
			}

		});
}

	
void task_description::draw_net()
{
	cv::Mat token_overlay(drawing_net_overlay.size(), CV_8UC4);
	for (const auto& entry : circle_centers)
	{
		cv::circle(drawing_net_overlay, entry.second, place_radius, cv::Scalar(255, 255, 255, 255));
		auto distribution = marking->get_distribution(entry.first);

		double max = 0.;
		pn_token::Ptr token;
		for (const auto& dist : distribution)
		{
			if (dist.second > max)
			{
				max = dist.second;
				token = dist.first;
			}
		}

		if (max > 0)
		{
			cv::Scalar color(0, 0, 0, 255);
			auto iter = token_to_prototype.find(token);
			if(iter != token_to_prototype.end())
			{
				auto col = iter->second->get_mean_color();
				color(0) = col.b;
				color(1) = col.g;
				color(2) = col.r;

				color(3) = 255 * max;

				cv::circle(token_overlay, entry.second, token_radius, color);
			}

			
		}
	}

	cv::Mat zeros = cv::Mat(drawing_net_overlay.size(), CV_8UC1, cv::Scalar(0));
	drawing_net_overlay = cv::Scalar(0, 0, 0, 0);

	std::vector<cv::Mat> channels;
	cv::split(token_overlay, channels);
	cv::Mat alpha;
	cv::merge(std::vector<cv::Mat>({ channels[3],channels[3],channels[3],zeros }), alpha);

	cv::Mat result = drawing_net_overlay.mul(cv::Scalar(255, 255, 255, 255) - alpha, 1. / 255) + token_overlay.mul(alpha, 1. / 255);
	(*emitter)(result, enact_priority::operation::UPDATE);

	drawing_net_overlay = result;
}
}
