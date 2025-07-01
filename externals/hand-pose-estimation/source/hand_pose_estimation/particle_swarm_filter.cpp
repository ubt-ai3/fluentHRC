#include "particle_swarm_filter.hpp"

#include <fstream>

#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>

#include <opencv2/imgproc.hpp>

namespace hand_pose_estimation
{

/////////////////////////////////////////////////////////////
//
//
//  Class: particle_filter_parameters
//
//
/////////////////////////////////////////////////////////////

particle_filter_parameters::particle_filter_parameters()
{
	filename_ = std::string("particle_filter_parameters.xml");

	std::ifstream file(folder_ + filename_);
	if (file.good()) {
		boost::archive::xml_iarchive ia{ file };
		ia >> BOOST_SERIALIZATION_NVP(*this);
	}
	else
	{
		cognitive_component = 2.8f;
		social_component = 1.3f;
		disturbance_interval = 3;
	}
}

particle_filter_parameters::~particle_filter_parameters()
{
	std::ofstream file(folder_ + filename_);
	boost::archive::xml_oarchive oa{ file };
	const particle_filter_parameters& particle_filter_params = *this; //passing *this to BOOST_SERIALIZATION_NVP will not work
	oa << BOOST_SERIALIZATION_NVP(particle_filter_params);
}


/////////////////////////////////////////////////////////////
//
//
//  Class: quality_criterion
//
//
/////////////////////////////////////////////////////////////

quality_criterion::quality_criterion(double weight)
	:
	weight(weight)
{
}


/////////////////////////////////////////////////////////////
//
//
//  Class: quality_2d_key_points
//
//
/////////////////////////////////////////////////////////////

double quality_2d_key_points::evaluate(const visual_input& input,
	const hand_pose_particle_instance& particle,
	const img_segment& seg) const
{
	Eigen::Matrix3f key_points = particle.pose->get_key_points();
	auto& net_eval = particle.pose->right_hand ? seg.net_eval_right : seg.net_eval_left;

	if (!net_eval)
		throw std::exception("Missing net evaluation");

	const net_evaluation::heatmaps& maps = net_eval->maps;
	const auto& box = net_eval->input_box;

	Eigen::Matrix<float, 3, 4> transform = Eigen::Scaling(maps.at(0).cols / (float)box.width,
		maps.at(0).rows / (float)box.height) *
		Eigen::Translation2f(-box.x, -box.y) *
		input.img_projection;

	double sum_prob = 0;

	for (int i = 0; i < key_points.cols(); i++)
	{
		Eigen::Vector2f p = (transform * key_points.col(i).homogeneous()).hnormalized();
		cv::Point2f pixel(p.x(), p.y());

		// extract interpolated value and handle out-of-image values
		cv::Mat patch;
		cv::remap(maps.at(i), patch, cv::Mat(1, 1, CV_32FC2, &pixel), cv::noArray(),
			cv::INTER_LINEAR, cv::BORDER_CONSTANT, -0.5f);

		sum_prob += 1 - patch.at<float>(0, 0);
	}

	return 2 * sum_prob / key_points.cols() * weight;
}


/////////////////////////////////////////////////////////////
//
//
//  Class: particle_swarm_filter
//
//
/////////////////////////////////////////////////////////////

particle_swarm_filter::particle_swarm_filter(unsigned int count_particles,
	unsigned int count_generations,
	std::vector<quality_criterion::Ptr> objectives)
	:
	count_particles(count_particles),
	count_generations(count_generations),
	objectives(std::move(objectives)),
	dist_uni_0_1(0.f,1.f),
	dist_bool(0,1)
{
}

hand_pose_particle_instance::Ptr particle_swarm_filter::update(const visual_input& input,
	const img_segment& seg,
	const std::vector<hand_pose_18DoF::Ptr>& seeds,
	const hand_pose_particle_instance::Ptr& prev_particle)
{
	auto particles = initiate_particles(seeds, prev_particle);
	best = particles.front()->current;

	for (auto& particle : particles)
	{
		evaluate_objectives(input, seg, *particle->current);
		if (particle->current->error < best->error)
			best = particle->current;
	}

	for (int generation = 0; generation < count_generations; generation++)
	{
		for (auto& particle : particles)
		{
			hand_pose_18DoF::Ptr model;

			if (generation % params.disturbance_interval == 0 && dist_bool(generator))
				model = disturb_finger(*particle->current->pose);
			else
				model = sample_unconstraint(*particle);

			model = constrain(*particle, model);
			auto next = std::make_shared<hand_pose_particle_instance>(std::move(model));
			model = nullptr;
			evaluate_objectives(input, seg, *next);
			particle->add(next);

			if (next->error < best->error)
				best = next;
		}
	}

	return best;
}

void particle_swarm_filter::update(float timestamp)
{
	prev_timestamp = this->timestamp;
	this->timestamp = timestamp;
}

hand_pose_18DoF::Ptr particle_swarm_filter::constrain(const hand_pose_particle& particle, const hand_pose_18DoF::Ptr& model)
{
	if (!particle.prev_frame)
		return model;

	hand_pose_18DoF::Vector18f velocity_bounds = dynamic_params.get_constraints_18DoF();
	hand_pose_18DoF::Vector18f pose = model->get_parameters();
	float diff = timestamp - prev_timestamp;

	hand_pose_18DoF::Vector18f speed = pose - particle.prev_frame->pose->get_parameters();
	speed /= diff;

	bool modified = false;
	for (int i = 0; i < pose.rows(); i++)
	{
		if (speed(i) > velocity_bounds(i))
		{
			pose(i) -= (speed(i) - velocity_bounds(i)) * diff;
			modified = true;
		}
		else if (speed(i) < -velocity_bounds(i))
		{
			pose(i) += (std::abs(speed(i)) - velocity_bounds(i)) * diff;
			modified = true;
		}
	}

	if (modified)
		return std::make_shared<hand_pose_18DoF>(model->hand_kinematic_params, pose, model->right_hand, model->bone_scaling);
	else
		return model;
}

hand_pose_18DoF::Ptr particle_swarm_filter::disturb(const hand_pose_18DoF& model, const hand_pose_particle_instance::Ptr& prev_frame)
{
	float diff = timestamp - prev_timestamp;
	auto pose = model.get_parameters();

	hand_pose_18DoF::Vector18f velocity_bounds = dynamic_params.get_constraints_18DoF();
	hand_pose_18DoF::Vector18f lower = model.get_lower_bounds();
	hand_pose_18DoF::Vector18f upper = model.get_upper_bounds();

	for (int i = 0; i < pose.rows(); i++)
	{
		float stdev = prev_frame ? std::max(0.05f * velocity_bounds(i), std::abs(prev_frame->velocities(i) * diff)) : 0.1f * velocity_bounds(i);

		std::normal_distribution<float> dist(pose(i), stdev);
		float sample;
		int j = 0;
		do {
			sample = dist(generator);
		} while (j++ < 10 && (lower(i) > sample || upper(i) < sample));

		pose(i) = sample;
	}

	return std::make_shared<hand_pose_18DoF>(model.hand_kinematic_params, pose, model.right_hand, model.bone_scaling);
}

hand_pose_18DoF::Ptr particle_swarm_filter::disturb_finger(const hand_pose_18DoF& model)
{
	auto pose = model.get_parameters();

	std::uniform_int_distribution<int> int_dist(6, 17);
	int index = int_dist(generator);

	std::uniform_real_distribution<float> real_dist(model.get_lower_bounds()(index), model.get_upper_bounds()(index));
	pose(index) = real_dist(generator);

	return std::make_shared<hand_pose_18DoF>(model.hand_kinematic_params, pose, model.right_hand, model.bone_scaling);
}

std::vector<hand_pose_particle::Ptr> particle_swarm_filter::initiate_particles(const std::vector<hand_pose_18DoF::Ptr>& seeds,
	const hand_pose_particle_instance::Ptr& prev_particle)
{
	std::vector<hand_pose_particle::Ptr> result;

	int i = 0;
	for (const auto& seed : seeds)
	{
		result.push_back(std::make_shared<hand_pose_particle>(
			std::make_shared<hand_pose_particle_instance>(seed),
			prev_particle
			));

		for (int j = result.size(); j < (i + 1) * count_particles / seeds.size(); j++)
		{
			result.push_back(std::make_shared<hand_pose_particle>(
				std::make_shared<hand_pose_particle_instance>(disturb(*seed)),
				prev_particle
				));
		}

		i++;
	}
	return result;
}

void particle_swarm_filter::evaluate_objectives(const visual_input& input,
	const img_segment& seg,
	hand_pose_particle_instance& particle) const
{
	double prob = 0;

	for (const auto& objective : objectives)
	{
		prob += objective->evaluate(input, particle, seg);
	}

	particle.error = prob;
}

hand_pose_18DoF::Ptr particle_swarm_filter::sample_unconstraint(const hand_pose_particle& particle)
{
	const float c_1 = params.cognitive_component;
	const float c_2 = params.social_component;

	const float psi = c_1 + c_2;
	const float w = 2 / std::abs(2 - psi - std::sqrt(psi * psi - 4 * psi));

	const float r_1 = dist_uni_0_1(generator);
	const float r_2 = dist_uni_0_1(generator);

	hand_pose_18DoF& pose = *particle.current->pose;

	return std::make_shared<hand_pose_18DoF>(pose.hand_kinematic_params,
		hand_pose_18DoF::Vector18f(w * (particle.current->velocities +
			c_1 * r_1 * particle.best->pose->get_parameters() +
			c_2 * r_2 * best->pose->get_parameters()) +
			(1 - w * c_1 * r_1 - w * c_2 * r_2) * pose.get_parameters()),
		pose.right_hand,
		pose.bone_scaling);
}

}