#pragma once

#include <random>

#include "hand_model.hpp"
#include "parameter_set.hpp"

namespace hand_pose_estimation
{

/**
 *************************************************************************
 *
 * @class particle_filter_parameters
 *
 * Parameters for the particle swarm filter algorithm
 *
 ************************************************************************/

class particle_filter_parameters : public parameter_set {
public:

	particle_filter_parameters();

	~particle_filter_parameters();

	float cognitive_component;
	float social_component;
	unsigned int disturbance_interval;


	template <typename Archive>
	void serialize(Archive& ar, const unsigned int version) {
		ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(parameter_set);
		ar& BOOST_SERIALIZATION_NVP(cognitive_component);
		ar& BOOST_SERIALIZATION_NVP(social_component);
		ar& BOOST_SERIALIZATION_NVP(disturbance_interval);
	}

};

/**
*************************************************************************
*
* @class quality_criterion
*
* Part of the objective function for class particle_swarm_filter.
* See function evaluate().
*
************************************************************************/
class  HANDPOSEESTIMATION_API quality_criterion {
public:
	typedef std::shared_ptr<quality_criterion> Ptr;
	typedef std::shared_ptr<const quality_criterion> ConstPtr;

	quality_criterion(double weight);

	quality_criterion(const quality_criterion&) = default;
	quality_criterion(quality_criterion&&) noexcept = default;

	virtual ~quality_criterion() = default;

	quality_criterion& operator=(const quality_criterion&) = default;

	/*
	* Returns a value within [0,\infty)
* A value of 1 should roughly correspend to all (key) points being offset
* by 1 cm. This should ensure that all quality measures have roughly the
* same magnitude. The weight is applied last before returning the value
	*/
	virtual double evaluate(const visual_input& input,
		const hand_pose_particle_instance& particle,
		const img_segment& seg) const = 0;

protected:

	double weight;
};


/**
*************************************************************************
*
* @class quality_2d_key_points
*
* Part of the objective function for class particle_swarm_filter.
* See function evaluate().
*
************************************************************************/
class  HANDPOSEESTIMATION_API quality_2d_key_points : public  quality_criterion {
public:

	quality_2d_key_points(double weight) :
		quality_criterion(weight)
	{}

	virtual double evaluate(const visual_input& input,
		const hand_pose_particle_instance& particle,
		const img_segment& seg) const;
};



/**
*************************************************************************
*
* @class particle_swarm_filter
*
* Runs particle swarm filter as described in 
* https://www.researchgate.net/publication/233950932_Efficient_model-based_3D_tracking_of_hand_articulations_using_Kinect
*
************************************************************************/
class particle_swarm_filter
{
public:

	HANDPOSEESTIMATION_API particle_swarm_filter(unsigned int count_particles = 50,
		unsigned int count_generations = 30,
		std::vector<quality_criterion::Ptr> objectives = { std::make_shared< quality_2d_key_points>(1.f) });

	HANDPOSEESTIMATION_API particle_swarm_filter(const particle_swarm_filter&) = default;
	HANDPOSEESTIMATION_API particle_swarm_filter(particle_swarm_filter&&) noexcept = default;

	HANDPOSEESTIMATION_API ~particle_swarm_filter() = default;

	HANDPOSEESTIMATION_API particle_swarm_filter& operator=(const particle_swarm_filter&) = default;

	const particle_filter_parameters params;
	const hand_dynamic_parameters dynamic_params;

	const int count_particles;
	const int count_generations;

	HANDPOSEESTIMATION_API hand_pose_particle_instance::Ptr update(const visual_input& input,
		const img_segment& seg,
		const std::vector<hand_pose_18DoF::Ptr>& seeds,
		const hand_pose_particle_instance::Ptr& prev_particle = nullptr);

	HANDPOSEESTIMATION_API void update(float timestamp);

private:
	
	std::vector<quality_criterion::Ptr> objectives;
	std::mt19937 generator;
	std::uniform_real_distribution<float> dist_uni_0_1;
	std::uniform_int_distribution<short> dist_bool;

	hand_pose_particle_instance::Ptr best;
	float timestamp;
	float prev_timestamp;

	void evaluate_objectives(const visual_input& input,
		const img_segment& seg,
		hand_pose_particle_instance& particle) const;

	hand_pose_18DoF::Ptr sample_unconstraint(const hand_pose_particle& particle);

	hand_pose_18DoF::Ptr constrain(const hand_pose_particle& particle, const hand_pose_18DoF::Ptr& model);

	hand_pose_18DoF::Ptr disturb(const hand_pose_18DoF& model, 
		const hand_pose_particle_instance::Ptr& prev_frame = nullptr);
	hand_pose_18DoF::Ptr disturb_finger(const hand_pose_18DoF& model);

	std::vector<hand_pose_particle::Ptr> initiate_particles(const std::vector<hand_pose_18DoF::Ptr>& seeds,
		const hand_pose_particle_instance::Ptr& prev_particle = nullptr);
};

}