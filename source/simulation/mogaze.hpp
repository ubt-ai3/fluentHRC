#pragma once


#include "task.hpp"

#include "state_observation/object_prototype_loader.hpp"
#include <enact_priority/signaling_actor.hpp>

namespace simulation
{

namespace mogaze
{

class SIMULATION_API mogaze_object : public state_observation::object_prototype{
public:
	typedef std::shared_ptr<object_prototype> Ptr;
	typedef std::shared_ptr<const object_prototype> ConstPtr;

	mogaze_object(const pcl::RGB & mean_color,
		const state_observation::mesh_wrapper::Ptr & base_mesh,
		const std::string & name = std::string(),
		const std::string & type = std::string());

	static state_observation::aabb compute_bounding_box(const pcl::PolygonMesh& mesh);
};

class SIMULATION_API object_prototype_loader : public state_observation::object_prototype_loader
{
public:
	static std::vector<state_observation::object_prototype::Ptr> generate_default_prototypes();

	object_prototype_loader();
	virtual ~object_prototype_loader() = default;
};

class SIMULATION_API furnishing : public scene_object
{
public:
	typedef std::shared_ptr<furnishing> Ptr;

	furnishing(const std::string& mesh_path, const state_observation::obb& pose, pcl::RGB color);
	virtual ~furnishing() = default;

	void render(pcl::simulation::Scene& scene, std::chrono::duration<float> timestamp) override;
	void render(pcl::visualization::PCLVisualizer& viewer, std::chrono::duration<float> timestamp, int viewport = 0) override;


protected:
	pcl::PolygonMesh::ConstPtr mesh;
};


struct SIMULATION_API action 
{
	std::chrono::duration<float> timestamp;
	bool pick; // false = place
	state_observation::object_prototype::ConstPtr object;
	state_observation::obb pose;

	action(unsigned int timestamp,
		bool pick,
		state_observation::object_prototype::ConstPtr object,
		state_observation::obb pose);
};

class SIMULATION_API predicate
{
public:
	using Ptr = std::shared_ptr<predicate>;
	using traces = std::map<state_observation::object_prototype::ConstPtr, state_observation::pn_object_token::Ptr>;

	static const Eigen::AlignedBox3f table;
	// big shelf
	static const Eigen::AlignedBox3f laiva_shelf;
	// small shelf
	static const Eigen::AlignedBox3f vesken_shelf;

	//static std::map<std::string, state_observation::pn_object_token::Ptr> name_to_token;

	static std::set<state_observation::pn_token::Ptr> find_matching_tokens(const std::string& substring, const traces& token_traces);

	static Ptr all_types_in_region(const std::string& type, const Eigen::AlignedBox3f& region, const traces& token_traces);
	static Ptr all_colors_in_region(const std::string& color, const Eigen::AlignedBox3f& region, const traces& token_traces);
	static Ptr count_types_in_region(int n, const std::string& type, const Eigen::AlignedBox3f& region, const traces& token_traces);
	static Ptr bowl_or_jug_on_table(bool placed, const traces& token_traces);

	predicate(std::set<state_observation::pn_token::Ptr> relevant_tokens,
		Eigen::AlignedBox3f region);

	predicate(const predicate&) = default;
	predicate(predicate&&) = default;
	predicate& operator=(const predicate&) = default;

	virtual ~predicate() = default;

	virtual bool operator()(const state_observation::pn_binary_marking::ConstPtr& marking) const = 0;
	virtual std::set<state_observation::pn_transition::Ptr> get_blocked(const state_observation::pn_binary_marking::ConstPtr& marking) const = 0;
	virtual std::set<state_observation::pn_transition::Ptr> get_feasible(const state_observation::pn_binary_marking::ConstPtr& marking) const = 0;

	bool in_region(const state_observation::pn_boxed_place::Ptr& place) const;

	const std::set<state_observation::pn_token::Ptr> relevant_tokens;
	const Eigen::AlignedBox3f region;

protected:
	predicate() = default;

	std::set<state_observation::pn_transition::Ptr> get_incoming(const state_observation::pn_binary_marking::ConstPtr& marking) const;
	std::set<state_observation::pn_transition::Ptr> get_outgoing(const state_observation::pn_binary_marking::ConstPtr& marking) const;
};

class SIMULATION_API pred_all_in_region : public predicate
{
public:
	using Ptr = std::shared_ptr<pred_all_in_region>;

	pred_all_in_region(std::set<state_observation::pn_token::Ptr> relevant_tokens,
		Eigen::AlignedBox3f region);

	virtual ~pred_all_in_region() = default;

	bool operator()(const state_observation::pn_binary_marking::ConstPtr& marking) const override;
	std::set<state_observation::pn_transition::Ptr> get_blocked(const state_observation::pn_binary_marking::ConstPtr& marking) const override;
	std::set<state_observation::pn_transition::Ptr> get_feasible(const state_observation::pn_binary_marking::ConstPtr& marking) const override;

};

class SIMULATION_API pred_count_in_region : public predicate
{
public:
	using Ptr = std::shared_ptr<pred_count_in_region>;

	pred_count_in_region(int count,
		std::set<state_observation::pn_token::Ptr> relevant_tokens,
		Eigen::AlignedBox3f region);

	virtual ~pred_count_in_region() = default;

	int current_count(const state_observation::pn_binary_marking::ConstPtr& marking) const;

	bool operator()(const state_observation::pn_binary_marking::ConstPtr& marking) const override;
	std::set<state_observation::pn_transition::Ptr> get_blocked(const state_observation::pn_binary_marking::ConstPtr& marking) const override;
	std::set<state_observation::pn_transition::Ptr> get_feasible(const state_observation::pn_binary_marking::ConstPtr& marking) const override;

	const int count;
};

class SIMULATION_API pred_bowl_or_jug_on_table : public predicate
{
public:
	using Ptr = std::shared_ptr<pred_bowl_or_jug_on_table>;

	pred_bowl_or_jug_on_table(bool placed, std::set<state_observation::pn_token::Ptr> relevant_tokens);

	virtual ~pred_bowl_or_jug_on_table() = default;

	bool operator()(const state_observation::pn_binary_marking::ConstPtr& marking) const override;
	std::set<state_observation::pn_transition::Ptr> get_blocked(const state_observation::pn_binary_marking::ConstPtr& marking) const override;
	std::set<state_observation::pn_transition::Ptr> get_feasible(const state_observation::pn_binary_marking::ConstPtr& marking) const override;

	const bool placed;
};

struct SIMULATION_API instruction
{
	std::chrono::duration<float> time;
	int id;
	std::string text;
};

class SIMULATION_API task_execution
{
public:
	static const std::map<std::string, Eigen::Vector3f> object_dimensions;
	std::vector<std::vector<predicate::Ptr>> instruction_predicates;

	task_execution(const state_observation::object_parameters& object_params, int person);

	void render(pcl::simulation::Scene& scene);
	void render(pcl::visualization::PCLVisualizer& viewer);

	state_observation::pn_binary_marking::Ptr get_marking() const;
	int get_task_id() const { return task_id; };

	/* Returns dummy action if there is no next */
	action peek_next_action() const;

	state_observation::pn_transition::Ptr next();
	state_observation::pn_transition::Ptr execute(const action& a);

	std::vector<state_observation::pn_transition::Ptr> get_action_candidates() const;
	std::vector<state_observation::pn_transition::Ptr> get_action_candidates(const state_observation::pn_binary_marking::ConstPtr& marking) const;
	std::set<state_observation::pn_transition::Ptr> all_feasible_candidates(const state_observation::pn_binary_marking::ConstPtr& marking) const;


	/* Returns whether another boxed place (with a token) overlaps the target place of @param{transitions}. In consequence, the transition cannot fire. */
	static bool is_blocked(const state_observation::pn_binary_marking::ConstPtr& marking, const state_observation::pn_transition::Ptr& transition);

	std::vector<std::vector<predicate::Ptr>> get_instruction_predicates();

	object_prototype_loader loader;	

	// all transitions have one incoming and one outgoing arc
	state_observation::pn_net::Ptr net;
	state_observation::pn_place::Ptr human;

protected:
	std::chrono::duration<float> timestamp;
	int task_id;
	
	std::map<std::string, std::vector<state_observation::object_prototype::ConstPtr>> instances;
	std::map<std::string, std::vector<state_observation::pn_boxed_place::Ptr>> locations;
	std::map<state_observation::object_prototype::ConstPtr, movable_object::Ptr> movable_objects;
	std::vector<furnishing::Ptr> furnishings;
	std::map<state_observation::object_prototype::ConstPtr, state_observation::pn_object_token::Ptr> token_traces;

	std::queue<instruction> instructions;
	std::queue<action> actions;
	std::vector<state_observation::pn_transition::Ptr> executed_transitions;

	static state_observation::obb get_box(const std::vector<std::string>& entries, int start = 0);
	state_observation::pn_boxed_place::Ptr get_location(const state_observation::object_prototype& object, const state_observation::obb& box) const;
};




}

}