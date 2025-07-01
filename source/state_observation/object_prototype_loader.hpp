#pragma once

#include <vector>

#include "workspace_objects.hpp"

namespace state_observation
{

class STATEOBSERVATION_API object_prototype_loader : public parameter_set
{
public:
	static std::vector<object_prototype::Ptr> generate_default_prototypes();

	object_prototype_loader();
	virtual ~object_prototype_loader() override = default;

	[[nodiscard]] std::vector<object_prototype::ConstPtr> get_prototypes() const;
	[[nodiscard]] object_prototype::ConstPtr get(const std::string& name) const;

	template <typename Archive>
	void serialize(Archive& av, const unsigned int version)
	{
		BOOST_SERIALIZATION_NVP(prototypes);
	}

protected:
	std::vector<object_prototype::Ptr> prototypes;

};

};