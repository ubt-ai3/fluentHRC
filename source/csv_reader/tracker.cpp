#include "tracker.hpp"

#include <vector>
#include <filesystem>

// manually tuned if <= 3 decimal places
const std::map<std::string, float> tracker::time_offsets_ = {
	{"Team.0.IDs.-1.-2.Trial.1.csv",	1.495f },
	{"Team.0.IDs.-1.-2.Trial.2.csv",	3.563f },
	{"Team.0.IDs.-1.-2.Trial.3.csv",	3.9f },
	{"Team.1.IDs.1.2.Trial.1.csv", 2.82f },
	{"Team.1.IDs.1.2.Trial.2.csv", 5.2175775f },
	{"Team.10.IDs.19.20.Trial.1.csv", 1.9f },
	{"Team.10.IDs.19.20.Trial.2.csv", 3.3f },
	{"Team.10.IDs.19.20.Trial.3.csv", 3.03f },
	{"Team.2.IDs.3.4.Trial.1.csv", 12.5929913f },
	{"Team.2.IDs.3.4.Trial.2.csv", 5.3181552f },
	{"Team.2.IDs.3.4.Trial.3.csv", 4.4905496f },
	{"Team.3.IDs.5.6.Trial.1.csv", 3.3369282f },
	{"Team.3.IDs.5.6.Trial.2.csv", 0.2200615f },
	{"Team.4.IDs.7.8.Trial.1.csv", -2.8f },
	{"Team.4.IDs.7.8.Trial.2.csv", 8.4725626f },
	{"Team.4.IDs.7.8.Trial.3.csv", 2.7875995f },
	{"Team.5.IDs.9.10.Trial.1.csv", 1.8f },
	{"Team.5.IDs.9.10.Trial.2.csv", -2.6f },
	{"Team.5.IDs.9.10.Trial.3.csv", 3.79f },
	{"Team.6.IDs.11.12.Trial.1.csv", 3.06f },
	{"Team.6.IDs.11.12.Trial.2.csv", 2.53f },
	{"Team.6.IDs.11.12.Trial.3.csv", 4.07f },
	{"Team.7.IDs.13.14.Trial.1.csv", 3.73f },
	{"Team.7.IDs.13.14.Trial.2.csv", 3.5303529f },
	{"Team.7.IDs.13.14.Trial.3.csv", 3.0639714f },
	{"Team.8.IDs.15.16.Trial.1.csv", 9.9f },
	{"Team.8.IDs.15.16.Trial.2.csv", 3.699f },
	{"Team.8.IDs.15.16.Trial.3.csv", 2.86f },
	{"Team.9.IDs.17.18.Trial.1.csv", 2.42f },
	{"Team.9.IDs.17.18.Trial.2.csv", 8.98f },
	{"Team.9.IDs.17.18.Trial.3.csv", 4.3f },
	{"Test.Single.csv", 2.388f }
};

/*
const std::map<std::string, std::map<hand_type, size_t>> team_to_tracker_mapping_ = {
	{0, {
		{hand_type::COLOR_LEFT, 2},
		{hand_type::COLOR_RIGHT, 1},
		{hand_type::TAN_LEFT, 3},
		{hand_type::TAN_RIGHT, 0}}
	},
		{1, {
		{hand_type::COLOR_LEFT, 0},
		{hand_type::COLOR_RIGHT, 1},
		{hand_type::TAN_LEFT, 3},
		{hand_type::TAN_RIGHT, 0}}
	},
};
*/


tracker::tracker(const std::filesystem::path& path, Eigen::Matrix<float, 2, 4> projection_matrix, Eigen::Matrix4f kinect_mat)
	:
	opti_track_to_rgb_(projection_matrix),
	opti_track_to_kinect_(kinect_mat)
{
	CsvWorker csv_worker;
	csv_worker.loadFromFile(path.string());
	frame_rate_ = std::stof(csv_worker.getField(0, 9));
	first_frame_ = std::stof(csv_worker.getField(6, 1));
	last_frame_ = std::stof(csv_worker.getField(csv_worker.getRowsCount() - 1, 1));

	//std::vector<std::string> path_elements = std::vector<std::string>();
	//boost::split(path_elements, path, [](char c) {return c == '\\' || c == '/'; });
	std::string path_element = (path / "").parent_path().string();

	time_offset_sec_ = tracker::time_offsets_.contains(path_element) ? tracker::time_offsets_.at(path_element) : 0;

	for (int col = 6; csv_worker.getField(2, col).find("RigidBody") != std::string::npos && csv_worker.getField(2, col).find("Marker") == std::string::npos; col += 20) {
		if (col == 26) // first listed rigid body has 4 markers
			col = 30;

		std::vector<std::vector<Eigen::Vector3f>> rigid_body_trajectory;
		for (unsigned int row = 6; row < csv_worker.getRowsCount(); row++) {
			std::vector<Eigen::Vector3f> rigid_body;
			for (int marker_col = col; marker_col < col + 16; marker_col += 4) // X, Y, Z, marker quality
			{
				rigid_body.emplace_back(
					parseFloat(csv_worker.getFieldOrNull(row, marker_col)),
					parseFloat(csv_worker.getFieldOrNull(row, marker_col + 1)),
					parseFloat(csv_worker.getFieldOrNull(row, marker_col + 2))
				);
			}

			rigid_body_trajectory.push_back(rigid_body);
		}
		movement_.push_back(rigid_body_trajectory);

		markers_.push_back(csv_worker.getField(2, col));
	}
}

tracker::tracker(const std::string& path, Eigen::Matrix<float, 2, 4> projection_matrix)
	:
	tracker(path, projection_matrix, Eigen::Matrix4f::Identity())
{
}

tracker::tracker(const std::string& path, Eigen::Matrix4f projection_matrix)
	:
	tracker(path, Eigen::Matrix<float,2,4>::Identity(), projection_matrix)
{
}

float tracker::parseFloat(const std::string* value) {
	if (!value || !*value->begin())
		return std::numeric_limits<float>::quiet_NaN();


	try {
		return std::stof(*value);
	}
	catch (const std::exception&) {
		return std::numeric_limits<float>::quiet_NaN();
	}
}

Eigen::Vector3f tracker::get_offset_opti_track_position(hand_type hand, float time) const
{
	const std::vector<std::vector<Eigen::Vector3f>>& trajectory(movement_[static_cast<int>(hand)]);

	auto trajectory_index = static_cast<size_t>((time + time_offset_sec_) * 100);
	if (trajectory_index >= trajectory.size())
		trajectory_index = trajectory.size() - 1;

	const std::vector<Eigen::Vector3f>& rigid_body(trajectory[trajectory_index]);

	switch (hand)
	{
	case hand_type::COLOR_RIGHT:
		return 0.8f * rigid_body[1] + 0.5f * rigid_body[2] - 0.3f* rigid_body[3] ;

	case hand_type::COLOR_LEFT:
		return -0.3f * rigid_body[1] + 0.8f * rigid_body[2] + 0.5f * rigid_body[3];

	case hand_type::TAN_LEFT:
		return 0.66f * rigid_body[1] + 0.58f * rigid_body[2] - 0.25f * rigid_body[3];

	case hand_type::TAN_RIGHT:
		return 1.5f * rigid_body[1] - 1.8f * rigid_body[2] + 1.3f * rigid_body[3];
	}
}

Eigen::Vector2f tracker::get_position_2d(hand_type hand, float time)
{
	const Eigen::Vector3f position = get_offset_opti_track_position(hand, time);

	Eigen::Vector4f homogen_position;
	homogen_position << position, 1;

	return opti_track_to_rgb_ * homogen_position;
}

Eigen::Vector3f tracker::get_position_3d(hand_type hand, float time)
{
	const Eigen::Vector3f position = get_offset_opti_track_position(hand, time);

	Eigen::Vector4f homogen_position;
	homogen_position << position, 1;

	Eigen::Vector4f result = (opti_track_to_kinect_ * homogen_position);
	return { result.x() / result(3), result.y() / result(3), result.z() / result(3) };
}
