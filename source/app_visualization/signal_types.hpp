typedef pcl::PointXYZRGBA PointT;
typedef boost::signals2::signal<void(const pcl::PointCloud<PointT>::ConstPtr&)> cloud_signal_t;
typedef boost::signals2::signal<void(const cv::Mat4b&)> image_signal_t;
typedef boost::signals2::signal<void(const std::shared_ptr<enact_core::entity_id>&)> hand_signal_t;
typedef boost::signals2::signal<void(const std::shared_ptr<enact_core::entity_id>&)> object_signal_t;
typedef boost::signals2::signal<void(const std::shared_ptr<enact_core::entity_id>&, enact_priority::operation)> hand_event_signal_t;
typedef boost::signals2::signal<void(const std::shared_ptr<enact_core::entity_id>&, enact_priority::operation)> object_event_signal_t;