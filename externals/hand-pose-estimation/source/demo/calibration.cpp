#include "calibration.hpp"



#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>

#include <fstream>
#include <iostream>
#include <Kinect.h>
#include <thread>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/projection_matrix.h>

/////////////////////////////////////////////////////////////
//
//
//  Class: kinect2_parameters
//
//
/////////////////////////////////////////////////////////////

template<class Interface>
inline void SafeRelease(Interface*& IRelease)
{
	if (IRelease != NULL) {
		IRelease->Release();
		IRelease = NULL;
	}
}

kinect2_parameters::kinect2_parameters()
{
	filename_ = std::string("kinect2_parameters.xml");

	std::ifstream file(folder_ + filename_);
	if (file.good()) {
		boost::archive::xml_iarchive ia{ file };
		ia >> BOOST_SERIALIZATION_NVP(*this);
	}
	else
	{
		// manually tuned values
		rgb_projection(0,0) =1.064943848e+03;
			rgb_projection(1,0) =0.000000000e+00;
			rgb_projection(2,0) =9.668615112e+02;
			rgb_projection(0,1) =5.000000000e+01;
			rgb_projection(1,1) =0.000000000e+00;
			rgb_projection(2,1) =1.062570923e+03;
			rgb_projection(0,2) =5.504958496e+02;
			rgb_projection(1,2) -5.000000000e+00;
			rgb_projection(2,2) =0.000000000e+00;
			rgb_projection(0,3) =0.000000000e+00;
			rgb_projection(1,3) =1.000000000e+00;
			rgb_projection(2,3) =0.000000000e+00;

		//values from https://github.com/shanilfernando/VRInteraction/tree/master/calibration
			 focal_length_x= 366.193f;
			 focal_length_y= 366.193f;
			 principal_point = Eigen::Vector2f(256.684f, 207.085f);
			 radial_distortion_second_order = 0.0893804f;
			 radial_distortion_fourth_order = -0.272566f;
			 radial_distortion_sixth_order = 0.0958438f;

			 depth_projection <<
				 7.1162970910343665e+02f , 0.f                     , 0.f , 256.684f ,
				 0.f                     , 7.1094384711878797e+02f , 0.f , 207.085f ,
				 5.0990097114150876e+02f , 4.0360735428585957e+02f , 1.f , 0.f
				 ;


		init_thread = std::make_shared<std::thread>([&]()
		{
				
				WAITABLE_HANDLE frameEvent;

				HRESULT result;
				IKinectSensor* sensor;
				ICoordinateMapper* mapper;
				IColorFrameSource* colorSource;
				IColorFrameReader* colorReader;
				IDepthFrameSource* depthSource;
				IDepthFrameReader* depthReader;
				IMultiSourceFrameReader* multiSourceFrameReader;

				int colorWidth;
				int colorHeight;

				int depthWidth;
				int depthHeight;
				UINT16 depthMinDist;
				UINT16 depthMaxDist;


			// Create Sensor Instance
				result = GetDefaultKinectSensor(&sensor);
				if (FAILED(result)) {
					throw std::exception("Exception : GetDefaultKinectSensor()");
				}

				// Open Sensor
				result = sensor->Open();
				if (FAILED(result)) {
					throw std::exception("Exception : IKinectSensor::Open()");
				}

				// Retrieved Coordinate Mapper
				result = sensor->get_CoordinateMapper(&mapper);
				if (FAILED(result)) {
					throw std::exception("Exception : IKinectSensor::get_CoordinateMapper()");
				}

				// Retrieved Color Frame Source
				result = sensor->get_ColorFrameSource(&colorSource);
				if (FAILED(result)) {
					throw std::exception("Exception : IKinectSensor::get_ColorFrameSource()");
				}

				// Retrieved Depth Frame Source
				result = sensor->get_DepthFrameSource(&depthSource);
				if (FAILED(result)) {
					throw std::exception("Exception : IKinectSensor::get_DepthFrameSource()");
				}

				result = sensor->OpenMultiSourceFrameReader(
					FrameSourceTypes::FrameSourceTypes_Depth | FrameSourceTypes::FrameSourceTypes_Color,
					&multiSourceFrameReader);
				if (FAILED(result)) {
					throw std::exception("Exception : IKinectSensor::OpenMultiSourceFrameReader()");
				}

				// Retrieved Color Frame Size
				IFrameDescription* colorDescription;
				result = colorSource->get_FrameDescription(&colorDescription);
				if (FAILED(result)) {
					throw std::exception("Exception : IColorFrameSource::get_FrameDescription()");
				}

				result = colorDescription->get_Width(&colorWidth); // 1920
				if (FAILED(result)) {
					throw std::exception("Exception : IFrameDescription::get_Width()");
				}

				result = colorDescription->get_Height(&colorHeight); // 1080
				if (FAILED(result)) {
					throw std::exception("Exception : IFrameDescription::get_Height()");
				}

				SafeRelease(colorDescription);

				// Retrieved Depth Frame Size
				IFrameDescription* depthDescription;
				result = depthSource->get_FrameDescription(&depthDescription);
				if (FAILED(result)) {
					throw std::exception("Exception : IDepthFrameSource::get_FrameDescription()");
				}

				result = depthDescription->get_Width(&depthWidth); // 512
				if (FAILED(result)) {
					throw std::exception("Exception : IDepthFrameSource::get_Width()");
				}

				result = depthDescription->get_Height(&depthHeight); // 424
				if (FAILED(result)) {
					throw std::exception("Exception : IDepthFrameSource::get_Height()");
				}

				result = depthSource->get_DepthMaxReliableDistance(&depthMaxDist);
				if (FAILED(result)) {
					throw std::exception("Exception : IDepthFrameSource::get_DepthMaxReliableDistance()");
				}

				result = depthSource->get_DepthMinReliableDistance(&depthMinDist);
				if (FAILED(result)) {
					throw std::exception("Exception : IDepthFrameSource::get_DepthMinReliableDistance()");
				}

				SafeRelease(depthDescription);

			
				// Open Color Frame Reader
				result = colorSource->OpenReader(&colorReader);
				if (FAILED(result)) {
					throw std::exception("Exception : IColorFrameSource::OpenReader()");
				}

				// Open Depth Frame Reader
				result = depthSource->OpenReader(&depthReader);
				if (FAILED(result)) {
					throw std::exception("Exception : IDepthFrameSource::OpenReader()");
				}

				// Open combined reader to listen for updates
				HRESULT hr = multiSourceFrameReader->SubscribeMultiSourceFrameArrived(&frameEvent);
				if (FAILED(hr)) {
					throw std::exception("Exception: Couldn't subscribe frame");
				}



				//retrieve first frame

				HANDLE handles[] = { reinterpret_cast<HANDLE>(frameEvent) };
				int idx;

				bool quit = false;
				while (!quit) {
					idx = WaitForMultipleObjects(1, handles, FALSE, 100);
					switch (idx) {
					case WAIT_TIMEOUT:
						continue;
					case WAIT_OBJECT_0:

						IMultiSourceFrameArrivedEventArgs* pFrameArgs = nullptr;
						HRESULT hr = multiSourceFrameReader->GetMultiSourceFrameArrivedEventData(frameEvent, &pFrameArgs);

						CameraIntrinsics intrinsics;
						mapper->GetDepthCameraIntrinsics(&intrinsics);

						focal_length_x = intrinsics.FocalLengthX;
						focal_length_y = intrinsics.FocalLengthY;
						principal_point = Eigen::Vector2f(intrinsics.PrincipalPointX, intrinsics.PrincipalPointY);
						radial_distortion_second_order = intrinsics.RadialDistortionSecondOrder;
						radial_distortion_fourth_order = intrinsics.RadialDistortionFourthOrder;
						radial_distortion_sixth_order = intrinsics.RadialDistortionSixthOrder;
						
						auto pc_color = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
						pc_color->resize(colorHeight* colorWidth);
						pc_color->height = colorHeight;
						pc_color->width = colorWidth;

						std::vector<int> indices_color;

						auto pc_depth = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
						pc_depth->resize(depthHeight * depthWidth);
						pc_depth->height = depthHeight;
						pc_depth->width = depthWidth;

						std::vector<int> indices_depth;

						for(int a = 0; a < 3; a++)
							for(int b = 0; b < 3; b++)
								for(int c = 0; c < 3; c++)
								{
									float x = depthWidth * (0.2f + 0.3f * a) + c; // avoid that points land on the same pixel
									float y = depthHeight * (0.2f + 0.3f * b);
									UINT16 depth = (depthMaxDist - depthMinDist) * (0.2f + 0.3f * c) + depthMinDist;
									
									pcl::PointXYZ point;

									DepthSpacePoint depthSpacePoint = { depthWidth - x - 1,y };
									

									// Coordinate Mapping Depth to Color Space, and Setting PointCloud RGBA
									ColorSpacePoint colorSpacePoint = { 0.0f, 0.0f };
									mapper->MapDepthPointToColorSpace(depthSpacePoint, depth, &colorSpacePoint);
									int colorX = colorWidth  - static_cast<int>(std::floor(colorSpacePoint.X + 0.5f)) -1;
									int colorY = static_cast<int>(std::floor(colorSpacePoint.Y + 0.5f));

									CameraSpacePoint cameraSpacePoint = { 0.0f, 0.0f, 0.0f };
									mapper->MapDepthPointToCameraSpace(depthSpacePoint, depth, &cameraSpacePoint);
									
									if ((0 <= colorX) && (colorX < colorWidth) && (0 <= colorY) && (colorY < colorHeight)) {
										point.x = cameraSpacePoint.X;
										point.y = cameraSpacePoint.Y;
										point.z = cameraSpacePoint.Z;

										pc_color->at(colorX, colorY) = point;									
										indices_color.push_back(colorY* colorWidth + colorX);

										pc_depth->at(static_cast<int>(roundf(x)), static_cast<int>(roundf(y))) = point;
										indices_depth.push_back(static_cast<int>(roundf(y))* depthWidth + static_cast<int>(roundf(x)));
									}
								}

						Eigen::Matrix<float, 3, 4, Eigen::RowMajor> projection;
						pcl::estimateProjectionMatrix<pcl::PointXYZ>(pc_color, projection, indices_color);
						rgb_projection = projection;

						Eigen::Matrix<float, 3, 4, Eigen::RowMajor> proj;
						pcl::estimateProjectionMatrix<pcl::PointXYZ>(pc_depth, proj, indices_depth);
						depth_projection = proj;
						
					}
					quit = true;
				}





			// cleanup
				multiSourceFrameReader->UnsubscribeMultiSourceFrameArrived(frameEvent);
			
				if (sensor) {
					sensor->Close();
				}
				SafeRelease(sensor);
				SafeRelease(mapper);
				SafeRelease(colorSource);
				SafeRelease(colorReader);
				SafeRelease(depthSource);
				SafeRelease(depthReader);
		});
	}
}

kinect2_parameters::~kinect2_parameters()
{
	if (init_thread)
		init_thread->detach();
	
	std::ofstream file(folder_ + filename_);
	boost::archive::xml_oarchive oa{ file };
	const kinect2_parameters& kinect2_params = *this; //passing *this to BOOST_SERIALIZATION_NVP will not work
	oa << BOOST_SERIALIZATION_NVP(kinect2_params);
}
