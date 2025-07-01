// Kinect2Grabber is pcl::Grabber to retrieve the point cloud data from Kinect v2 using Kinect for Windows SDK 2.x.
// This source code is licensed under the MIT license. Please see the License in License.txt.

#ifndef KINECT2_GRABBER
#define KINECT2_GRABBER

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <memory>
#include <thread>
#include <mutex>
#include <Windows.h>
#include <Kinect.h>

#include <opencv2/core.hpp>

//#include <pcl/io/boost.h>
#include <pcl/io/grabber.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <boost/signals2.hpp>

namespace std
{
    namespace chrono 
    {
        typedef std::chrono::duration<uint64_t, std::ratio<1, 10'000'000>> filetime_period;
    }
}

namespace pcl
{
    struct pcl::PointXYZ;
    struct pcl::PointXYZI;
    struct pcl::PointXYZRGB;
    struct pcl::PointXYZRGBA;
    template <typename T> class pcl::PointCloud;
    
    template<class Interface>
    inline void SafeRelease( Interface *& IRelease )
    {
        if( IRelease != NULL ){
            IRelease->Release();
            IRelease = NULL;
        }
    }

    class Kinect2Grabber : public pcl::Grabber
    {
        public:
            Kinect2Grabber();
            virtual ~Kinect2Grabber() throw ();
            virtual void start();
            virtual void stop();
            virtual bool isRunning() const;
            virtual std::string getName() const;
            virtual float getFramesPerSecond() const;
			std::chrono::duration<float> getRelativeTimestampInSeconds() const;
			virtual void setMirroring(bool enable);
			virtual bool isMirroring() const;

            typedef void ( signal_Kinect2_PointXYZ )( const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& );
            typedef void ( signal_Kinect2_PointXYZI )( const pcl::PointCloud<pcl::PointXYZI>::ConstPtr& );
            typedef void ( signal_Kinect2_PointXYZRGB )( const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr& );
            typedef void ( signal_Kinect2_PointXYZRGBA )( const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr& );
			typedef void ( signal_Kinect2_MatRGB )(const cv::Mat4b&);

        protected:
            boost::signals2::signal<signal_Kinect2_PointXYZ>* signal_PointXYZ;
            boost::signals2::signal<signal_Kinect2_PointXYZI>* signal_PointXYZI;
            boost::signals2::signal<signal_Kinect2_PointXYZRGB>* signal_PointXYZRGB;
            boost::signals2::signal<signal_Kinect2_PointXYZRGBA>* signal_PointXYZRGBA;
            boost::signals2::signal<signal_Kinect2_MatRGB>* signal_MatRGB;

            pcl::PointCloud<pcl::PointXYZ>::Ptr convertDepthToPointXYZ(std::chrono::microseconds timestamp, UINT16* depthBuffer, bool mirror);
            pcl::PointCloud<pcl::PointXYZI>::Ptr convertInfraredDepthToPointXYZI(std::chrono::microseconds timestamp, UINT16* infraredBuffer, UINT16* depthBuffer, bool mirror);
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr convertRGBDepthToPointXYZRGB(std::chrono::microseconds timestamp, RGBQUAD* colorBuffer, UINT16* depthBuffer, bool mirror);
            pcl::PointCloud<pcl::PointXYZRGBA>::Ptr convertRGBADepthToPointXYZRGBA(std::chrono::microseconds timestamp, RGBQUAD* colorBuffer, UINT16* depthBuffer, bool mirror);
			cv::Mat4b convertRGBToMatRGB(std::chrono::microseconds timestamp, RGBQUAD* colorBuffer, bool mirror);

            static std::chrono::filetime_period getAbsFileTime();
            static std::chrono::filetime_period fileTimeToUnix(const std::chrono::filetime_period& fileTime);

            std::thread thread;
            mutable std::mutex mutex;

            void threadFunction();

            bool quit;
            bool running;
			WAITABLE_HANDLE frameEvent;

            HRESULT result;
            IKinectSensor* sensor;
            ICoordinateMapper* mapper;
            IColorFrameSource* colorSource;
            IColorFrameReader* colorReader;
            IDepthFrameSource* depthSource;
            IDepthFrameReader* depthReader;
            IInfraredFrameSource* infraredSource;
            IInfraredFrameReader* infraredReader;
			IMultiSourceFrameReader* multiSourceFrameReader;

            int colorWidth;
            int colorHeight;
            std::vector<RGBQUAD> colorBuffer;

            int depthWidth;
            int depthHeight;
            std::vector<UINT16> depthBuffer;

            int infraredWidth;
            int infraredHeight;
            std::vector<UINT16> infraredBuffer;

			std::vector<DepthSpacePoint> colorToDepth;
			std::vector<CameraSpacePoint> projectedDepthBuffer;

			std::chrono::filetime_period relativeTimestamp;
            INT64 timestamp = std::numeric_limits<INT64>::min();
			bool mirror = false;
    };

    pcl::Kinect2Grabber::Kinect2Grabber()
        : sensor( nullptr )
        , mapper( nullptr )
        , colorSource( nullptr )
        , colorReader( nullptr )
        , depthSource( nullptr )
        , depthReader( nullptr )
        , infraredSource( nullptr )
        , infraredReader( nullptr )
		, multiSourceFrameReader( nullptr )
        , result( S_OK )
        , colorWidth( 1920 )
        , colorHeight( 1080 )
        , colorBuffer()
        , depthWidth( 512 )
        , depthHeight( 424 )
        , depthBuffer()
        , infraredWidth( 512 )
        , infraredHeight( 424 )
        , infraredBuffer()
        , running( false )
        , quit( false )
		, frameEvent( NULL )
		, mirror( false )
        , signal_PointXYZ( nullptr )
        , signal_PointXYZI( nullptr )
        , signal_PointXYZRGB( nullptr )
        , signal_PointXYZRGBA( nullptr )
		, signal_MatRGB( nullptr )
    {
        // Create Sensor Instance
        result = GetDefaultKinectSensor( &sensor );
        if( FAILED( result ) ){
            throw std::exception( "Exception : GetDefaultKinectSensor()" );
        }

        // Open Sensor
        result = sensor->Open();
        if( FAILED( result ) ){
            throw std::exception( "Exception : IKinectSensor::Open()" );
        }

        // Retrieved Coordinate Mapper
        result = sensor->get_CoordinateMapper( &mapper );
        if( FAILED( result ) ){
            throw std::exception( "Exception : IKinectSensor::get_CoordinateMapper()" );
        }

        // Retrieved Color Frame Source
        result = sensor->get_ColorFrameSource( &colorSource );
        if( FAILED( result ) ){
            throw std::exception( "Exception : IKinectSensor::get_ColorFrameSource()" );
        }

        // Retrieved Depth Frame Source
        result = sensor->get_DepthFrameSource( &depthSource );
        if( FAILED( result ) ){
            throw std::exception( "Exception : IKinectSensor::get_DepthFrameSource()" );
        }

        // Retrieved Infrared Frame Source
        result = sensor->get_InfraredFrameSource( &infraredSource );
        if( FAILED( result ) ){
            throw std::exception( "Exception : IKinectSensor::get_InfraredFrameSource()" );
        }

		result = sensor->OpenMultiSourceFrameReader(
			FrameSourceTypes::FrameSourceTypes_Depth | FrameSourceTypes::FrameSourceTypes_Color,
			&multiSourceFrameReader);
		if (FAILED(result)) {
			throw std::exception("Exception : IKinectSensor::OpenMultiSourceFrameReader()");
		}

        // Retrieved Color Frame Size
        IFrameDescription* colorDescription;
        result = colorSource->get_FrameDescription( &colorDescription );
        if( FAILED( result ) ){
            throw std::exception( "Exception : IColorFrameSource::get_FrameDescription()" );
        }

        result = colorDescription->get_Width( &colorWidth ); // 1920
        if( FAILED( result ) ){
            throw std::exception( "Exception : IFrameDescription::get_Width()" );
        }

        result = colorDescription->get_Height( &colorHeight ); // 1080
        if( FAILED( result ) ){
            throw std::exception( "Exception : IFrameDescription::get_Height()" );
        }

        SafeRelease( colorDescription );

        // To Reserve Color Frame Buffer
        colorBuffer.resize( colorWidth * colorHeight );

        // Retrieved Depth Frame Size
        IFrameDescription* depthDescription;
        result = depthSource->get_FrameDescription( &depthDescription );
        if( FAILED( result ) ){
            throw std::exception( "Exception : IDepthFrameSource::get_FrameDescription()" );
        }

        result = depthDescription->get_Width( &depthWidth ); // 512
        if( FAILED( result ) ){
            throw std::exception( "Exception : IFrameDescription::get_Width()" );
        }

        result = depthDescription->get_Height( &depthHeight ); // 424
        if( FAILED( result ) ){
            throw std::exception( "Exception : IFrameDescription::get_Height()" );
        }

        SafeRelease( depthDescription );

        // To Reserve Depth Frame Buffer
        depthBuffer.resize( depthWidth * depthHeight );

        // Retrieved Infrared Frame Size
        IFrameDescription* infraredDescription;
        result = infraredSource->get_FrameDescription( &infraredDescription );
        if( FAILED( result ) ){
            throw std::exception( "Exception : IInfraredFrameSource::get_FrameDescription()" );
        }

        result = infraredDescription->get_Width( &infraredWidth ); // 512
        if( FAILED( result ) ){
            throw std::exception( "Exception : IFrameDescription::get_Width()" );
        }

        result = infraredDescription->get_Height( &infraredHeight ); // 424
        if( FAILED( result ) ){
            throw std::exception( "Exception : IFrameDescription::get_Height()" );
        }

        SafeRelease( infraredDescription );

        // To Reserve Infrared Frame Buffer
        infraredBuffer.resize( infraredWidth * infraredHeight );

		colorToDepth.resize(colorWidth* colorHeight);

		projectedDepthBuffer.resize(depthWidth* depthHeight);

        signal_PointXYZ = createSignal<signal_Kinect2_PointXYZ>();
        signal_PointXYZI = createSignal<signal_Kinect2_PointXYZI>();
        signal_PointXYZRGB = createSignal<signal_Kinect2_PointXYZRGB>();
        signal_PointXYZRGBA = createSignal<signal_Kinect2_PointXYZRGBA>();
		signal_MatRGB = createSignal<signal_Kinect2_MatRGB>();
    }

    pcl::Kinect2Grabber::~Kinect2Grabber() throw()
    {
        stop();

        disconnect_all_slots<signal_Kinect2_PointXYZ>();
        disconnect_all_slots<signal_Kinect2_PointXYZI>();
        disconnect_all_slots<signal_Kinect2_PointXYZRGB>();
        disconnect_all_slots<signal_Kinect2_PointXYZRGBA>();

        if (thread.joinable())
            thread.join();

        // End Processing
        if( sensor ){
            sensor->Close();
        }
        SafeRelease( sensor );
        SafeRelease( mapper );
        SafeRelease( colorSource );
        SafeRelease( colorReader );
        SafeRelease( depthSource );
        SafeRelease( depthReader );
        SafeRelease( infraredSource );
        SafeRelease( infraredReader );
    }

    void pcl::Kinect2Grabber::start()
    {
        // Open Color Frame Reader
        result = colorSource->OpenReader( &colorReader );
        if( FAILED( result ) ){
            throw std::exception( "Exception : IColorFrameSource::OpenReader()" );
        }

        // Open Depth Frame Reader
        result = depthSource->OpenReader( &depthReader );
        if( FAILED( result ) ){
            throw std::exception( "Exception : IDepthFrameSource::OpenReader()" );
        }

        // Open Infrared Frame Reader
        result = infraredSource->OpenReader( &infraredReader );
        if( FAILED( result ) ){
            throw std::exception( "Exception : IInfraredFrameSource::OpenReader()" );
        }

		// Open combined reader to listen for updates
		HRESULT hr = multiSourceFrameReader->SubscribeMultiSourceFrameArrived(&frameEvent);
		if (FAILED(hr)) {
			throw std::exception("Exception: Couldn't subscribe frame");
		}

        quit = false;
        running = true;

        thread = std::thread( &Kinect2Grabber::threadFunction, this );
    }

    void pcl::Kinect2Grabber::stop()
    {
        std::unique_lock<std::mutex> lock( mutex );

        quit = true;
        running = false;

		multiSourceFrameReader->UnsubscribeMultiSourceFrameArrived(frameEvent);


		frameEvent = NULL;

        lock.unlock();

        if (thread.joinable())
            thread.join();
    }

    bool pcl::Kinect2Grabber::isRunning() const
    {
        std::unique_lock<std::mutex> lock( mutex );

        return running;

        lock.unlock();
    }

    std::string pcl::Kinect2Grabber::getName() const
    {
        return std::string( "Kinect2Grabber" );
    }

    float pcl::Kinect2Grabber::getFramesPerSecond() const
    {
        return 30.0f;
    }

	std::chrono::duration<float> pcl::Kinect2Grabber::getRelativeTimestampInSeconds() const
	{
		std::unique_lock<std::mutex> lock(mutex);

		return std::chrono::duration_cast<std::chrono::duration<float>>(relativeTimestamp);

		lock.unlock();
	}

	void pcl::Kinect2Grabber::setMirroring(bool enable) 
	{
		std::unique_lock<std::mutex> lock(mutex);

		mirror = enable;

		lock.unlock();
	}

	bool pcl::Kinect2Grabber::isMirroring() const
	{
		return mirror;
	}

    void pcl::Kinect2Grabber::threadFunction()
    {
		HANDLE handles[] = { reinterpret_cast<HANDLE>(frameEvent) };
		int idx;

		while (!quit) {
			idx = WaitForMultipleObjects(1, handles, FALSE, 100);
			switch (idx) {
			case WAIT_TIMEOUT:
				continue;
			case WAIT_OBJECT_0:
                std::chrono::filetime_period abs_file_time = getAbsFileTime();

				IMultiSourceFrameArrivedEventArgs* pFrameArgs = nullptr;
				HRESULT hr = multiSourceFrameReader->GetMultiSourceFrameArrivedEventData(frameEvent, &pFrameArgs);


				// Acquire Latest Color Frame
				IColorFrame* colorFrame = nullptr;
				result = colorReader->AcquireLatestFrame(&colorFrame);
				INT64 color_stamp = 0, depth_stamp = 0;
				if (SUCCEEDED(result)) {
					// Retrieved Color Data
					result = colorFrame->CopyConvertedFrameDataToArray(colorBuffer.size() * sizeof(RGBQUAD), reinterpret_cast<BYTE*>(&colorBuffer[0]), ColorImageFormat::ColorImageFormat_Bgra);

					result &= colorFrame->get_RelativeTime(&color_stamp);

					if (FAILED(result)) {
						throw std::exception("Exception : IColorFrame::CopyConvertedFrameDataToArray()");
					}

				}
				SafeRelease(colorFrame);

				// Acquire Latest Depth Frame
				IDepthFrame* depthFrame = nullptr;
				result = depthReader->AcquireLatestFrame(&depthFrame);
				if (SUCCEEDED(result)) {
					// Retrieved Depth Data
					result = depthFrame->CopyFrameDataToArray(depthBuffer.size(), &depthBuffer[0]);
                    result &= depthFrame->get_RelativeTime(&depth_stamp);
					if (FAILED(result)) {
						throw std::exception("Exception : IDepthFrame::CopyFrameDataToArray()");
					}
				}
				SafeRelease(depthFrame);

                INT64 stamp = std::max(color_stamp, depth_stamp);
                if (!stamp || stamp <= timestamp)
                    continue;

                std::unique_lock<std::mutex> lock(mutex);

                if (quit)
                    return;

                timestamp = stamp;
                relativeTimestamp = std::chrono::filetime_period(timestamp);

                std::chrono::microseconds micro_timestamp = 
                    std::chrono::duration_cast<std::chrono::microseconds>(relativeTimestamp);

                /*
                 * Replace with see below?
                 * std::chrono::microseconds micro_timestamp = 
                 *      std::chrono::duration_cast<std::chrono::microseconds>(
                 *      fileTimeToUnix(abs_file_time));
                 */

				// Acquire Latest Infrared Frame
				IInfraredFrame* infraredFrame = nullptr;
				result = infraredReader->AcquireLatestFrame(&infraredFrame);
				if (SUCCEEDED(result)) {
					// Retrieved Infrared Data
					result = infraredFrame->CopyFrameDataToArray(infraredBuffer.size(), &infraredBuffer[0]);
					if (FAILED(result)) {
						throw std::exception("Exception : IInfraredFrame::CopyFrameDataToArray()");
					}
				}
				SafeRelease(infraredFrame);

				bool mirror = this->mirror;

				lock.unlock();

				if (signal_PointXYZ->num_slots() > 0) {
					signal_PointXYZ->operator()(convertDepthToPointXYZ(micro_timestamp, &depthBuffer[0], mirror));
				}

				if (signal_PointXYZI->num_slots() > 0) {
					signal_PointXYZI->operator()(convertInfraredDepthToPointXYZI(micro_timestamp, &infraredBuffer[0], &depthBuffer[0], mirror));
				}

				if (signal_PointXYZRGB->num_slots() > 0) {
					signal_PointXYZRGB->operator()(convertRGBDepthToPointXYZRGB(micro_timestamp, &colorBuffer[0], &depthBuffer[0], mirror));
				}

				if (signal_PointXYZRGBA->num_slots() > 0) {
					signal_PointXYZRGBA->operator()(convertRGBADepthToPointXYZRGBA(micro_timestamp, &colorBuffer[0], &depthBuffer[0], mirror));
				}

				if (signal_MatRGB->num_slots() > 0) {
					signal_MatRGB->operator()(convertRGBToMatRGB(micro_timestamp, &colorBuffer[0], mirror));
				}
			}
		}
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl::Kinect2Grabber::convertDepthToPointXYZ(std::chrono::microseconds timestamp, UINT16* depthBuffer, bool mirror )
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud( new pcl::PointCloud<pcl::PointXYZ>() );

        cloud->width = static_cast<uint32_t>( depthWidth );
        cloud->height = static_cast<uint32_t>( depthHeight );
        cloud->is_dense = false;
		cloud->header.stamp = timestamp.count();

        cloud->points.resize( cloud->height * cloud->width );

        pcl::PointXYZ* pt = &cloud->points[0];
        for( int y = 0; y < depthHeight; y++ ){
            for( int x = 0; x < depthWidth; x++, pt++ ){
                pcl::PointXYZ point;

                DepthSpacePoint depthSpacePoint = { static_cast<float>( x ), static_cast<float>( y ) };
                UINT16 depth = depthBuffer[y * depthWidth + (mirror ? (depthWidth - x - 1) : x)];

                // Coordinate Mapping Depth to Camera Space, and Setting PointCloud XYZ
                CameraSpacePoint cameraSpacePoint = { 0.0f, 0.0f, 0.0f };
                mapper->MapDepthPointToCameraSpace( depthSpacePoint, depth, &cameraSpacePoint );
                point.x = cameraSpacePoint.X;
                point.y = cameraSpacePoint.Y;
                point.z = cameraSpacePoint.Z;

                *pt = point;
            }
        }

        return cloud;
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr pcl::Kinect2Grabber::convertInfraredDepthToPointXYZI(std::chrono::microseconds timestamp, UINT16* infraredBuffer, UINT16* depthBuffer, bool mirror)
    {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud( new pcl::PointCloud<pcl::PointXYZI>() );

        cloud->width = static_cast<uint32_t>( depthWidth );
        cloud->height = static_cast<uint32_t>( depthHeight );
        cloud->is_dense = false;
		cloud->header.stamp = timestamp.count();

        cloud->points.resize( cloud->height * cloud->width );

        pcl::PointXYZI* pt = &cloud->points[0];
        for( int y = 0; y < depthHeight; y++ ){
            for( int x = 0; x < depthWidth; x++, pt++ ){
                pcl::PointXYZI point;

                DepthSpacePoint depthSpacePoint = { static_cast<float>( x ), static_cast<float>( y ) };
                UINT16 depth = depthBuffer[y * depthWidth + (mirror ? (depthWidth - x - 1) : x)];

                // Setting PointCloud Intensity
                point.intensity = static_cast<float>( infraredBuffer[y * depthWidth + (mirror ? (depthWidth - x - 1) : x)] );

                // Coordinate Mapping Depth to Camera Space, and Setting PointCloud XYZ
                CameraSpacePoint cameraSpacePoint = { 0.0f, 0.0f, 0.0f };
                mapper->MapDepthPointToCameraSpace( depthSpacePoint, depth, &cameraSpacePoint );
                point.x = cameraSpacePoint.X;
                point.y = cameraSpacePoint.Y;
                point.z = cameraSpacePoint.Z;

                *pt = point;
            }
        }

        return cloud;
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl::Kinect2Grabber::convertRGBDepthToPointXYZRGB(std::chrono::microseconds timestamp, RGBQUAD* colorBuffer, UINT16* depthBuffer, bool mirror)
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());

        cloud->width = static_cast<uint32_t>(depthWidth);
        cloud->height = static_cast<uint32_t>(depthHeight);
        cloud->is_dense = false;
        cloud->header.stamp = timestamp.count();

        cloud->points.resize(cloud->height * cloud->width);

        pcl::PointXYZRGB* pt = &cloud->points[0];
        for (int y = 0; y < depthHeight; y++) {
            for (int x = 0; x < depthWidth; x++, pt++) {
                pcl::PointXYZRGB point;

                DepthSpacePoint depthSpacePoint = { static_cast<float>(x), static_cast<float>(y) };
                UINT16 depth = depthBuffer[y * depthWidth + (mirror ? (depthWidth - x - 1) : x)];

                // Coordinate Mapping Depth to Color Space, and Setting PointCloud RGBA
                ColorSpacePoint colorSpacePoint = { 0.0f, 0.0f };
                mapper->MapDepthPointToColorSpace(depthSpacePoint, depth, &colorSpacePoint);
                int colorX = static_cast<int>(std::floor(colorSpacePoint.X + 0.5f));
                int colorY = static_cast<int>(std::floor(colorSpacePoint.Y + 0.5f));
                if ((0 <= colorX) && (colorX < colorWidth) && (0 <= colorY) && (colorY < colorHeight)) {
                    RGBQUAD color = colorBuffer[colorY * colorWidth + (mirror ? (colorWidth - colorX - 1) : colorX)];
                    point.b = color.rgbBlue;
                    point.g = color.rgbGreen;
                    point.r = color.rgbRed;
                }

                // Coordinate Mapping Depth to Camera Space, and Setting PointCloud XYZ
                CameraSpacePoint cameraSpacePoint = { 0.0f, 0.0f, 0.0f };
                mapper->MapDepthPointToCameraSpace(depthSpacePoint, depth, &cameraSpacePoint);
                if ((0 <= colorX) && (colorX < colorWidth) && (0 <= colorY) && (colorY < colorHeight)) {
                    point.x = cameraSpacePoint.X;
                    point.y = cameraSpacePoint.Y;
                    point.z = cameraSpacePoint.Z;
                }

                *pt = point;
            }
        }

        return cloud;
    }

    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr pcl::Kinect2Grabber::convertRGBADepthToPointXYZRGBA(std::chrono::microseconds timestamp, RGBQUAD* colorBuffer, UINT16* depthBuffer, bool mirror)
    {
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud( new pcl::PointCloud<pcl::PointXYZRGBA>() );

        cloud->width = static_cast<uint32_t>( depthWidth );
        cloud->height = static_cast<uint32_t>( depthHeight );
        cloud->is_dense = false;
		cloud->header.stamp = timestamp.count();

        cloud->points.resize( cloud->height * cloud->width );

        pcl::PointXYZRGBA* pt = &cloud->points[0];
        for( int y = 0; y < depthHeight; y++ ){
            for( int x = 0; x < depthWidth; x++, pt++ ){
                pcl::PointXYZRGBA point;

                DepthSpacePoint depthSpacePoint = { static_cast<float>( x ), static_cast<float>( y ) };
                UINT16 depth = depthBuffer[y * depthWidth + (mirror ? (depthWidth - x - 1) : x)];

                // Coordinate Mapping Depth to Color Space, and Setting PointCloud RGBA
                ColorSpacePoint colorSpacePoint = { 0.0f, 0.0f };
                mapper->MapDepthPointToColorSpace( depthSpacePoint, depth, &colorSpacePoint );
                int colorX = static_cast<int>( std::floor( colorSpacePoint.X + 0.5f ) );
                int colorY = static_cast<int>( std::floor( colorSpacePoint.Y + 0.5f ) );
                if( ( 0 <= colorX ) && ( colorX < colorWidth ) && ( 0 <= colorY ) && ( colorY < colorHeight ) ){
                    RGBQUAD color = colorBuffer[colorY * colorWidth + (mirror ? (colorWidth - colorX - 1) : colorX)];
                    point.b = color.rgbBlue;
                    point.g = color.rgbGreen;
                    point.r = color.rgbRed;
                    point.a = color.rgbReserved;
                }

                // Coordinate Mapping Depth to Camera Space, and Setting PointCloud XYZ
                CameraSpacePoint cameraSpacePoint = { 0.0f, 0.0f, 0.0f };
                mapper->MapDepthPointToCameraSpace( depthSpacePoint, depth, &cameraSpacePoint );
                if( ( 0 <= colorX ) && ( colorX < colorWidth ) && ( 0 <= colorY ) && ( colorY < colorHeight ) ){
                    point.x = cameraSpacePoint.X;
                    point.y = cameraSpacePoint.Y;
                    point.z = cameraSpacePoint.Z;
                }

                *pt = point;
            }
        }

        return cloud;
    }
	
	cv::Mat4b pcl::Kinect2Grabber::convertRGBToMatRGB(std::chrono::microseconds timestamp, RGBQUAD* colorBuffer, bool mirror)
	{
		cv::Mat4b mat(colorHeight, colorWidth);

		memcpy(mat.data, colorBuffer, 4 * colorWidth * colorHeight);

		if(!mirror)
			cv::flip(mat, mat, +1);

		return mat;
	}

    std::chrono::filetime_period pcl::Kinect2Grabber::getAbsFileTime()
    {
        FILETIME abs_file_time;
        #if (_WIN32_WINNT >= _WIN32_WINNT_WIN8)
            GetSystemTimePreciseAsFileTime(&abs_file_time);
        #else
            GetSystemTimeAsFileTime(&abs_file_time);
        #endif

        ULARGE_INTEGER abs_int_timestamp;
        abs_int_timestamp.HighPart = abs_file_time.dwHighDateTime;
        abs_int_timestamp.LowPart = abs_file_time.dwLowDateTime;

        return std::chrono::filetime_period(abs_int_timestamp.QuadPart);
    }

    std::chrono::filetime_period pcl::Kinect2Grabber::fileTimeToUnix(const std::chrono::filetime_period& fileTime)
    {
        return fileTime - std::chrono::seconds(11'644'473'600);
    }
}

#endif KINECT2_GRABBER

