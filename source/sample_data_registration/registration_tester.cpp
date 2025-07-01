#include "registration_tester.hpp"

#include <iostream>
#include "stdafx.h"
#include <strsafe.h>
#include "resource.h"

#include "../csv_reader/tracker.hpp"
#include "projection_matrix.hpp"

/// <summary>
/// Constructor
/// </summary>
registration_tester::registration_tester(Eigen::Matrix<float, 2, 4> projection_matrix) :
	m_nStartTime(0),
	m_nLastCounter(0),
	m_nFramesSinceUpdate(0),
	m_fFreq(0),
	m_nNextStatusTime(0LL),
	m_bSaveScreenshot(false),
	m_pKinectSensor(NULL),
	m_pColorFrameReader(NULL),
	m_pColorRGBX(NULL),
	m_colorImage(cColorHeight, cColorWidth, CV_8UC3),
	m_tracker(
		"W:\\DB_Forschung\\FlexCobot\\11.Unterprojekte\\DS_Nico_Höllerich\\05.Rohdaten\\MMK.Bausteine\\Motion Tracking.no_backup\\Team.8.IDs.15.16.Trial.1.csv",
		projection_matrix
	)
{
	LARGE_INTEGER qpf = { 0 };
	if (QueryPerformanceFrequency(&qpf))
	{
		m_fFreq = double(qpf.QuadPart);
	}


}


/// <summary>
/// Destructor
/// </summary>
registration_tester::~registration_tester()
{
	if (m_pColorRGBX)
	{
		delete[] m_pColorRGBX;
		m_pColorRGBX = NULL;
	}

	// done with depth frame reader
	SafeRelease(m_pColorFrameReader);

	// close the Kinect Sensor
	if (m_pKinectSensor)
	{
		m_pKinectSensor->Close();
	}

	SafeRelease(m_pKinectSensor);
}


/// <summary>
/// Creates the main window and begins processing
/// </summary>
/// <param name="hInstance">handle to the application instance</param>
/// <param name="nCmdShow">whether to display minimized, maximized, or normally</param>
int main()
{
	
	registration_tester application(compute_projection_opti_track_to_rgb());

	cv::namedWindow("color image");



	// Main message loop
	while (true)
	{
		application.Update();

		int c = cv::waitKey(1);
		if (c == 27 || c == 'q' || c == 'Q')
			break;
	}


	cv::destroyWindow("depth image");


	return static_cast<int>(0);
}


/// <summary>
/// Main processing function
/// </summary>
void registration_tester::Update()
{
	if (!m_pColorFrameReader)
	{
		// Get and initialize the default Kinect sensor
		InitializeDefaultSensor();

		if (!m_pColorFrameReader)
			return;
	}

	IColorFrame* pColorFrame = NULL;

	HRESULT hr = m_pColorFrameReader->AcquireLatestFrame(&pColorFrame);

	if (SUCCEEDED(hr))
	{
		INT64 nTime = 0;
		IFrameDescription* pFrameDescription = NULL;
		int nWidth = 0;
		int nHeight = 0;
		USHORT nDepthMinReliableDistance = 0;
		USHORT nDepthMaxDistance = 0;
		UINT nBufferSize = 0;
		ColorImageFormat imageFormat = ColorImageFormat_None;
		RGBQUAD* pBuffer = NULL;

		hr = pColorFrame->get_RelativeTime(&nTime);

		if (SUCCEEDED(hr))
		{
			hr = pColorFrame->get_FrameDescription(&pFrameDescription);
		}

		if (SUCCEEDED(hr))
		{
			hr = pFrameDescription->get_Width(&nWidth);
		}

		if (SUCCEEDED(hr))
		{
			hr = pFrameDescription->get_Height(&nHeight);

			if (!m_pColorRGBX)
				m_pColorRGBX = new RGBQUAD[nWidth * nHeight];
		}


		if (SUCCEEDED(hr))
		{
			hr = pColorFrame->get_RawColorImageFormat(&imageFormat);
		}

		if (SUCCEEDED(hr))
		{
			if (imageFormat == ColorImageFormat_Bgra)
			{
				hr = pColorFrame->AccessRawUnderlyingBuffer(&nBufferSize, reinterpret_cast<BYTE * *>(&pBuffer));
			}
			else if (m_pColorRGBX)
			{
				pBuffer = m_pColorRGBX;
				nBufferSize = cColorWidth * cColorHeight * sizeof(RGBQUAD);
				hr = pColorFrame->CopyConvertedFrameDataToArray(nBufferSize, reinterpret_cast<BYTE*>(pBuffer), ColorImageFormat_Bgra);
			}
			else
			{
				hr = E_FAIL;
			}
		}

		if (SUCCEEDED(hr))
		{
			ProcessColor(nTime, pBuffer, nWidth, nHeight);
		}

		SafeRelease(pFrameDescription);
	}

	SafeRelease(pColorFrame);
}



/// <summary>
/// Initializes the default Kinect sensor
/// </summary>
/// <returns>indicates success or failure</returns>
HRESULT registration_tester::InitializeDefaultSensor()
{
	HRESULT hr;

	hr = GetDefaultKinectSensor(&m_pKinectSensor);
	if (FAILED(hr))
	{
		return hr;
	}

	if (m_pKinectSensor)
	{
		// Initialize the Kinect and get the color reader
		IColorFrameSource* pColorFrameSource = NULL;

		hr = m_pKinectSensor->Open();

		if (SUCCEEDED(hr))
		{
			hr = m_pKinectSensor->get_ColorFrameSource(&pColorFrameSource);
		}

		if (SUCCEEDED(hr))
		{
			hr = pColorFrameSource->OpenReader(&m_pColorFrameReader);
		}

		SafeRelease(pColorFrameSource);
	}

	if (!m_pKinectSensor || FAILED(hr))
	{
		std::cout << "No ready Kinect found!" << std::endl;
		return E_FAIL;
	}

	return hr;
}

/// <summary>
/// Handle new depth data
/// <param name="nTime">timestamp of frame</param>
/// <param name="pBuffer">pointer to frame data</param>
/// <param name="nWidth">width (in pixels) of input image data</param>
/// <param name="nHeight">height (in pixels) of input image data</param>
/// <param name="nMinDepth">minimum reliable depth</param>
/// <param name="nMaxDepth">maximum reliable depth</param>
/// </summary>
void registration_tester::ProcessColor(INT64 nTime, const RGBQUAD* pBuff, int nWidth, int nHeight)
{
	if (!m_nStartTime)
	{
		m_nStartTime = nTime;
	}

	float time = static_cast<float>(nTime - m_nStartTime) / 10000000.f;

	// Make sure we've received valid data
	if (pBuff && (nWidth == cColorWidth) && (nHeight == cColorHeight))
	{

		for (int i = 0; i < cColorWidth * cColorHeight; i++)
		{
			cv::Vec3b pixel;
			pixel[0] = pBuff[i].rgbBlue;
			pixel[1] = pBuff[i].rgbGreen;
			pixel[2] = pBuff[i].rgbRed;
			m_colorImage.at<cv::Vec3b>(cv::Point(cColorWidth - i % cColorWidth - 1, i / cColorWidth)) = pixel;

		}

		for (hand_type hand : {hand_type::TAN_LEFT, hand_type::TAN_RIGHT, hand_type::COLOR_LEFT, hand_type::COLOR_RIGHT}) {
			Eigen::Vector2f pos(m_tracker.get_position_2d(hand, time));
			int hand_num = static_cast<int>(hand) + 1;
			cv::circle(m_colorImage,
				cv::Point(pos(0), pos(1)),
				5, // radius
				cv::Scalar(255 * (hand_num & 4), 255 * (hand_num & 2), 255 * (hand_num & 1)),
				-1, // filled (thickness)
				8); //line_type
		}
		cv::imshow("color image", m_colorImage);

	}
}


