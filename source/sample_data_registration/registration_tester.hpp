#pragma once

#ifndef SAMPLE_DATA_REGISTRATION
#define SAMPLE_DATA_REGISTRATION

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include "resource.h"
#include "stdafx.h"

#include "../csv_reader/tracker.hpp"

class registration_tester
{
	static const int        cColorWidth = 1920;
	static const int        cColorHeight = 1080;

public:
	/// <summary>
	/// Constructor
	/// </summary>
	registration_tester(Eigen::Matrix<float, 2, 4> projection_matrix);

	/// <summary>
	/// Destructor
	/// </summary>
	~registration_tester();


	/// <summary>
/// Main processing function
/// </summary>
	void                    Update();


private:
	INT64                   m_nStartTime;
	INT64                   m_nLastCounter;
	double                  m_fFreq;
	INT64                   m_nNextStatusTime;
	DWORD                   m_nFramesSinceUpdate;
	bool                    m_bSaveScreenshot;

	// Current Kinect
	IKinectSensor*			m_pKinectSensor;

	// Color reader
	IColorFrameReader*		m_pColorFrameReader;

	RGBQUAD*				m_pColorRGBX;

	cv::Mat					m_colorImage;
	tracker					m_tracker;




	/// <summary>
	/// Initializes the default Kinect sensor
	/// </summary>
	/// <returns>S_OK on success, otherwise failure code</returns>
	HRESULT                 InitializeDefaultSensor();

	/// <summary>
	/// Handle new color data
	/// <param name="nTime">timestamp of frame</param>
	/// <param name="pBuffer">pointer to frame data</param>
	/// <param name="nWidth">width (in pixels) of input image data</param>
	/// <param name="nHeight">height (in pixels) of input image data</param>

	/// </summary>
	void                    ProcessColor(INT64 nTime, const RGBQUAD* pBuffer, int nHeight, int nWidth);


};


#endif // !SAMPLE_DATA_REGISTRATION