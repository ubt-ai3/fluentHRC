#pragma once

#ifndef OPENCV_SERIALIZATION
#define OPENCV_SERIALIZATION

#include <boost/serialization/serialization.hpp>
#include <opencv2/core/core.hpp>

/**
* by Dan Mašek
* on https://stackoverflow.com/a/50072041
*/

namespace boost {
	namespace serialization {

		template<class Archive>
		void serialize(Archive& ar, cv::Mat& mat, const unsigned int)
		{
			int cols, rows, type;
			bool continuous;

			if (Archive::is_saving::value) {
				cols = mat.cols; rows = mat.rows; type = mat.type();
				continuous = mat.isContinuous();
			}

			ar& boost::serialization::make_nvp("cols", cols);
			ar& boost::serialization::make_nvp("rows", rows);
			ar& boost::serialization::make_nvp("type", type);
			ar& boost::serialization::make_nvp("continuous", continuous);

			if (Archive::is_loading::value)
				mat.create(rows, cols, type);

			if (continuous) {
				size_t const data_size(rows * cols * mat.elemSize());
				ar& boost::serialization::make_array(mat.ptr(), data_size);
			}
			else {
				size_t const row_size(cols * mat.elemSize());
				for (int i = 0; i < rows; i++) {
					ar& boost::serialization::make_array(mat.ptr(i), row_size);
				}
			}
		}

		template<class Archive>
		void serialize(Archive& ar, cv::Point& pt, const unsigned int)
		{
			ar& boost::serialization::make_nvp("x", pt.x);
			ar& boost::serialization::make_nvp("y", pt.y);
		}

		template<class Archive, typename _Tp, int cn >
		void serialize(Archive& ar, cv::Vec<_Tp, cn>& vec, const unsigned int)
		{
			ar& boost::serialization::make_nvp("Mat", boost::serialization::base_object<cv::Matx<_Tp, cn, 1>>(vec));
		}

		template<class Archive, typename _Tp, int m, int n >
		void serialize(Archive& ar, cv::Matx<_Tp, m, n>& mat, const unsigned int ver)
		{
			cv::Mat mat2 = cv::Mat(mat);
			serialize(ar, mat2, ver);
		}
	}
}



#endif // !OPENCV_SERIALIZATION
