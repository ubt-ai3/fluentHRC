/**
* by Jonas Witt
* on https://stackoverflow.com/a/35074759
*/

#ifndef EIGEN_SERIALIZATION
#define EIGEN_SERIALIZATION

#include <Eigen/Core>

namespace boost {
	namespace serialization {

		template<   class Archive,
			class S,
			int Rows_,
			int Cols_,
			int Ops_,
			int MaxRows_,
			int MaxCols_>
			inline void serialize(Archive& ar,
				Eigen::Matrix<S, Rows_, Cols_, Ops_, MaxRows_, MaxCols_>& matrix,
				const unsigned int version)
		{
			int rows = static_cast<int>(matrix.rows());
			int cols = static_cast<int>(matrix.cols());
			ar& make_nvp("rows", rows);
			ar& make_nvp("cols", cols);
			matrix.resize(rows, cols); // no-op if size does not change!

			// always save/load row-major
			for (int r = 0; r < rows; ++r)
				for (int c = 0; c < cols; ++c)
					ar& make_nvp("val", matrix(r, c));
		}

		template<   class Archive,
			class S,
			int Dim_,
			int Mode_,
			int Options_>
			inline void serialize(Archive& ar,
				Eigen::Transform<S, Dim_, Mode_, Options_>& transform,
				const unsigned int version)
		{
			serialize(ar, transform.matrix(), version);
		}

		template<   class Archive,
			class S,
			int Options_>
			inline void serialize(Archive& ar,
				Eigen::Quaternion<S, Options_>& quaternion,
				const unsigned int version)
		{
			ar& make_nvp("x", quaternion.x());
			ar& make_nvp("y", quaternion.y());
			ar& make_nvp("z", quaternion.z());
			ar& make_nvp("w", quaternion.w());
		}
	}
} // namespace boost::serialization


#endif // !EIGEN_SERIALIZATION