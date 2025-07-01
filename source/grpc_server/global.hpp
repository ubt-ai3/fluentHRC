/*
 * Grpc does and will not support dll's in the forseeable future
 * https://github.com/googleapis/google-cloud-cpp/issues/5849
 *
 *#ifndef GRPC_SERVER_API
	#ifdef _DLL
		#ifdef GRPC_SERVER_EXPORT
			#define GRPC_SERVER_API __declspec(dllexport)
		#elif
			#define GRPC_SERVER_API __declspec(dllimport)
		#endif
	#elif
		#define GRPC_SERVER_API
	#endif
#endif*/