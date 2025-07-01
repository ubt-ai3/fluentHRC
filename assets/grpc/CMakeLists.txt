add_library(proto-objects OBJECT)

target_sources(proto-objects
    PUBLIC
        proto/debug.proto
        proto/depth_image.proto
        proto/hand_tracking.proto
        proto/meta_data.proto
        proto/object.proto
        proto/object_prototype.proto
        proto/robot.proto
        proto/services.proto
        proto/vertex.proto
)

find_package(gRPC CONFIG REQUIRED)
find_package(Protobuf CONFIG REQUIRED)

target_link_libraries(proto-objects
    PUBLIC
        protobuf::libprotobuf
        gRPC::grpc++
)

set(PROTO_BINARY_DIR "${GrpcDir}/generated")
set(PROTO_IMPORT_DIRS "${GrpcDir}/proto")

target_include_directories(proto-objects PUBLIC "$<BUILD_INTERFACE:${PROTO_BINARY_DIR}>")

protobuf_generate(
    TARGET proto-objects
    IMPORT_DIRS ${PROTO_IMPORT_DIRS}
    PROTOC_OUT_DIR ${PROTO_BINARY_DIR}
)
protobuf_generate(
    TARGET proto-objects
    LANGUAGE grpc
    GENERATE_EXTENSIONS .grpc.pb.h .grpc.pb.cc
    PLUGIN protoc-gen-grpc=$<TARGET_FILE:gRPC::grpc_cpp_plugin>
    IMPORT_DIRS ${PROTO_IMPORT_DIRS}
    PROTOC_OUT_DIR "${PROTO_BINARY_DIR}")