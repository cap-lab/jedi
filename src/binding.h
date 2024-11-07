#ifndef BINDING_H_
#define BINDING_H_

#include <cassert>
#include <iostream>

#include <NvInfer.h>

#include <cuda_runtime_api.h>

#include "cuda_jedi.h"

int getDataTypeSize(nvinfer1::DataType data_type);

class TensorAllocator : public nvinfer1::IOutputAllocator {
	public:
		TensorAllocator(bool is_host_allocated, void *_buf, void *_host_buf, uint64_t _size, nvinfer1::DataType _data_type) : is_host_allocated(is_host_allocated), buf(_buf), host_buf(_host_buf), size(_size), data_type(_data_type) {}

#if NV_TENSORRT_MAJOR > 8
		void allocateWithStream(uint64_t _size, cudaStream_t stream)
		{
			cudaError_t status;

			if(is_host_allocated) {
				if(buf != nullptr)
					cudaFreeHost(host_buf);

				status = cudaHostAlloc((void **) &host_buf, _size, cudaHostAllocMapped);
				check_error(status);
				status = cudaHostGetDevicePointer((void **) &(buf), host_buf, 0);
				check_error(status);
			}
			else {
				if(buf != nullptr)
					cudaFree(buf);

				status = cudaMallocAsync((void **) &buf, _size, stream);
				check_error(status);
				host_buf = nullptr;
			}
			this->size = _size;
			is_reallocated = true;

			//std::cerr<<"["<<__FILE__<<":"<<__func__<<":"<<__LINE__<<"]"<< "ptr: "<< this  << ", size(in): "<< this->size <<std::endl;
		}

		void* reallocateOutputAsync(char const* tensorName, void* currentMemory, uint64_t size, uint64_t alignment, cudaStream_t stream) noexcept override
		{
			int data_type_size = getDataTypeSize(data_type);
			uint64_t allocatedSizeInBytes;
			//std::cerr<<"["<<__FILE__<<":"<<__func__<<":"<<__LINE__<<"]"<<" tensorName: "<<tensorName<<std::endl;
			//std::cerr<<"["<<__FILE__<<":"<<__func__<<":"<<__LINE__<<"]"<<" size: "<<size<<std::endl;
			//std::cerr<<"["<<__FILE__<<":"<<__func__<<":"<<__LINE__<<"]"<<" alignment: "<<alignment<<std::endl;

			is_reallocated = false;
			size = std::max(size, static_cast<uint64_t>(1));
			if ((size % alignment) != 0) {
				allocatedSizeInBytes = (size / alignment) * (alignment + 1);
			}
			else {
				allocatedSizeInBytes = size;
			}

			if (allocatedSizeInBytes > this->size) {
				std::cerr<<"["<<__FILE__<<":"<<__func__<<":"<<__LINE__<<"]"<<" tensorName: "<<tensorName<<", ptr: "<<this<<std::endl;
				std::cerr<<"["<<__FILE__<<":"<<__func__<<":"<<__LINE__<<"]"<<" size: "<<allocatedSizeInBytes<<", size2:"<< this->size <<std::endl;
				assert(allocatedSizeInBytes % data_type_size == 0);
				allocateWithStream(allocatedSizeInBytes, stream);
			}

			return buf;
		}
#else
		void allocate(uint64_t _size)
		{
			int data_type_size = getDataTypeSize(data_type);

			if(is_host_allocated) {
				if(buf != nullptr)
					cudaFreeHost(host_buf);

				host_buf = cuda_make_generic_array_host(_size, data_type_size);
				cudaHostGetDevicePointer((void **) &(buf), host_buf, 0);
			}	
			else {
				if(buf != nullptr)
					cudaFree(buf);

				buf = cuda_make_generic_array(nullptr, _size, data_type_size);
				host_buf = nullptr;
			}
			this->size = _size * data_type_size;
			is_reallocated = true;

			//std::cerr<<"["<<__FILE__<<":"<<__func__<<":"<<__LINE__<<"]"<< "ptr: "<< this  << ", size(in): "<< this->size <<std::endl;
		}

		void* reallocateOutput(char const* tensorName, void* currentMemory, uint64_t size, uint64_t alignment) noexcept override
		{
			int data_type_size = getDataTypeSize(data_type);
			uint64_t allocatedSizeInBytes;
			//std::cerr<<"["<<__FILE__<<":"<<__func__<<":"<<__LINE__<<"]"<<" tensorName: "<<tensorName<<std::endl;
			//std::cerr<<"["<<__FILE__<<":"<<__func__<<":"<<__LINE__<<"]"<<" size: "<<size<<std::endl;
			//std::cerr<<"["<<__FILE__<<":"<<__func__<<":"<<__LINE__<<"]"<<" alignment: "<<alignment<<std::endl;

			is_reallocated = false;
			size = std::max(size, static_cast<uint64_t>(1));
			if ((size % alignment) != 0) {
				allocatedSizeInBytes = (size / alignment) * (alignment + 1);
			}
			else {
				allocatedSizeInBytes = size;
			}

			if (allocatedSizeInBytes > this->size) {
				std::cerr<<"["<<__FILE__<<":"<<__func__<<":"<<__LINE__<<"]"<<" tensorName: "<<tensorName<<", ptr: "<<this<<std::endl;
				std::cerr<<"["<<__FILE__<<":"<<__func__<<":"<<__LINE__<<"]"<<" size: "<<allocatedSizeInBytes<<", size2:"<< this->size * data_type_size<<std::endl;
				assert(allocatedSizeInBytes % data_type_size == 0);
				allocate(allocatedSizeInBytes / data_type_size);
			}

			return buf;
		}
#endif


		void notifyShape(char const* tensorName, nvinfer1::Dims const& dims) noexcept override {}

		void* getBuf() { return buf; }

		void* getHostBuf() { return host_buf; }

		bool getIsReallocated() { return is_reallocated; }

		uint64_t getSize() { return size;  }

		virtual ~TensorAllocator() {}

	private:
		bool is_host_allocated{false};
		void *buf{nullptr};
		void *host_buf{nullptr};
		uint64_t size{0};
		nvinfer1::DataType data_type{nvinfer1::DataType::kFLOAT};
		bool is_reallocated{false};
};

#endif
