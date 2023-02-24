#ifndef BINDING_H_
#define BINDING_H_

#include <iostream>

#include <NvInfer.h>

#include "cuda.h"

class TensorAllocator : public nvinfer1::IOutputAllocator {
	public:
		TensorAllocator(bool is_host_allocated, void *_buf, float *_host_buf, uint64_t _size, nvinfer1::DataType _data_type) : is_host_allocated(is_host_allocated), buf(_buf), host_buf(_host_buf), size(_size), data_type(_data_type) {}

		void allocate(uint64_t _size)
		{
			if(is_host_allocated) {
				if(buf != nullptr)
					cudaFreeHost(buf);	
				host_buf = (float *)cuda_make_array_host(_size);
				cudaHostGetDevicePointer(&(buf), host_buf, 0);
			}	
			else {
				if(buf != nullptr)
					cudaFree(buf);
				if(data_type == nvinfer1::DataType::kFLOAT)
					buf = (void *)cuda_make_array(nullptr, _size);
				else
					buf = (void *)cuda_make_array_16(nullptr, _size);
				host_buf = nullptr;
			}
			size = _size;
			is_reallocated = true;
		}

		void* reallocateOutput(char const* tensorName, void* currentMemory, uint64_t size, uint64_t alignment) noexcept override
		{
			// std::cerr<<"["<<__FILE__<<":"<<__func__<<":"<<__LINE__<<"]"<<" size: "<<size<<std::endl;

			is_reallocated = false;

			size = std::max(size, static_cast<uint64_t>(1));
			if (size > this->size) {
				// std::cerr<<"["<<__FILE__<<":"<<__func__<<":"<<__LINE__<<"]"<<" size: "<<size<<", size2:"<<this->size<<std::endl;
				allocate(size);
			}

			return buf;
		}

		void notifyShape(char const* tensorName, nvinfer1::Dims const& dims) noexcept override {}

		void* getBuf() { return buf; }

		float* getHostBuf() { return host_buf; }

		bool getIsReallocated() { return is_reallocated; }

		virtual ~TensorAllocator() {}

	private:
		bool is_host_allocated{false};
		void *buf{nullptr};
		float *host_buf{nullptr};
		uint64_t size{0};
		nvinfer1::DataType data_type{nvinfer1::DataType::kFLOAT};
		bool is_reallocated{false};
};

#endif
