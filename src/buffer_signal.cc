#include "buffer_signal.h"

BufferSignal::BufferSignal()
{
    this->current_reference_count = 0;
    this->max_reference_count = 0;
    this->signal = false;
}

BufferSignal::~BufferSignal()
{
}

void BufferSignal::produceSignal()
{
    std::lock_guard<std::mutex> lock(this->mutex);
	if(this->max_reference_count > 0) {
	    this->current_reference_count = this->max_reference_count;
		this->signal = true;
	}
}

void BufferSignal::consumeSignal()
{
    std::lock_guard<std::mutex> lock(this->mutex);
    if (this->current_reference_count > 0) {
        this->current_reference_count--;
    }

    if (this->current_reference_count == 0) {
        this->signal = false;
    }
}

bool BufferSignal::isSignalSet()
{
    return this->signal;
}

void BufferSignal::increaseMaxReference()
{
    this->max_reference_count++;
}

void BufferSignal::updateSignal(bool value)
{
    if (value == false) {
        consumeSignal();
    } else {
        produceSignal();
    }
}
