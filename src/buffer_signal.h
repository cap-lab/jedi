#ifndef BUFFER_SIGNAL_H_
#define BUFFER_SIGNAL_H_
#include <mutex>

class BufferSignal {
    public:
        BufferSignal();
        ~BufferSignal();
        void produceSignal();
        void consumeSignal();
        bool isSignalSet();
        void increaseMaxReference();
        void updateSignal(bool value);
    private:
        bool signal;
        int current_reference_count;
        int max_reference_count;
        std::mutex mutex;
};

#endif