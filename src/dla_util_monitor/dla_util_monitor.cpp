#include "daemon.hpp"

#include <unistd.h>
#include <limits.h>

#include <iostream>
#include <chrono>
#include <string>
#include <cstring>

using namespace daemonpp;
using namespace std::chrono_literals;

constexpr std::string_view dla0_path = "/sys/kernel/debug/nvdla0/firmware/utilization_rate";
constexpr std::string_view dla1_path = "/sys/kernel/debug/nvdla1/firmware/utilization_rate";

class DLAUtilMonitor : public daemon
{
public:
    void on_start(const dconfig& cfg) override {
        util_logging_file.open(this->util_logging_path);
    }

    void on_update() override {
        std::string dla0_util;
        std::string dla1_util;
        dla0_input_file.open(std::string(dla0_path));
        std::getline(dla0_input_file, dla0_util);
        dla1_input_file.open(std::string(dla1_path));
        std::getline(dla1_input_file, dla1_util);
        dla0_input_file.close();
        dla1_input_file.close();
        util_logging_file << dla0_util << ":" <<  dla1_util << std::endl;
        util_logging_file.flush();
    }

    void on_stop() override {
        util_logging_file.close();
    }

    void on_reload(const dconfig& cfg) override {
    }

    void set_util_logging_path(std::string util_logging_path) {
        this->util_logging_path = util_logging_path;
    }
private:
    std::string util_logging_path = "dla_util.log";
    std::ifstream dla0_input_file;
    std::ifstream dla1_input_file;
    std::ofstream util_logging_file;
};

std::string getExecutablePath(const char* argv0) {
    char result[PATH_MAX];
    if (realpath(argv0, result) == NULL) {
        std::cerr << "Error resolving the real path." << std::endl;
        exit(1);
    }
    std::string fullPath(result);
    size_t pos = fullPath.find_last_of("\\/");
    return (std::string::npos == pos) ? "" : fullPath.substr(0, pos);
}

int main(int argc, const char* argv[]) {
    char buffer[PATH_MAX];
    std::string executable_path = getExecutablePath(argv[0]);
    std::string pid_file_path = executable_path + "/dla_util_monitor.pid";

    if (argc >= 4 && std::strcmp(argv[1], "start") == 0) {
        DLAUtilMonitor dla_util_daemon; 
        int update_duration = atoi(argv[2]);
        std::chrono::milliseconds duration(update_duration);
        dla_util_daemon.set_update_duration(duration);
        dla_util_daemon.set_name("dla_util_monitor");
        dla_util_daemon.set_util_logging_path(std::string(argv[3]));
        getcwd(buffer, sizeof(buffer));
        dla_util_daemon.set_cwd(std::string(buffer));
        dla_util_daemon.set_pid_file_path(pid_file_path);
        dla_util_daemon.run(argc, argv);
    } else if (argc >= 2 && std::strcmp(argv[1], "stop") == 0) {
        std::ifstream pid_file(pid_file_path);
        pid_t pid;
        pid_file >> pid;
        pid_file.close();
        if (kill(pid, SIGTERM) == 0) {
            std::cout << "Successfully terminated dla_util_monitor process with PID: " << pid << std::endl;
        } else {
            //std::cerr << "Failed to terminate dla_util_monitor process with PID: " << pid << std::endl;
            //perror("Error");
        }
    }
    else {
        std::cout<<"usage:"<<std::endl;
        std::cout<<"	./dla_util_monitor start [time interval in milliseconds] [log file name]" <<std::endl;
        std::cout<<"	./dla_util_monitor stop" <<std::endl;
        std::cout<<"example:"<<std::endl;
        std::cout<<"	./dla_util_monitor start 100 dla_util.log"<<std::endl;
        std::cout<<"	./dla_util_monitor stop"<<std::endl;
    }
    return EXIT_SUCCESS;
}

