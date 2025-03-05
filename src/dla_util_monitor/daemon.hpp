//
// Created by bader on 4/18/23.
//
#pragma once
#include <iostream>
#include <unistd.h>
#include <cerrno>
#include <cstring>
#include <string>
#include <csignal>
#include <sys/stat.h>
#include <stdexcept>
#include <chrono>
#include <thread>
#include <atomic>
#include <condition_variable>
#include "dlog.hpp"
#include "dconfig.hpp"

namespace daemonpp {
  class daemon {
    private:
        static daemon* instance;

    public:
       /** Construct a new daemon process.
         * @note: only one daemon per application is possible.
         * @param name: name of daemon process
         * @param cwd: daemon current working directory, root "/" directory by default.
         * @param update_duration: duration to sleep before waking up the on_update() callback every time, default 10 seconds
         */
        daemon(const std::string &name, const std::string& cwd = "/", const std::chrono::high_resolution_clock::duration& update_duration = std::chrono::seconds(10), const std::string& pid_file_path="dla_util_monitor.pid") :
        m_name(name), m_cwd(cwd), m_update_duration(update_duration), m_is_running(false), m_pid_file_path(pid_file_path), m_exit_code(EXIT_SUCCESS)
        {
          if(instance) {
            //dlog::error("Only one daemon instance is possible.");
            std::exit(EXIT_FAILURE);
          }
          instance = this;
        }

        daemon() : m_name("<unknown_daemon>"), m_cwd("/"), m_update_duration(std::chrono::seconds(10)), m_is_running(false), m_pid_file_path("dla_util_monitor.pid")
        {
          if(instance) {
            //dlog::error("Only one daemon instance is possible.");
            std::exit(EXIT_FAILURE);
          }
          instance = this;
        }

        void run(const int argc, const char* argv[])
        {
          if(m_is_running.load()) {
           // dlog::error("Daemon '"+m_name+"' is already running.");
            return;
          }

          // Daemonize this program by forking parent process.
          daemonize();

          // Mark as running (better to have it before on_start() as user may call stop() inside on_start()).
          m_is_running = true;
          on_start(dconfig::from_file(m_config_file));
          while(m_is_running.load())
          {
            on_update();
            // On long sleeps, if we want to exit we need a cv to wake up the thread from sleep to carry on exiting.
            std::unique_lock<std::mutex> lock(m_mutex);
            m_update_cv.wait_for(lock, m_update_duration, [this]() {
              return !m_is_running.load();
            });
          }
          on_stop();
        }

        void stop(std::int32_t code = EXIT_SUCCESS)
        {
          m_exit_code = code;
          m_is_running.store(false);
          m_update_cv.notify_all();
        }

        ~daemon() {
          //dlog::shutdown();
          // Terminate the child process when the daemon completes (loop stopped)
          // note that calling std::exit() inside the run function will not call dtor.
          std::exit(m_exit_code);
        }

    protected: // Callbacks
        /**
         * @brief Called once on daemon starts
         * @scenarios:
         *  - when systems starts
         *  - when your run `$ systemctl start your_daemon` manually
         * @param cfg: Installed daemon config file
         * Initialize your code here...
         */
        virtual void on_start(const dconfig& cfg) = 0;

        /**
         * @brief Called every DURATION which was set by set_update_duration(DURATION).
         * Update your code here...
         */
        virtual void on_update() = 0;

        /**
         * @brief Called once before daemon is about to exit.
         * @scenarios:
         *  - when you call stop(exit_code)
         *  - when you run `$ systemctl stop your_daemon` manually
         *  - when the system kills your daemon for some reason
         * Cleanup your code here...
         */
        virtual void on_stop() = 0;

        /**
         * @brief Called once when daemon's config or service files are updated.
         * @scenarios:
         *  - when you run `$ systemctl daemon-reload` after you have changed your .conf or .service files (after reinstalling your daemon with `$ sudo make install` for example)
         * Reinitialize your code here...
         */
        virtual void on_reload(const dconfig& cfg) = 0;

    private:
        static void signal_handler(std::int32_t sig) {
          //dlog::info("Signal " + std::to_string(sig) + " received.");
          switch (sig) {
            // daemon.service handler: ExecStop=/bin/kill -s SIGTERM $MAINPID
            // When daemon is stopped, system sends SIGTERM first, if daemon didn't respond during 90 seconds, it will send a SIGKILL signal
            case SIGTERM:
            case SIGKILL: {
              instance->stop();
              break;
            }
            // daemon.service handler: ExecReload=/bin/kill -s SIGHUP $MAINPID
            // When daemon is reloaded due updates in .service or .conf, system sends SIGHUB signal.
            case SIGHUP: {
              instance->on_reload(dconfig::from_file(instance->m_config_file));
              break;
            }
            default:
              break;
          }
        }

    public: // getters & setters
        void set_update_duration(const std::chrono::high_resolution_clock::duration& duration) noexcept {
          m_update_duration = duration;
        }
        const std::chrono::high_resolution_clock::duration& get_update_duration() const noexcept {
          return m_update_duration;
        }

        void set_name(const std::string& daemon_name) noexcept {
          m_name = daemon_name;
        }
        const std::string& get_name() const noexcept { return m_name; }

        void set_cwd(const std::string& current_working_dir) noexcept {
          // Change the current working directory to a directory guaranteed to exist, provided by the user.
          if(chdir(current_working_dir.c_str()) < 0)
          {
            //dlog::error("Could not change current working directory to '" +current_working_dir+ "': " + std::string(std::strerror(errno)));
            return;
          }
          m_cwd = current_working_dir;
        }

        void set_pid_file_path(const std::string& pid_file_path) noexcept {
          m_pid_file_path = pid_file_path;
        }

        const std::string& get_cwd() const noexcept { return m_cwd; }

        pid_t get_pid() const noexcept { return m_pid; }
        pid_t get_sid() const noexcept { return m_sid; }

    private:
        /**
         * Daemonize this program
         * @note: It is also possible to use glibc function deamon()
         * at this point, but it is useful to customize your daemon. Like
         * for example handle signals, set working directory..
         */
        void daemonize() {
          // Fork off the parent process https://linux.die.net/man/3/fork
          m_pid = fork();
          // Success: The parent process continues with a process ID greater than 0
          if (m_pid > 0) {
            std::ofstream pid_file(m_pid_file_path);
            pid_file << m_pid;
            pid_file.close();
            std::exit(EXIT_SUCCESS);
          }
          // An error occurred. A process ID lower than 0 indicates a failure in either process
          else if (m_pid < 0) {
            std::exit(EXIT_FAILURE);
          }
          // The parent process has now terminated, and the forked child process will continue
          // (the pid of the child process was 0)

          // Since the child process is a daemon, the umask needs to be set so files and logs can be written
          umask(0);

          // Initialize syslog for this daemon, here it's a good place to do so.
          //dlog::init(m_name);
          //dlog::notice("Daemon '" + m_name + "' started successfully.");

          // On success: The child process becomes session leader. Generate a session ID for the child process
          m_sid = setsid();
          if (m_sid < 0) {
            //dlog::error("Could not set SID to child process: " + std::string(std::strerror(errno)));
            std::exit(EXIT_FAILURE);
          }

          // Ignore the Child terminated or stopped signal.
          std::signal(SIGCHLD, SIG_IGN);
          // Set signal handlers to detect daemon interrupt, restart..
          std::signal(SIGHUP, signal_handler);
          // when a sudo systemctl stop my_daemon.service is ran, by default,
          // a SIGTERM is sent, followed by 90 seconds of waiting followed by a SIGKILL.
          std::signal(SIGTERM, signal_handler); // if ran: sudo systemctl stop my_daemon.service
          std::signal(SIGKILL, signal_handler); // if ran: sudo systemctl stop my_daemon.service (after 90 seconds)

          // Change the current working directory to a directory guaranteed to exist, provided by the user
          if(chdir(m_cwd.c_str()) < 0)
          {
            //dlog::error("Could not change current working directory to '" +m_cwd+ "': " + std::string(std::strerror(errno)));
            // If our guaranteed directory does not exist, terminate the child process to ensure
            // the daemon has not been hijacked
            std::exit(EXIT_FAILURE);
          }

          // A daemon cannot use the terminal, so close standard file descriptors for security reasons
          close(STDIN_FILENO);
          close(STDOUT_FILENO);
          close(STDERR_FILENO);
        }

    private:
        pid_t m_pid;
        pid_t m_sid;
        std::string m_config_file;
        std::string m_name;
        std::string m_cwd;
        std::chrono::high_resolution_clock::duration m_update_duration;
        std::atomic<bool> m_is_running;
        std::string m_pid_file_path;
        std::condition_variable m_update_cv;
        std::mutex m_mutex;
        std::int32_t m_exit_code;
    };
   daemon* daemon::instance = nullptr;
} // !namespace daemonpp
