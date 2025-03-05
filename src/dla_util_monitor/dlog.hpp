//
// Created by bader on 4/18/23.
//

#pragma once

#include <syslog.h>
#include <cstdio>
#include <string>

namespace daemonpp {
    class dlog {
    public:
        /**
         * initialize the logger
         * @param daemon_name
         */
        static void init(const std::string &daemon_name) {
          m_daemon_name = daemon_name;
          openlog(m_daemon_name.c_str(), LOG_PID, LOG_DAEMON);
        }

        /**
         * main logger with priority LOG_X
         * @param message
         * @param priority
         */
        static void log(const std::string &message, std::int32_t priority) {
          syslog(priority, "%s", message.c_str());
        }

        /**
         * debug-level messages
         * @param message
         */
        static void debug(const std::string &message) {
          log(message, LOG_DEBUG);
        }

        /**
         * informational
         * @param message
         */
        static void info(const std::string &message) {
          log(message, LOG_INFO);
        }

        /**
         * normal but significant condition
         * @param message
         */
        static void notice(const std::string &message) {
          log(message, LOG_NOTICE);
        }

        /**
         * warning conditions
         * @param message
         */
        static void warning(const std::string &message) {
          log(message, LOG_WARNING);
        }

        /**
         * error conditions
         * @param message
         */
        static void error(const std::string &message) {
          log(message, LOG_ERR);
        }

        /**
         * critical conditions
         * @param message
         */
        static void critical(const std::string &message) {
          log(message, LOG_CRIT);
        }

        /**
         * action must be taken immediately
         * @param message
         */
        static void alert(const std::string &message) {
          log(message, LOG_ALERT);
        }

        /**
         * system is unusable
         * @param message
         */
        static void emergency(const std::string &message) {
          log(message, LOG_EMERG);
        }

        /**
         * shutdown the logger
         */
        static void shutdown() {
          closelog();
        }

    private:
      static std::string priority_str(std::int32_t priority)
      {
        switch (priority) {
          case LOG_EMERG: return "emergency";
          case LOG_ALERT: return "alert";
          case LOG_CRIT: return "critical";
          case LOG_ERR: return "error";
          case LOG_WARNING: return "warning";
          case LOG_NOTICE: return "notice";
          case LOG_INFO: return "info";
          case LOG_DEBUG: return "debug";
          default: return "unknown_priority";
        }
      }

    private:
        static std::string m_daemon_name;
    };
    std::string dlog::m_daemon_name{};
}
