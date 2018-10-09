#include <iomanip>
#include <chrono>
#include <ctime>
#include <stdexcept>
#include "image_proc.h"

LogLevel get_log_level(std::string str)
{
    if (str == "debug") return LogLevel::debug;
    if (str == "info") return LogLevel::info;
    if (str == "warn") return LogLevel::warn;
    if (str == "error") return LogLevel::error;

    throw std::runtime_error("Bad log level: " + str);
}

std::string to_string(LogLevel level)
{
    switch (level) {
        case debug: return "debug";
        case info: return "info";
        case warn: return "warn";
        case error: return "error";
    }
    throw;
}

// printing time in HH:MM:SS.mmm, according to
// https://stackoverflow.com/questions/24686846/get-current-time-in-milliseconds-or-hhmmssmmm-format
// should not be this difficult
void log_time()
{
    int mod = 100000;
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % mod;

    auto timer = std::chrono::system_clock::to_time_t(now);
    std::tm bt = *std::localtime(&timer);

    std::cout << std::put_time(&bt, "%T"); // HH:MM:SS
    std::cout << "." << std::setfill('0') << std::setw(3) << ms.count();
}

void log(std::string message, LogLevel level)
{
    if (level < LOG_LEVEL) return;
    log_time();
    std::cout << " [" << to_string(level) << "] ";
    std::cout << message << std::endl;
}
