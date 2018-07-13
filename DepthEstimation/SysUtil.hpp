/**
@brief class for basic system operations
@author: Shane Yuan
@date: Dec 11, 2017
*/

#ifndef __TINY_GIGA_SYS_UTIL_HPP__
#define __TINY_GIGA_SYS_UTIL_HPP__

#include <iostream>
#include <string>
#include <fstream>
#include <map>

#ifdef WIN32
#include <Windows.h>
#endif
#include <direct.h>
#include <chrono>
#include <memory>
#include <thread>

enum class ConsoleColor {
	red = 12,
	blue = 9,
	green = 10,
	yellow = 14,
	white = 15,
	pink = 13,
	cyan = 11
};

#ifndef WIN32
#define BLACK_TEXT(x) "\033[30;1m" x "\033[0m"
#define RED_TEXT(x) "\033[31;1m" x "\033[0m"
#define GREEN_TEXT(x) "\033[32;1m" x "\033[0m"
#define YELLOW_TEXT(x) "\033[33;1m" x "\033[0m"
#define BLUE_TEXT(x) "\033[34;1m" x "\033[0m"
#define MAGENTA_TEXT(x) "\033[35;1m" x "\033[0m"
#define CYAN_TEXT(x) "\033[36;1m" x "\033[0m"
#define WHITE_TEXT(x) "\033[37;1m" x "\033[0m"
#endif

class SysUtil {
public:
	/***********************************************************/
	/*                    mkdir function                       */
	/***********************************************************/
	static int mkdir(char* dir) {
#ifdef WIN32
		_mkdir(dir);
#else
		char command[COMMAND_STRING_LENGTH];
		sprintf(command, "mkdir %s", dir);
		system(command);
#endif
		return 0;
	}
	static int mkdir(std::string dir) {
		return mkdir((char *)dir.c_str());
	}

	/***********************************************************/
	/*                      format output                      */
	/***********************************************************/
	static std::string sprintf(const char *format, ...) {
		char str[512];
		va_list arg;
		va_start(arg, format);
		vsprintf(str, format, arg);
		va_end(arg);
		return std::string(str);
	}

	/***********************************************************/
	/*                    sleep function                       */
	/***********************************************************/
	static int sleep(size_t miliseconds) {
		std::this_thread::sleep_for(std::chrono::milliseconds(miliseconds));
		return 0;
	}

	/***********************************************************/
	/*             make colorful console output                */
	/***********************************************************/
	static int setConsoleColor(ConsoleColor color) {
#ifdef WIN32
		SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), static_cast<int>(color));
#endif
		return 0;
	}

	/***********************************************************/
	/*                 warning error output                    */
	/***********************************************************/
	static int errorOutput(std::string info) {
#ifdef WIN32
		SysUtil::setConsoleColor(ConsoleColor::red);
		std::cerr << "ERROR: " << info.c_str() << std::endl;
		SysUtil::setConsoleColor(ConsoleColor::white);
#else
		std::cerr << RED_TEXT("ERROR: ") << RED_TEXT(info.c_str()) << std::endl;
#endif
		return 0;
	}

	static int warningOutput(std::string info) {
#ifdef WIN32
		SysUtil::setConsoleColor(ConsoleColor::yellow);
		std::cerr << "WARNING: " << info.c_str() << std::endl;
		SysUtil::setConsoleColor(ConsoleColor::white);
#else
		std::cerr << YELLOW_TEXT("ERROR: ") << YELLOW_TEXT(info.c_str()) << std::endl;
#endif
		return 0;
	}

	static int infoOutput(std::string info) {
#ifdef WIN32
		SysUtil::setConsoleColor(ConsoleColor::green);
		std::cerr << "INFO: " << info.c_str() << std::endl;
		SysUtil::setConsoleColor(ConsoleColor::white);
#else
		std::cerr << GREEN_TEXT("ERROR: ") << GREEN_TEXT(info.c_str()) << std::endl;
#endif
		return 0;
	}

};

/***********************************************************/
/*                      config class                       */
/***********************************************************/
class Config {
private:
	std::map<std::string, std::string> config;

	/**
	@brief functions to remove space
	@param std::string str: input string
	@param int direction: from first 0 or from last 1
	@return std::string
	*/
	std::string removeSpace(std::string str, int direction = 0) {
		std::string outstr("");
		for (size_t i = 0; i < str.length(); i++) {
			int ind = i;
			if (direction == 1)
				ind = str.length() - 1 - ind;
			char c = str.at(ind);
			if (c != ' ') {
				if (direction == 0)
					outstr = str.substr(i, str.length() - i);
				else
					outstr = str.substr(0, str.length() - i);
				return outstr;
			}
		}
		return outstr;
	}

public:
	/**
	@brief get functions
	get value in map with input key
	@return
	*/
	bool getString(std::string key, std::string & str) {
		auto iter = config.find(key);
		if (iter != config.end()) {
			str = iter->second;
			return true;
		}
		else {
			str = "";
			return false;
		}
	}

	bool getInt(std::string key, int & str) {
		auto iter = config.find(key);
		if (iter != config.end()) {
			str = atoi(iter->second.c_str());
			return true;
		}
		else {
			str = 0;
			return false;
		}
	}

	template <typename T>
	bool getInt(std::string key, T & str) {
		auto iter = config.find(key);
		if (iter != config.end()) {
			str = static_cast<T>(atoi(iter->second.c_str()));
			return true;
		}
		else {
			str = static_cast<T>(0);
			return false;
		}
	}

	bool getFloat(std::string key, float & str) {
		auto iter = config.find(key);
		if (iter != config.end()) {
			str = atof(iter->second.c_str());
			return true;
		}
		else {
			str = 0;
			return false;
		}
	}

	/**
	@breif function to load config information from file
	*/
	int	loadConfigFile(std::string filename, char separator = '=') {
		std::fstream fs(filename, std::ios::in);
		std::string line;
		std::string::size_type pos;
		// get lines and process
		while (std::getline(fs, line)) {
			if (line.length() == 0)
				continue;
			// find first separator
			pos = line.find(separator);
			// check if this line is comment
			if (line.at(0) == '#')
				continue;
			// find separator
			if (pos == std::string::npos) {
				// can not find separator in this line, discard
				char info[256];
				snprintf(info, 256, "Can not find separator %c , " \
					" discard this line !!!", separator);
				SysUtil::warningOutput(std::string(info));
			}
			else {
				std::string keystr = line.substr(0, pos);
				keystr = removeSpace(keystr, 0);
				keystr = removeSpace(keystr, 1);
				std::string valstr = line.substr(pos + 1, line.length() - pos - 1);
				valstr = removeSpace(valstr, 0);
				valstr = removeSpace(valstr, 1);
				config.insert(std::pair<std::string, std::string>(keystr, valstr));
			}
		}
		fs.close();
		return 0;
	}
};

#endif