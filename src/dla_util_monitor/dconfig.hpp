//
// Created by bader on 4/21/23.
//

#pragma once
#include <fstream>
#include <cstring>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <map>
#include "dlog.hpp"

// TODO: better config parsing
namespace daemonpp
{
  struct dconfig {
      std::map<std::string, std::string> values;

      std::string get(const std::string& key) const {
        auto it = values.find(key);
        if(it != values.end())
          return it->second;
        return "";
      }
  
      static dconfig from_file(const std::string& filename){
          dconfig cfg;
          if(filename.empty()) return cfg;
          auto split = [](const std::string& str, char delim) {
              std::vector<std::string> parts;
              std::stringstream oss{str};
              std::string part;
              while(std::getline(oss, part, delim)){
                parts.push_back(part);
              }
              return parts;
          };
          auto trim = [](std::string &s) {
            s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
                return !std::isspace(ch);
            }));
            s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
                return !std::isspace(ch);
            }).base(), s.end());
          };
          
          std::ifstream ifs{filename};
          std::string line;
          while(std::getline(ifs, line)){
            trim(line);
            if(line.empty()) continue;
            if(line[0] == '#') continue; // skip comments
            auto parts = split(line, '=');
            std::string key = parts[0];
            std::string value = parts[1];
            trim(key);
            trim(value);
            cfg.values[key] = value;
          }
          ifs.close();
          return cfg;
      }

   };
} 
