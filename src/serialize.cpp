#include <fstream>
#include <iostream>
#include "image_proc.h"
#include "geometry.h"
#include "json.hpp"

using json = nlohmann::json;

std::string MIDWAY_PATH;
std::string DEBUG_PATH;
std::string NAME;
LogLevel LOG_LEVEL;
bool DRAW_DEBUG_IMAGES;
bool OUTPUT_CROPPED_IMAGES;
std::unordered_map<std::string, double> PARAMS;
double RESOLUTION;

// read all the arguments from the json parser
void read_args(std::string args_string, std::string& map_path, std::vector<cv::Point>& boundary, std::vector<Street>& streets)
{
    std::cout << "here4" << std::endl;
    std::cout << args_string << std::endl;

    json j = json::parse(args_string);

    //std::cout << "here5" << std::endl;

    NAME = j["name"].get<std::string>();
    MIDWAY_PATH = j["midway_path"].get<std::string>();
    DEBUG_PATH = j["debug_path"].get<std::string>();
    LOG_LEVEL = get_log_level(j["log_level"].get<std::string>());
    RESOLUTION = stod(j["resolution"].get<std::string>());
    OUTPUT_CROPPED_IMAGES = j["output_cropped_images"].get<bool>();
    DRAW_DEBUG_IMAGES = j["debug"].get<bool>();

    //std::cout << "here6" << std::endl;

    map_path = j["map_path"].get<std::string>();
    boundary = j["boundary_points"].get<std::vector<cv::Point>>();
    streets = j["modern_streets"].get<std::vector<Street>>();
}

static json j_out;

// reading in streets

std::vector<cv::Point> read_boundary(const std::string& boundary_string)
{
    json j = json::parse(boundary_string);
    return j.get<std::vector<cv::Point>>();
}

std::vector<Street> read_streets(const std::string& streets_string)
{
    json j = json::parse(streets_string);
    return j.get<std::vector<Street>>();
}



void from_json(const json& j, Street& s)
{
    s.id = j.at("id").get<uint>();
    s.feature_id = j.at("id").get<uint>();
    s.segment = j.at("segment").get<Segment>();
}

void from_json(const json& j, Segment& s)
{
    s.p1 = j.at("p1").get<cv::Point>();
    s.p2 = j.at("p2").get<cv::Point>();
}

void to_json(json& j, const MyLine& line) {
    j = json {
        {"mid", line.mid_},
        {"domi", line.domi_},
        {"norm", line.norm_},
        {"width", line.width_},
        {"height", line.height_},
        {"isStreetName", line.isStreetName_},
        {"isHouseNumber", line.isHouseNumber_},
        {"boundingBox", line.boundingBox()},
        {"endpoints", line.endpoints()},
        {"id", line.streetId_},
        {"word_id", line.wordId_},
        {"relative_center", line.wordRelativeToStreet_},
        {"texturePackX", line.texturePackX},
        {"texturePackY", line.texturePackY},
        {"texturePackW", line.texturePackW},
        {"texturePackH", line.texturePackH},
        {"texturePackSheet", line.texturePackSheet},
        {"connectedComponentPass", line.connectedComponentPass_},
        {"heightConsistencyScore", line.heightConsistencyScore},
        {"spacingStreetNameScore", line.spacingStreetNameScore},
        {"spacingHouseNumScore", line.spacingHouseNumScore},
        {"areaStreetNameScore", line.areaStreetNameScore},
        {"areaHouseNumScore", line.areaHouseNumScore},
        {"connCompScore", line.connCompScore},
        {"wordRelativeToExtendedStreet", line.wordRelativeToExtendedStreet_},
    };
}

void to_json(json& j, const LongLine& line) {
    j = line.line_ids;
}

namespace cv {
    void to_json(json& j, const cv::Point2f& pt) {
        j = json {{"x", pt.x}, {"y", pt.y}};
    }

    void from_json(const json& j, cv::Point2f& pt) {
        pt.x = j.at("x").get<float>();
        pt.y = j.at("y").get<float>();
    }

    void to_json(json& j, const cv::Point& pt) {
        j = json {{"x", pt.x}, {"y", pt.y}};
    }

    void from_json(const json& j, cv::Point& pt) {
        pt.x = j.at("x").get<uint>();
        pt.y = j.at("y").get<uint>();
    }
}

void write_streets(const std::vector<MyLine>& streets)
{
    j_out["streets"] = streets;
}

void write_words(const std::vector<MyLine>& words)
{
    j_out["words"] = words;
}

void write_matches(const std::unordered_map<uint, std::vector<int>>& matches)
{
    j_out["matches"] = matches;
}

void write_extensions(const std::vector<LongLine>& extended_streets)
{
    std::vector<LongLine> lines_to_write;
    for (const auto& ll : extended_streets) {
        if (!ll.ignore_me) {
            lines_to_write.push_back(ll);
        }
    }
    j_out["extended_streets"] = lines_to_write;
}


void dump_json()
{
    //std::cout << j_out.dump(4) << std::endl;
    // Note: output_path must exist for this to work
    std::ofstream fs;
    fs.open(MIDWAY_PATH + PATH_SEP + NAME + ".txt");
    fs << std::setw(4) << j_out << std::endl;

    if (fs.fail()) {
        std::cout << "failed to write file output textfile" << std::endl;
    }
}
