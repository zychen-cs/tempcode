#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include "ekf.h"
// 字符串分割
static void split(const std::string& s, std::vector<std::string>& tokens, const std::string& delimiters = " ")
{
    std::string::size_type lastPos = s.find_first_not_of(delimiters, 0);
    std::string::size_type pos = s.find_first_of(delimiters, lastPos);
    while (std::string::npos != pos || std::string::npos != lastPos) {
        tokens.push_back(s.substr(lastPos, pos - lastPos));//use emplace_back after C++1
        lastPos = s.find_first_not_of(delimiters, pos);
        pos = s.find_first_of(delimiters, lastPos);
    }
}
// 全局变量
std::vector<Eigen::Matrix<double, 5,1>> poses;
std::vector<Eigen::Matrix<double, 30,1>> meass;
// 加载文件中的数据
void LoadDataFromFile() {
    std::ifstream file1("../0112pose.csv");
    std::string line;
    std::getline(file1, line);
    std::vector<std::string> tokens;
    while (std::getline(file1, line)) {
        tokens.clear();
        split(line, tokens, ",");
        Eigen::Matrix<double, 5,1> pos;
        pos<<stof(tokens[2]),stof(tokens[3]),stof(tokens[4]),stof(tokens[5]),stof(tokens[6]);
        poses.emplace_back(pos);
    }

    std::ifstream file2("../0112sensor.csv");
    std::getline(file2, line);
    while (std::getline(file2, line)) {
        tokens.clear();
        split(line, tokens, ",");
        Eigen::Matrix<double, 30,1> meas = Eigen::Matrix<double, 30,1>::Zero(30,1);
        for(int i = 0; i < 30; ++i)
        {
            auto str = tokens[i+2];
            if(str.front() == '"')
            {
                meas(i) = stof(str.substr(2, str.length()-2));
            }else if(str.back() == '"')
            {
                meas(i) = stof(str.substr(0, str.length()-2));
            }else{
                meas(i) = stof(str);
            }
        }
        meass.emplace_back(meas);
    }
}

double offset[10][3] = {  {7.97211849 , 6.6357796  ,  0.70268908},
                          {3.92386126 , 23.69922648 ,  1.34865825},
                          {50.15567932, 39.46317429 , 54.67707398},
                          {16.97569252, -30.99201639,   8.32914118},
                          {5.2589972  , 7.08048109,   9.41717116},
                          {-18.18148847,-13.35689644,  18.01217468},
                          {-5.17809366, -8.26765124,  16.1910124 },
                          {6.63456889,  -2.48714757,  10.79503975},
                          {4.61783353,  8.47660365,  -9.46816979},
                          {-3.83599326, -42.42874346,  -7.5117813 }};
double scale[10][3] = {   {34.20682548, 34.31255772, 35.01314558},
                          {32.8655348 , 35.11646587, 34.84757859},
                          {24.58609985, 26.82656228, 26.09768036},
                          {32.45645399, 34.7738967 , 34.61187494},
                          {32.53203975, 35.48868686, 34.25862027},
                          {33.29412708, 35.77212393, 34.97134392},
                          {33.98938892, 35.56978734, 35.43890517},
                          {33.44979704, 35.76081192, 35.451998  },
                          {32.50748362, 34.68123958, 34.31192605},
                          {33.54922046, 36.29036765, 35.43402999}};
// 主函数
int main() {
    // 加载文件
    LoadDataFromFile();
    // 初始化EKF
    Eigen::Matrix<double, 5,1> X0 = poses.front();
    EKF ekf(X0);
    // 计算平均scale
    double scale_avg = 0.0;
    for(int idx = 0; idx < 10; ++idx) {
        scale_avg += scale[idx][0] + scale[idx][1] + scale[idx][2];
    }
    scale_avg = scale_avg / 30.0;
    // 计算Rs
    Eigen::Matrix<double, 30,1> Rs = Eigen::Matrix<double, 30,1>::Zero(30,1);
    // 计算100条数据
    for(int i = 0; i < 100; ++i)
    {
        // 先计算观测的Z
        Eigen::Matrix<double, 30,1> Z = meass[i];
        for(int idx = 0; idx < 10; ++idx)
        {
            Z(idx*3+0) = (Z(idx*3+0) - offset[idx][0])/scale[idx][0]*scale_avg;
            Z(idx*3+1) = (Z(idx*3+1) - offset[idx][1])/scale[idx][1]*scale_avg;
            Z(idx*3+2) = (Z(idx*3+2) - offset[idx][2])/scale[idx][2]*scale_avg;
        }
        // 再计算理论计算的z
        auto z = ekf.OutputEquation(poses[i]);
        // 做差
        auto y = Z-z;
        for(int idx = 0; idx < 30; ++idx)
        {
            // 取误差绝对值
             Rs(idx) += fabs(y(idx));
            // 取误差平方
            // Rs(idx) += y(idx)*y(idx);
        }
    }
    // 保存，打印
    Rs = Rs/100.0;
    std::cout<<Rs<<std::endl;
    return 0;
}
