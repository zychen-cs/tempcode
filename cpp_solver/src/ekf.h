#ifndef EKF_H
#define EKF_H

#include <Eigen/Dense>

class EKF
{
public:
    // 构造函数
    EKF(Eigen::Matrix<double, 5,1> S0);
    // 更新
    void update(const Eigen::Matrix<double, 30,1>& Z);
    // 获取状态
    Eigen::Matrix<double, 5,1> getState(){return S;}
    // 观测方程
    Eigen::Matrix<double, 30,1> OutputEquation(Eigen::Matrix<double, 5,1> X);
private:
    // 更新Q矩阵
    void updateQMat(Eigen::Matrix<double, 30,1> Z);
    // 更新H矩阵
    void updateHMat();
    // 状态量
    Eigen::Matrix<double, 5,1> S;
    // 协方差
    Eigen::Matrix<double, 5,5> P;
    // 状态矩阵 S_{k+1} = F*S_{k}，这里F是单位阵
    Eigen::Matrix<double, 5,5> F;
    // 观测雅可比
    Eigen::Matrix<double, 30,5> H;
    // 过程噪声
    Eigen::Matrix<double, 5,5>  Q;
    // 观测噪声
    Eigen::Matrix<double, 30,30> R;
    // 首次运行标志
    bool first = true;
    // 上一回合的观测值
    Eigen::Matrix<double, 30,1> prevZ;

    double xyzs[10][3] = {
            {0, -0.8, 0},
            {0.5, -0.5, 0},
            {-0.5, -0.5, 0},
            {1, 0, 0},
            {0, 0, 0},
            {-1, 0, 0},
            {0.5, 0.5, 0},
            {-0.5, 0.5, 0},
            {1, 1, 0},
            {-1, 1, 0}
    };
    // 误差平方
    double Rs1[30] = { 63.022
            ,595.002
            ,861.264
            ,60.0086
            ,453.319
            ,1072.65
            ,49.7802
            ,483.356
            ,1451.06
            ,79.6229
            ,407.259
            ,346.164
            ,88.2023
            ,392.516
            ,519.493
            ,65.2735
            ,414.168
            ,462.252
            ,70.4998
            ,319.219
            ,222.598
            ,91.5389
            ,328.857
            ,239.689
            ,95.4321
            ,308.539
            ,129.594
            ,88.5684
            ,416.405
            ,131.625};
    // 误差绝对值
    double Rs2[30] = {
            7.57025
            ,24.0581
            ,27.8227
            ,7.14037
            ,21.0445
            ,26.7924
            ,6.33279
            ,21.2538
            ,30.6494
            ,8.43118
            ,20.1256
            ,14.6724
            ,9.08287
            ,19.6725
            ,18.6743
            ,7.50473
            ,20.172
            ,16.6301
            ,8.01405
            ,17.8078
            ,11.7164
            ,9.35602
            ,18.0584
            ,12.4596
            ,9.55445
            ,17.5202
            ,9.20374
            ,9.23977
            ,20.3683
            ,9.01319
    };
};

#endif // EKF_H
