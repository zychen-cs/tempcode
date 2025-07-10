#include "ekf.h"
#include <iostream>
EKF::EKF(Eigen::Matrix<double, 5,1> S0)
{
    S = S0;
    P = Eigen::Matrix<double, 5,5>::Identity(5,5);
    Q = Eigen::Matrix<double, 5,5>::Identity(5,5);
    R = Eigen::Matrix<double, 30,30>::Identity(30,30);
    for(int i = 0; i < 30; ++i)
    {
        R(i,i) = Rs2[i];
    }
    F = Eigen::Matrix<double, 5,5>::Identity(5,5);
    H = Eigen::Matrix<double, 30,5>::Zero(30,5);
}

void EKF::updateQMat(Eigen::Matrix<double, 30,1> Z)
{
    if(first)
    {
        first = false;
    }else{
        double N = 0.0;
        for(int i = 0; i < 30; ++i)
        {
            double dB = Z(i) - prevZ(i);
            N += dB*dB;
        }
        double M = -4.623*1e-7*N*N+0.001622*N-8.128;
        Q(4) = pow(10,M);
        Q(0) = 100*Q(4);
        Q(1) = 100*Q(4);
        Q(2) = 100*Q(4);
        Q(3) = Q(4);
    }
    prevZ = Z;
}
Eigen::Matrix<double, 30,1> EKF::OutputEquation(Eigen::Matrix<double, 5,1> X)
{
    double a = X(0);
    double b = X(1);
    double c = X(2);
    double theta = X(3);
    double phi   = X(4);
    double m     = sin(theta)*cos(phi);
    double n     = sin(theta)*sin(phi);
    double p     = cos(theta);
    double BT    = 1e5*0.27;
    Eigen::Matrix<double, 30,1> Z = Eigen::Matrix<double, 30,1>::Zero(30,1);
    for(int i = 0; i < 10; ++i)
    {
        double xl  = xyzs[i][0];
        double yl  = xyzs[i][1];
        double zl  = xyzs[i][2];
        double Tl  = m*(xl-a)+n*(yl-b)+p*(zl-c);
        double Rl  = sqrt( (xl-a)*(xl-a) + (yl-b)*(yl-b) + (zl-c)*(zl-c));
        double Rl3 = Rl*Rl*Rl;
        double Rl5 = Rl3*Rl*Rl;
        double Blx = BT*(3*Tl*(xl - a)/Rl5 - m/Rl3);
        double Bly = BT*(3*Tl*(yl - b)/Rl5 - n/Rl3);
        double Blz = BT*(3*Tl*(zl - c)/Rl5 - p/Rl3);
        Z(i*3+0) = Blx;
        Z(i*3+1) = Bly;
        Z(i*3+2) = Blz;
    }
    return Z;
}
// 这里实时计算观测方程的雅可比矩阵（观测量对状态量的偏导)
void EKF::updateHMat()
{
    double a = S(0);
    double b = S(1);
    double c = S(2);
    double theta = S(3);
    double phi   = S(4);
    double m     = sin(theta)*cos(phi);
    double n     = sin(theta)*sin(phi);
    double p     = cos(theta);
    double BT    = 1e5*0.27;
    for(int i = 0; i < 10; ++i)
    {
        double xl  = xyzs[i][0];
        double yl  = xyzs[i][1];
        double zl  = xyzs[i][2];
        double t[5];
        t[0] = BT*(1.0/pow(pow(a-xl,2.0)+pow(b-yl,2.0)+pow(c-zl,2.0),5.0/2.0)*(cos(theta)*(c-zl)*3.0+cos(phi)*sin(theta)*(a-xl)*3.0+sin(phi)*sin(theta)*(b-yl)*3.0)-(a-xl)*(a*2.0-xl*2.0)*1.0/pow(pow(a-xl,2.0)+pow(b-yl,2.0)+pow(c-zl,2.0),7.0/2.0)*(cos(theta)*(c-zl)*3.0+cos(phi)*sin(theta)*(a-xl)*3.0+sin(phi)*sin(theta)*(b-yl)*3.0)*(5.0/2.0)+cos(phi)*sin(theta)*(a-xl)*1.0/pow(pow(a-xl,2.0)+pow(b-yl,2.0)+pow(c-zl,2.0),5.0/2.0)*3.0+cos(phi)*sin(theta)*(a*2.0-xl*2.0)*1.0/pow(pow(a-xl,2.0)+pow(b-yl,2.0)+pow(c-zl,2.0),5.0/2.0)*(3.0/2.0));
        t[1] = BT*((a-xl)*(b*2.0-yl*2.0)*1.0/pow(pow(a-xl,2.0)+pow(b-yl,2.0)+pow(c-zl,2.0),7.0/2.0)*(cos(theta)*(c-zl)*3.0+cos(phi)*sin(theta)*(a-xl)*3.0+sin(phi)*sin(theta)*(b-yl)*3.0)*(-5.0/2.0)+sin(phi)*sin(theta)*(a-xl)*1.0/pow(pow(a-xl,2.0)+pow(b-yl,2.0)+pow(c-zl,2.0),5.0/2.0)*3.0+cos(phi)*sin(theta)*(b*2.0-yl*2.0)*1.0/pow(pow(a-xl,2.0)+pow(b-yl,2.0)+pow(c-zl,2.0),5.0/2.0)*(3.0/2.0));
        t[2] = BT*(cos(theta)*(a-xl)*1.0/pow(pow(a-xl,2.0)+pow(b-yl,2.0)+pow(c-zl,2.0),5.0/2.0)*3.0-(a-xl)*(c*2.0-zl*2.0)*1.0/pow(pow(a-xl,2.0)+pow(b-yl,2.0)+pow(c-zl,2.0),7.0/2.0)*(cos(theta)*(c-zl)*3.0+cos(phi)*sin(theta)*(a-xl)*3.0+sin(phi)*sin(theta)*(b-yl)*3.0)*(5.0/2.0)+cos(phi)*sin(theta)*(c*2.0-zl*2.0)*1.0/pow(pow(a-xl,2.0)+pow(b-yl,2.0)+pow(c-zl,2.0),5.0/2.0)*(3.0/2.0));
        t[3] = BT*((a-xl)*1.0/pow(pow(a-xl,2.0)+pow(b-yl,2.0)+pow(c-zl,2.0),5.0/2.0)*(sin(theta)*(c-zl)*-3.0+cos(phi)*cos(theta)*(a-xl)*3.0+cos(theta)*sin(phi)*(b-yl)*3.0)-cos(phi)*cos(theta)*1.0/pow(pow(a-xl,2.0)+pow(b-yl,2.0)+pow(c-zl,2.0),3.0/2.0));
        t[4] = BT*((a-xl)*(cos(phi)*sin(theta)*(b-yl)*3.0-sin(phi)*sin(theta)*(a-xl)*3.0)*1.0/pow(pow(a-xl,2.0)+pow(b-yl,2.0)+pow(c-zl,2.0),5.0/2.0)+sin(phi)*sin(theta)*1.0/pow(pow(a-xl,2.0)+pow(b-yl,2.0)+pow(c-zl,2.0),3.0/2.0));
        for(int idx = 0; idx < 5; ++idx)
        {
            H(i*3+0,idx) = t[idx];
        }

        t[0] = BT*((b-yl)*(a*2.0-xl*2.0)*1.0/pow(pow(a-xl,2.0)+pow(b-yl,2.0)+pow(c-zl,2.0),7.0/2.0)*(cos(theta)*(c-zl)*3.0+cos(phi)*sin(theta)*(a-xl)*3.0+sin(phi)*sin(theta)*(b-yl)*3.0)*(-5.0/2.0)+cos(phi)*sin(theta)*(b-yl)*1.0/pow(pow(a-xl,2.0)+pow(b-yl,2.0)+pow(c-zl,2.0),5.0/2.0)*3.0+sin(phi)*sin(theta)*(a*2.0-xl*2.0)*1.0/pow(pow(a-xl,2.0)+pow(b-yl,2.0)+pow(c-zl,2.0),5.0/2.0)*(3.0/2.0));
        t[1] = BT*(1.0/pow(pow(a-xl,2.0)+pow(b-yl,2.0)+pow(c-zl,2.0),5.0/2.0)*(cos(theta)*(c-zl)*3.0+cos(phi)*sin(theta)*(a-xl)*3.0+sin(phi)*sin(theta)*(b-yl)*3.0)-(b-yl)*(b*2.0-yl*2.0)*1.0/pow(pow(a-xl,2.0)+pow(b-yl,2.0)+pow(c-zl,2.0),7.0/2.0)*(cos(theta)*(c-zl)*3.0+cos(phi)*sin(theta)*(a-xl)*3.0+sin(phi)*sin(theta)*(b-yl)*3.0)*(5.0/2.0)+sin(phi)*sin(theta)*(b-yl)*1.0/pow(pow(a-xl,2.0)+pow(b-yl,2.0)+pow(c-zl,2.0),5.0/2.0)*3.0+sin(phi)*sin(theta)*(b*2.0-yl*2.0)*1.0/pow(pow(a-xl,2.0)+pow(b-yl,2.0)+pow(c-zl,2.0),5.0/2.0)*(3.0/2.0));
        t[2] = BT*(cos(theta)*(b-yl)*1.0/pow(pow(a-xl,2.0)+pow(b-yl,2.0)+pow(c-zl,2.0),5.0/2.0)*3.0-(b-yl)*(c*2.0-zl*2.0)*1.0/pow(pow(a-xl,2.0)+pow(b-yl,2.0)+pow(c-zl,2.0),7.0/2.0)*(cos(theta)*(c-zl)*3.0+cos(phi)*sin(theta)*(a-xl)*3.0+sin(phi)*sin(theta)*(b-yl)*3.0)*(5.0/2.0)+sin(phi)*sin(theta)*(c*2.0-zl*2.0)*1.0/pow(pow(a-xl,2.0)+pow(b-yl,2.0)+pow(c-zl,2.0),5.0/2.0)*(3.0/2.0));
        t[3] = BT*((b-yl)*1.0/pow(pow(a-xl,2.0)+pow(b-yl,2.0)+pow(c-zl,2.0),5.0/2.0)*(sin(theta)*(c-zl)*-3.0+cos(phi)*cos(theta)*(a-xl)*3.0+cos(theta)*sin(phi)*(b-yl)*3.0)-cos(theta)*sin(phi)*1.0/pow(pow(a-xl,2.0)+pow(b-yl,2.0)+pow(c-zl,2.0),3.0/2.0));
        t[4] = BT*((b-yl)*(cos(phi)*sin(theta)*(b-yl)*3.0-sin(phi)*sin(theta)*(a-xl)*3.0)*1.0/pow(pow(a-xl,2.0)+pow(b-yl,2.0)+pow(c-zl,2.0),5.0/2.0)-cos(phi)*sin(theta)*1.0/pow(pow(a-xl,2.0)+pow(b-yl,2.0)+pow(c-zl,2.0),3.0/2.0));
        for(int idx = 0; idx < 5; ++idx)
        {
            H(i*3+1,idx) = t[idx];
        }

        t[0] = BT*(cos(theta)*(a*2.0-xl*2.0)*1.0/pow(pow(a-xl,2.0)+pow(b-yl,2.0)+pow(c-zl,2.0),5.0/2.0)*(3.0/2.0)-(c-zl)*(a*2.0-xl*2.0)*1.0/pow(pow(a-xl,2.0)+pow(b-yl,2.0)+pow(c-zl,2.0),7.0/2.0)*(cos(theta)*(c-zl)*3.0+cos(phi)*sin(theta)*(a-xl)*3.0+sin(phi)*sin(theta)*(b-yl)*3.0)*(5.0/2.0)+cos(phi)*sin(theta)*(c-zl)*1.0/pow(pow(a-xl,2.0)+pow(b-yl,2.0)+pow(c-zl,2.0),5.0/2.0)*3.0);
        t[1] = BT*(cos(theta)*(b*2.0-yl*2.0)*1.0/pow(pow(a-xl,2.0)+pow(b-yl,2.0)+pow(c-zl,2.0),5.0/2.0)*(3.0/2.0)-(c-zl)*(b*2.0-yl*2.0)*1.0/pow(pow(a-xl,2.0)+pow(b-yl,2.0)+pow(c-zl,2.0),7.0/2.0)*(cos(theta)*(c-zl)*3.0+cos(phi)*sin(theta)*(a-xl)*3.0+sin(phi)*sin(theta)*(b-yl)*3.0)*(5.0/2.0)+sin(phi)*sin(theta)*(c-zl)*1.0/pow(pow(a-xl,2.0)+pow(b-yl,2.0)+pow(c-zl,2.0),5.0/2.0)*3.0);
        t[2] = BT*(1.0/pow(pow(a-xl,2.0)+pow(b-yl,2.0)+pow(c-zl,2.0),5.0/2.0)*(cos(theta)*(c-zl)*3.0+cos(phi)*sin(theta)*(a-xl)*3.0+sin(phi)*sin(theta)*(b-yl)*3.0)+cos(theta)*(c-zl)*1.0/pow(pow(a-xl,2.0)+pow(b-yl,2.0)+pow(c-zl,2.0),5.0/2.0)*3.0+cos(theta)*(c*2.0-zl*2.0)*1.0/pow(pow(a-xl,2.0)+pow(b-yl,2.0)+pow(c-zl,2.0),5.0/2.0)*(3.0/2.0)-(c-zl)*(c*2.0-zl*2.0)*1.0/pow(pow(a-xl,2.0)+pow(b-yl,2.0)+pow(c-zl,2.0),7.0/2.0)*(cos(theta)*(c-zl)*3.0+cos(phi)*sin(theta)*(a-xl)*3.0+sin(phi)*sin(theta)*(b-yl)*3.0)*(5.0/2.0));
        t[3] = BT*(sin(theta)*1.0/pow(pow(a-xl,2.0)+pow(b-yl,2.0)+pow(c-zl,2.0),3.0/2.0)+(c-zl)*1.0/pow(pow(a-xl,2.0)+pow(b-yl,2.0)+pow(c-zl,2.0),5.0/2.0)*(sin(theta)*(c-zl)*-3.0+cos(phi)*cos(theta)*(a-xl)*3.0+cos(theta)*sin(phi)*(b-yl)*3.0));
        t[4] = BT*(c-zl)*(cos(phi)*sin(theta)*(b-yl)*3.0-sin(phi)*sin(theta)*(a-xl)*3.0)*1.0/pow(pow(a-xl,2.0)+pow(b-yl,2.0)+pow(c-zl,2.0),5.0/2.0);
        for(int idx = 0; idx < 5; ++idx)
        {
            H(i*3+2,idx) = t[idx];
        }
    }
}
// EKF计算函数
void EKF::update(const Eigen::Matrix<double, 30,1>& Z)
{
    // predict
    // 状态预测
    S = F * S;
    // 预测协方差更新
    P = F* P* F.transpose() + Q;

    // update
    // 计算观测和预测观测的差
    auto z = OutputEquation(S);
    auto y = Z - z;
    updateHMat();
    // 计算卡尔曼增益
    auto T = H* P * H.transpose() + R;
    auto K = P * H.transpose()*T.inverse();
    // 更新状态估计
    S = S + K*y;
    // 更新协方差
    P = (Eigen::Matrix<double, 5,5>::Identity(5,5) - K*H)*P;
}
