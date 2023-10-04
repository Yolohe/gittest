#include "../inc/filter.h"

// 除去最大最小值后求平均值
template <typename T>
T comb_fil<T>::combination_filter()
{
  sum = 0;
  max = data[0];
  min = data[0];
  for (int i = 0; i < maxsize; i++)
  {
    if (data[i] > max)
      max = data[i];
    if (data[i] < min)
      min = data[i];
    sum += data[i];
  }
  average = (sum - max - min) / (maxsize - 2);
  flag = 1;
  return average;
}

// 卡尔曼滤波
template <typename L>
kal_fil<L>::kal_fil(float _R, float _Q)
{
  R = _R;
  Q = _Q;
  A = 1;
  H = 1;
  C_last = 1;
}
template <typename L>
L kal_fil<L>::kalman_filter()
{
  x_last = A * x_last;
  C_last = C_last + Q;
  kg = C_last / (C_last + R);              // 计算卡尔曼增益
  result = x_last + kg * (input - x_last); // 得到状态的最优化估计值
  x_last = result;
  C_last = (1 - kg) * C_last;
  return result;
}
