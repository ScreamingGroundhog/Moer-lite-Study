#pragma once
#include <vector>

//* 小型全连接神经网络(ReLU激活) + Adam优化器
//* 参数布局: W1[h1*in], b1[h1], W2[h2*h1], b2[h2], W3[out*h2], b3[out]
//* 配合多分辨率哈希编码用于拟合 radiance 函数
class MLP {
public:
  MLP() = default;

  //* 初始化网络结构与参数(Xavier风格)
  void init(int inDim, int h1, int h2, int outDim);

  //* 前向传播: in -> out(out维)
  void forward(const float *in, float *out) const;

  //* 前向+反向: 输入in与输出的梯度gradOut
  //* 梯度累积到参数梯度缓冲gradParams；输入的梯度写入gradIn(inDim维，可空)
  void forwardBackward(const float *in, const float *gradOut,
                       float *gradParams, float *gradIn = nullptr) const;

  int numParams() const { return (int)params.size(); }

public:
  int inDim = 0, h1 = 0, h2 = 0, outDim = 0;
  std::vector<float> params;

private:
  //* 各参数块在params中的偏移
  int offW1, offB1, offW2, offB2, offW3, offB3;

  friend class AdamOptimizer;
};

//* Adam优化器，维护一阶、二阶矩估计
class AdamOptimizer {
public:
  AdamOptimizer() = default;

  //* 初始化矩估计，numParams为参数个数
  void init(int numParams);

  //* 开始新一步迭代(递增时间步t)，随后可并行调用stepRange
  void beginStep() { ++t; }

  //* 使用梯度grad更新参数params(原地更新)
  void step(float *params, const float *grad, float learningRate);

  //* 对一段连续参数区间使用梯度更新(用于并行更新)
  void stepRange(float *params, const float *grad, float learningRate,
                 int begin, int end);

private:
  std::vector<float> m, v;
  int t = 0;
  static constexpr float beta1 = .9f, beta2 = .999f, eps = 1e-8f;
};
