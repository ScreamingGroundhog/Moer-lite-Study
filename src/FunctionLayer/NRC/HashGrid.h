#pragma once
#include <CoreLayer/Math/Math.h>
#include <atomic>
#include <vector>

//* 多分辨率哈希编码(Multi-resolution Hash Encoding, 参考 Instant-NGP)
//* 将 [0,1]^3 空间中的点通过L层多分辨率网格编码为低维特征向量
//* - 低分辨率层使用直接索引(线性索引，保证无碰撞)
//* - 高分辨率层使用哈希索引(碰撞通过平均特征缓解)
//* - 每层使用三线性插值聚合8个顶点的特征
class HashGrid {
public:
  HashGrid() = default;

  HashGrid(int baseRes, int numLevels, int featuresPerLevel, int tableSize);

  //* 前向：将点p(需在[0,1]^3内)编码到out
  //* out的长度为 numLevels * featuresPerLevel
  void encode(const Vector3f &p, float *out) const;

  //* 反向：将out的梯度gradOut累积到参数梯度缓冲gradParams中
  //* gradParams的长度为 params.size()
  void backward(const Vector3f &p, const float *gradOut,
                float *gradParams) const;

  //* 反向（原子版本）：用于多线程并行训练
  //* 梯度以CAS方式累积到以uint32存储的原子梯度缓冲中
  void backwardAtomic(const Vector3f &p, const float *gradOut,
                      std::atomic<uint32_t> *gradParams) const;

  //* 参数初始化(均匀随机)
  void initParameters(float minVal, float maxVal);

  int numParams() const { return (int)params.size(); }

public:
  int baseRes = 16;         // 基础分辨率
  int numLevels = 16;       // 层数
  int featuresPerLevel = 2; // 每层特征数
  int tableSize = 1 << 18;  // 哈希表大小(2的幂)
  std::vector<float> params;
  int outputDim = 32; // numLevels * featuresPerLevel

private:
  //* 第level层某顶点的参数下标
  //* level <= 阈值时使用线性索引，否则使用哈希索引
  int parameterIndex(int level, int x, int y, int z) const;
};
