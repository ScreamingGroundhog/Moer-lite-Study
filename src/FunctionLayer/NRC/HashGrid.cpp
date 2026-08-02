#include "HashGrid.h"
#include <cmath>

namespace {
//* 整数坐标 + 层数的哈希函数(参考 "A Fast and Simple Hash Function")
inline uint32_t hashCoord(uint32_t x, uint32_t y, uint32_t z, uint32_t level) {
  uint32_t h = x * 0x8da6b343u ^ y * 0xd8163841u ^ z * 0xcb1ab31fu;
  h ^= level * 0x1234567fu;
  h ^= h >> 16;
  h *= 0x85ebca6bu;
  h ^= h >> 13;
  h *= 0xc2b2ae35u;
  h ^= h >> 16;
  return h;
}

//* 以CAS方式向原子缓冲中的浮点数做累加
inline void atomicAddFloat(std::atomic<uint32_t> *addr, float value) {
  uint32_t expected = addr->load(std::memory_order_relaxed);
  uint32_t desired;
  do {
    float current;
    std::memcpy(&current, &expected, sizeof(float));
    float sum = current + value;
    std::memcpy(&desired, &sum, sizeof(float));
  } while (!addr->compare_exchange_weak(expected, desired,
                                        std::memory_order_relaxed));
}

//* 第level层的网格分辨率
inline int gridRes(int baseRes, int level) { return baseRes << level; }

//* 将点p(需在[0,1]^3内)映射到第level层的网格坐标(x0,y0,z0, wx,wy,wz)
//* 保证 0 <= x0 <= res-2, 0 <= x0+1 <= res-1, 0 <= w <= 1
inline void gridCoordinates(const Vector3f &p, int res, int &x0, int &y0,
                            int &z0, float &wx, float &wy, float &wz) {
  const float pv[3] = {p[0], p[1], p[2]};
  float *w[3] = {&wx, &wy, &wz};
  int *c[3] = {&x0, &y0, &z0};
  for (int d = 0; d < 3; ++d) {
    float s = pv[d] * (res - 1);
    int cell = (int)std::floor(s);
    cell = std::max(0, std::min(cell, res - 2));
    *c[d] = cell;
    *w[d] = std::max(0.f, std::min(1.f, s - cell));
  }
}
} // namespace

HashGrid::HashGrid(int baseRes, int numLevels, int featuresPerLevel,
                   int tableSize)
    : baseRes(baseRes), numLevels(numLevels),
      featuresPerLevel(featuresPerLevel), tableSize(tableSize) {
  params.assign((size_t)numLevels * tableSize * featuresPerLevel, .0f);
  outputDim = numLevels * featuresPerLevel;
}

void HashGrid::initParameters(float minVal, float maxVal) {
  //* 参数多、维度高，这里使用简单的固定种子伪随机序列保证可复现
  uint32_t seed = 0x9e3779b9u;
  auto nextRand = [&seed]() {
    seed = seed * 1664525u + 1013904223u;
    return (seed >> 8) / 16777216.f; // [0, 1)
  };
  for (auto &p : params) {
    p = minVal + (maxVal - minVal) * nextRand();
  }
}

int HashGrid::parameterIndex(int level, int x, int y, int z) const {
  int res = gridRes(baseRes, level);
  //* 直接索引：分辨率^3 <= 表大小时所有网格顶点都有唯一的参数槽
  if ((int64_t)res * res * res <= tableSize) {
    return level * tableSize + (x * res + y) * res + z;
  }
  return level * tableSize +
         (int)(hashCoord((uint32_t)x, (uint32_t)y, (uint32_t)z, (uint32_t)level) &
               (uint32_t)(tableSize - 1));
}

void HashGrid::encode(const Vector3f &p, float *out) const {
  int F = featuresPerLevel;
  for (int level = 0; level < numLevels; ++level) {
    int res = gridRes(baseRes, level);
    int x0, y0, z0;
    float wx, wy, wz;
    gridCoordinates(p, res, x0, y0, z0, wx, wy, wz);
    //* 三线性插值权重
    float w[8];
    int i = 0;
    for (int dx = 0; dx <= 1; ++dx)
      for (int dy = 0; dy <= 1; ++dy)
        for (int dz = 0; dz <= 1; ++dz)
          w[i++] = (dx ? wx : 1.f - wx) * (dy ? wy : 1.f - wy) *
                   (dz ? wz : 1.f - wz);
    //* 8个顶点的参数下标
    int idx[8];
    i = 0;
    for (int dx = 0; dx <= 1; ++dx)
      for (int dy = 0; dy <= 1; ++dy)
        for (int dz = 0; dz <= 1; ++dz)
          idx[i++] = parameterIndex(level, x0 + dx, y0 + dy, z0 + dz);
    for (int f = 0; f < F; ++f) {
      float v = .0f;
      for (int c = 0; c < 8; ++c) {
        v += w[c] * params[(size_t)idx[c] * F + f];
      }
      out[level * F + f] = v;
    }
  }
}

void HashGrid::backward(const Vector3f &p, const float *gradOut,
                        float *gradParams) const {
  int F = featuresPerLevel;
  for (int level = 0; level < numLevels; ++level) {
    int res = gridRes(baseRes, level);
    int x0, y0, z0;
    float wx, wy, wz;
    gridCoordinates(p, res, x0, y0, z0, wx, wy, wz);
    float w[8];
    int i = 0;
    for (int dx = 0; dx <= 1; ++dx)
      for (int dy = 0; dy <= 1; ++dy)
        for (int dz = 0; dz <= 1; ++dz)
          w[i++] = (dx ? wx : 1.f - wx) * (dy ? wy : 1.f - wy) *
                   (dz ? wz : 1.f - wz);
    int idx[8];
    i = 0;
    for (int dx = 0; dx <= 1; ++dx)
      for (int dy = 0; dy <= 1; ++dy)
        for (int dz = 0; dz <= 1; ++dz)
          idx[i++] = parameterIndex(level, x0 + dx, y0 + dy, z0 + dz);
    for (int f = 0; f < F; ++f) {
      float g = gradOut[level * F + f];
      if (g == .0f)
        continue;
      for (int c = 0; c < 8; ++c) {
        gradParams[(size_t)idx[c] * F + f] += w[c] * g;
      }
    }
  }
}

void HashGrid::backwardAtomic(const Vector3f &p, const float *gradOut,
                              std::atomic<uint32_t> *gradParams) const {
  int F = featuresPerLevel;
  for (int level = 0; level < numLevels; ++level) {
    int res = gridRes(baseRes, level);
    int x0, y0, z0;
    float wx, wy, wz;
    gridCoordinates(p, res, x0, y0, z0, wx, wy, wz);
    float w[8];
    int i = 0;
    for (int dx = 0; dx <= 1; ++dx)
      for (int dy = 0; dy <= 1; ++dy)
        for (int dz = 0; dz <= 1; ++dz)
          w[i++] = (dx ? wx : 1.f - wx) * (dy ? wy : 1.f - wy) *
                   (dz ? wz : 1.f - wz);
    int idx[8];
    i = 0;
    for (int dx = 0; dx <= 1; ++dx)
      for (int dy = 0; dy <= 1; ++dy)
        for (int dz = 0; dz <= 1; ++dz)
          idx[i++] = parameterIndex(level, x0 + dx, y0 + dy, z0 + dz);
    for (int f = 0; f < F; ++f) {
      float g = gradOut[level * F + f];
      if (g == .0f)
        continue;
      for (int c = 0; c < 8; ++c) {
        atomicAddFloat(&gradParams[(size_t)idx[c] * F + f], w[c] * g);
      }
    }
  }
}
