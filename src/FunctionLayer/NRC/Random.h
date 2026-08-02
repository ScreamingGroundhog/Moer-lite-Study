#pragma once
#include <CoreLayer/Math/Math.h>
#include <FunctionLayer/Sampler/Sampler.h>
#include <random>

//* 线程局部随机数工具
class Random {
public:
  Random(uint64_t seed) : gen(seed) {}

  float next1D() {
    return std::uniform_real_distribution<float>(.0f, 1.f)(gen);
  }

  Vector2f next2D() { return Vector2f{next1D(), next1D()}; }

  std::mt19937_64 gen;
};

//* 将Random包装为框架的Sampler接口
class RandomSampler : public Sampler {
public:
  RandomSampler(Random &rng)
      : Sampler(Json{{"xSamples", 1}, {"ySamples", 1}}), rng(rng) {}

  virtual float next1D() override { return rng.next1D(); }

  virtual Vector2f next2D() override { return rng.next2D(); }

  Random &rng;
};
