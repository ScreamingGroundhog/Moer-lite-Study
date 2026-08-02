#include "MLP.h"
#include <algorithm>
#include <cmath>
#include <cstdint>

void MLP::init(int inDim, int h1, int h2, int outDim) {
  this->inDim = inDim;
  this->h1 = h1;
  this->h2 = h2;
  this->outDim = outDim;
  offW1 = 0;
  offB1 = h1 * inDim;
  offW2 = offB1 + h1;
  offB2 = offW2 + h2 * h1;
  offW3 = offB2 + h2;
  offB3 = offW3 + outDim * h2;
  params.assign(offB3 + outDim, .0f);
  //* Xavier初始化
  auto initWeight = [](float *w, int fanIn, int fanOut) {
    float scale = std::sqrt(6.f / (fanIn + fanOut));
    uint32_t seed = (uint32_t)(fanIn * 73856093u ^ fanOut * 19349663u) + 1u;
    auto nextRand = [&seed]() {
      seed = seed * 1664525u + 1013904223u;
      return (seed >> 8) / 16777216.f; // [0, 1)
    };
    for (int i = 0; i < fanIn * fanOut; ++i) {
      w[i] = (2.f * nextRand() - 1.f) * scale;
    }
  };
  initWeight(&params[offW1], inDim, h1);
  initWeight(&params[offW2], h1, h2);
  initWeight(&params[offW3], h2, outDim);
}

void MLP::forward(const float *in, float *out) const {
  const float *W1 = &params[offW1], *b1 = &params[offB1];
  const float *W2 = &params[offW2], *b2 = &params[offB2];
  const float *W3 = &params[offW3], *b3 = &params[offB3];
  //* 隐藏层1
  float a1[512];
  for (int j = 0; j < h1; ++j) {
    float sum = b1[j];
    const float *row = W1 + j * inDim;
    for (int i = 0; i < inDim; ++i)
      sum += row[i] * in[i];
    a1[j] = std::max(sum, .0f);
  }
  //* 隐藏层2
  float a2[512];
  for (int j = 0; j < h2; ++j) {
    float sum = b2[j];
    const float *row = W2 + j * h1;
    for (int i = 0; i < h1; ++i)
      sum += row[i] * a1[i];
    a2[j] = std::max(sum, .0f);
  }
  //* 输出层(线性)
  for (int j = 0; j < outDim; ++j) {
    float sum = b3[j];
    const float *row = W3 + j * h2;
    for (int i = 0; i < h2; ++i)
      sum += row[i] * a2[i];
    out[j] = sum;
  }
}

void MLP::forwardBackward(const float *in, const float *gradOut,
                          float *gradParams, float *gradIn) const {
  const float *W1 = &params[offW1];
  const float *W2 = &params[offW2];
  const float *W3 = &params[offW3];
  float *gW1 = gradParams + offW1, *gB1 = gradParams + offB1;
  float *gW2 = gradParams + offW2, *gB2 = gradParams + offB2;
  float *gW3 = gradParams + offW3, *gB3 = gradParams + offB3;
  //* 前向
  float a1[512], z1[512], a2[512], z2[512];
  for (int j = 0; j < h1; ++j) {
    float sum = 0.f;
    const float *row = W1 + j * inDim;
    for (int i = 0; i < inDim; ++i)
      sum += row[i] * in[i];
    z1[j] = sum;
    a1[j] = std::max(sum, .0f);
  }
  for (int j = 0; j < h2; ++j) {
    float sum = 0.f;
    const float *row = W2 + j * h1;
    for (int i = 0; i < h1; ++i)
      sum += row[i] * a1[i];
    z2[j] = sum;
    a2[j] = std::max(sum, .0f);
  }
  //* 输出层反向
  float da2[512];
  for (int j = 0; j < h2; ++j)
    da2[j] = .0f;
  for (int j = 0; j < outDim; ++j) {
    float g = gradOut[j];
    const float *row = W3 + j * h2;
    for (int i = 0; i < h2; ++i) {
      gW3[j * h2 + i] += g * a2[i];
      da2[i] += g * row[i];
    }
    gB3[j] += g;
  }
  //* 隐藏层2反向
  float da1[512];
  for (int j = 0; j < h1; ++j)
    da1[j] = .0f;
  for (int j = 0; j < h2; ++j) {
    float g = da2[j] * (z2[j] > .0f ? 1.f : .0f);
    const float *row = W2 + j * h1;
    for (int i = 0; i < h1; ++i) {
      gW2[j * h1 + i] += g * a1[i];
      da1[i] += g * row[i];
    }
    gB2[j] += g;
  }
  //* 隐藏层1反向
  for (int j = 0; j < h1; ++j) {
    float g = da1[j] * (z1[j] > .0f ? 1.f : .0f);
    const float *row = W1 + j * inDim;
    for (int i = 0; i < inDim; ++i) {
      gW1[j * inDim + i] += g * in[i];
      if (gradIn)
        gradIn[i] += g * row[i];
    }
    gB1[j] += g;
  }
}

void AdamOptimizer::init(int numParams) {
  m.assign(numParams, .0f);
  v.assign(numParams, .0f);
  t = 0;
}

void AdamOptimizer::step(float *params, const float *grad, float lr) {
  ++t;
  stepRange(params, grad, lr, 0, (int)m.size());
}

void AdamOptimizer::stepRange(float *params, const float *grad, float lr,
                              int begin, int end) {
  float bc1 = 1.f - std::pow(beta1, (float)t);
  float bc2 = 1.f - std::pow(beta2, (float)t);
  for (int i = begin; i < end; ++i) {
    float g = grad[i];
    float mi = beta1 * m[i] + (1.f - beta1) * g;
    float vi = beta2 * v[i] + (1.f - beta2) * g * g;
    m[i] = mi;
    v[i] = vi;
    params[i] -= lr * (mi / bc1) / (std::sqrt(vi / bc2) + eps);
  }
}
