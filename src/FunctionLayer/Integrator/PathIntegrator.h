#pragma once
#include "Integrator.h"
#include <functional>

//* 路径追踪过程中顶点处的radiance样本，用于NRC训练数据的收集
struct RadianceSample {
  Point3f position;   // 顶点位置
  Vector3f normal;    // 顶点法线
  Vector3f wo;        // 出射方向(指向上一顶点)
  Spectrum radiance;  // 沿wo方向的出射radiance估计值
  int depth;          // 顶点所在的弹射深度(0为相机光线的第一个交点)
};

//* 蒙特卡洛路径追踪
//* - 直接光照通过NEE(对光源采样)得到，环境光通过BSDF采样后的miss得到
//* - delta BSDF(镜面)无法通过NEE采样直接光照，在镜面弹射击中光源时以权重1加入发光
//* - 可以注册回调收集每个顶点处的radiance样本(用于神经辐射缓存的训练数据)
class PathIntegrator : public Integrator {
public:
  PathIntegrator() = default;

  PathIntegrator(const Json &json);

  virtual Spectrum li(Ray &ray, const Scene &scene,
                      std::shared_ptr<Sampler> sampler) const override;

  //* 带样本收集的路径追踪：在每个顶点计算完radiance后调用onSample
  //* onSample为空时与li完全一致
  Spectrum liWithSamples(Ray &ray, const Scene &scene,
                         std::shared_ptr<Sampler> sampler,
                         const std::function<void(const RadianceSample &)>
                             &onSample) const;

public:
  int maxDepth = 8;       // 最大弹射深度
  int rrDepth = 3;        // 深度大于rrDepth后开始俄罗斯轮盘赌
  float rrProbability = 0.8f; // 轮盘赌继续的概率
};
