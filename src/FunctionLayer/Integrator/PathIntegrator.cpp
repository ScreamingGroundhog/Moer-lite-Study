#include "PathIntegrator.h"
#include <FunctionLayer/Material/Material.h>
#include <ResourceLayer/Factory.h>

PathIntegrator::PathIntegrator(const Json &json) {
  maxDepth = fetchOptional<int>(json, "maxDepth", maxDepth);
  rrDepth = fetchOptional<int>(json, "rrDepth", rrDepth);
  rrProbability =
      fetchOptional<float>(json, "rrProbability", rrProbability);
}

namespace {
//* 光线未击中任何几何体时，累加环境光的radiance
inline Spectrum evaluateInfiniteLights(const Ray &ray, const Scene &scene) {
  Spectrum s(.0f);
  for (auto light : scene.infiniteLights) {
    s += light->evaluateEmission(ray);
  }
  return s;
}
} // namespace

Spectrum PathIntegrator::li(Ray &ray, const Scene &scene,
                            std::shared_ptr<Sampler> sampler) const {
  return liWithSamples(ray, scene, sampler, nullptr);
}

Spectrum
PathIntegrator::liWithSamples(Ray &ray, const Scene &scene,
                              std::shared_ptr<Sampler> sampler,
                              const std::function<void(const RadianceSample &)>
                                  &onSample) const {
  //* 递归路径追踪：
  //* L(p, wo) = Le + NEE(直接光照) + BSDF采样延续路径(间接光照)
  //* 顶点处的radiance估计值在计算出全部贡献后通过onSample记录，供NRC训练使用
  std::function<Spectrum(Ray &, int, bool)> trace =
      [&](Ray &r, int depth, bool specularBounce) -> Spectrum {
    auto hitOpt = scene.rayIntersect(r);
    //* 未击中任何物体：环境光
    if (!hitOpt.has_value()) {
      return evaluateInfiniteLights(r, scene);
    }
    Intersection its = hitOpt.value();
    Spectrum L(.0f);

    //* 直接击中光源：
    //* - 相机光线(depth==0)直接击中光源
    //* - 镜面弹射击中光源(镜面无法通过NEE采样直接光照，这里以权重1加入发光)
    bool isLight = its.shape->light != nullptr;
    if (isLight && (depth == 0 || specularBounce)) {
      L += its.shape->light->evaluateEmission(its, -r.direction);
    }

    auto material = its.shape->material;
    if (!material) {
      return L;
    }
    auto bsdf = material->computeBSDF(its);
    bool specular = bsdf->isSpecular();

    //* 直接光照：从交点采样光源并连接(NEE)
    //* delta BSDF(镜面)对NEE的贡献为0，跳过
    if (!specular) {
      float pdfLight = .0f;
      auto light = scene.sampleLight(sampler->next1D(), &pdfLight);
      if (light && pdfLight != .0f) {
        auto lightSampleResult = light->sample(its, sampler->next2D());
        Ray shadowRay{its.position, lightSampleResult.direction, 1e-4f,
                      lightSampleResult.distance};
        if (!scene.rayIntersect(shadowRay).has_value()) {
          Spectrum f = bsdf->f(-r.direction, lightSampleResult.direction);
          lightSampleResult.pdf *= pdfLight;
          float pdf = convertPDF(lightSampleResult, its);
          L += lightSampleResult.energy * f / pdf;
        }
      }
    }

    //* 俄罗斯轮盘赌：深度较深后以一定概率终止路径
    bool useRR = depth >= rrDepth;
    bool continuePath = !useRR || sampler->next1D() <= rrProbability;

    //* 采样BSDF延展路径
    if (continuePath && depth < maxDepth) {
      auto bsdfSampleResult = bsdf->sample(-r.direction, sampler->next2D());
      if (bsdfSampleResult.pdf > .0f && !bsdfSampleResult.weight.isZero()) {
        float weightScale = useRR ? 1.f / rrProbability : 1.f;
        Ray nextRay{its.position, bsdfSampleResult.wi, 1e-4f};
        auto nextHitOpt = scene.rayIntersect(nextRay);
        Spectrum continuation(.0f);
        if (!nextHitOpt.has_value()) {
          //* 环境光
          continuation = evaluateInfiniteLights(nextRay, scene);
        } else {
          auto &nextHit = nextHitOpt.value();
          bool nextIsLight = nextHit.shape->light != nullptr;
          if (specular && nextIsLight) {
            //* 镜面弹射击中光源：直接加入发光
            continuation =
                nextHit.shape->light->evaluateEmission(nextHit, -nextRay.direction);
          } else {
            //* 继续递归路径追踪(镜面弹射后若击中光源不再发光，由上一顶点的NEE负责)
            continuation = trace(nextRay, depth + 1, specular);
          }
        }
        L += bsdfSampleResult.weight * continuation * weightScale;
      }
    }

    //* 记录该顶点沿wo方向的radiance估计值(即本函数返回的L)
    if (onSample) {
      onSample(RadianceSample{its.position, its.normal, -r.direction, L,
                              depth});
    }
    return L;
  };
  return trace(ray, 0, false);
}

REGISTER_CLASS(PathIntegrator, "pathTracer")
