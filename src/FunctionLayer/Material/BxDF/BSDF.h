#pragma once
#include <CoreLayer/ColorSpace/Spectrum.h>
#include <FunctionLayer/Shape/Intersection.h>

enum class BSDFType { Diffuse, Specular };

struct BSDFSampleResult {
  Spectrum weight;
  Vector3f wi;
  float pdf;
  BSDFType type;
};

class BSDF {
public:
  BSDF(const Vector3f &_normal, const Vector3f &_tangent,
       const Vector3f &_bitangent) {
    normal = _normal;
    tangent = _tangent;
    bitangent = _bitangent;
  }
  virtual Spectrum f(const Vector3f &wo, const Vector3f &wi) const = 0;
  virtual BSDFSampleResult sample(const Vector3f &wo,
                                  const Vector2f &sample) const = 0;

  //* 是否为delta(dirac)类型的BSDF(如镜面反射)
  //* delta类型的BSDF无法通过NEE得到直接光照，需要特殊处理
  virtual bool isSpecular() const { return false; }

public:
  Vector3f normal, tangent, bitangent; // 构成局部坐标系
protected:
  Vector3f toLocal(Vector3f world) const {
    return Vector3f{dot(tangent, world), dot(normal, world),
                    dot(bitangent, world)};
  }
  Vector3f toWorld(Vector3f local) const {
    return local[0] * tangent + local[1] * normal + local[2] * bitangent;
  }
};