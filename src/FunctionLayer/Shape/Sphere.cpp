#include "Sphere.h"
#include <ResourceLayer/Factory.h>

Sphere::Sphere(const Json &json) : Shape(json) {
  center = fetchRequired<Point3f>(json, "center");
  radius = fetchRequired<float>(json, "radius");

  center = transform.toWorld(center);
  boundingBox = AABB(center - Vector3f(radius), center + Vector3f(radius));
}

bool Sphere::rayIntersectShape(Ray &ray, int *primID, float *u,
                               float *v) const {
  Point3f origin = ray.origin;
  Vector3f direction = ray.direction;
  Vector3f o2c = center - origin;
  float b = dot(o2c, direction);
  float c = o2c.length() * o2c.length() - radius * radius;
  float delta = b * b - c;
  if (delta <= 0)
    return false; // 不相交
  float sqrtDelta = fm::sqrt(delta);
  float t1 = b - sqrtDelta;
  float t2 = b + sqrtDelta;

  bool hit = false;
  if (ray.tNear <= t2 && t2 <= ray.tFar) {
    ray.tFar = t2;
    hit = true;
  }
  if (ray.tNear <= t1 && t1 <= ray.tFar) {
    ray.tFar = t1;
    hit = true;
  }
  if (hit) {
    *primID = 0;
    //* 计算u,v
    // TODO 需要考虑旋转
    Vector3f worldNormal = normalize(ray.at(ray.tFar) - center);
    // 将世界空间法线变换到局部空间以正确处理旋转
    vecmat::vec4f localN = transform.invRotate * vecmat::vec4f(worldNormal[0], worldNormal[1], worldNormal[2], 0.f);
    Vector3f localNormal = normalize(Vector3f(localN[0], localN[1], localN[2]));
    float cosTheta = localNormal[1];
    *v = fm::acos(cosTheta);
    if (std::abs(localNormal[2]) < 1e-4f) {
      *u = (localNormal[0] > .0f) ? (PI * .5f) : (PI * 1.5f);
    } else {
      float tanPhi = localNormal[0] / localNormal[2];
      *u = fm::atan(tanPhi); // u in [-.5f * PI, .5f * PI]
      if (localNormal[2] < .0f)
        *u += PI;
    }
  }
  return hit;
}

void Sphere::fillIntersection(float distance, int primID, float u, float v,
                              Intersection *intersection) const {
  // u->phi, v->theta

  intersection->shape = this;
  intersection->distance = distance;
  // 从uv重建局部空间法线，然后变换到世界空间
  Vector3f localNormal = Vector3f{std::sin(v) * std::sin(u), std::cos(v),
                                  std::sin(v) * std::cos(u)};
  Vector3f worldNormal = normalize(transform.toWorld(localNormal));

  intersection->normal = worldNormal;

  //* 计算交点
  Point3f position = center + radius * worldNormal;
  intersection->position = position;

  //* 计算纹理坐标
  intersection->texCoord = Vector2f{u * INV_PI * .5f, v * INV_PI};

  // TODO 计算交点的切线和副切线
  Vector3f dpdu_local =
      radius * Vector3f{std::sin(v) * std::cos(u), 0.f, -std::sin(v) * std::sin(u)};
  Vector3f dpdv_local =
      radius * Vector3f{std::cos(v) * std::sin(u), -std::sin(v), std::cos(v) * std::cos(u)};
  intersection->dpdu = transform.toWorld(dpdu_local);
  intersection->dpdv = transform.toWorld(dpdv_local);
  intersection->tangent = normalize(intersection->dpdu);
  intersection->bitangent =
      normalize(cross(intersection->tangent, intersection->normal));
}

REGISTER_CLASS(Sphere, "sphere")