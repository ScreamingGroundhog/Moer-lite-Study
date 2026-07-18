#include "Cone.h"
#include "ResourceLayer/Factory.h"

bool Cone::rayIntersectShape(Ray &ray, int *primID, float *u, float *v) const {
    //* todo 完成光线与圆柱的相交 填充primId,u,v.如果相交，更新光线的tFar
    //* 1.光线变换到局部空间
    auto inv_ray = std::move(transform.inverseRay(ray));
    //* 2.联立方程求解
    float k = radius / height;
    float dx = inv_ray.direction[0], dy = inv_ray.direction[1], dz = inv_ray.direction[2];
    float ox = inv_ray.origin[0], oy = inv_ray.origin[1], oz = inv_ray.origin[2];

    float A = dx * dx + dy * dy - k * k * dz * dz;
    float B = 2.f * (ox * dx + oy * dy + k * dz * (radius - k * oz));
    float C = ox * ox + oy * oy - (radius - k * oz) * (radius - k * oz);

    float t0, t1;
    if (!Quadratic(A, B, C, &t0, &t1)) {
        return false;
    }
    if (t0 > inv_ray.tFar || t1 < inv_ray.tNear) {
        return false;
    }
    //* 3.检验交点是否在圆锥范围内
    float tHit = inv_ray.tFar;
    for (float t : {t0, t1}) {
        if (t < inv_ray.tNear || t > inv_ray.tFar)
            continue;
        float z = oz + t * dz;
        if (z < 0 || z > height)
            continue;
        if (t < tHit)
            tHit = t;
    }
    if (tHit >= inv_ray.tFar)
        return false;
    //* 4.更新ray的tFar,减少光线和其他物体的相交计算次数
    ray.tFar = tHit;
    Point3f hitPoint = inv_ray.at(tHit);
    float phi = std::atan2(hitPoint[1], hitPoint[0]);
    phi = phi > 0 ? phi : phi + PI * 2;
    if (phi > phiMax)
        return false;
    *primID = 0;
    *u = phi / phiMax;
    *v = hitPoint[2] / height;
    return true;
}

void Cone::fillIntersection(float distance, int primID, float u, float v, Intersection *intersection) const {
    /// ----------------------------------------------------
    //* todo 填充圆锥相交信息中的法线以及相交位置信息
    //* 1.法线可以先计算出局部空间的法线，然后变换到世界空间
    //* 2.位置信息可以根据uv计算出，同样需要变换
    //* Write your code here.
    /// ----------------------------------------------------
    float phi = u * phiMax;
    float z = v * height;
    float r_xy = radius * (1.f - v);
    float sinTheta = std::sqrt(1.f - cosTheta * cosTheta);

    Point3f localPos(r_xy * std::cos(phi), r_xy * std::sin(phi), z);
    intersection->position = transform.toWorld(localPos);
    Vector3f localNormal(cosTheta * std::cos(phi), cosTheta * std::sin(phi), sinTheta);
    intersection->normal = normalize(transform.toWorld(localNormal));

    intersection->shape = this;
    intersection->distance = distance;
    intersection->texCoord = Vector2f{u, v};
    Vector3f tangent{1.f, 0.f, .0f};
    Vector3f bitangent;
    if (std::abs(dot(tangent, intersection->normal)) > .9f) {
        tangent = Vector3f(.0f, 1.f, .0f);
    }
    bitangent = normalize(cross(tangent, intersection->normal));
    tangent = normalize(cross(intersection->normal, bitangent));
    intersection->tangent = tangent;
    intersection->bitangent = bitangent;
}

void Cone::uniformSampleOnSurface(Vector2f sample, Intersection *result, float *pdf) const {

}

Cone::Cone(const Json &json) : Shape(json) {
    radius = fetchOptional(json, "radius", 1.f);
    height = fetchOptional(json, "height", 1.f);
    phiMax = fetchOptional(json, "phi_max", 2 * PI);
    float tanTheta = radius / height;
    cosTheta = sqrt(1/(1+tanTheta * tanTheta));
    //theta = fetchOptional(json,)
    AABB localAABB = AABB(Point3f(-radius,-radius,0),Point3f(radius,radius,height));
    boundingBox = transform.toWorld(localAABB);
    boundingBox = AABB(Point3f(-100,-100,-100),Point3f(100,100,100));
}

REGISTER_CLASS(Cone, "cone")
