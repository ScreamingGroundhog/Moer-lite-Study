#include "Cylinder.h"
#include "ResourceLayer/Factory.h"
bool Cylinder::rayIntersectShape(Ray &ray, int *primID, float *u,
								 float *v) const {
	//* todo 完成光线与圆柱的相交 填充primId,u,v.如果相交，更新光线的tFar
	//* 1.光线变换到局部空间
	auto inv_ray = std::move(transform.inverseRay(ray));
	//* 2.联立方程求解
	float A = inv_ray.direction[0] * inv_ray.direction[0] +
			  inv_ray.direction[1] * inv_ray.direction[1];
	float B = 2 * (inv_ray.origin[0] * inv_ray.direction[0] +
				   inv_ray.origin[1] * inv_ray.direction[1]);
	float C = inv_ray.origin[0] * inv_ray.origin[0] +
			  inv_ray.origin[1] * inv_ray.origin[1] - radius * radius;
	float t0, t1;
	if (!Quadratic(A, B, C, &t0, &t1)) {
		return false;
	}
	if (t0 > inv_ray.tFar || t1 < inv_ray.tNear) {
		return false;
	}
	//* 3.检验交点是否在圆柱范围内
	float tHit = inv_ray.tFar;
	for (float t : {t0, t1}) {
		if (t < inv_ray.tNear || t > inv_ray.tFar)
			continue;
		float z = inv_ray.origin[2] + t * inv_ray.direction[2];
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
	*primID = 0;
	*u = phi / phiMax;
	*v = (hitPoint[2]) / height;
	return true;
}

void Cylinder::fillIntersection(float distance, int primID, float u, float v,
								Intersection *intersection) const {
	/// ----------------------------------------------------
	//* todo 填充圆柱相交信息中的法线以及相交位置信息
	//* 1.法线可以先计算出局部空间的法线，然后变换到世界空间
	//* 2.位置信息可以根据uv计算出，同样需要变换
	//* Write your code here.
	/// ----------------------------------------------------
	float phi = u * phiMax;
	float z = v * height;

	intersection->position = transform.toWorld(
		Point3f(radius * std::cos(phi), radius * std::sin(phi), z));
	intersection->normal = normalize(
		transform.toWorld(Vector3f(std::cos(phi), std::sin(phi), 0.f)));

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

void Cylinder::uniformSampleOnSurface(Vector2f sample, Intersection *result,
									  float *pdf) const {}

Cylinder::Cylinder(const Json &json) : Shape(json) {
	radius = fetchOptional(json, "radius", 1.f);
	height = fetchOptional(json, "height", 1.f);
	phiMax = fetchOptional(json, "phi_max", 2 * PI);
	AABB localAABB =
		AABB(Point3f(-radius, -radius, 0), Point3f(radius, radius, height));
	boundingBox = transform.toWorld(localAABB);
}

REGISTER_CLASS(Cylinder, "cylinder")
