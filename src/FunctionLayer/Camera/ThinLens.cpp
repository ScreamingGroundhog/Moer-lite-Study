#include "ThinLens.h"
#include <CoreLayer/Math/Constant.h>
#include <cmath>

ThinLensCamera::ThinLensCamera(const Json& json) : PerspectiveCamera(json) {
    lensRadius    = json["lensRadius"].get<float>();
    focalDistance = json["focalDistance"].get<float>();
}

Ray ThinLensCamera::sampleRay(const CameraSample& sample, Vector2f NDC) const {
    // Step 1: Compute the film point (same as pinhole camera)
    float x = (NDC[0] - 0.5f) * film->size[0] + sample.xy[0],
          y = (0.5f - NDC[1]) * film->size[1] + sample.xy[1];

    float tanHalfFov = fm::tan(verticalFov * 0.5f);
    float z = -film->size[1] * 0.5f / tanHalfFov;

    // Step 2: Compute pFocus -- the intersection of pinhole ray with focal plane
    // Pinhole ray from origin through film point: O + t * (x, y, z)
    // Focal plane is at z = -focalDistance
    // t * z = -focalDistance  =>  t = -focalDistance / z
    float t = -focalDistance / z;
    Point3f pFocus(t * x, t * y, -focalDistance);

    // Step 3: Uniformly sample a point on the lens (disk of radius lensRadius)
    float r     = lensRadius * std::sqrt(sample.lens[0]);
    float theta = 2.0f * PI * sample.lens[1];
    Point3f lensPoint(r * std::cos(theta), r * std::sin(theta), 0.0f);

    // Step 4: Ray direction from lens point toward pFocus
    Vector3f direction = pFocus - lensPoint;

    // Step 5: Transform to world space
    direction = transform.toWorld(direction);
    Point3f origin = transform.toWorld(lensPoint);

    return Ray(origin, direction, tNear, tFar, timeStart);
}

Ray ThinLensCamera::sampleRayDifferentials(const CameraSample& sample, Vector2f NDC) const {
    // Compute the film point (same as sampleRay)
    float x = (NDC[0] - 0.5f) * film->size[0] + sample.xy[0],
          y = (0.5f - NDC[1]) * film->size[1] + sample.xy[1];

    float tanHalfFov = fm::tan(verticalFov * 0.5f);
    float z = -film->size[1] * 0.5f / tanHalfFov;

    // Compute pFocus and offset versions for ray differentials
    float t = -focalDistance / z;
    Point3f pFocus(t * x, t * y, -focalDistance);
    Point3f pFocusX(t * (x + 1.f), t * y, -focalDistance);
    Point3f pFocusY(t * x, t * (y + 1.f), -focalDistance);

    // Sample the lens (use the same lens point for all three rays)
    float r     = lensRadius * std::sqrt(sample.lens[0]);
    float theta = 2.0f * PI * sample.lens[1];
    Point3f lensPoint(r * std::cos(theta), r * std::sin(theta), 0.0f);

    // Directions for main and offset rays
    Vector3f direction  = transform.toWorld(pFocus - lensPoint);
    Vector3f directionX = transform.toWorld(pFocusX - lensPoint);
    Vector3f directionY = transform.toWorld(pFocusY - lensPoint);
    Point3f origin = transform.toWorld(lensPoint);

    Ray ret = Ray(origin, direction, tNear, tFar, timeStart);
    ret.hasDifferentials = true;
    ret.directionX = directionX;
    ret.directionY = directionY;
    ret.originX = ret.originY = origin;
    return ret;
}

REGISTER_CLASS(ThinLensCamera, "thinlens")