#include "Triangle.h"
#include <FunctionLayer/Acceleration/Linear.h>
//--- Triangle ---
Triangle::Triangle(int _primID, int _vtx0Idx, int _vtx1Idx, int _vtx2Idx,
                   const TriangleMesh *_mesh)
    : primID(_primID), vtx0Idx(_vtx0Idx), vtx1Idx(_vtx1Idx), vtx2Idx(_vtx2Idx),
      mesh(_mesh) {
  Point3f vtx0 = mesh->transform.toWorld(mesh->meshData->vertexBuffer[vtx0Idx]),
          vtx1 = mesh->transform.toWorld(mesh->meshData->vertexBuffer[vtx1Idx]),
          vtx2 = mesh->transform.toWorld(mesh->meshData->vertexBuffer[vtx2Idx]);
  boundingBox.Expand(vtx0);
  boundingBox.Expand(vtx1);
  boundingBox.Expand(vtx2);
  this->geometryID = mesh->geometryID;
}

bool Triangle::rayIntersectShape(Ray &ray, int *primID, float *u,
                                 float *v) const {
  //* todo 实现三角形与光线求交
  Point3f v0 = mesh->transform.toWorld(mesh->meshData->vertexBuffer[vtx0Idx]);
  Point3f v1 = mesh->transform.toWorld(mesh->meshData->vertexBuffer[vtx1Idx]);
  Point3f v2 = mesh->transform.toWorld(mesh->meshData->vertexBuffer[vtx2Idx]);

  Vector3f e1 = v1 - v0;
  Vector3f e2 = v2 - v0;
  Vector3f s = ray.origin - v0;
  Vector3f s1 = cross(ray.direction, e2);
  Vector3f s2 = cross(s, e1);

  float invDet = 1.f / dot(s1, e1);

  float t = dot(s2, e2) * invDet;
  float b1 = dot(s1, s) * invDet;
  float b2 = dot(s2, ray.direction) * invDet;

  if (t < ray.tNear || t > ray.tFar)
    return false;
  if (b1 < 0 || b2 < 0 || b1 + b2 > 1)
    return false;

  ray.tFar = t;
  *primID = this->primID;
  *u = b1;
  *v = b2;
  return true;
}

void Triangle::fillIntersection(float distance, int primID, float u, float v,
                                Intersection *intersection) const {
  // 该函数实际上不会被调用
  return;
}

//--- TriangleMesh ---
TriangleMesh::TriangleMesh(const Json &json) : Shape(json) {
  const auto &filepath = fetchRequired<std::string>(json, "file");
  meshData = MeshData::loadFromFile(filepath);
}

RTCGeometry TriangleMesh::getEmbreeGeometry(RTCDevice device) const {
  RTCGeometry geometry = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);

  float *vertexBuffer = (float *)rtcSetNewGeometryBuffer(
      geometry, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, 3 * sizeof(float),
      meshData->vertexCount);
  for (int i = 0; i < meshData->vertexCount; ++i) {
    Point3f vertex = transform.toWorld(meshData->vertexBuffer[i]);
    vertexBuffer[3 * i] = vertex[0];
    vertexBuffer[3 * i + 1] = vertex[1];
    vertexBuffer[3 * i + 2] = vertex[2];
  }

  unsigned *indexBuffer = (unsigned *)rtcSetNewGeometryBuffer(
      geometry, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
      3 * sizeof(unsigned), meshData->faceCount);
  for (int i = 0; i < meshData->faceCount; ++i) {
    indexBuffer[i * 3] = meshData->faceBuffer[i][0].vertexIndex;
    indexBuffer[i * 3 + 1] = meshData->faceBuffer[i][1].vertexIndex;
    indexBuffer[i * 3 + 2] = meshData->faceBuffer[i][2].vertexIndex;
  }
  rtcCommitGeometry(geometry);
  return geometry;
}

bool TriangleMesh::rayIntersectShape(Ray &ray, int *primID, float *u,
                                     float *v) const {
  //* 当使用embree加速时，该方法不会被调用
  int geomID = -1;
  return acceleration->rayIntersect(ray, &geomID, primID, u, v);
}

void TriangleMesh::fillIntersection(float distance, int primID, float u,
                                    float v, Intersection *intersection) const {
  //* todo 填充光线与三角网格求交得到的交点信息
  intersection->distance = distance;
  intersection->shape = this;
  //* 1. 在三角形内部用插值计算交点坐标
  //* 2. 在三角形内部用插值计算法线
  //* 3. 在三角形内部用插值计算纹理坐标
  //* 4. 在三角形内部用插值计算交点的切线和副切线
  const auto &face = meshData->faceBuffer[primID];
  int i0 = face[0].vertexIndex, i1 = face[1].vertexIndex, i2 = face[2].vertexIndex;
  Point3f v0 = transform.toWorld(meshData->vertexBuffer[i0]);
  Point3f v1 = transform.toWorld(meshData->vertexBuffer[i1]);
  Point3f v2 = transform.toWorld(meshData->vertexBuffer[i2]);

  float w = 1.f - u - v;

  //* 1. 插值计算交点坐标
  intersection->position = v0 + u * (v1 - v0) + v * (v2 - v0);

  //* 2. 插值计算法线
  if (!meshData->normalBuffer.empty()) {
    int ni0 = face[0].normalIndex, ni1 = face[1].normalIndex, ni2 = face[2].normalIndex;
    Vector3f n0 = normalize(transform.toWorld(meshData->normalBuffer[ni0]));
    Vector3f n1 = normalize(transform.toWorld(meshData->normalBuffer[ni1]));
    Vector3f n2 = normalize(transform.toWorld(meshData->normalBuffer[ni2]));
    intersection->normal = normalize(n0 + u * (n1 - n0) + v * (n2 - n0));
  } else {
    intersection->normal = normalize(cross(v1 - v0, v2 - v0));
  }

  //* 3. 插值计算纹理坐标
  if (!meshData->texcodBuffer.empty()) {
    int ti0 = face[0].texcodIndex, ti1 = face[1].texcodIndex, ti2 = face[2].texcodIndex;
    Vector2f t0 = meshData->texcodBuffer[ti0];
    Vector2f t1 = meshData->texcodBuffer[ti1];
    Vector2f t2 = meshData->texcodBuffer[ti2];
    intersection->texCoord = t0 + u * (t1 - t0) + v * (t2 - t0);
  } else {
    intersection->texCoord = Vector2f{u, v};
  }

  //* 4. 计算交点的切线和副切线
  Vector3f dpdu = v1 - v0;
  Vector3f dpdv = v2 - v0;
  intersection->dpdu = dpdu;
  intersection->dpdv = dpdv;
  intersection->tangent = normalize(dpdu);
  intersection->bitangent = normalize(cross(intersection->tangent, intersection->normal));
}

void TriangleMesh::initInternalAcceleration() {
  acceleration = Acceleration::createAcceleration();
  int primCount = meshData->faceCount;
  for (int primID = 0; primID < primCount; ++primID) {
    int vtx0Idx = meshData->faceBuffer[primID][0].vertexIndex,
        vtx1Idx = meshData->faceBuffer[primID][1].vertexIndex,
        vtx2Idx = meshData->faceBuffer[primID][2].vertexIndex;
    std::shared_ptr<Triangle> triangle =
        std::make_shared<Triangle>(primID, vtx0Idx, vtx1Idx, vtx2Idx, this);
    acceleration->attachShape(triangle);
  }
  acceleration->build();
  // TriangleMesh的包围盒就是其内部加速结构的包围盒
  boundingBox = acceleration->boundingBox;
}
REGISTER_CLASS(TriangleMesh, "triangle")