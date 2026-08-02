#pragma once
#include <CoreLayer/ColorSpace/Spectrum.h>
#include <CoreLayer/Math/Math.h>
#include <FunctionLayer/Camera/Camera.h>
#include <FunctionLayer/Scene/Scene.h>
#include <ResourceLayer/JsonUtil.h>
#include <memory>
#include <string>
#include <vector>

class HashGrid;
class MLP;
class AdamOptimizer;

//* Neural Radiance Cache (神经辐射缓存)
//* 使用一个小型神经网络(多分辨率哈希编码 + MLP)拟合场景中任意点、
//* 沿任意出射方向的radiance，然后在路径追踪中查询该缓存以提前终止路径。
//*
//* 流水线(通过scene.json的"nrc"字段配置)：
//*   1. collect：用路径追踪生成训练数据(按弹射深度分层、按像素组织)，
//*      每层数据先按像素平均再空间盒式滤波，存储到dataFile
//*   2. train：用收集的数据训练网络(相对L1损失 + Adam)，模型存储到modelFile
//*   3. render：用训练好的网络渲染
//*      - cache:  相机光线在第一个非镜面交点处查询缓存并终止路径
//*      - hybrid: 前hybridDepth次弹射用路径追踪计算，随后查询缓存终止
//*      - path:   纯路径追踪参考图
class NRC {
public:
  NRC() = default;

  NRC(const Json &json);

  //* 执行配置的pipeline阶段，json为scene.json的根对象
  void run(const Scene &scene, const Camera &camera, const Json &json);

private:
  struct NRCConfig {
    //* 数据收集
    int width = 512, height = 512; // 训练数据分辨率
    int collectSPP = 32;           // 每个像素的路径追踪样本数
    int maxDepth = 6;              // 路径最大弹射深度(即数据层数)
    int filterRadius = 2;          // 训练标签的空间盒式滤波半径
    uint64_t seed = 2026;          // 随机数种子
    //* 训练
    int epochs = 4;          // 训练轮数
    int iterations = 0;      // 迭代次数，0表示由epochs自动计算
    int batchSize = 4096;    // 每个mini-batch的样本数
    float learningRate = 0.01f; // Adam学习率
    int hashBaseRes = 16;    // 哈希网格基础分辨率
    int hashLevels = 16;     // 哈希网格层数
    int hashFeatures = 2;    // 每层特征数
    int hashTableSize = 1 << 18; // 哈希表大小
    int hidden1 = 64;        // 网络隐藏层宽度
    int hidden2 = 64;
    //* 渲染
    int renderSPP = 256;      // 参考路径追踪的每像素样本数
    int hybridDepth = 1;      // hybrid模式中先路径追踪的弹射数
    std::string renderMode = "cache"; // cache / hybrid / path / all
    //* 文件
    std::string dataFile = "nrc_data.bin";
    std::string modelFile = "nrc_model.bin";
    //* 阶段开关
    bool doCollect = true, doTrain = true, doRender = true;
    //* 线程数，0表示使用全部
    int numThreads = 0;
  };

  //* 某一弹射深度的训练数据，按像素存储
  struct LayerData {
    std::vector<float> position; // W*H*3
    std::vector<float> normal;   // W*H*3
    std::vector<float> wo;       // W*H*3
    std::vector<float> radiance; // W*H*3
    std::vector<float> count;    // W*H
  };

  //* 各阶段
  void phaseCollect(const Scene &scene, const Camera &camera);
  void phaseTrain();
  void phaseRender(const Scene &scene, const Camera &camera);

  //* 渲染模式
  void renderCache(const Scene &scene, const Camera &camera);
  void renderPathTraced(const Scene &scene, const Camera &camera);
  void renderHybrid(const Scene &scene, const Camera &camera);

  //* 查询缓存：p为世界坐标，返回沿wo方向的radiance
  Spectrum query(const Point3f &p, const Vector3f &normal,
                 const Vector3f &wo) const;

  //* 将世界坐标归一化到[0,1]^3
  Vector3f normalizePosition(const Point3f &p) const;

  //* 顶点处是否为delta BSDF(镜面)
  bool isSpecular(const Intersection &its) const;

  //* 训练数据的加载与保存
  void saveData(const std::string &path) const;
  void loadData(const std::string &path);
  void saveModel(const std::string &path) const;
  void loadModel(const std::string &path);

  //* 保存tone mapping后的PNG(Reinhard + gamma)
  void saveToneMapped(const std::string &path, const std::vector<float> &rgb,
                      int w, int h) const;

  NRCConfig config;
  std::vector<LayerData> layers;
  AABB sceneBounds;
  Point3f boundsMin;
  Vector3f boundsExtent;
  std::shared_ptr<HashGrid> hashGrid;
  std::shared_ptr<MLP> mlp;
};
