#include "NRC.h"
#include "HashGrid.h"
#include "MLP.h"
#include "Random.h"
#include <FunctionLayer/Integrator/PathIntegrator.h>
#include <FunctionLayer/Material/Material.h>
#include <ResourceLayer/Factory.h>
#include <ResourceLayer/Image.h>
#include <atomic>
#include <cstring>
#include <fstream>
#include <omp.h>

#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

namespace {
inline void printProgress(float percentage) {
  int val = (int)(percentage * 100);
  int lpad = (int)(percentage * PBWIDTH);
  int rpad = PBWIDTH - lpad;
  printf("\r%3d%% [%.*s%*s]", val, lpad, PBSTR, rpad, "");
  fflush(stdout);
}

inline Spectrum evaluateInfiniteLights(const Ray &ray, const Scene &scene) {
  Spectrum s(.0f);
  for (auto light : scene.infiniteLights) {
    s += light->evaluateEmission(ray);
  }
  return s;
}

inline bool fileExists(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  return f.good();
}

inline void writeFloats(std::ofstream &f, const std::vector<float> &data) {
  f.write((const char *)data.data(), (std::streamsize)data.size() * sizeof(float));
}

inline void readFloats(std::ifstream &f, std::vector<float> &data) {
  f.read((char *)data.data(), (std::streamsize)data.size() * sizeof(float));
}
} // namespace

NRC::NRC(const Json &json) {
  config.width = fetchOptional<int>(json, "width", config.width);
  config.height = fetchOptional<int>(json, "height", config.height);
  config.collectSPP = fetchOptional<int>(json, "collectSPP", config.collectSPP);
  config.maxDepth = fetchOptional<int>(json, "maxDepth", config.maxDepth);
  config.filterRadius =
      fetchOptional<int>(json, "filterRadius", config.filterRadius);
  config.seed = fetchOptional<uint64_t>(json, "seed", config.seed);
  config.epochs = fetchOptional<int>(json, "epochs", config.epochs);
  config.iterations = fetchOptional<int>(json, "iterations", config.iterations);
  config.batchSize = fetchOptional<int>(json, "batchSize", config.batchSize);
  config.learningRate =
      fetchOptional<float>(json, "learningRate", config.learningRate);
  config.hashBaseRes =
      fetchOptional<int>(json, "hashBaseRes", config.hashBaseRes);
  config.hashLevels = fetchOptional<int>(json, "hashLevels", config.hashLevels);
  config.hashFeatures =
      fetchOptional<int>(json, "hashFeatures", config.hashFeatures);
  config.hashTableSize =
      fetchOptional<int>(json, "hashTableSize", config.hashTableSize);
  config.hidden1 = fetchOptional<int>(json, "hidden1", config.hidden1);
  config.hidden2 = fetchOptional<int>(json, "hidden2", config.hidden2);
  config.renderSPP = fetchOptional<int>(json, "renderSPP", config.renderSPP);
  config.hybridDepth =
      fetchOptional<int>(json, "hybridDepth", config.hybridDepth);
  config.renderMode =
      fetchOptional<std::string>(json, "renderMode", config.renderMode);
  config.dataFile =
      fetchOptional<std::string>(json, "dataFile", config.dataFile);
  config.modelFile =
      fetchOptional<std::string>(json, "modelFile", config.modelFile);
  config.doCollect = fetchOptional<bool>(json, "doCollect", config.doCollect);
  config.doTrain = fetchOptional<bool>(json, "doTrain", config.doTrain);
  config.doRender = fetchOptional<bool>(json, "doRender", config.doRender);
  config.numThreads =
      fetchOptional<int>(json, "numThreads", config.numThreads);
}

void NRC::run(const Scene &scene, const Camera &camera, const Json &json) {
  if (config.numThreads > 0) {
    omp_set_num_threads(config.numThreads);
  }
  if (config.doCollect) {
    phaseCollect(scene, camera);
  } else if (!fileExists(config.dataFile)) {
    printf("Data file %s not found, collecting anyway.\n",
           config.dataFile.c_str());
    phaseCollect(scene, camera);
  } else {
    loadData(config.dataFile);
  }
  if (config.doTrain) {
    phaseTrain();
  } else if (!fileExists(config.modelFile)) {
    printf("Model file %s not found, training anyway.\n",
           config.modelFile.c_str());
    phaseTrain();
  } else {
    loadModel(config.modelFile);
  }
  if (config.doRender) {
    phaseRender(scene, camera);
  }
}

void NRC::phaseCollect(const Scene &scene, const Camera &camera) {
  //* 计算场景包围盒，并略微扩大以避免表面点落在边界上
  AABB bounds = scene.getAABB();
  Vector3f size = bounds.pMax - bounds.pMin;
  Vector3f pad = size * .01f;
  bounds.pMin -= pad;
  bounds.pMax += pad;
  sceneBounds = bounds;
  boundsMin = bounds.pMin;
  boundsExtent = bounds.pMax - bounds.pMin;

  int W = config.width, H = config.height;
  layers.assign(config.maxDepth, LayerData());
  for (auto &layer : layers) {
    layer.position.assign((size_t)W * H * 3, .0f);
    layer.normal.assign((size_t)W * H * 3, .0f);
    layer.wo.assign((size_t)W * H * 3, .0f);
    layer.radiance.assign((size_t)W * H * 3, .0f);
    layer.count.assign((size_t)W * H, .0f);
  }

  PathIntegrator pathTracer;
  pathTracer.maxDepth = config.maxDepth;

  printf("Collecting training data (%dx%d, %d spp, %d layers)...\n", W, H,
         config.collectSPP, config.maxDepth);
  auto start = std::chrono::system_clock::now();

#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    Random rng(config.seed + (uint64_t)tid * 0x9e3779b97f4a7c15ull);
    auto sampler = std::make_shared<RandomSampler>(rng);
#pragma omp for schedule(dynamic, 4)
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        for (int s = 0; s < config.collectSPP; ++s) {
          Vector2f NDC{(x + rng.next1D()) / W, (y + rng.next1D()) / H};
          Ray ray = camera.sampleRayDifferentials(
              CameraSample{rng.next2D(), rng.next2D(), .0f}, NDC);
          pathTracer.liWithSamples(
              ray, scene, sampler, [&](const RadianceSample &rs) {
                if (rs.depth >= config.maxDepth) {
                  return;
                }
                LayerData &layer = layers[rs.depth];
                size_t base = (size_t)(y * W + x) * 3;
                layer.position[base + 0] += rs.position[0];
                layer.position[base + 1] += rs.position[1];
                layer.position[base + 2] += rs.position[2];
                layer.normal[base + 0] += rs.normal[0];
                layer.normal[base + 1] += rs.normal[1];
                layer.normal[base + 2] += rs.normal[2];
                layer.wo[base + 0] += rs.wo[0];
                layer.wo[base + 1] += rs.wo[1];
                layer.wo[base + 2] += rs.wo[2];
                layer.radiance[base + 0] += rs.radiance[0];
                layer.radiance[base + 1] += rs.radiance[1];
                layer.radiance[base + 2] += rs.radiance[2];
                layer.count[y * W + x] += 1.f;
              });
        }
      }
      if (tid == 0) {
        printProgress((float)(y + 1) / H);
      }
    }
  }
  printf("\n");

  //* 按像素平均并归一化
  int count0 = 0;
  for (auto &layer : layers) {
    for (int i = 0; i < W * H; ++i) {
      float c = layer.count[i];
      if (c <= .0f) {
        continue;
      }
      ++count0;
      size_t base = (size_t)i * 3;
      Vector3f pos{layer.position[base + 0] / c, layer.position[base + 1] / c,
                   layer.position[base + 2] / c};
      Vector3f n{layer.normal[base + 0] / c, layer.normal[base + 1] / c,
                 layer.normal[base + 2] / c};
      Vector3f wo{layer.wo[base + 0] / c, layer.wo[base + 1] / c,
                  layer.wo[base + 2] / c};
      layer.position[base + 0] = pos[0];
      layer.position[base + 1] = pos[1];
      layer.position[base + 2] = pos[2];
      if (n.length() > 1e-6f) {
        n = normalize(n);
      }
      layer.normal[base + 0] = n[0];
      layer.normal[base + 1] = n[1];
      layer.normal[base + 2] = n[2];
      if (wo.length() > 1e-6f) {
        wo = normalize(wo);
      }
      layer.wo[base + 0] = wo[0];
      layer.wo[base + 1] = wo[1];
      layer.wo[base + 2] = wo[2];
      layer.radiance[base + 0] /= c;
      layer.radiance[base + 1] /= c;
      layer.radiance[base + 2] /= c;
    }
  }
  printf("Valid samples: %d\n", count0);

  //* 对radiance标签做空间盒式滤波(与NRC论文一致)，只统计有效像素
  if (config.filterRadius > 0) {
    for (auto &layer : layers) {
      std::vector<float> filtered(layer.radiance.size(), .0f);
      for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
          if (layer.count[y * W + x] <= .0f) {
            continue;
          }
          float sum[3] = {.0f, .0f, .0f};
          int n = 0;
          for (int dy = -config.filterRadius; dy <= config.filterRadius; ++dy) {
            for (int dx = -config.filterRadius; dx <= config.filterRadius;
                 ++dx) {
              int xx = x + dx, yy = y + dy;
              if (xx < 0 || xx >= W || yy < 0 || yy >= H) {
                continue;
              }
              if (layer.count[yy * W + xx] <= .0f) {
                continue;
              }
              size_t base = (size_t)(yy * W + xx) * 3;
              sum[0] += layer.radiance[base + 0];
              sum[1] += layer.radiance[base + 1];
              sum[2] += layer.radiance[base + 2];
              ++n;
            }
          }
          if (n > 0) {
            size_t base = (size_t)(y * W + x) * 3;
            filtered[base + 0] = sum[0] / n;
            filtered[base + 1] = sum[1] / n;
            filtered[base + 2] = sum[2] / n;
          }
        }
      }
      for (int i = 0; i < W * H; ++i) {
        if (layer.count[i] > .0f) {
          layer.radiance[i * 3 + 0] = filtered[i * 3 + 0];
          layer.radiance[i * 3 + 1] = filtered[i * 3 + 1];
          layer.radiance[i * 3 + 2] = filtered[i * 3 + 2];
        }
      }
    }
  }

  //* 保存第0层标签的可视化，方便检查数据质量
  saveToneMapped("nrc_labels.png", layers[0].radiance, W, H);

  auto end = std::chrono::system_clock::now();
  printf("Collection costs %.2fs\n",
         std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                 .count() /
             1000.f);
  saveData(config.dataFile);
}

void NRC::phaseTrain() {
  loadData(config.dataFile);

  int W = config.width, H = config.height;
  //* 将分层数据展开为平面样本数组(每个样本12个float：位置3+法线3+出射方向3+radiance3)
  std::vector<float> samples;
  samples.reserve((size_t)W * H * config.maxDepth * 12);
  for (auto &layer : layers) {
    for (int i = 0; i < W * H; ++i) {
      if (layer.count[i] <= .0f) {
        continue;
      }
      size_t base = (size_t)i * 3;
      for (int j = 0; j < 3; ++j) {
        samples.push_back(layer.position[base + j]);
      }
      for (int j = 0; j < 3; ++j) {
        samples.push_back(layer.normal[base + j]);
      }
      for (int j = 0; j < 3; ++j) {
        samples.push_back(layer.wo[base + j]);
      }
      samples.push_back(layer.radiance[base + 0]);
      samples.push_back(layer.radiance[base + 1]);
      samples.push_back(layer.radiance[base + 2]);
    }
  }
  size_t nSamples = samples.size() / 12;
  if (nSamples == 0) {
    std::cerr << "No training samples!\n";
    exit(1);
  }
  //* 样本数过多时均匀抽样，保持空间覆盖的同时控制训练时间
  size_t maxSamples = 2u << 20;
  if (nSamples > maxSamples) {
    std::vector<float> subset;
    subset.reserve(maxSamples * 12);
    for (size_t i = 0; i < maxSamples; ++i) {
      size_t idx = i * nSamples / maxSamples;
      size_t base = idx * 12;
      for (int j = 0; j < 12; ++j) {
        subset.push_back(samples[base + j]);
      }
    }
    samples.swap(subset);
    nSamples = samples.size() / 12;
  }

  //* 初始化网络
  hashGrid = std::make_shared<HashGrid>(config.hashBaseRes, config.hashLevels,
                                        config.hashFeatures,
                                        config.hashTableSize);
  hashGrid->initParameters(-0.1f, 0.1f);
  mlp = std::make_shared<MLP>();
  int encDim = hashGrid->outputDim;
  int inDim = encDim + 6; // 哈希编码 + 法线 + 出射方向
  mlp->init(inDim, config.hidden1, config.hidden2, 3);

  AdamOptimizer adamHash, adamMlp;
  adamHash.init(hashGrid->numParams());
  adamMlp.init(mlp->numParams());

  int T = omp_get_max_threads();
  int iterations = config.iterations > 0
                       ? config.iterations
                       : (int)(nSamples * config.epochs / config.batchSize) + 1;

  //* 梯度缓冲：MLP每线程一份，哈希网格共享一份(原子累加)
  std::vector<std::vector<float>> mlpGrads(
      T, std::vector<float>(mlp->numParams(), .0f));
  std::vector<std::atomic<uint32_t>> hashGrad(hashGrid->numParams());
  std::vector<float> hashGradF(hashGrid->numParams(), .0f);
  std::vector<double> losses(T, .0);

  printf("Training: %zu samples, %d iterations, batch %d, %d threads\n",
         nSamples, iterations, config.batchSize, T);
  auto start = std::chrono::system_clock::now();

  for (int iter = 1; iter <= iterations; ++iter) {
    //* 余弦学习率衰减(从learningRate衰减到其1/10)，提高训练后期的稳定性
    float lr = config.learningRate *
               (.1f + .9f * .5f *
                          (1.f + std::cos((float)iter / iterations * PI)));
    //* 清零梯度
#pragma omp parallel num_threads(T)
    {
      int tid = omp_get_thread_num();
      std::fill(mlpGrads[tid].begin(), mlpGrads[tid].end(), .0f);
    }
#pragma omp parallel for num_threads(T)
    for (size_t i = 0; i < hashGrad.size(); ++i) {
      hashGrad[i].store(0u, std::memory_order_relaxed);
    }

    //* 并行计算mini-batch的前向与反向
#pragma omp parallel num_threads(T)
    {
      int tid = omp_get_thread_num();
      Random rng(config.seed ^ (uint64_t)iter * 0x9e3779b97f4a7c15ull +
                 (uint64_t)tid * 0xbf58476d1ce4e5b9ull);
      std::vector<float> enc(encDim), gradIn(inDim), in(inDim), pred(3),
          gradOut(3);
      float *mg = mlpGrads[tid].data();
      double localLoss = .0;
#pragma omp for schedule(static)
      for (int b = 0; b < config.batchSize; ++b) {
        uint32_t idx = (uint32_t)(rng.next1D() * (nSamples - 1));
        const float *s = samples.data() + (size_t)idx * 12;
        Vector3f u = normalizePosition(
            Point3f{s[0], s[1], s[2]});
        hashGrid->encode(u, enc.data());
        for (int i = 0; i < encDim; ++i) {
          in[i] = enc[i];
        }
        for (int i = 0; i < 6; ++i) {
          in[encDim + i] = s[3 + i];
        }
        mlp->forward(in.data(), pred.data());
        //* 相对L1损失 (NRC论文)
        float mPred = (pred[0] + pred[1] + pred[2]) / 3.f;
        float mRad = (s[9] + s[10] + s[11]) / 3.f;
        float D = mPred + mRad + 1e-2f;
        float N = std::abs(pred[0] - s[9]) + std::abs(pred[1] - s[10]) +
                  std::abs(pred[2] - s[11]);
        localLoss += N / D;
        float invD = 1.f / D;
        float f3 = N / (3.f * D * D);
        for (int c = 0; c < 3; ++c) {
          gradOut[c] = (pred[c] > s[9 + c] ? 1.f : -1.f) * invD - f3;
        }
        std::fill(gradIn.begin(), gradIn.end(), .0f);
        mlp->forwardBackward(in.data(), gradOut.data(), mg, gradIn.data());
        hashGrid->backwardAtomic(u, gradIn.data(), hashGrad.data());
      }
      losses[tid] = localLoss;
    }

    //* 归约MLP梯度并更新
    std::vector<float> mlpGradTotal(mlp->numParams(), .0f);
#pragma omp parallel for num_threads(T)
    for (int i = 0; i < mlp->numParams(); ++i) {
      float sum = .0f;
      for (int t = 0; t < T; ++t) {
        sum += mlpGrads[t][i];
      }
      mlpGradTotal[i] = sum;
    }
    adamMlp.step(mlp->params.data(), mlpGradTotal.data(), lr);

    //* 拷贝原子梯度缓冲并更新哈希网格参数
#pragma omp parallel for num_threads(T)
    for (size_t i = 0; i < hashGrad.size(); ++i) {
      uint32_t bits = hashGrad[i].load(std::memory_order_relaxed);
      float value;
      std::memcpy(&value, &bits, sizeof(float));
      hashGradF[i] = value;
    }
    adamHash.beginStep();
#pragma omp parallel for num_threads(T) schedule(static, 4096)
    for (size_t i = 0; i < hashGradF.size(); i += 4096) {
      adamHash.stepRange(hashGrid->params.data(), hashGradF.data(), lr,
                         (int)i, (int)std::min(i + 4096, hashGradF.size()));
    }

    if (iter % 100 == 0 || iter == iterations) {
      double loss = .0;
      for (int t = 0; t < T; ++t) {
        loss += losses[t];
      }
      loss /= config.batchSize;
      printf("\rIter %6d / %d, loss = %.4f", iter, iterations, loss);
      fflush(stdout);
    }
  }
  printf("\n");

  auto end = std::chrono::system_clock::now();
  printf("Training costs %.2fs\n",
         std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                 .count() /
             1000.f);
  saveModel(config.modelFile);
}

void NRC::phaseRender(const Scene &scene, const Camera &camera) {
  loadModel(config.modelFile);
  printf("Rendering mode: %s\n", config.renderMode.c_str());
  if (config.renderMode == "cache") {
    renderCache(scene, camera);
  } else if (config.renderMode == "hybrid") {
    renderHybrid(scene, camera);
  } else if (config.renderMode == "path") {
    renderPathTraced(scene, camera);
  } else if (config.renderMode == "all") {
    renderCache(scene, camera);
    renderHybrid(scene, camera);
    renderPathTraced(scene, camera);
  } else {
    std::cerr << "Unknown renderMode: " << config.renderMode << "\n";
    exit(1);
  }
}

void NRC::renderCache(const Scene &scene, const Camera &camera) {
  int W = camera.film->size[0], H = camera.film->size[1];
  std::vector<float> image((size_t)W * H * 3, .0f);

  printf("Rendering with cache query (1 spp)...\n");
  auto start = std::chrono::system_clock::now();
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    Random rng(config.seed + 0x12345678ull + (uint64_t)tid * 0x9e3779b97f4a7c15ull);
    RandomSampler sampler(rng);
#pragma omp for schedule(dynamic, 8)
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        Vector2f NDC{(x + .5f) / W, (y + .5f) / H};
        Ray ray = camera.sampleRayDifferentials(
            CameraSample{Vector2f{.5f, .5f}, Vector2f{.0f, .0f}, .0f}, NDC);
        Spectrum L(.0f);
        //* 穿过镜面链，在第一个非镜面顶点查询缓存并终止路径
        int bounce = 0;
        while (true) {
          auto hitOpt = scene.rayIntersect(ray);
          if (!hitOpt.has_value()) {
            L = evaluateInfiniteLights(ray, scene);
            break;
          }
          Intersection its = hitOpt.value();
          if (!isSpecular(its)) {
            L = query(its.position, its.normal, -ray.direction);
            break;
          }
          //* 镜面顶点：加入发光后继续反射
          if (its.shape->light) {
            L += its.shape->light->evaluateEmission(its, -ray.direction);
            break;
          }
          auto material = its.shape->material;
          auto bsdf = material->computeBSDF(its);
          auto res = bsdf->sample(-ray.direction, sampler.next2D());
          if (res.pdf <= .0f || res.weight.isZero() || ++bounce > 8) {
            break;
          }
          ray = Ray{its.position, res.wi, 1e-4f};
        }
        size_t base = (size_t)(y * W + x) * 3;
        image[base + 0] = L[0];
        image[base + 1] = L[1];
        image[base + 2] = L[2];
      }
      if (tid == 0) {
        printProgress((float)(y + 1) / H);
      }
    }
  }
  printf("\n");
  auto end = std::chrono::system_clock::now();
  printf("Cache rendering costs %.2fs\n",
         std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                 .count() /
             1000.f);
  saveToneMapped("nrc_cache.png", image, W, H);
}

void NRC::renderPathTraced(const Scene &scene, const Camera &camera) {
  int W = camera.film->size[0], H = camera.film->size[1];
  std::vector<float> image((size_t)W * H * 3, .0f);

  PathIntegrator pathTracer;
  pathTracer.maxDepth = config.maxDepth;

  printf("Rendering path tracing reference (%d spp)...\n", config.renderSPP);
  auto start = std::chrono::system_clock::now();
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    Random rng(config.seed + 0x55555555ull + (uint64_t)tid * 0x9e3779b97f4a7c15ull);
    auto sampler = std::make_shared<RandomSampler>(rng);
#pragma omp for schedule(dynamic, 8)
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        Spectrum L(.0f);
        for (int s = 0; s < config.renderSPP; ++s) {
          Vector2f NDC{(x + rng.next1D()) / W, (y + rng.next1D()) / H};
          Ray ray = camera.sampleRayDifferentials(
              CameraSample{rng.next2D(), rng.next2D(), .0f}, NDC);
          L += pathTracer.li(ray, scene, sampler);
        }
        L /= config.renderSPP;
        size_t base = (size_t)(y * W + x) * 3;
        image[base + 0] = L[0];
        image[base + 1] = L[1];
        image[base + 2] = L[2];
      }
      if (tid == 0) {
        printProgress((float)(y + 1) / H);
      }
    }
  }
  printf("\n");
  auto end = std::chrono::system_clock::now();
  printf("Path tracing costs %.2fs\n",
         std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                 .count() /
             1000.f);
  saveToneMapped("nrc_path_ref.png", image, W, H);
}

void NRC::renderHybrid(const Scene &scene, const Camera &camera) {
  int W = camera.film->size[0], H = camera.film->size[1];
  std::vector<float> image((size_t)W * H * 3, .0f);

  printf("Rendering hybrid (%d PT bounces then cache query)...\n",
         config.hybridDepth);
  auto start = std::chrono::system_clock::now();
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    Random rng(config.seed + 0x77777777ull + (uint64_t)tid * 0x9e3779b97f4a7c15ull);
    RandomSampler sampler(rng);
    //* 与PathIntegrator相同结构的路径追踪，但在深度>=hybridDepth时查询缓存终止
    std::function<Spectrum(Ray &, int, bool)> trace =
        [&](Ray &r, int depth, bool specularBounce) -> Spectrum {
      auto hitOpt = scene.rayIntersect(r);
      if (!hitOpt.has_value()) {
        return evaluateInfiniteLights(r, scene);
      }
      Intersection its = hitOpt.value();

      //* 查询缓存并终止路径(镜面顶点继续反射，直到非镜面顶点)
      if (depth >= config.hybridDepth) {
        if (!isSpecular(its)) {
          return query(its.position, its.normal, -r.direction);
        }
        Spectrum L(.0f);
        if (its.shape->light &&
            (depth == 0 || specularBounce)) {
          L += its.shape->light->evaluateEmission(its, -r.direction);
        }
        if (its.shape->material) {
          auto bsdf = its.shape->material->computeBSDF(its);
          auto res = bsdf->sample(-r.direction, sampler.next2D());
          if (res.pdf > .0f && !res.weight.isZero()) {
            Ray nextRay{its.position, res.wi, 1e-4f};
            return L + res.weight * trace(nextRay, depth + 1, true);
          }
        }
        return L;
      }

      //* 完整路径追踪(与PathIntegrator一致)
      Spectrum L(.0f);
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
      if (!specular) {
        float pdfLight = .0f;
        auto light = scene.sampleLight(sampler.next1D(), &pdfLight);
        if (light && pdfLight != .0f) {
          auto lightSampleResult = light->sample(its, sampler.next2D());
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
      bool useRR = depth >= 3;
      bool continuePath = !useRR || sampler.next1D() <= .8f;
      if (continuePath && depth < config.maxDepth) {
        auto bsdfSampleResult = bsdf->sample(-r.direction, sampler.next2D());
        if (bsdfSampleResult.pdf > .0f && !bsdfSampleResult.weight.isZero()) {
          float weightScale = useRR ? 1.f / .8f : 1.f;
          Ray nextRay{its.position, bsdfSampleResult.wi, 1e-4f};
          auto nextHitOpt = scene.rayIntersect(nextRay);
          Spectrum continuation(.0f);
          if (!nextHitOpt.has_value()) {
            continuation = evaluateInfiniteLights(nextRay, scene);
          } else {
            auto &nextHit = nextHitOpt.value();
            if (specular && nextHit.shape->light) {
              continuation = nextHit.shape->light->evaluateEmission(
                  nextHit, -nextRay.direction);
            } else {
              continuation = trace(nextRay, depth + 1, specular);
            }
          }
          L += bsdfSampleResult.weight * continuation * weightScale;
        }
      }
      return L;
    };
#pragma omp for schedule(dynamic, 8)
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        Spectrum L(.0f);
        for (int s = 0; s < config.renderSPP; ++s) {
          Vector2f NDC{(x + rng.next1D()) / W, (y + rng.next1D()) / H};
          Ray ray = camera.sampleRayDifferentials(
              CameraSample{rng.next2D(), rng.next2D(), .0f}, NDC);
          L += trace(ray, 0, false);
        }
        L /= config.renderSPP;
        size_t base = (size_t)(y * W + x) * 3;
        image[base + 0] = L[0];
        image[base + 1] = L[1];
        image[base + 2] = L[2];
      }
      if (tid == 0) {
        printProgress((float)(y + 1) / H);
      }
    }
  }
  printf("\n");
  auto end = std::chrono::system_clock::now();
  printf("Hybrid rendering costs %.2fs\n",
         std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                 .count() /
             1000.f);
  saveToneMapped("nrc_hybrid.png", image, W, H);
}

Spectrum NRC::query(const Point3f &p, const Vector3f &normal,
                    const Vector3f &wo) const {
  Vector3f u = normalizePosition(p);
  int encDim = hashGrid->outputDim;
  std::vector<float> enc(encDim), in(encDim + 6), out(3);
  hashGrid->encode(u, enc.data());
  for (int i = 0; i < encDim; ++i) {
    in[i] = enc[i];
  }
  in[encDim + 0] = normal[0];
  in[encDim + 1] = normal[1];
  in[encDim + 2] = normal[2];
  in[encDim + 3] = wo[0];
  in[encDim + 4] = wo[1];
  in[encDim + 5] = wo[2];
  mlp->forward(in.data(), out.data());
  return Spectrum(std::max(out[0], .0f), std::max(out[1], .0f),
                  std::max(out[2], .0f));
}

Vector3f NRC::normalizePosition(const Point3f &p) const {
  Vector3f u = (p - boundsMin) / boundsExtent;
  for (int i = 0; i < 3; ++i) {
    u[i] = std::max(.0f, std::min(1.f, u[i]));
  }
  return u;
}

bool NRC::isSpecular(const Intersection &its) const {
  if (!its.shape->material) {
    return false;
  }
  return its.shape->material->computeBSDF(its)->isSpecular();
}

void NRC::saveData(const std::string &path) const {
  std::ofstream f(path, std::ios::binary);
  const char magic[10] = "NRCDATA1";
  f.write(magic, 8);
  int W = config.width, H = config.height;
  f.write((const char *)&W, sizeof(int));
  f.write((const char *)&H, sizeof(int));
  int depth = (int)layers.size();
  f.write((const char *)&depth, sizeof(int));
  writeFloats(f, {boundsMin[0], boundsMin[1], boundsMin[2]});
  writeFloats(f, {boundsExtent[0], boundsExtent[1], boundsExtent[2]});
  for (auto &layer : layers) {
    writeFloats(f, layer.position);
    writeFloats(f, layer.normal);
    writeFloats(f, layer.wo);
    writeFloats(f, layer.radiance);
    writeFloats(f, layer.count);
  }
  printf("Saved training data to %s\n", path.c_str());
}

void NRC::loadData(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f.good()) {
    std::cerr << "Failed to open data file: " << path << "\n";
    exit(1);
  }
  char magic[9] = {0};
  f.read(magic, 8);
  int W = 0, H = 0, depth = 0;
  f.read((char *)&W, sizeof(int));
  f.read((char *)&H, sizeof(int));
  f.read((char *)&depth, sizeof(int));
  std::vector<float> minF(3), extF(3);
  readFloats(f, minF);
  readFloats(f, extF);
  boundsMin = Point3f{minF[0], minF[1], minF[2]};
  boundsExtent = Vector3f{extF[0], extF[1], extF[2]};
  sceneBounds = AABB(boundsMin, boundsMin + boundsExtent);
  layers.assign(depth, LayerData());
  for (auto &layer : layers) {
    layer.position.resize((size_t)W * H * 3);
    layer.normal.resize((size_t)W * H * 3);
    layer.wo.resize((size_t)W * H * 3);
    layer.radiance.resize((size_t)W * H * 3);
    layer.count.resize((size_t)W * H);
    readFloats(f, layer.position);
    readFloats(f, layer.normal);
    readFloats(f, layer.wo);
    readFloats(f, layer.radiance);
    readFloats(f, layer.count);
  }
  config.width = W;
  config.height = H;
  printf("Loaded training data from %s (%dx%d, %d layers)\n", path.c_str(), W,
         H, depth);
}

void NRC::saveModel(const std::string &path) const {
  std::ofstream f(path, std::ios::binary);
  const char magic[10] = "NRCMODEL1";
  f.write(magic, 8);
  writeFloats(f, {boundsMin[0], boundsMin[1], boundsMin[2]});
  writeFloats(f, {boundsExtent[0], boundsExtent[1], boundsExtent[2]});
  int baseRes = hashGrid->baseRes, levels = hashGrid->numLevels,
      features = hashGrid->featuresPerLevel, table = hashGrid->tableSize;
  f.write((const char *)&baseRes, sizeof(int));
  f.write((const char *)&levels, sizeof(int));
  f.write((const char *)&features, sizeof(int));
  f.write((const char *)&table, sizeof(int));
  writeFloats(f, hashGrid->params);
  int inDim = mlp->inDim, h1 = mlp->h1, h2 = mlp->h2, outDim = mlp->outDim;
  f.write((const char *)&inDim, sizeof(int));
  f.write((const char *)&h1, sizeof(int));
  f.write((const char *)&h2, sizeof(int));
  f.write((const char *)&outDim, sizeof(int));
  writeFloats(f, mlp->params);
  printf("Saved model to %s\n", path.c_str());
}

void NRC::loadModel(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f.good()) {
    std::cerr << "Failed to open model file: " << path << "\n";
    exit(1);
  }
  char magic[9] = {0};
  f.read(magic, 8);
  std::vector<float> minF(3), extF(3);
  readFloats(f, minF);
  readFloats(f, extF);
  boundsMin = Point3f{minF[0], minF[1], minF[2]};
  boundsExtent = Vector3f{extF[0], extF[1], extF[2]};
  sceneBounds = AABB(boundsMin, boundsMin + boundsExtent);
  int baseRes = 0, levels = 0, features = 0, table = 0;
  f.read((char *)&baseRes, sizeof(int));
  f.read((char *)&levels, sizeof(int));
  f.read((char *)&features, sizeof(int));
  f.read((char *)&table, sizeof(int));
  hashGrid = std::make_shared<HashGrid>(baseRes, levels, features, table);
  readFloats(f, hashGrid->params);
  int inDim = 0, h1 = 0, h2 = 0, outDim = 0;
  f.read((char *)&inDim, sizeof(int));
  f.read((char *)&h1, sizeof(int));
  f.read((char *)&h2, sizeof(int));
  f.read((char *)&outDim, sizeof(int));
  mlp = std::make_shared<MLP>();
  mlp->init(inDim, h1, h2, outDim);
  readFloats(f, mlp->params);
  printf("Loaded model from %s\n", path.c_str());
}

void NRC::saveToneMapped(const std::string &path, const std::vector<float> &rgb,
                         int w, int h) const {
  Image img(Vector2i{w, h});
  const float gamma = 1.f / 2.2f;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      size_t base = (size_t)(y * w + x) * 3;
      Vector3f v;
      for (int c = 0; c < 3; ++c) {
        float t = rgb[base + c];
        if (t < .0f)
          t = .0f;
        t = t / (1.f + t); // Reinhard tone mapping
        v[c] = std::pow(t, gamma);
      }
      img.setValue(Vector2i{x, y}, v);
    }
  }
  img.savePNG(path.c_str());
  printf("Saved %s\n", path.c_str());
}

REGISTER_CLASS(NRC, "nrc")
