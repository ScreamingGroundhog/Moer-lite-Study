# Moer-lite：面向教学的蒙特卡洛路径追踪渲染框架

## 简介

Moer-lite是一个面向教学的蒙特卡洛路径追踪框架。它提供了很多已有的功能，可以避免从零开始构建渲染器时遇到的一些工程性障碍。  
Moer-lite支持的功能不多，它只提供基础的积分方法、材质以及纹理，一些更强大的方法需要依靠同学们自己实现。  
Moer-lite的框架基于Moer主干：https://github.com/NJUCG/Moer .

这是一片自由发挥的乐园，希望同学们学习愉快。

## 编译和使用

请参考Lab0的编译使用过程。

## 示例场景
目前有六个示例场景：
- `examples/area-lights`：提供了一个面光源照亮一个Stanford bunny的场景；
- `examples/two-spotlights`：提供了一个有着两个点光源的场景；
- `examples/bunny`：提供了一个使用环境光（environment lighting）照亮Stanford bunny的场景。
- `examples/cornell-box`：提供了一个使用whitted-style ray tracing的cornell-box场景
- `examples/lab1-test0`：提供了一个简单的mesh场景用来进行求交测试
- `examples/nrc-cornell`：神经辐射缓存（Neural Radiance Cache, NRC）示例场景，对应大作业离线渲染选题二，运行 `./target/bin/Moer examples/nrc-cornell` 即可

## 大作业：离线渲染选题二 —— 神经辐射缓存 (NRC)

在 `scene.json` 中配置 `"nrc"` 字段后，渲染器会执行 收集数据 -> 训练网络 -> 渲染 的完整流程：

1. **collect**：使用路径追踪（`pathTracer`）从相机发射光线，在路径的每个顶点记录
   `(位置, 法线, 出射方向, radiance)`，按弹射深度分层、按像素组织；每层数据先按像素
   平均，再做空间盒式滤波以降低标签噪声，结果存入 `nrc_data.bin`。
2. **train**：用一个小型神经网络拟合 radiance 函数 —— 输入为多分辨率哈希编码
   （参考 Instant-NGP）拼接法线与出射方向（38维），输出3通道radiance；使用相对L1损失
   + Adam优化器（OpenMP并行训练），模型存入 `nrc_model.bin`。
3. **render**（`renderMode`）：
   - `cache`：相机光线在第一个非镜面交点处查询缓存并终止路径（验证网络拟合效果，
     对应题目中的图2左侧）；
   - `hybrid`：前 `hybridDepth` 次弹射用路径追踪计算，之后查询缓存终止路径；
   - `path`：纯路径追踪参考图（图2右侧）。

主要代码位于 `src/FunctionLayer/NRC/`（`NRC`、`HashGrid`、`MLP`），路径追踪位于
`src/FunctionLayer/Integrator/PathIntegrator.cpp`（注册名 `pathTracer`，也可作为普通
积分器使用）。常用配置项：`collectSPP`、`maxDepth`、`filterRadius`、`epochs`、
`learningRate`、`renderSPP`、`renderMode`、`doCollect/doTrain/doRender`（可跳过已完成的
阶段）。
## BUG反馈

如果发现Moer-lite框架存在bug，或者有任何架构上的改进意见，任何更好的第三方库……
请不要犹豫，你有多种渠道可以向我们反馈：
1. 直接在QQ上联系助教；
2. 向助教发邮件；
3. 在这个Github仓库的issues页面提出建议。

## Update
- 2023-3-2 添加了进度条，whitted-style积分器，镜面材质以及一个新场景cornell-box。

- 2023-3-10
  - 修改了Cube.cpp 28、29行bug（之前代码正确性不影响，但是逻辑不对）
  - 修改了Triangle.cpp中的bug（之前当mesh文件不存在法线和纹理信息时会发生段错误）
  - 对AABB进行封装（见Shape::getAABB）
  - Scene::rayIntersect、Shape::rayIntersectShape、Acceleration::rayIntersect、Integrator::li中const Ray& 修改为Ray&
  - 修改了Acceleration求交的接口，现在每个加速结构需要实现bool Acceleration::rayIntersect(Ray &ray, int *geomID, int *primID, float *u, float *v) const函数
  
- 2023-3-11
  - 增加了Octree八叉树，实现八叉树需要你实现以下两个方法
    - std::shared_ptr<OctreeNode> Octree::recursiveBuild
    - bool Octree::rayIntersect
  - 增加了`examples/lab1-test0`测试场景，用于测试加速结构的效率
  - 目前可以修改配置文件中scene的acceleration参数用来测试不同的加速结构，现在可选的参数有`embree`（default）、`linear`（没有任何求交加速）、`octree`（需要实现）
  - 修改了面光源存在时运行发生段错误
  
- 2023-3-18
  lab1有关内容更新 具体请看实验手册

- 2023-4-18
  更新了lab2，新增代码以及实验手册请切换到本仓库lab-2分支进行查看
  添加了lab0的手册
  
- 2023-10-6

  修正了lab2对于微表面模型公式中，G项的描述错误

## 纠错
- Parallelogram.cpp 中的 Parallelogram::uniformSampleOnSurface方法，其中 area 应为 `cross(edge0, edge1).length()`， 即去掉.5f的系数（感谢苏尧同学的指正）

## TODO
- 对mesh和sphere实现表面采样（目前只有parallelogram可以配置为面光源）
- 加速环境光的采样