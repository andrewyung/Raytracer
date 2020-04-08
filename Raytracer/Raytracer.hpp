#pragma once

#include <atlas/core/Float.hpp>
#include <atlas/math/Math.hpp>
#include <atlas/math/Random.hpp>
#include <atlas/math/Ray.hpp>
#include <atlas/utils/LoadObjFile.hpp>
#include <atlas/core/Timer.hpp>

#include <fmt/printf.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <limits>
#include <memory>
#include <vector>
#include <iostream>
#include <thread>
#include <sstream>
#include <mutex>
#include <queue>
#include <atomic>

using atlas::core::areEqual;

using ColourAlpha = atlas::math::Vector4;
using Colour = atlas::math::Vector;
using Point = atlas::math::Point;
using Vector = atlas::math::Vector;
using Vector2 = atlas::math::Vector2;

void saveToFile(std::string const& filename,
    std::size_t width,
    std::size_t height,
    std::vector<Colour> const& image);

// Declarations
class BRDF;
class Camera;
class Material;
class Textured;
class Light;
class Shape;
class Mesh;
class Sampler;
class BoundingVolumeBox;

static std::string modelRoot{ "C:/Users/Andrew/Documents/GitHub/Raytracer/Raytracer/Raytracer/models/" };
static const int maxBounceDepth = 3;
static const int lightAttenuationFactor = 500000;

/** Thread safe cout class
  * Exemple of use:
  *    PrintThread{} << "Hello world!" << std::endl;
  */
class PrintThread : public std::ostringstream
{
public:
    PrintThread() = default;

    ~PrintThread()
    {
        std::lock_guard<std::mutex> guard(_mutexPrint);
        std::cout << this->str();
    }

private:
    static std::mutex _mutexPrint;
};
std::mutex PrintThread::_mutexPrint{};

struct World
{
    std::size_t width, height;
    Colour background;
    std::shared_ptr<Sampler> sampler;
    std::vector<std::shared_ptr<Shape>> scene;
    std::vector<Colour> image;
    std::vector<std::shared_ptr<Light>> lights;
    std::shared_ptr<Light> ambient;
    std::shared_ptr<Camera> camera;

    int threadsAccessed = 0;
};

struct ShadeRec
{
    Colour color;
    float t;
    atlas::math::Normal normal;
    atlas::math::Ray<atlas::math::Vector> ray;
    std::shared_ptr<Material> material;
    std::shared_ptr<World> world;
    Vector2 uvCoord;
    unsigned int depth = 0;
    float indexOfRefraction = 1;

    bool shadowRay = false;
};

class ImageTexture
{
public:
    ImageTexture(std::string const& imageFilePath);

    ColourAlpha getColour(Vector2 uv);

    ~ImageTexture();
private:
    bool mSet;
    unsigned char* mImage;
    int mWidth, mHeight, mChannels;
};


class ThreadPool
{
public:
    ThreadPool(unsigned int const& numThreads);

    void startJob(std::function<void()> job);

    void waitDone();

    ~ThreadPool();
private:
    void loopFunction();

    bool mTerminatePool;

    std::atomic<int> mJobsRunning;
    std::vector<std::thread> mThreads;
    std::queue<std::function<void()>> mQueue;

    std::mutex mQueueMutex;
    std::condition_variable mCondition;

    std::mutex mComplMutex;
    std::condition_variable mComplCondition;
};

// Abstract classes defining the interfaces for concrete entities

class Camera
{
public:
    Camera(Point position, Point lookAt, Vector up, float frustrumDist);

    Point getEye() const;

    void calculateRay(float x, float y, atlas::math::Ray<Vector>& ray) const;
protected:
    Point mEye;
    float mFrustrumDist;
    Vector mU, mV, mW;
};

class Sampler
{
public:
    Sampler(int numSamples, int numSets);
    virtual ~Sampler() = default;

    int getNumSamples() const;

    void setupShuffledIndices();

    virtual void generateSamples() = 0;

    atlas::math::Point sampleUnitSquare();

protected:
    std::vector<atlas::math::Point> mSamples;
    std::vector<int> mShuffledIndices;

    int mNumSamples;
    int mNumSets;
    unsigned long mCount;
    int mJump;
};

class Shape
{
public:
    Shape();
    virtual ~Shape() = default;

    // if t computed is less than the t in sr, it and the color should be
    // updated in sr
    virtual bool hit(atlas::math::Ray<atlas::math::Vector> const& ray,
        ShadeRec& sr) const = 0;

    void setColour(Colour const& col);

    Colour getColour() const;

    void setMaterial(std::shared_ptr<Material> const& material);

    std::shared_ptr<Material> getMaterial() const;

protected:
    virtual bool intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
        float& tMin) const = 0;

    Colour mColour;
    std::shared_ptr<Material> mMaterial;
};

class BRDF
{
public:
    virtual ~BRDF() = default;

    virtual Colour fn(ShadeRec const& sr,
        atlas::math::Vector const& reflected,
        atlas::math::Vector const& incoming) const = 0;
    virtual Colour rho(ShadeRec const& sr,
        atlas::math::Vector const& reflected) const = 0;
};

class Material
{
public:
    virtual ~Material() = default;

    virtual Colour shade(ShadeRec& sr) = 0;

    float shadowed(ShadeRec const& sr, std::shared_ptr<Light> const& light);

    bool mTransparent = false;
};

class Light
{
public:
    virtual atlas::math::Vector getDirection(ShadeRec const& sr) = 0;
    virtual atlas::math::Vector getSourcePoint(ShadeRec const& sr) = 0;

    virtual Colour L(ShadeRec& sr);

    void scaleRadiance(float b);

    void setColour(Colour const& c);

protected:
    Colour mColour;
    float mRadiance;
};

// Concrete classes which we can construct and use in our ray tracer

class Sphere : public Shape
{
public:
    Sphere(atlas::math::Point center, float radius);

    bool hit(atlas::math::Ray<atlas::math::Vector> const& ray,
        ShadeRec& sr) const;

    std::vector<Point> getBoundingBoxPoints();
private:
    bool intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
        float& tMin) const;

    atlas::math::Point mCentre;
    float mRadius;
    float mRadiusSqr;
};

class Triangle : public Shape
{
public:
    Triangle(Point p0, Point p1, Point p2, Vector2 p0UV = { 0,0 }, Vector2 p1UV = { 0,0 }, Vector2 p2UV = { 0,0 });

    bool hit(atlas::math::Ray<atlas::math::Vector> const& ray,
        ShadeRec& sr) const;

    Vector getV0Point() const;
    Vector getV1Point() const;
    Vector getV2Point() const;

    std::vector<Point> getBoundingBoxPoints();
private:
    bool intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
        float& tMin) const;
    bool intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
        float& tMin, Vector& barycentricCoords) const;

    Point mV0, mV1, mV2;
    Vector2 mUV0, mUV1, mUV2;
};

class MultiMesh : public Shape
{
public:
    MultiMesh(atlas::utils::ObjMesh const& mesh, std::string modelSubDirName = "", Vector offset = { 0,0,0 });

    bool hit(atlas::math::Ray<atlas::math::Vector> const& ray,
        ShadeRec& sr) const;

    std::vector<Point> getBoundingBoxPoints();
private:
    bool intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
        float& tMin) const;

    std::vector<Mesh> meshes;
    std::unique_ptr<BoundingVolumeBox> mBoundVolumeBox;
};

class Mesh : public Shape
{
public:
    Mesh(atlas::utils::ObjMesh const& mesh, std::string modelSubDirName = "", Vector offset = { 0,0,0 });
    Mesh(atlas::utils::Shape shape, unsigned int matIndex, const std::vector<std::shared_ptr<Textured>>& loadedMaterials, std::string modelSubDirName, Vector offset = { 0,0,0 });

    bool hit(atlas::math::Ray<atlas::math::Vector> const& ray,
        ShadeRec& sr) const;

    std::vector<Point> getBoundingBoxPoints();
private:
    bool intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
        float& tMin) const;

    std::vector<Triangle> mMeshTriangles;
    std::unique_ptr<BoundingVolumeBox> mBoundVolumeBox;
};

class Plane : public Shape
{
public:
    Plane(Point point, Vector normal);

    bool hit(atlas::math::Ray<atlas::math::Vector> const& ray,
        ShadeRec& sr) const;

    void setPoint(Point const& point);

    const Vector& getPoint() const;
private:
    bool intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
        float& tMin) const;

    Point mPoint;
    Vector mNormal;
};

class BoundingVolumeBox
{
public:
    BoundingVolumeBox(Vector const& startPoint);

    void addVolumePoint(Point const& point);

    bool intersects(atlas::math::Ray<Vector> const& ray);

    std::vector<Point> getBoundingBoxPoints();
    Point getCentroid();
private:
    Plane mXLeftPlane;
    Plane mXRightPlane;
    Plane mYLeftPlane;
    Plane mYRightPlane;
    Plane mZLeftPlane;
    Plane mZRightPlane;

    float xMin;
    float xMax;
    float yMin;
    float yMax;
    float zMin;
    float zMax;

    Point mCentroid;

    void updateMinMaxCentroid();
};

class BVHAccel : public Shape
{
public:
    BVHAccel();

    void addShape(std::vector<Point> boundingBoxPoints, std::shared_ptr<Shape> shape);

    void generateBVH();

    virtual bool hit(atlas::math::Ray<atlas::math::Vector> const& ray,
        ShadeRec& sr) const;

protected:
    virtual bool intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
        float& tMin) const;

private:
    struct BVHNode
    {
        BVHNode(BoundingVolumeBox bvb);

        bool accesed;
        bool leaf; // node or leaf
        std::shared_ptr<Shape> shape;
        BoundingVolumeBox nodeBvb;
        size_t left;
        size_t right;
    };

    bool mDirty;
    bool mGenerated;
    mutable std::vector<BVHNode> mHeirarchy;
};

class Regular : public Sampler
{
public:
    Regular(int numSamples, int numSets);

    void generateSamples();
};

class Jitter : public Sampler
{
public:
    Jitter(int numSamples, int numSets);

    void generateSamples();
};

class Random : public Sampler
{
public:
    Random(int numSamples, int numSets);

    void generateSamples();
};

class Lambertian : public BRDF
{
public:
    Lambertian();
    Lambertian(Colour diffuseColor, float diffuseReflection);

    Colour fn(ShadeRec const& sr,
        atlas::math::Vector const& reflected,
        atlas::math::Vector const& incoming) const override;

    Colour rho(ShadeRec const& sr,
        atlas::math::Vector const& reflected) const override;

    void setDiffuseReflection(float kd);

    void setDiffuseColour(Colour const& colour);

private:
    Colour mDiffuseColour;
    float mDiffuseReflection;
};

class SpecularReflection : public BRDF
{
public:
    SpecularReflection();

    virtual Colour fn(ShadeRec const& sr,
        atlas::math::Vector const& reflected,
        atlas::math::Vector const& incoming) const override;

    virtual Colour rho(ShadeRec const& sr,
        atlas::math::Vector const& reflected) const override;

    void setSpecularCoefficient(float kd);
    void setSpecularExp(int exp);
    void setSpecularColour(Colour const& colour);

private:
    float mSpecularCofficient;
    int mSpecularExp;
    Colour mSpecularColour;
};

class Reflective : public BRDF
{
public:
    Reflective();

    Colour fn(ShadeRec const& sr,
        atlas::math::Vector const& reflected,
        atlas::math::Vector const& incoming) const override;

    Colour rho(ShadeRec const& sr,
        atlas::math::Vector const& reflected) const override;
};

class Transparency : public BRDF
{
public:
    Transparency(float internalRefraction = 1.7f);

    void setIndexOfRefraction(float index);

    Colour fn(ShadeRec const& sr,
        atlas::math::Vector const& reflected,
        atlas::math::Vector const& incoming) const override;

    Colour rho(ShadeRec const& sr,
        atlas::math::Vector const& reflected) const override;

private:
    Vector refraction(ShadeRec const& sr, float& enterIndexOfRefract) const;

    float mIndexOfRefraction;
};

class Matte : public Material
{
public:
    Matte();
    Matte(float kd, float ka, Colour color);

    void setDiffuseReflection(float k);

    void setAmbientReflection(float k);

    void setDiffuseColour(Colour colour);

    Colour shade(ShadeRec& sr) override;

private:
    std::shared_ptr<Lambertian> mDiffuseBRDF;
    std::shared_ptr<Lambertian> mAmbientBRDF;
};

class Specular : public Material
{
public:
    Specular();
    Specular(float ks, float kd, float ka, Colour color);

    void setSpecularCoefficient(float k);
    void setSpecularExp(int k);

    void setDiffuseReflection(float k);

    void setAmbientReflection(float k);

    void setDiffuseColour(Colour colour);

    Colour shade(ShadeRec& sr) override;

private:
    std::shared_ptr<SpecularReflection> mSpecularBRDF;
    std::shared_ptr<Lambertian> mDiffuseBRDF;
    std::shared_ptr<Lambertian> mAmbientBRDF;
};

class Mirror : public Material
{
public:
    Mirror();

    void setAmbientColour(Colour colour);
    void setAmbientReflection(float ka);

    Colour shade(ShadeRec& sr) override;
private:
    std::shared_ptr<Reflective> mReflectiveBRDF;
    std::shared_ptr<SpecularReflection> mSpecularBRDF;
    std::shared_ptr<Lambertian> mAmbientBRDF;
};

class Refraction : public Material
{
public:
    Refraction();

    void setIndexOfRefraction(float index);

    Colour shade(ShadeRec& sr) override;
private:
    std::shared_ptr<Transparency> mTransparencyBRDF;
};

class Textured : public Material
{
public:
    Textured(tinyobj::material_t const& material, std::string const& modelSubDirName = "");

    void setAmbientReflection(float ka);
    void setSpecularReflection(float ks);
    void setSpecularExp(int exp);
    void setSpecularColour(Colour col);

    Colour shade(ShadeRec& sr) override;
    ImageTexture mTexture;
private:
    std::shared_ptr<SpecularReflection> mSpecularBRDF;
    std::shared_ptr<Lambertian> mAmbientBRDF;
};

class Directional : public Light
{
public:
    Directional();
    Directional(atlas::math::Vector const& d);

    void setDirection(atlas::math::Vector const& d);

    atlas::math::Vector getDirection(ShadeRec const& sr) override;
    atlas::math::Vector getSourcePoint(ShadeRec const& sr) override;

private:
    atlas::math::Vector mDirection;
};

class PointLight : public Light
{
public:
    PointLight();
    PointLight(Point const& p);

    atlas::math::Vector getDirection(ShadeRec const& sr) override;
    atlas::math::Vector getSourcePoint(ShadeRec const& sr) override;
private:
    Point mPoint;
};

class Ambient : public Light
{
public:
    Ambient();

    atlas::math::Vector getDirection(ShadeRec const& sr) override;
    atlas::math::Vector getSourcePoint(ShadeRec const& sr) override;

    Colour L(ShadeRec& sr) override;
private:
    unsigned int mOcclusionSamples = 8; // n x n
    float mOcclusionRayDist = 1000;
};