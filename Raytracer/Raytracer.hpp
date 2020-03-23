#pragma once

#include <atlas/core/Float.hpp>
#include <atlas/math/Math.hpp>
#include <atlas/math/Random.hpp>
#include <atlas/math/Ray.hpp>
#include <atlas/utils/LoadObjFile.hpp>

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
class Light;
class Shape;
class Sampler;
class BoundingVolumeBox;

static std::string modelRoot{ "C:/Users/Andrew/Documents/GitHub/Raytracer/Raytracer/Raytracer/models/" };

struct World
{
    std::size_t width, height;
    Colour background;
    std::shared_ptr<Sampler> sampler;
    std::vector<std::shared_ptr<Shape>> scene;
    std::vector<Colour> image;
    std::vector<std::shared_ptr<Light>> lights;
    std::shared_ptr<Light> ambient;
};

struct ShadeRec
{
    Colour color;
    float t;
    atlas::math::Normal normal;
    atlas::math::Ray<atlas::math::Vector> ray;
    std::shared_ptr<Material> material;
    std::shared_ptr<World> world;
    float uCoord;
    float vCoord;
};

class ImageTexture
{
public:
    ImageTexture(std::string const& imageFilePath);

    ColourAlpha getColour(int x, int y);

    ~ImageTexture();
private:
    bool mSet;
    unsigned char* mImage;
    int mWidth, mHeight, mChannels;
};

// Abstract classes defining the interfaces for concrete entities

class Camera
{
public:
    Camera(Point position, Point lookAt, Vector up, float frustrumDist);

    void calculateRay(float x, float y, atlas::math::Ray<Vector>& ray) const;
protected:
    Point mEye;
    Point mLookAt;
    float mFrustrumDist;
    Vector mU, mV, mW;
};

class Sampler
{
public:
    Sampler(int numSamples, int numSets);
    virtual ~Sampler() = default;

    int getNumSamples() const;

    void setupShuffledIndeces();

    virtual void generateSamples() = 0;

    atlas::math::Point sampleUnitSquare();

protected:
    std::vector<atlas::math::Point> mSamples;
    std::vector<int> mShuffledIndeces;

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
};

class Light
{
public:
    virtual atlas::math::Vector getDirection(ShadeRec& sr) = 0;

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
private:
    bool intersectRay(atlas::math::Ray<atlas::math::Vector> const& ray,
        float& tMin) const;

    Point mV0, mV1, mV2;
    Vector2 mUV0, mUV1, mUV2;
    mutable Vector mBarycenCoords;
};

class Mesh : public Shape
{
public:
    Mesh(atlas::utils::ObjMesh const& mesh, std::string modelSubDirName = "");

    bool hit(atlas::math::Ray<atlas::math::Vector> const& ray,
        ShadeRec& sr) const;

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
private:
    Plane mXLeftPlane;
    Plane mXRightPlane;
    Plane mYLeftPlane;
    Plane mYRightPlane;
    Plane mZLeftPlane;
    Plane mZRightPlane;
};

class Regular : public Sampler
{
public:
    Regular(int numSamples, int numSets);

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

class Textured : public Material
{
public:
    Textured(tinyobj::material_t const& material, std::string const& modelSubDirName);

    Colour shade(ShadeRec& sr) override;
private:
    ImageTexture mTexture;
};

class Directional : public Light
{
public:
    Directional();
    Directional(atlas::math::Vector const& d);

    void setDirection(atlas::math::Vector const& d);

    atlas::math::Vector getDirection(ShadeRec& sr) override;

private:
    atlas::math::Vector mDirection;
};

class Ambient : public Light
{
public:
    Ambient();

    atlas::math::Vector getDirection(ShadeRec& sr) override;

private:
    atlas::math::Vector mDirection;
};