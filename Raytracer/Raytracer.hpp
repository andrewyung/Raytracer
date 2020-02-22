#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp >
#include <limits>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <iostream>

#include <vector>

using Point = glm::vec3;
using Vector = glm::vec3;
using Colour = glm::vec3;

// Declarations
class Material;
class Camera;
class Shape;
class Sampler;
class Light;
struct SurfaceData;

class Light
{
public:
	Light(Colour colour);
	virtual ~Light() = default;

	Colour getColour();
	void setRadiance(float radiance);
	virtual float getRadiance([[maybe_unused]] SurfaceData sd);
	virtual Vector getDirection([[maybe_unused]] SurfaceData sd);
	
protected:
	Colour colour;
	float radiance;
};

class PointLight : public Light
{
public:
	PointLight(Colour colour = { 0,0,0 }, Point point = { 0,0,0 });

	float getRadiance([[maybe_unused]] SurfaceData sd);
	Vector getDirection([[maybe_unused]] SurfaceData sd) override;
protected:
	Point point;
};

class DirectionalLight : public Light
{
public:
	DirectionalLight(Colour colour = { 0,0,0 }, Vector direction = { 0,0,0 });
	Vector getDirection([[maybe_unused]] SurfaceData sd) override;
protected:
	Vector direction;
};

class AmbientLight : public Light
{
public:
	AmbientLight(Colour colour);

	Vector getDirection([[maybe_unused]] SurfaceData sd) override;
};

class Camera
{
public:
	Camera();

	virtual ~Camera() = default;

	void setEye(Point const& eye);

	void setLookAt(Point const& lookAt);

	void setUpVector(Vector const& up);

	void setFrustrumDistance(float frustumDistance);

	void computeUVW();

	const Point& getmEye();

	const Vector& getmW();
	const Vector& getmV();
	const Vector& getmU();
	const float& getFrustrumDistance();
protected:
	Point mEye;
	Point mLookAt;
	Point mUp;
	Vector mU, mV, mW;
	float mFrustrumDistance = 100;
};

struct Ray
{
	Point o;
	Vector d;

	Point eval(float t)
	{
		return o + t * d;
	}
};


struct SurfaceData
{
	std::shared_ptr<Material> material;
	Vector normal{ 0,0,0 };
	float t = FLT_MAX;
	Point intersection{ 0,0,0 };
	std::shared_ptr<std::vector<std::shared_ptr<Light>>> lights;
	std::shared_ptr<Light> ambient;
};

class Material
{
public:
	Material(Colour diffuseColour);
	virtual ~Material() = default;

	virtual Colour shade(Ray& const ray, SurfaceData& const surfaceData) = 0;

protected:
	Colour diffuseColour;
};

class Shape
{
public:
	Shape();
	virtual ~Shape() = default;

	// if t computed is less than the t in sr, it and the color should be updated in sr
	virtual bool hit(Ray const& ray, SurfaceData& data) const = 0;

	void setColour(Colour const& col);
	void setMaterial(std::shared_ptr<Material> material);

	Colour getColour() const;

protected:
	Colour mColour;
	std::shared_ptr<Material> mMaterial;
};

class Matte : public Material
{
public:
	Matte(Colour diffuseColour);
	Colour shade(Ray& const ray, SurfaceData& const surfaceData);
};

// Concrete classes which we can construct and use in our ray tracer

class Sphere : public Shape
{
public:
	Sphere(Point center, float radius);

	bool hit(Ray const& ray, SurfaceData& data) const;

private:
	Point mCentre;
	float mRadius;
	float mRadiusSqr;
};

class Plane : public Shape
{
public:
	Plane(Point point, Vector normal);

	bool hit(Ray const& ray, SurfaceData& data) const;

private:
	Point point;
	Vector normal;
};

class Triangle : public Shape
{
public:
	Triangle(Point p0, Point p1, Point p2);

	bool hit(Ray const& ray, SurfaceData& data) const;

private:
	Point p0, p1, p2;
};

void saveToFile(std::string const& filename,
	std::size_t width,
	std::size_t height,
	std::vector<Colour> const& image);

void populateGeometry(std::vector<Shape>& shapes);

float randomRange(float min, float max);