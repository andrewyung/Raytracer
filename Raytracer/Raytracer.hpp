#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtx/transform.hpp>
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
class Ray;
struct SurfaceData;

class PointLight : public Light
{
public:
	PointLight(Colour colour = { 0,0,0 }, Point point = { 0,0,0 });

	float getRadiance([[maybe_unused]] SurfaceData& const sd);
	Vector getDirection([[maybe_unused]] SurfaceData& const sd);
	Point getPoint([[maybe_unused]] SurfaceData& const sd);
protected:
	Point point;
};

class DirectionalLight : public Light
{
public:
	DirectionalLight(Colour colour = { 0,0,0 }, Vector direction = { 0,0,0 });
	Vector getDirection([[maybe_unused]] SurfaceData& const sd);
	Point getPoint([[maybe_unused]] SurfaceData& const sd);
protected:
	Vector direction;
};

class AmbientLight : public Light
{
public:
	AmbientLight(Colour colour);

	Vector getDirection([[maybe_unused]] SurfaceData& const sd);
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

	virtual Ray calculateRay(float x, float y);

	const Vector& getmW();
	const Vector& getmV();
	const Vector& getmU();
	const float& getFrustrumDistance();

	const int height = 600;
	const int width = 600;
protected:
	Point mEye;
	Point mLookAt;
	Point mUp;
	Vector mU, mV, mW;
	float mFrustrumDistance = 90.0f;
};

class FishEye : public Camera
{
public:
	Ray calculateRay(float x, float y);

	void setFOV(float fov);
protected:
	float psi_max = 180.0f; //fov
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
	bool shadowed = false;
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

	Colour specular(Vector wi, Vector camDir, SurfaceData& const surfaceData);
protected:
	float specularExp = 10.0f;
	float specularCoefficient = 0.2f;
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